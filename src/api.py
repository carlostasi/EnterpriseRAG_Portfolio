import os
import re
import secrets
import unicodedata
from typing import List, Dict, Optional

import boto3
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.responses import StreamingResponse
from fastapi.security.api_key import APIKeyHeader

from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.storage.mongodb import MongoDBStore
from pymongo import MongoClient

load_dotenv()

# RAG_DEBUG=true stampa input e output di ogni passaggio della catena LangChain (molto verboso)
if os.environ.get("RAG_DEBUG", "").lower() == "true":
    from langchain_core.globals import set_debug
    set_debug(True)

# Inizializzo il server FastAPI
app = FastAPI(title="Enterprise RAG API")
API_KEY = os.environ.get("API_KEY")
api_key_header = APIKeyHeader(name="Portfolio-API-Key")

def verifica_api_key(api_key: str = Security(api_key_header)):
    # compare_digest evita timing attack sul confronto della chiave
    if not API_KEY or not secrets.compare_digest(api_key, API_KEY):
        raise HTTPException(status_code=403, detail="Accesso negato.")
    return api_key

print("Avvio del motore RAG...")

# Setup AWS e LangChain
bedrock_client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
embeddings = BedrockEmbeddings(client=bedrock_client, model_id="amazon.titan-embed-text-v2:0")

llm = ChatBedrock(
    client=bedrock_client,
    model="global.anthropic.claude-sonnet-4-6",
    temperature=0,
    max_tokens=4096,
)

MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI:
    raise ValueError("ERRORE: manca la variabile d'ambiente MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["enterprise_rag"]

# ==========================================
# SETUP DATABASE: VETTORI E DOCSTORE
# ==========================================
# 1. Il Vector Store (I Figli)
collection_vettori = db["Manuali_Child"]
vectorStore = MongoDBAtlasVectorSearch(
    collection=collection_vettori,
    embedding=embeddings,
    index_name="vector_index",
    relevance_score_fn="cosine",
    text_key="text",
    embedding_key="embedding",
)

# 2. Il Doc Store (I Genitori)
store = MongoDBStore(
    db_name="enterprise_rag",
    connection_string=MONGO_URI,
    collection_name="Manuali_Parent")

# 3. Il ParentDocumentRetriever richiede il child splitter anche in sola lettura
child_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)


# ==========================================
# ROUTER: RICONOSCIMENTO DEL VEICOLO
# ==========================================
def normalizza(testo: str) -> str:
    """'GLE_Coupé' -> 'gle coupe'. Stessa forma usata da popola_db.py per il campo 'model'."""
    testo = unicodedata.normalize("NFKD", testo).encode("ascii", "ignore").decode()
    return re.sub(r"[\s_\-]+", " ", testo).strip().lower()

# Catalogo dei modelli realmente presenti nel DB: il router può scegliere solo tra questi
MODELLI_DISPONIBILI = sorted(m for m in collection_vettori.distinct("model") if m)
print(f"[ROUTER] Veicoli nel database: {MODELLI_DISPONIBILI}")

def risolvi_veicolo(veicolo: Optional[str]) -> Optional[str]:
    """Mappa l'output dell'LLM (es. 'Honda Civic', 'GLE Coupé') su un valore esatto del catalogo."""
    if not veicolo:
        return None
    veicolo = normalizza(veicolo)
    if veicolo in MODELLI_DISPONIBILI:
        return veicolo
    candidati = [m for m in MODELLI_DISPONIBILI if m in veicolo or veicolo in m]
    return candidati[0] if len(candidati) == 1 else None

class AnalisiDomanda(BaseModel):
    serve_ricerca: bool = Field(
        description="True se per rispondere servono informazioni dai manuali. False per saluti, ringraziamenti, chiacchiere o domande sull'assistente stesso."
    )
    veicolo_rilevato: Optional[str] = Field(
        description="Il modello del veicolo, scritto esattamente come nell'elenco dei veicoli disponibili. Null se la domanda è generica o il veicolo non è in elenco."
    )

router_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Il tuo compito è analizzare l'ultimo messaggio dell'utente.\n"
     "1. Decidi se per rispondere bisogna consultare i manuali dei veicoli. Saluti ('ciao'), "
     "ringraziamenti, chiacchiere o domande su cosa sai fare NON richiedono ricerca; "
     "qualunque domanda tecnica o sui veicoli sì. Nel dubbio, scegli di cercare.\n"
     "2. Capisci a quale veicolo si riferisce. Veicoli disponibili nel database: {modelli}.\n"
     "Usa anche la cronologia: se la domanda è un follow-up (es. 'e quanto olio serve?') "
     "il veicolo è quello di cui si stava parlando.\n"
     "Restituisci il modello scritto esattamente come nell'elenco, oppure null se la domanda "
     "è generica o riguarda un veicolo non presente."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
]).partial(modelli=", ".join(MODELLI_DISPONIBILI) or "nessuno")

router_chain = router_prompt | llm.with_structured_output(AnalisiDomanda)

# Risposta diretta, senza database, per i messaggi che non richiedono ricerca
chat_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Sei un assistente tecnico automobilistico che risponde consultando i manuali d'uso di questi veicoli: {modelli}.\n"
     "L'utente ha scritto un messaggio che non richiede di consultare i manuali (es. un saluto). "
     "Rispondi in modo breve e cordiale nella sua lingua e, se opportuno, invitalo a fare una domanda sui veicoli. "
     "NON fornire informazioni tecniche: quelle vanno sempre prese dai manuali."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
]).partial(modelli=", ".join(MODELLI_DISPONIBILI) or "nessuno")

chat_chain = chat_prompt | llm | StrOutputParser()


# ==========================================
# PROMPT GLOBALI
# ==========================================
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Data la cronologia della chat e l'ultima domanda dell'utente, che potrebbe fare riferimento "
     "al contesto della cronologia, formula una domanda autonoma comprensibile senza la cronologia. "
     "Se nella conversazione è stato menzionato un veicolo, includi esplicitamente marca e modello.\n"
     "NON rispondere alla domanda: restituisci solo la domanda riformulata, "
     "oppure la domanda così com'è se è già autonoma."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

sys_prompt = (
    "Sei un esperto assistente tecnico automobilistico, preciso e affidabile.\n"
    "Rispondi usando ESCLUSIVAMENTE i frammenti dei manuali riportati nel CONTESTO qui sotto e la cronologia della conversazione.\n"
    "REGOLA D'ORO: ogni frammento indica il manuale da cui proviene. Quando rispondi cita in modo naturale "
    "il veicolo o il manuale che stai usando (es. 'Secondo il manuale della Honda Civic...'), "
    "ma NON scrivere MAI stringhe grezze come 'FONTE: Civic.pdf' o percorsi di file.\n"
    "PERICOLO CRITICO: NON mescolare MAI informazioni di veicoli o marche diverse. "
    "Se il contesto riguarda un veicolo diverso da quello richiesto, dichiaralo esplicitamente.\n"
    "Se il contesto non contiene le informazioni necessarie, dillo chiaramente: non inventare MAI nulla. "
    "Se una procedura si interrompe nel frammento, fermati e scrivi: 'La procedura si interrompe qui.'\n"
    "Rispondi nella lingua della domanda, in modo professionale; usa elenchi numerati per le procedure.\n\n"
    "CONTESTO TROVATO NEL DATABASE:\n"
    "{context}"
)

user_prompt = ChatPromptTemplate.from_messages([
    ("system", sys_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

document_separator_prompt = PromptTemplate.from_template(
    "--- INIZIO FONTE: {source} (Pagina {page_number}) ---\n{page_content}\n--- FINE FONTE ---\n"
)

# Il blocco fonti viene accodato alla risposta: lo togliamo dalla history prima di ridarla all'LLM
MARCATORE_FONTI = "\n\n**📚 Fonti utilizzate:**\n"

def aggiungi_numero_pagina(docs: List[Document]) -> List[Document]:
    # PyPDF numera le pagine da 0: per l'utente (e per l'LLM) partiamo da 1
    for doc in docs:
        pagina = doc.metadata.get("page")
        doc.metadata["page_number"] = pagina + 1 if isinstance(pagina, int) else "N/A"
    return docs

def crea_retriever(pre_filter: Optional[dict]) -> ParentDocumentRetriever:
    search_kwargs = {"k": 4}  # 4 figli, deduplicati in al massimo 4 genitori
    if pre_filter:
        search_kwargs["pre_filter"] = pre_filter
    return ParentDocumentRetriever(
        vectorstore=vectorStore,
        docstore=store,
        child_splitter=child_splitter,
        search_kwargs=search_kwargs,
    )

# Definizione API
class UserQuestion(BaseModel):
    text: str
    history: List[Dict[str, str]] = []


# ==========================================
# ENDPOINT STREAMING (usato da Streamlit)
# ==========================================
@app.post("/chat_stream")
async def chat_with_docs_stream(domanda: UserQuestion, chiave: str = Depends(verifica_api_key)):
    async def generatore_risposta():
        try:
            chat_history = []
            for msg in domanda.history:
                contenuto = msg.get("content", "").split(MARCATORE_FONTI)[0]
                if msg.get("role") == "user": chat_history.append(HumanMessage(content=contenuto))
                elif msg.get("role") == "assistant": chat_history.append(AIMessage(content=contenuto))

            # --- FASE 1: ROUTER INTELLIGENTE (vede anche la cronologia) ---
            print(f"[ROUTER] Analizzo la domanda: '{domanda.text}'")
            analisi = await router_chain.ainvoke({"input": domanda.text, "chat_history": chat_history})

            # Nel dubbio (analisi fallita) si cerca comunque
            if analisi and not analisi.serve_ricerca:
                print("[ROUTER] Messaggio conversazionale: rispondo senza consultare il database.")
                async for testo in chat_chain.astream({"input": domanda.text, "chat_history": chat_history}):
                    yield testo
                return

            car_target = risolvi_veicolo(analisi.veicolo_rilevato if analisi else None)

            # --- FASE 2: COSTRUZIONE FILTRO ---
            if car_target:
                print(f"[FILTRO] Veicolo rilevato: {car_target}.")
                retriever = crea_retriever({"model": {"$eq": car_target}})
            else:
                print("[FILTRO] Nessun veicolo rilevato. Ricerca globale.")
                retriever = crea_retriever(None)

            def recupera_documenti(query: str) -> List[Document]:
                # Con la cronologia, qui arriva la domanda già riformulata dall'LLM
                print(f"[RICERCA] Query usata: '{query}'")
                docs = retriever.invoke(query)
                if not docs and car_target:
                    # Il filtro non ha trovato nulla: meglio un contesto globale che nessun contesto
                    print(f"[FILTRO] 0 risultati per '{car_target}', ripiego sulla ricerca globale.")
                    docs = crea_retriever(None).invoke(query)
                docs = aggiungi_numero_pagina(docs)
                print(f"[RICERCA] {len(docs)} documenti genitori recuperati:")
                for doc in docs:
                    anteprima = doc.page_content[:80].replace("\n", " ")
                    print(f"   - {doc.metadata.get('model')} p.{doc.metadata['page_number']} | {anteprima}...")
                return docs

            # --- FASE 3: ASSEMBLAGGIO CATENA ---
            history_aware_retriever = create_history_aware_retriever(llm, RunnableLambda(recupera_documenti), contextualize_q_prompt)
            question_answer_chain = create_stuff_documents_chain(llm, user_prompt, document_prompt=document_separator_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            # --- FASE 4: ESECUZIONE E STREAMING ---
            completed_text = ""
            fonti_grezze = []

            async for chunk in rag_chain.astream({"input": domanda.text, "chat_history": chat_history}):
                if "answer" in chunk:
                    completed_text += chunk["answer"]
                    yield chunk["answer"]

                if "context" in chunk:
                    fonti_grezze = chunk["context"]

            # --- FASE 5: FILTRAGGIO FONTI FINALI ---
            # Una fonte è "usata" se è del veicolo filtrato oppure se la risposta nomina il suo modello
            testo_normalizzato = normalizza(completed_text)
            fonti_reali = set()
            for doc in fonti_grezze:
                nome_file = doc.metadata.get("source", "Sconosciuto").replace("\\", "/").split("/")[-1]
                modello = normalizza(doc.metadata.get("model") or nome_file.rsplit(".", 1)[0])
                if modello == car_target or modello in testo_normalizzato:
                    fonti_reali.add(f"- {nome_file} (Pagina {doc.metadata.get('page_number', 'N/A')})")

            if fonti_reali:
                yield MARCATORE_FONTI + "\n".join(sorted(fonti_reali))

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"\n\nErrore durante lo streaming: {str(e)}"

    return StreamingResponse(generatore_risposta(), media_type="text/event-stream")
