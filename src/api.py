from langchain_classic.retrievers import ParentDocumentRetriever
import boto3
import os
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.responses import StreamingResponse
from fastapi.security.api_key import APIKeyHeader

from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.storage.mongodb import MongoDBStore
from pymongo import MongoClient
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

# Inizializzo il server FastAPI
app = FastAPI(title="Enterprise RAG API")
API_KEY = os.environ.get("API_KEY")
api_key_header = APIKeyHeader(name="Portfolio-API-Key")

def verifica_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Accesso negato.")
    return api_key

print("Avvio del motore RAG...")

# Setup AWS e LangChain
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
embeddings = BedrockEmbeddings(client=bedrock_client, model_id="amazon.titan-embed-text-v2:0")

llm = ChatBedrock(
    client=bedrock_client,
    model="us.anthropic.claude-sonnet-4-20250514-v1:0",
    temperature=0,
    max_tokens=4096,
)

# Struttura per l'estrazione
class EstrazioneVeicolo(BaseModel):
    veicolo_rilevato: Optional[str] = Field(
        description="Il modello esatto del veicolo menzionato (es. 'Civic', 'Focus', 'Jazz'). Se generico, restituisci null."
    )
extract_llm = llm.with_structured_output(EstrazioneVeicolo)

MONGO_URI = os.environ.get("MONGO_URI")
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
collection_genitori = db["Manuali_Parent"]
store = MongoDBStore(
    db_name="enterprise_rag",
    connection_string=MONGO_URI,
    collection_name="Manuali_Parent")

# 3. Splitter (Servono al ParentDocumentRetriever per sapere le dimensioni)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)


# ==========================================
# PROMPT GLOBALI
# ==========================================
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Data una cronologia... (inserisci qui il tuo prompt di contextualize)"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

sys_prompt = "Sei un esperto assistente tecnico... (inserisci qui il tuo system prompt con la regola d'oro)\nCONTESTO:\n{context}"

user_prompt = ChatPromptTemplate.from_messages([
    ("system", sys_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

document_separator_prompt = PromptTemplate.from_template(
    "--- INIZIO FONTE: {source} (Pagina {page}) ---\n{page_content}\n--- FINE FONTE ---\n"
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
                if msg["role"] == "user": chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant": chat_history.append(AIMessage(content=msg["content"]))

            # --- FASE 1: ROUTER INTELLIGENTE ---
            print(f"[ROUTER] Analizzo la domanda: '{domanda.text}'")
            # RISOLTO CORTOCIRCUITO 3: Uso .invoke()
            result_extract = extract_llm.invoke(domanda.text) 
            car_target = result_extract.veicolo_rilevato

            # --- FASE 2: COSTRUZIONE FILTRO ---
            search_kwargs = {"k": 4} # Cerchiamo 4 genitori
            if car_target:
                print(f"[FILTRO] Veicolo rilevato da Claude: {car_target}.")
                # Match esatto sul campo 'model' che abbiamo appena creato!
                search_kwargs["pre_filter"] = {"model": {"$eq": car_target.lower()}}
            else:
                print("[FILTRO] Nessun veicolo rilevato. Ricerca globale.")

            # --- FASE 3: ASSEMBLAGGIO CATENA ---
            # Creiamo il retriever dinamicamente ORA, con il filtro applicato
            filtered_retriever = ParentDocumentRetriever(
                vectorstore=vectorStore,
                docstore=store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
                search_kwargs=search_kwargs
            )

            # Costruiamo la catena ORA, collegandola al retriever filtrato
            history_aware_retriever = create_history_aware_retriever(llm, filtered_retriever, contextualize_q_prompt)
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
            if fonti_grezze:
                fonti_reali = set()
                for doc in fonti_grezze:
                    path = doc.metadata.get("source", "Sconosciuto")
                    nome_file_completo = path.replace("\\", "/").split("/")[-1]
                    parola_chiave = nome_file_completo.replace(".pdf", "").replace(".PDF", "").lower()
                    
                    if parola_chiave in completed_text.lower():
                        pagina = doc.metadata.get('page', 'N/A')
                        fonti_reali.add(f"- {nome_file_completo} (Pagina {pagina})")
                
                if fonti_reali:
                    yield "\n\n**📚 Fonti utilizzate:**\n" + "\n".join(fonti_reali)

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"\n\nErrore durante lo streaming: {str(e)}"

    return StreamingResponse(generatore_risposta(), media_type="text/event-stream")