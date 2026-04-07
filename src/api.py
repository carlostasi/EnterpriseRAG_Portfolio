import boto3
import os
import json
from typing import List, Dict
from pydantic import ValidationError, BaseModel
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.responses import StreamingResponse
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from fastapi.security.api_key import APIKeyHeader
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# Inizializzo il server FastAPI
app = FastAPI(title="Enterprise RAG API", description="API custom per interrogare documenti aziendali tramite AWS Bedrock")
API_KEY = os.environ.get("API_KEY")
api_key_header = APIKeyHeader(name="Portfolio-API-Key")

def verifica_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Accesso negato. Chiave API errata." )
    return api_key
print("Avvio del motore RAG...")

# Setup AWS e LangChain
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

embeddings = BedrockEmbeddings(
    client=bedrock_client,
    model_id="amazon.titan-embed-text-v2:0"
)

llm = ChatBedrock(
    client=bedrock_client,
    model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
    # model_kwargs={"temperature": 0}
    temperature=0,
    max_tokens=4096
)
MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI:
    raise ValueError("ERRORE: Manca la variabile d'ambiente MONGO_URI")

# Ci connettiamo al cloud
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["enterprise_rag"]
collection = db["Procedure"]

# Carico dalla memoria
vectorStore = MongoDBAtlasVectorSearch(
    collection = collection,
    embedding=embeddings,
    index_name="vector_index",
    relevance_score_fn="cosine",
    text_key="text",
    embedding_key="embedding"
)
retriever = vectorStore.as_retriever(
    search_type="mmr", # Attiva la logica di diversificazione
    search_kwargs={
        "k": 10,
        "fetch_k": 30, # Numeri di documenti iniziali che prende da Mongo,
        "lambda_mult": 0.5 # è il bilanciamento perfetto tra "simile alla domanda" e "diverso dagli altri chunk"
    }
    # search_type="similarity",
    # search_kwargs={
    #     "k": 15
    # }
)

# PROMPT
contextualize_q_system_prompt = (
    "Data una cronologia della chat e l'ultima domanda dell'utente"
    "che potrebbe fare riferimento al contesto nella cronologia della chat,"
    "formula una domanda autonoma che possa essere compresa senza la cronologia."
    "NON rispondere alla domanda, riformulala solo se necessario per la ricerca, altrimenti restiuiscila cosi com'è."
)

"""Questo template serve esclusivamente per parlare con il database. Non si interessa di rispondere all'utente.
Passa la cronologia e l'ultima domanda a Claude e si riscrive la domanda ottimizzata per la ricerca
 """
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt), # Regole e premesse generali 
    MessagesPlaceholder("chat_history"), # Passiamo la cronologia
    ("human", "{input}") # Prompt dell'utente alla fine
])

"""
input: LLM, database, prompt_contesto
output: Rielabora la domanda aggiungendo eventuale contesto ricavato dalla cronologia
"""
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

sys_prompt = (
    "Sei un assistente aziendale super preciso."
    "Usa i seguenti frammenti di contesto recuperati QUI SOTTO e la CRONOLOGIA della chat per rispondere alla domanda dell'utente."
    "ATTENZIONE: Se ti viene chiesta una procedura passo-passo, assicurati che i passaggi appartengano allo stesso documento logico."
    "NON mescolare mai procedure di file o macchinari diversi. Se il contesto si interrompe prima della fine della procedura, scrivi:"
    "'La procedura nei miei documenti si interrompe qui.'"
    "Se l'utente fa riferimento a informazioni già discusse nella cronologia, usale. "
    "Se non conosci la risposta basandoti su questi die elementi, dì semplicemente che non lo sai."
    "Non inventare MAI informazioni fuori dal contesto o dalla cronologia. Sii conciso e professionale.\n\n"
    "CONTESTO TROVATO NEL DATABASE:\n"
    "{context}"
)

""""
Questo template genera la risposta finale. Prende i frammenti di testo che tornano dalla ricerca semantica nel db, li aggiunge alla cronologia,
prende la domanda originale e prepara il pacchetto completo con il prompt dell'utente
"""
user_prompt = ChatPromptTemplate.from_messages([
    ("system", sys_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Creazione della PIPELINE RAG
# Questa catena dice a Claude come leggere i documenti.
# Prende i 5 frammenti di pdf dal db e li estrae mettendoli nel context
"""
input: LLM, user_prompt
output: prende i 5 frammenti che 'history_aware_retriever' porta, ne estrae il testo puro e lo mette nel {context} del sys prompt.
Crea il super pacchetto da mandare a Claude per generare la risposta finale
"""
question_answer_chain = create_stuff_documents_chain(llm, user_prompt)
# Questa catena unisce la ricerca nel database con la generazione della risposta
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Definizione struttura dati API in ingresso
class UserQuestion(BaseModel):
    text: str
    history: List[Dict[str, str]] = [] # Memoria della chat

# Creo l'endpoint
@app.post("/chat")
async def chat_with_docs(domanda: UserQuestion, chiave: str = Depends(verifica_api_key)):
    try:
        # Arriva una richiesta e uso il RAG per rispondere
        chat_history = []
        for msg in domanda.history:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))
        result = rag_chain.invoke({
            "input": domanda.text,
            "chat_history": chat_history
            })
        fonti_unificate = set()
        for doc in result["context"]:
            path = doc.metadata.get('source', 'Sconosciuto')
            nome_file = os.path.basename(path)
            pagina = doc.metadata.get('page', 'N/A')
            fonti_unificate.add(f"- {nome_file} (Pagina {pagina})")

        return {
            "status": "success",
            "ai_answer": result["answer"],
            "fonti": list(fonti_unificate)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat_stream")
async def chat_with_docs_stream(domanda: UserQuestion, chiave: str = Depends(verifica_api_key)):
    async def generatore_risposta():
        try:
            chat_history = []
            for msg in domanda.history:
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history.append(AIMessage(content=msg["content"]))
            
            # Usiamo .astream() invece di .invoke()
            fonti_unificate = set()
            async for chunk in rag_chain.astream({
                "input": domanda.text,
                "chat_history": chat_history
            }):
                # LangChain divide il flusso in pezzi. Cerchiamo la risposta finale.
                if "answer" in chunk:
                    # Inviamo il pezzettino di testo al frontend
                    yield chunk["answer"]
                
                # Raccogliamo le fonti man mano che passano nel flusso
                if "context" in chunk:
                    for doc in chunk["context"]:
                        path = doc.metadata.get('source', 'Sconosciuto')
                        nome_file = os.path.basename(path)
                        pagina = doc.metadata.get('page', 'N/A')
                        fonti_unificate.add(f"- {nome_file} (Pagina {pagina})")
            
            # Alla fine del flusso, mandiamo le fonti formattate
            if fonti_unificate:
                yield "\n\n**📚 Fonti utilizzate:**\n" + "\n".join(fonti_unificate)

        except Exception as e:
            yield f"Errore durante lo streaming: {str(e)}"

    # Restituiamo un flusso continuo di testo
    return StreamingResponse(generatore_risposta(), media_type="text/event-stream")
