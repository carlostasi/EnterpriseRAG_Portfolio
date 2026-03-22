from pydantic import ValidationError
from chromadb import HttpClient
from fastapi import FastAPI, HTTPException, Security, Depends
from pydantic import BaseModel
import boto3
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from fastapi.security.api_key import APIKeyHeader
import os
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage

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
    model_kwargs={"temperature": 0}
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
retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

# PROMPT
contextualize_q_system_prompt = (
    "Data una cronologia della chat e l'ultima domanda dell'utente"
    "che potrebbe fare riferimento al contesto nella cronologia della chat,"
    "formula una domanda autonoma che possa essere compresa senza la cronologia."
    "NON rispondere alla domanda, riformulala solo se necessario per la ricerca, altrimenti restiuiscila cosi com'è."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


sys_prompt = (
    "Sei un assistente aziendale super preciso."
    "Usa i seguenti frammenti di contesto recuperati per rispondere alla domanda dell'utente."
    "Se non conosci la risposta basandoti ESCLUSIVAMENTE sul contesto dì semplicemente che non lo sai."
    "Non inventare MAI informazioni. Sii conciso e professionale.\n\n"
    "CONTESTO TROVATO NEL DATABASE:\n"
    "{context}"
)

user_prompt = ChatPromptTemplate.from_messages([
    ("system", sys_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Creazione della PIPELINE RAG
# Questa catena dice a Claude come leggere i documenti
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
        raise HTTPException(status_code=500, detail=str(e))
