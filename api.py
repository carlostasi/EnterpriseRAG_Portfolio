from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# Inizializzo il server FastAPI
app = FastAPI(title="Enterprise RAG API", description="API custom per interrogare documenti aziendali tramite AWS Bedrock")
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

# Carico dalla memoria
vectorStore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

# PROMPT
sys_prompt = (
    "Sei un assistente aziendale super preciso."
    "Usa i seguenti frammenti di contesto recuperati per rispondere alla domanda dell'utente."
    "Se non conosci la risposta basandoti ESCLUSIVAMENTE sul contesto dì semplicemente che non lo sai."
    "Non inventare MAI informazioni. Sii conciso e professionale.\n\n"
    "CONTESTO TROVATO NEL DATABASE:\n"
    "{context}"
)

propmpt = ChatPromptTemplate.from_messages([
    ("system", sys_prompt),
    ("human", "{input}"),
])

# Creazione della PIPELINE RAG
# Questa catena dice a Claude come leggere i documenti
question_answer_chain = create_stuff_documents_chain(llm, propmpt)
# Questa catena unisce la ricerca nel database con la generazione della risposta
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Definizione struttura dati API in ingresso
class UserQuestion(BaseModel):
    text: str

# Creo l'endpoint
@app.post("/chat")
async def chat_with_docs(domanda: UserQuestion):
    try:
        # Arriva una richiesta e uso il RAG per rispondere
        result = rag_chain.invoke({"input": domanda.text})
        testo_fonti = ""
        for doc in result["context"]:
            riga = f"- File: {doc.metadata.get('source', 'Sconosciuto')} (Pagina {doc.metadata.get('page', 'N/A')})\n"
            testo_fonti += riga

        return {
            "status": "success",
            "ai_answer": result["answer"],
            "fonti": testo_fonti
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
