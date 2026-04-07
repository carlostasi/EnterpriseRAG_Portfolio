import os 
from langchain_community.document_loaders import S3DirectoryLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch, MongoDBByteStore
from langchain_core.retrievers import ParentDocumentRetriever
from pymongo import MongoClient
import boto3
from dotenv import load_dotenv
from tqdm import tqdm
import dns.resolver

load_dotenv()
# CONF
MONGO_URI = os.environ.get("MONGO_URI")
BUCKET_NAME = "portfolio-rag-documenti-carlo"

if not MONGO_URI:
    raise ValueError("ERRORE: manca la stringa di connessione MONGO_URI")

print(f"Connessione a MongoDB in corso...")
client = MongoClient(MONGO_URI)
db = client["enterprise_rag"]

os.makedirs("temp_pdf", exist_ok=True)

# print("\nSto resettando MongoDB...")
# db["Procedure"].delete_many({})
loader = PyPDFDirectoryLoader("temp_pdf")
documenti = loader.load()
print(f"Trovate {len(documenti)} pagine di PDF su S3.")

# Chunking
print("Chunking...")
# text_splitter = RecursiveCharacterTextSplitter(chunk_size= 1500, chunk_overlap=250)

# chunks = text_splitter.split_documents(documenti)

# Inizio embedding
embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
batch_size = 500
# total_chunks = len(chunks)

vector_search = MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=db["Procedure"],
    index_name="vector_index"
)

print("Caricamento su MongoDB... (potrebbe volerci molto)")

# for i in tqdm(range(0, total_chunks, batch_size), desc="Upload Progress"):
#     batch = chunks[i : i + batch_size]
#     vector_search.add_documents(documents=batch)


# Proviamo ad usare i nuovi splitter per genitori e figli
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200)

child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

# Creiamo il nuovo VectorStore (dove vivranno i child)
collection_vettori = db["Procedure_ParentChild"]
vectorStore = MongoDBAtlasVectorSearch(
    collection=collection_vettori,
    embedding=embeddings,
    index_name="vector_index_2"
)

# Inizializziamo il DocStore (dove vivranno i parent)
store = MongoDBByteStore(
    connection_string=MONGO_URI,
    db_name="enterprise_rag",
    collection_name="Procedure_Genitori"
)

retriever = ParentDocumentRetriever(
    vectorstore = vectorStore,
    docstore = store,
    child_splitter = child_splitter,
    parent_splitter = parent_splitter
)
retriever.add_documents(documenti)


print("Caricamento su MongoDB completato!")