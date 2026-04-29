import os, json
from langchain_community.document_loaders import S3DirectoryLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.storage.mongodb import MongoDBByteStore
from langchain_core.documents import Document
from langchain_classic.storage import EncoderBackedStore
from langchain_classic.retrievers import ParentDocumentRetriever
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
# Proviamo ad usare i nuovi splitter per genitori e figli
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

# Creiamo il nuovo VectorStore (dove vivranno i child)
collection_vettori = db["Procedure_ParentChild"]
vectorStore = MongoDBAtlasVectorSearch(
    collection=collection_vettori,
    embedding=embeddings,
    index_name="vector_index_2"
)

# Inizializziamo il DocStore (dove vivranno i parent)
mongo_byte_store = MongoDBByteStore(
    connection_string=MONGO_URI,
    db_name="enterprise_rag",
    collection_name="Procedure_Genitori"
)

store = EncoderBackedStore(
    store=mongo_byte_store,
    key_encoder=lambda x: x,
    value_serializer=lambda doc: json.dumps({"page_content": doc.page_content, "metadata": doc.metadata}).encode("utf-8"),
    value_deserializer=lambda b: Document(**json.loads(b.decode("utf-8")))
)

# vector_search = MongoDBAtlasVectorSearch(
#     embedding=embeddings,
#     collection=db["Procedure"],
#     index_name="vector_index"
# )

retriever = ParentDocumentRetriever(
    vectorstore = vectorStore,
    docstore = store,
    child_splitter = child_splitter,
    parent_splitter = parent_splitter
)
print("Caricamento su MongoDB... (potrebbe volerci molto)")
batch_size = 2
total_docs = len(documenti)


for i in tqdm(range(0, total_docs, batch_size), desc="Upload Progress"):
    batch = documenti[i : i + batch_size]
    retriever.add_documents(batch)

print("Caricamento su MongoDB completato!")