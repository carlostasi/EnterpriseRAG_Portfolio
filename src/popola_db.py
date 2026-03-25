import os 
from langchain_community.document_loaders import S3DirectoryLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
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
collection = client["enterprise_rag"]["Procedure"]

# Scarico i documenti da S3
# print(f"Scaricando i documenti dal bucket S3 '{BUCKET_NAME}...")

# Langchain ha un loader appsota per S3
# loader = S3DirectoryLoader(BUCKET_NAME)
# s3 = boto3.client("s3")
os.makedirs("temp_pdf", exist_ok=True)
# Prendi la lista dei file su S3 e scaricali
# oggetti = s3.list_objects_v2(Bucket=BUCKET_NAME).get('Contents', [])
# for obj in oggetti:
#     chiave_file = obj['Key']
#     if chiave_file.endswith(".pdf"): # Prende solo i PDF!
#         path_locale = os.path.join("temp_pdf", chiave_file)
#         os.makedirs(os.path.dirname(path_locale), exist_ok=True)
#         s3.download_file(BUCKET_NAME, chiave_file, path_locale)
#         print(f"⬇️ Scaricato: {chiave_file}")

print("\nSto resettando MongoDB...")
collection.delete_many({})
loader = PyPDFDirectoryLoader("temp_pdf")
documenti = loader.load()
print(f"Trovate {len(documenti)} pagine di PDF su S3.")

# Chunking
print("Chunking...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
chunks = text_splitter.split_documents(documenti)

# Inizio embedding
embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
batch_size = 500
total_chunks = len(chunks)

vector_search = MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=collection,
    index_name="vector_index"
)

print("Caricamento su MongoDB... (potrebbe volerci molto)")

# MongoDBAtlasVectorSearch.from_documents(
#     documents=chunks,   
#     embedding=embeddings,
#     collection=collection,
#     index_name="vector_index"
# )
for i in tqdm(range(0, total_chunks, batch_size), desc="Upload Progress"):
    batch = chunks[i : i + batch_size]
    vector_search.add_documents(documents=batch)

print("Caricamento su MongoDB completato!")