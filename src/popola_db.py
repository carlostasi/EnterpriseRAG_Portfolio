import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.storage.mongodb import MongoDBStore
from langchain_classic.retrievers import ParentDocumentRetriever
from pymongo import MongoClient
import boto3
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import datetime

def time_log():
    return datetime.now().strftime("[%H:%M:%S]")    
load_dotenv()
# CONF
MONGO_URI = os.environ.get("MONGO_URI")

if not MONGO_URI:
    raise ValueError("ERRORE: manca la stringa di connessione MONGO_URI")


print(f"{time_log()} Connessione a MongoDB in corso...")
client = MongoClient(MONGO_URI)
db = client["enterprise_rag"]

os.makedirs("inputs", exist_ok=True)

print(f"\n{time_log()} Leggo i PDF...")
loader = PyPDFDirectoryLoader("inputs")
documenti = loader.load()

print(f"{time_log()} Trovate {len(documenti)} pagine di PDF")


print(f"{time_log()} Creo i metadata...")
for doc in documenti:
    # Esempio path: "inputs/Mercedes/Classe_A.pdf"
    path = doc.metadata["source"].replace("\\", "/") # Normalizziamo per Windows/Linux
    parti = path.split("/")
    
    # La marca è solitamente la cartella (es. Mercedes)
    marca = parti[-2].lower() if len(parti) >= 2 else "generico"
    
    # Il modello è il nome del file pulito (es. Classe A)
    nome_file = parti[-1].replace(".pdf", "").replace(".PDF", "")
    modello = nome_file.replace("_", " ").replace("-", " ").lower()
    
    # Aggiungiamo i metadati strutturati
    doc.metadata["brand"] = marca
    doc.metadata["model"] = modello
# Chunking
print(f"{time_log()} Chunking...")
# Inizio embedding
embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
# Proviamo ad usare i nuovi splitter per genitori e figli
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

# Creiamo il nuovo VectorStore (dove vivranno i child)
collection_vettori = db["Manuali_Child"]
vectorStore = MongoDBAtlasVectorSearch(
    collection=collection_vettori,
    embedding=embeddings,
    index_name="vector_index"
)

# Inizializziamo il DocStore (dove vivranno i parent)
store = MongoDBStore(
    connection_string=MONGO_URI,
    db_name="enterprise_rag",
    collection_name="Manuali_Parent"
)

retriever = ParentDocumentRetriever(
    vectorstore = vectorStore,
    docstore = store,
    child_splitter = child_splitter,
    parent_splitter = parent_splitter
)
print(f"\n{time_log()} Caricamento su MongoDB... (potrebbe volerci molto)")
batch_size = 20
total_docs = len(documenti)


for i in tqdm(range(0, total_docs, batch_size), desc="Upload Progress"):
    batch = documenti[i : i + batch_size]
    retriever.add_documents(batch)

print(f"\n{time_log()} Caricamento su MongoDB completato!")