import boto3
from langchain_community.document_loaders import PyPDFLoader        
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma

print("Inizio il processo di Ingestion...\n")

# Mi connetto ad AWS per usare gli embeddings (traduzione da "documento" a vettore)
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
embeddings = BedrockEmbeddings(
    client = bedrock_client,
    model_id="amazon.titan-embed-text-v2:0"
)

# Carico pdf
print("Lettura pdf in corso...")
loader = PyPDFLoader("data/PROCEDURE/08_LT_FIBER-LTFIBER_CO2/LT_FIBER/SERVICE MANUAL_LT_FIBER_v13.01_ITA.pdf")
documenti = loader.load()
print(f"Fatto! Trovate {len(documenti)} pagine\n")

# CHUNKING (Spezzettamento)
print("Chunking del testo...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
frammenti = text_splitter.split_documents(documenti)
print(f"Fatto! Il PDF è stato diviso in {len(frammenti)} frammenti di testo\n")

# EMBEDDING
# print("Procedo a richiamare l'embedding con AWS")
# test_text = frammenti[0].page_content
# vettore_test = embeddings.embed_query(test_text)

vectorStore = Chroma.from_documents(
    documents = frammenti,
    embedding = embeddings,
    persist_directory = "./chroma_db"
)

print(f"Success! Tutti i vettori sono stati salvati nel database")
