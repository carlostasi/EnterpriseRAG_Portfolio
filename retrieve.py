import boto3
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma

print("Avvio motore ricerca semantica")

# Riconnetto il modello di embedding
# Devo usare lo stesso modello che ho suato in ingest.py, altrimenti le coordinate non coincidono
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
embeddings = BedrockEmbeddings(
    client=bedrock_client,
    model_id="amazon.titan-embed-text-v2:0"
)

# Mi collego al database
print("Leggo memoria ChromaDB...\n")
vectorStore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# PROPMPT
queryPrompt = "Vorrei sapere come dovrei impostare l'indirizzo ip della macchina LT Fiber"
print(f"Ho chiesto: {queryPrompt}\n")

# Similarity search
results = vectorStore.similarity_search(queryPrompt, k=3)

print("RISULTATI TROVATI:\n")
for i, doc in enumerate(results):
    print(f"--- Rilevanza {i+1} ---")
    print(f"Testo estratto: {doc.page_content}")
    print(f"Metadati: File '{doc.metadata.get('source', 'Sconosciuto')}' - Pagina {doc.metadata.get('page', 'N/A')}\n")