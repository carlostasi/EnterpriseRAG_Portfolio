import os, time
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

# Crea (o aggiorna) il Vector Search index sui figli. Va lanciato una volta dopo popola_db.py:
#   python src/create_vector_index.py
load_dotenv()
MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI:
    raise ValueError("ERRORE: manca la stringa di connessione MONGO_URI")

INDEX_NAME = "vector_index"
DEFINIZIONE = {
    "fields": [
        # Titan Text Embeddings v2 produce vettori da 1024 dimensioni
        {"type": "vector", "path": "embedding", "numDimensions": 1024, "similarity": "cosine"},
        # Campi usati dal pre_filter del router in api.py
        {"type": "filter", "path": "model"},
        {"type": "filter", "path": "brand"},
    ]
}

collection = MongoClient(MONGO_URI)["enterprise_rag"]["Manuali_Child"]

esistenti = {idx["name"] for idx in collection.list_search_indexes()}
if INDEX_NAME in esistenti:
    print(f"Indice '{INDEX_NAME}' già presente: aggiorno la definizione...")
    collection.update_search_index(INDEX_NAME, DEFINIZIONE)
else:
    print(f"Creo l'indice '{INDEX_NAME}'...")
    collection.create_search_index(SearchIndexModel(definition=DEFINIZIONE, name=INDEX_NAME, type="vectorSearch"))

# L'indice si costruisce in background: aspettiamo che sia interrogabile
while True:
    indice = next(collection.list_search_indexes(INDEX_NAME), None)
    if indice and indice.get("queryable") and indice.get("status") == "READY":
        print("Indice pronto!")
        break
    print(f"Stato: {indice.get('status') if indice else 'in creazione'}... attendo")
    time.sleep(10)
