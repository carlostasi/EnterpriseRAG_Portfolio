# Automotive Enterprise RAG (v1.0.0)

Un sistema RAG (Retrieval-Augmented Generation) di livello Enterprise, progettato per interrogare manuali d'uso e manutenzione automobilistici con precisione. 
Questo sistema può essere applicato a qualsiasi tipo di documento aziendale. L'organizzazione e la pulizia dei file ricopre il ruolo chiave affinchè le informazioni vengano estratte in maniera chiara e semplice.

Questa architettura risolve i problemi classici dei RAG standard (allucinazioni, collisioni semantiche tra marchi diversi e perdita di contesto) implementando pattern avanzati come il **Parent-Child Retrieval** e un **Agentic Router** basato su LLM per il filtraggio preventivo dei metadati.

## Architettura e Core Features

- **🧠 Agentic Routing & Pre-Filtering:** Prima della ricerca vettoriale, un LLM veloce analizza l'intento dell'utente, estrae il veicolo bersaglio (tramite `Pydantic` Structured Output) e applica un preciso filtro su MongoDB Atlas, filtrando i vettori a livello di database. Zero possibilità di mischiare ricambi Ford su una Mercedes.
- **📚 Parent-Child Document Retrieval:** Il database è diviso in due. I vettori vengono calcolati su "Figli" piccoli (800 caratteri) per garantire una precisione matematica altissima. Durante la generazione, l'LLM riceve i "Genitori" interi (4000 caratteri) recuperati da un DocStore separato, garantendo procedure discorsive e mai tagliate a metà.
- **🛡️ Strict Source Tracking:** Il sistema non si limita a stampare i documenti estratti dal DB, ma verifica in tempo reale quali faldoni l'LLM ha *effettivamente* utilizzato per formulare la risposta, scartando i falsi positivi.
- **⚡ Streaming Responses:** Backend FastAPI asincrono integrato con un frontend Streamlit per risposte in tempo reale stile ChatGPT.

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI, Docker
- **Frontend:** Streamlit
- **AI & Orchestration:** LangChain, AWS Bedrock (Claude Sonnet per LLM, Amazon Titan Text V2 per Embeddings)
- **Database:** MongoDB Atlas (Vector Search + Nativo Document Store)

## 📁 Struttura del Progetto

```text
├── inputs/                # Cartella per i PDF (es. /Ford/Focus.pdf)
├── popola_db.py           # Script di Data Ingestion e arricchimento metadati
├── api.py                 # Backend FastAPI con logica RAG
├── frontend.py            # Frontend Streamlit
├── Dockerfile             # Configurazione container per il backend
├── requirements.txt       # Dipendenze Python
└── .env.example           # Template per le variabili d'ambiente
```

## 🚀 Guida all'Installazione Step-by-Step

Questa guida ti accompagnerà dalla clonazione del repository fino all'avvio dell'interfaccia chat nel tuo browser.

### 📋 Prerequisiti
Prima di iniziare, assicurati di avere installato e configurato:
- **Python 3.10+**
- **Docker** e Docker Compose
- Un account **MongoDB Atlas** (tier gratuito M0 va benissimo) con la feature *Vector Search* abilitata.
- Un account **AWS** con l'accesso abilitato ai modelli **Claude 3 Sonnet** e **Titan Text Embeddings v2** tramite *Amazon Bedrock*.

---

### ⚙️ 1. Setup dell'Ambiente

**Clona il repository**
```bash
git clone https://github.com/carlostasi/EnterpriseRAG_Portfolio.git
cd TUO_REPO
```
### ⚙️ 2. Crea l'ambiente virtuale (consigliato)
**Su Windows**
```bash
python -m venv venv
.\venv\Scripts\activate
```
**Su macOS/Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```
### ⚙️ 3. Installa le dipendenze
```bash
pip install -r requirements.txt
```
### ⚙️ 4. Configura le variabili d'ambiente
Crea un file chiamato .env nella cartella principale del progetto e compila i campi necessari

### ⚙️ 5. Caricamento dati
1. Crea una cartella chiamata inputs/ nella root del progetto.

2. Inserisci i tuoi manuali in formato PDF all'interno di sottocartelle per marca (es. inputs/Ford/Focus.pdf, inputs/Honda/Civic.pdf). Il sistema usa questi percorsi per generare i metadati.

3. Lancia lo script di popolamento:
```bash
python popola_db.py
```
Ricorda di inizializzare su MongoDB un vector search index sulla collezione dei manuali figli
### ⚙️ 6. Avvio dell'applicazione
```bash
docker build -t automotive-rag-api .
docker run -p 8000:8000 --env-file .env automotive-rag-api
```
In un altro terminale
```bash
streamlit run frontend.py
```

