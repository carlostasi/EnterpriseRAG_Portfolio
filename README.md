# Automotive Enterprise RAG (v1.1.0)

Un sistema RAG (Retrieval-Augmented Generation) di livello Enterprise, progettato per interrogare manuali d'uso e manutenzione automobilistici con precisione. 
Questo sistema può essere applicato a qualsiasi tipo di documento aziendale. L'organizzazione e la pulizia dei file ricopre il ruolo chiave affinchè le informazioni vengano estratte in maniera chiara e semplice.

Questa architettura risolve i problemi classici dei RAG standard (allucinazioni, collisioni semantiche tra marchi diversi e perdita di contesto) implementando pattern avanzati come il **Parent-Child Retrieval** e un **Agentic Router** basato su LLM per il filtraggio preventivo dei metadati.

## Architettura e Core Features

- **🧠 Agentic Routing & Pre-Filtering:** Prima della ricerca vettoriale, un LLM analizza l'intento dell'utente *tenendo conto della cronologia della chat*, estrae il veicolo bersaglio (tramite `Pydantic` Structured Output) e lo confronta con i modelli realmente presenti nel database. Il filtro viene applicato direttamente su MongoDB Atlas, a livello di indice vettoriale. Zero possibilità di mischiare ricambi Ford su una Mercedes.
- **💬 Conversazione naturale:** Saluti, ringraziamenti e domande generiche ("cosa sai fare?") vengono riconosciuti dal router e gestiti senza interrogare il database, risparmiando tempo e token.
- **🔁 Follow-up contestuali:** Le domande di approfondimento ("E ogni quanto cambio l'olio?") vengono riformulate in domande autonome prima della ricerca, mantenendo il veicolo della conversazione. Se il filtro non trova nulla, il sistema ripiega automaticamente su una ricerca globale.
- **📚 Parent-Child Document Retrieval:** Il database è diviso in due. I vettori vengono calcolati su "Figli" piccoli (800 caratteri) per garantire una precisione matematica altissima. Durante la generazione, l'LLM riceve i "Genitori" interi (4000 caratteri) recuperati da un DocStore separato, garantendo procedure discorsive e mai tagliate a metà.
- **🛡️ Strict Source Tracking:** Il sistema non si limita a stampare i documenti estratti dal DB, ma mostra solo le fonti coerenti con il veicolo richiesto o citato nella risposta, con il numero di pagina reale del manuale.
- **⚡ Streaming Responses:** Backend FastAPI asincrono integrato con un frontend Streamlit per risposte in tempo reale stile ChatGPT.
- **🔍 Debug trasparente:** Ogni passaggio (router, filtro, query usata, documenti recuperati) viene loggato; con `RAG_DEBUG=true` si attiva anche il tracing completo di LangChain.

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI
- **Frontend:** Streamlit
- **AI & Orchestration:** LangChain, AWS Bedrock (Claude Sonnet 4.6 per LLM, Amazon Titan Text V2 per Embeddings)
- **Database:** MongoDB Atlas (Vector Search + Nativo Document Store)
- **DevOps:** Docker, Docker Compose, Makefile

## 📁 Struttura del Progetto

```text
├── inputs/                       # Cartella per i PDF (es. /Ford/Focus.pdf)
├── src/
│   ├── popola_db.py              # Script di Data Ingestion e arricchimento metadati
│   ├── create_vector_index.py    # Crea l'indice vettoriale su MongoDB Atlas
│   ├── api.py                    # Backend FastAPI con logica RAG
│   └── frontend.py               # Frontend Streamlit
├── Dockerfile                    # Container del backend
├── Dockerfile.frontend           # Container del frontend
├── docker-compose.yml            # Avvia backend e frontend insieme
├── Makefile                      # Scorciatoie per tutti i comandi del progetto
├── requirements.txt              # Dipendenze Python (versioni fissate)
└── .env.example                  # Template per le variabili d'ambiente
```

## 🚀 Guida all'Installazione Step-by-Step

Questa guida ti accompagnerà dalla clonazione del repository fino all'avvio dell'interfaccia chat nel tuo browser.

### 📋 Prerequisiti
Prima di iniziare, assicurati di avere installato e configurato:
- **Python 3.11+**
- **Docker** e Docker Compose
- **GNU Make** (opzionale ma consigliato; su Windows: `winget install ezwinports.make`)
- Un account **MongoDB Atlas** (tier gratuito M0 va benissimo) con la feature *Vector Search* abilitata.
- Un account **AWS** con l'accesso abilitato ai modelli **Claude Sonnet 4.6** e **Titan Text Embeddings v2** tramite *Amazon Bedrock*.

---

### ⚙️ 1. Setup dell'Ambiente

**Clona il repository**
```bash
git clone https://github.com/carlostasi/EnterpriseRAG_Portfolio.git
cd EnterpriseRAG_Portfolio
```
### ⚙️ 2. Crea l'ambiente virtuale (consigliato)
**Con Make (qualsiasi sistema)** — crea il venv e installa le dipendenze, puoi saltare lo step 3
```bash
make install
```
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
Copia `.env.example` in un file chiamato `.env` nella cartella principale del progetto e compila i campi necessari

### ⚙️ 5. Caricamento dati
1. Crea una cartella chiamata inputs/ nella root del progetto.

2. Inserisci i tuoi manuali in formato PDF all'interno di sottocartelle per marca (es. inputs/Ford/Focus.pdf, inputs/Honda/Civic.pdf). Il sistema usa questi percorsi per generare i metadati.

3. Lancia lo script di popolamento e poi crea l'indice vettoriale sulla collezione dei manuali figli (senza indice ogni ricerca restituisce zero risultati):
```bash
python src/popola_db.py
python src/create_vector_index.py
```
Oppure, con Make, entrambi in un colpo solo:
```bash
make setup-db
```
### ⚙️ 6. Avvio dell'applicazione
**Con Docker (consigliato)** — avvia backend e frontend in background
```bash
docker compose up -d --build     # oppure: make up
```
- Chat: http://localhost:8501
- API: http://localhost:8000

Per fermare tutto: `docker compose down` (oppure `make down`).

**In locale (sviluppo)** — il backend si ricarica automaticamente a ogni modifica del codice
```bash
make dev
```
Oppure in due terminali separati:
```bash
uvicorn src.api:app --reload --port 8000
streamlit run src/frontend.py
```

### 🧰 Comandi Make disponibili
Lancia `make` per vedere la lista completa.

| Comando | Descrizione |
|---|---|
| `make install` | Crea il venv e installa le dipendenze |
| `make api` / `make frontend` / `make dev` | Avvia backend, frontend o entrambi in locale |
| `make ingest` / `make index` / `make setup-db` | Popola il database, crea l'indice vettoriale o fa entrambe le cose |
| `make build` / `make up` / `make down` / `make restart` | Gestione dei container Docker |
| `make rebuild-api` | Ricostruisce e riavvia solo il backend |
| `make logs` / `make logs-api` / `make logs-frontend` / `make ps` | Log e stato dei container |

### 🐞 Debug
Imposta `RAG_DEBUG=true` nel `.env` per attivare il tracing completo di LangChain. I log di router, filtro e ricerca sono sempre visibili nel terminale del backend (o con `make logs-api` se usi Docker).

