# Comandi del progetto. Lancia "make" (o "make help") per la lista.
# Su Windows serve GNU make: winget install ezwinports.make

ifeq ($(OS),Windows_NT)
    PY_SYS := python
    VENV_BIN := venv/Scripts
else
    PY_SYS := python3
    VENV_BIN := venv/bin
endif

PYTHON := $(VENV_BIN)/python
API_PORT := 8000
FRONTEND_PORT := 8501

# $(info) stampa uguale con cmd e sh; $(E) evita che make tolga gli spazi iniziali
E :=

.DEFAULT_GOAL := help
.PHONY: help install api frontend dev ingest index setup-db build up down restart rebuild-api logs logs-api logs-frontend ps

help:
	$(info $(E)Uso: make [comando])
	$(info )
	$(info $(E)  Locale)
	$(info $(E)    install        crea il venv e installa le dipendenze)
	$(info $(E)    api            avvia FastAPI su localhost:$(API_PORT) con reload)
	$(info $(E)    frontend       avvia Streamlit su localhost:$(FRONTEND_PORT))
	$(info $(E)    dev            avvia api e frontend insieme)
	$(info )
	$(info $(E)  Database)
	$(info $(E)    ingest         carica i PDF su MongoDB Atlas)
	$(info $(E)    index          crea indice vettoriale su Manuali_Child)
	$(info $(E)    setup-db       ingest + index)
	$(info )
	$(info $(E)  Docker)
	$(info $(E)    build          builda le immagini)
	$(info $(E)    up             build + avvio dei container in background)
	$(info $(E)    down           ferma e rimuove i container)
	$(info $(E)    restart        down + up)
	$(info $(E)    rebuild-api    ricostruisce e riavvia solo rag-api)
	$(info $(E)    logs           log di tutti i container)
	$(info $(E)    logs-api       log di rag-api)
	$(info $(E)    logs-frontend  log di rag-frontend)
	$(info $(E)    ps             stato dei container)

# --- Locale ---

install:
	$(PY_SYS) -m venv venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

api:
	$(PYTHON) -m uvicorn src.api:app --reload --port $(API_PORT)

frontend:
	$(PYTHON) -m streamlit run src/frontend.py --server.port $(FRONTEND_PORT)

# -j2 lancia i due target in parallelo; Ctrl+C li ferma entrambi
dev:
	$(MAKE) -j2 api frontend

# --- Database ---

ingest:
	$(PYTHON) src/popola_db.py

index:
	$(PYTHON) src/create_vector_index.py

setup-db: ingest index

# --- Docker ---

build:
	docker compose build

up:
	docker compose up -d --build

down:
	docker compose down

restart: down up

rebuild-api:
	docker compose up -d --build api

logs:
	docker compose logs -f

logs-api:
	docker compose logs -f api

logs-frontend:
	docker compose logs -f frontend

ps:
	docker compose ps
