# 1. Usa una versione di Python leggera e ufficiale come base
FROM python:3.11-slim

# 2. Crea una cartella "app" dentro il container e lavora da lì
WORKDIR /app

# 3. Copia SOLO il file dei requisiti per ora (trucco da pro per usare la cache di Docker e velocizzare le build future)
COPY requirements.txt .

# 4. Installa tutte le librerie necessarie senza salvare roba inutile (mantiene il container leggero)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Ora copia tutto il resto del tuo codice (api.py, la cartella chroma_db, ecc.) dentro il container
COPY . .

# 6. Diciamo a Docker che il nostro server comunicherà sulla porta 8000
EXPOSE 8000

# 7. Il comando magico che Docker lancerà quando si accende
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]