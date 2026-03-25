import os
import win32com.client

# --- CONFIGURAZIONE ---
# win32com è schizzinoso, vuole percorsi ASSOLUTI, quindi usiamo os.path.abspath
cartella_input = os.path.abspath(r"C:\\Users\\Utente\\Desktop\\Repos\\EnterpriseRAG_Portfolio\\data\\PROCEDURE\\01_LT651\\Procedure")
cartella_output = os.path.abspath(r"C:\\Users\\Utente\\Desktop\\Repos\\EnterpriseRAG_Portfolio\\data\\P_CONVERTITE\\PROCEDURE\\01_LT651\\Procedure")

# Crea la cartella di output se non esiste
os.makedirs(cartella_output, exist_ok=True)

print("🤖 Avvio di Microsoft Word in modalità fantasma...")
# Evoca Microsoft Word in background
word = win32com.client.Dispatch("Word.Application")
word.Visible = False # Zero interfacce grafiche, fa tutto di nascosto!

try:
    file_trovati = [f for f in os.listdir(cartella_input) if f.lower().endswith((".doc", ".docx"))]
    
    if not file_trovati:
        print("⚠️ Nessun file .doc o .docx trovato nella cartella 'da_convertire'.")
    else:
        for file in file_trovati:
            percorso_doc = os.path.join(cartella_input, file)
            # Cambia l'estensione in .pdf
            nome_pdf = file.rsplit('.', 1)[0] + ".pdf"
            percorso_pdf = os.path.join(cartella_output, nome_pdf)

            print(f"🔄 Convertendo: {file} -> {nome_pdf}...")
            
            # Apri il documento
            doc = word.Documents.Open(percorso_doc)
            # Salvalo come PDF (Il codice magico per il PDF in Word è il numero 17)
            doc.SaveAs(percorso_pdf, FileFormat=17)
            # Chiudilo
            doc.Close()
            
        print(f"{len(file_trovati)} file convertiti con successo!")

except Exception as e:
    print(f"❌ Errore durante la conversione: {e}")

finally:
    # Chiudi il processo di Word, sennò ti rimane incastrato in RAM!
    word.Quit()
    print("🛑 Processo Word terminato.")