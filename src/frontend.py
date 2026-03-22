import streamlit as st
import requests
import os
# --- CONFIGURAZIONE ---
# API_URL = "http://18.234.83.166:8000/chat" # L'indirizzo del tuo backend
API_URL = "http://localhost:8000/chat"
API_KEY = os.environ.get("API_KEY")# La chiave per parlare col backend

# 🛑 LA NUOVA PASSWORD PER ENTRARE NEL SITO 🛑
PASSWORD_SITO = os.environ.get("PASSWORD_SITO")

st.set_page_config(page_title="Enterprise RAG", page_icon="🏢")

# --- SISTEMA DI LOGIN FRONTEND ---
def controlla_password():
    """Ritorna True se l'utente ha inserito la password corretta"""
    if "autenticato" not in st.session_state:
        st.session_state.autenticato = False

    if not st.session_state.autenticato:
        st.title("🔒 Area Riservata")
        st.write("Inserisci la password per accedere alla chat RAG.")
        
        # Casella di testo che nasconde i caratteri (tipo="password")
        pwd_inserita = st.text_input("Password:", type="password")
        
        if st.button("Entra"):
            if pwd_inserita == PASSWORD_SITO:
                st.session_state.autenticato = True
                st.rerun() # Ricarica la pagina mostrando la chat
            else:
                st.error("❌ Password errata. Riprova.")
        return False
    return True

# --- SE LA PASSWORD È GIUSTA, MOSTRA LA CHAT ---
if controlla_password():
    
    # Da qui in giù è esattamente il tuo codice di prima!
    st.title("🏢 Enterprise Knowledge Hub")
    st.caption("Fai domande sui documenti aziendali. Il sistema risponderà citando le fonti.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Scrivi qui la tua domanda..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Sto consultando il database vettoriale... 🧠"):
                headers = {"Portfolio-API-Key": API_KEY}
                payload = {"text": prompt} # Assicurati che qui ci sia la parola giusta (es. "testo" o "domanda")
                
                try:
                    response = requests.post(API_URL, json=payload, headers=headers)
                    if response.status_code == 200:
                        dati = response.json()
                        risposta_ai = dati["ai_answer"] # Assicurati dei nomi corretti
                        fonti = dati["fonti"]
                        
                        st.markdown(risposta_ai)
                        fonti_format = "\n".join(fonti)
                        sezione_fonti = f"\n\n**📚 Fonti utilizzate:**\n{fonti_format}"
                        st.caption(sezione_fonti)
                        
                        st.session_state.messages.append({"role": "assistant", "content": risposta_ai + sezione_fonti})
                    elif response.status_code == 401 or response.status_code == 403:
                        st.error("Accesso Negato dal Backend: API Key non valida. 🛑")
                    else:
                        st.error(f"Errore del server: {response.status_code}")
                except Exception as e:
                    st.error(f"Errore interno del Frontend: {str(e)}")