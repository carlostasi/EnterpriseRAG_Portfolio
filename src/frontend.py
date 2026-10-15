import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()
# --- CONFIGURAZIONE ---
API_URL = os.environ.get("API_URL", "http://localhost:8000") # L'indirizzo del backend
API_KEY = os.environ.get("API_KEY")# La chiave per parlare col backend


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
                payload = {"text": prompt,
                            "history": st.session_state.messages[:-1]}

                try:
                    # ATTENZIONE: stream=True è vitale!
                    with requests.post(f"{API_URL}/chat_stream", json=payload, headers=headers, stream=True) as response:
                        if response.status_code == 200:
                            
                            def read_stream():
                                # Legge il fiume di testo pezzettino per pezzettino
                                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                                    if chunk:
                                        yield chunk
                            
                            # STREAMLIT: full_response catturerà TUTTO il testo che esce dal tubo,
                            # includendo sia la risposta di Claude sia le fonti che arrivano alla fine
                            full_response = st.write_stream(read_stream())
                            
                            # Salviamo TUTTO (risposta + fonti) nella memoria come un unico blocco di testo
                            st.session_state.messages.append({"role": "assistant", "content": full_response})
                            
                        elif response.status_code in [401, 403]:
                            st.error("Accesso Negato dal Backend: API Key non valida.")
                        else:
                            st.error(f"Errore del server: {response.status_code}")
                            
                except Exception as e:
                    st.error(f"Errore interno del Frontend: {str(e)}")