import boto3
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

print("Avvio del sistema RAG...")

bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

embeddings = BedrockEmbeddings(
    client=bedrock_client,
    model_id="amazon.titan-embed-text-v2:0"
)

llm = ChatBedrock(
    client=bedrock_client,
    model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
    model_kwargs={"temperature": 0}
)

# Carico dalla memoria
vectorStore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

# PROMPT
sys_prompt = (
    "Sei un assistente aziendale super preciso."
    "Usa i seguenti frammenti di contesto recuperati per rispondere alla domanda dell'utente."
    "Se non conosci la risposta basandoti ESCLUSIVAMENTE sul contesto dì semplicemente che non lo sai."
    "Non inventare MAI informazioni. Sii conciso e professionale.\n\n"
    "CONTESTO TROVATO NEL DATABASE:\n"
    "{context}"
)

propmpt = ChatPromptTemplate.from_messages([
    ("system", sys_prompt),
    ("human", "{input}"),
])

# Creazione della PIPELINE RAG
# Questa catena dice a Claude come leggere i documenti
question_answer_chain = create_stuff_documents_chain(llm, propmpt)
# Questa catena unisce la ricerca nel database con la generazione della risposta
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# queryPrompt = "Vorrei sapere come dovrei impostare l'indirizzo ip della macchina LT Fiber"
print("\nInserisci domanda:")
queryPrompt = input()
print(f"Utente: {queryPrompt}")

print("Claude sta leggendo i documenti e scrivendo la risposta...\n")
result = rag_chain.invoke({"input": queryPrompt})

print(result["answer"])

print("\n--------------------------------------------------")
print("📚 FONTI UTILIZZATE PER QUESTA RISPOSTA:")
for doc in result["context"]:
    print(f"- File: {doc.metadata.get('source', 'Sconosciuto')} (Pagina {doc.metadata.get('page', 'N/A')})")