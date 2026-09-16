import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from app.core.config import CHROMA_DB_DIR, GOOGLE_API_KEY
import logging

# Initialize Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# Initialize Chroma Vector Store
vectorstore = Chroma(
    collection_name="it_knowledge_base",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DB_DIR)
)

def populate_initial_kb():
    """Populates the vector store with some dummy IT knowledge base articles if it's empty."""
    existing_docs = vectorstore.get()
    if not existing_docs['ids']:
        logging.info("Populating initial Knowledge Base into ChromaDB...")
        dummy_articles = [
            Document(page_content="To reset a user's password, verify their identity first, then use the 'reset_password' tool.", metadata={"source": "IT Policy 101"}),
            Document(page_content="If a user reports the VPN is slow, check the server status first using the 'check_server_status' tool. If it's a known issue, inform the user.", metadata={"source": "Network Troubleshooting"}),
            Document(page_content="For hardware failures (e.g., broken screen, laptop won't turn on), a ticket must be created using the 'create_jira_ticket' tool.", metadata={"source": "Hardware Support"}),
            Document(page_content="To clear browser cache in Chrome: Settings -> Privacy and security -> Clear browsing data.", metadata={"source": "Software Support"}),
        ]
        vectorstore.add_documents(dummy_articles)
        logging.info("Knowledge Base populated.")

# Run population on startup
populate_initial_kb()

def retrieve_knowledge(query: str) -> str:
    """Retrieves relevant IT documentation for a given query."""
    logging.info(f"Retrieving KB documents for: {query}")
    docs = vectorstore.similarity_search(query, k=2)
    if docs:
        context = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in docs])
        return context
    return "No relevant IT documentation found."
