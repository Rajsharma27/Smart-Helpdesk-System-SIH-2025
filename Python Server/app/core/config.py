import os
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# ----------------- Basic Configuration -----------------
# Root of the project (2 levels up from app/core)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
dotenv_path = PROJECT_ROOT / '.env'
load_dotenv(dotenv_path=dotenv_path)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# AIOps: Set LangSmith Tracing if key exists
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Smart-Helpdesk-System"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if not GOOGLE_API_KEY:
    logging.error("FATAL ERROR: GOOGLE_API_KEY environment variable not found.")
    logging.error(f"Please ensure a .env file exists at {dotenv_path} and contains GOOGLE_API_KEY='your_api_key'")
    sys.exit(1)

# Create a directory for persistent chat histories and ChromaDB
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

CHAT_SESSIONS_DIR = DATA_DIR / "chat_sessions"
CHAT_SESSIONS_DIR.mkdir(exist_ok=True)

CHROMA_DB_DIR = DATA_DIR / "chroma_db"
CHROMA_DB_DIR.mkdir(exist_ok=True)
