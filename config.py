import os
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# ----------------- Basic Configuration -----------------
dotenv_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if not GOOGLE_API_KEY:
    logging.error("FATAL ERROR: GOOGLE_API_KEY environment variable not found.")
    logging.error(f"Please ensure a .env file exists at {dotenv_path} and contains GOOGLE_API_KEY='your_api_key'")
    sys.exit(1)

# Create a directory for persistent chat histories if it doesn't exist
CHAT_SESSIONS_DIR = Path(__file__).parent / "chat_sessions"
CHAT_SESSIONS_DIR.mkdir(exist_ok=True)
