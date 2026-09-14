import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env if present
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

def get_gemini_api_key() -> str:
    """Dynamically fetch GEMINI_API_KEY from environment or .env file."""
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
    return os.getenv("GEMINI_API_KEY", "")

GEMINI_API_KEY = get_gemini_api_key()
DB_PATH = Path(os.getenv("DB_PATH", "data/rag.db"))
if not DB_PATH.is_absolute():
    DB_PATH = BASE_DIR / DB_PATH

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/uploads"))
if not UPLOAD_DIR.is_absolute():
    UPLOAD_DIR = BASE_DIR / UPLOAD_DIR

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

# Ensure directories exist
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
