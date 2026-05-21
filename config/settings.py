"""
config/settings.py — Central configuration for all modules.
Loads from .env and exposes typed constants.
"""
import os
from dotenv import load_dotenv

load_dotenv()


GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")


PRIMARY_MODEL = "llama-3.1-70b-versatile"

FALLBACK_MODEL = "gemini-1.5-flash"

VISION_MODEL = "gemini-1.5-flash"

MAX_TOKENS = 2048
TEMPERATURE = 0.1           


CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = "medical_knowledge"
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


MEMORY_DB_PATH: str = os.getenv("MEMORY_DB_PATH", "./data/memory.db")
MAX_HISTORY_TURNS = 10      


NCBI_EMAIL: str = os.getenv("NCBI_EMAIL", "medassistant@demo.com")
OPENFDA_BASE = "https://api.fda.gov/drug"
PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NLM_CLINICAL_TABLES_BASE = "https://clinicaltables.nlm.nih.gov/api"

PUBMED_MAX_RESULTS = 5
OPENFDA_MAX_RESULTS = 3


AGENT_TIMEOUT_SECONDS = 30
MAX_REFLECTION_LOOPS = 2


LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")