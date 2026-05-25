"""
config/settings.py — Central configuration for all modules.
Loads from .env and exposes typed constants.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── LLM ─────────────────────────────────────────────────────────────────────
GROQ_API_KEY:   str = os.getenv("GROQ_API_KEY",   "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

PRIMARY_MODEL  = "llama-3.3-70b-versatile"  # Groq — free, fast
FALLBACK_MODEL = "gemini-1.5-flash"           # Gemini — free tier, 1M context
VISION_MODEL   = "gemini-1.5-flash"           # Vision

MAX_TOKENS  = 2048
TEMPERATURE = 0.1    # Low temp for clinical accuracy

# ─── RAG ──────────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
EMBED_MODEL:        str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = "medical_knowledge"
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100

# ─── Memory ───────────────────────────────────────────────────────────────────
MEMORY_DB_PATH:    str = os.getenv("MEMORY_DB_PATH", "./data/memory.db")
MAX_HISTORY_TURNS: int = 10

# ─── External APIs ────────────────────────────────────────────────────────────
NCBI_EMAIL:              str = os.getenv("NCBI_EMAIL", "medassistant@demo.com")
OPENFDA_BASE                 = "https://api.fda.gov/drug"
PUBMED_BASE                  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NLM_CLINICAL_TABLES_BASE     = "https://clinicaltables.nlm.nih.gov/api"
PUBMED_MAX_RESULTS: int  = 5
OPENFDA_MAX_RESULTS: int = 3

# ─── Agent / MCP ──────────────────────────────────────────────────────────────
AGENT_TIMEOUT_SECONDS:     int   = 30
MAX_REFLECTION_LOOPS:      int   = 2
MAX_TOOL_ITERATIONS:       int   = 10   # Max ReAct tool-call rounds per agent

# Confidence threshold — below this, Tavily web search fires as fallback
FALLBACK_CONFIDENCE_THR: float = 0.5

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")