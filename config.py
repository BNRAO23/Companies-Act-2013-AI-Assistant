"""
Configuration settings for Companies Act 2013 Local RAG Assistant.
All settings are centralized here for easy tuning and portability.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "Companies Act 2013.pdf"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "companies_act_2013"

# Ingestion & Chunking Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Local Embedding Settings (runs 100% locally via HuggingFace / sentence-transformers)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Alternate high-accuracy choice: "BAAI/bge-small-en-v1.5"

# Local LLM Runner Settings (Ollama)
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0.1
TOP_K_CHUNKS = 4
