import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).parent

_key_file = BASE_DIR / "api_key_apiyi.txt"
OPENAI_API_KEY: str = (
    _key_file.read_text(encoding="utf-8").strip()
    if _key_file.exists()
    else os.getenv("OPENAI_API_KEY", "")
)
OPENAI_BASE_URL = "https://api.apiyi.com/v1"

# LLM for entity extraction and Q&A
# Agent: entity extraction, Q&A, inference, research
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite-preview")

# Embedding (gemini-embedding-2-preview via apiyi)
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-2-preview")
EMBED_DIM = int(os.getenv("EMBED_DIM", "3072"))

# Graph construction + transcription / captioning (gemini via apiyi)
OMNI_MODEL = os.getenv("OMNI_MODEL", "gemini-3.1-flash-lite-preview")

# Knowledge base settings
KB_CHUNK_SIZE = 512        # tokens per chunk
KB_CHUNK_OVERLAP = 64      # overlap tokens
KB_EXTRACT_BATCH = 5       # chunks per extraction call
KB_TOP_K = 10              # retrieval results

# Media expression directory (video clips / page images saved here during ingestion)
EXPR_DIR = os.getenv("EXPR_DIR", str(BASE_DIR / "expr"))

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
