"""Simple configuration for local Multimodal RAG."""
import os
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
UPLOADS_DIR = DATA_DIR / "uploads"
IMAGES_DIR = DATA_DIR / "extracted_images"

for path in [DATA_DIR, CHROMA_DIR, UPLOADS_DIR, IMAGES_DIR]:
    path.mkdir(parents=True, exist_ok=True)


def get_ollama_url() -> str:
    """Retrieve Ollama endpoint from environment variable or local default."""
    return os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")


OLLAMA_URL = get_ollama_url()

# Local Ollama models
LLM_MODEL = "gemma4:e2b-mlx"
EMBEDDING_MODEL = "nomic-embed-text:latest"
