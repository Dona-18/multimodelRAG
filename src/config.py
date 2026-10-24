"""Configuration for local Multimodal RAG with model profile support."""
import os
from pathlib import Path
from typing import Dict, Any

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

# Available model profiles
MODEL_PROFILES: Dict[str, Dict[str, Any]] = {
    "qwen3": {
        "id": "qwen3",
        "name": "Qwen3 + Qwen3-Embedding (Recommended)",
        "llm": "qwen3:latest",
        "embedding": "qwen3-embedding:0.6b",
        "vision": "gemma4:e2b-mlx",
        "collection": "docs_qwen3_embedding_0_6b",
        "embedding_dim": 1024,
    },
    "gemma_nomic": {
        "id": "gemma_nomic",
        "name": "Gemma 4 + Nomic (Legacy Fallback)",
        "llm": "gemma4:e2b-mlx",
        "embedding": "nomic-embed-text:latest",
        "vision": "gemma4:e2b-mlx",
        "collection": "docs_nomic_embed_text_latest",
        "embedding_dim": 768,
    },
}

DEFAULT_PROFILE = "qwen3"
ACTIVE_PROFILE_KEY = os.environ.get("ACTIVE_PROFILE", DEFAULT_PROFILE)
if ACTIVE_PROFILE_KEY not in MODEL_PROFILES:
    ACTIVE_PROFILE_KEY = DEFAULT_PROFILE

_active = MODEL_PROFILES[ACTIVE_PROFILE_KEY]
LLM_MODEL = os.environ.get("LLM_MODEL", _active["llm"])
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", _active["embedding"])
VISION_MODEL = os.environ.get("VISION_MODEL", _active["vision"])
DEFAULT_COLLECTION_NAME = os.environ.get("COLLECTION_NAME", _active["collection"])


def get_profile(profile_key: str = None) -> Dict[str, Any]:
    """Get model configuration by profile key."""
    key = profile_key or ACTIVE_PROFILE_KEY
    return MODEL_PROFILES.get(key, MODEL_PROFILES[DEFAULT_PROFILE])
