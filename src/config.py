"""Small, explicit configuration for the local-first assistant."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
RENDERED_PAGES_DIR = DATA_DIR / "rendered_pages"
INDEX_SETTINGS_PATH = DATA_DIR / "index_settings.json"
ACTIVE_SOURCE_PATH = DATA_DIR / "active_source.json"

for directory in (DATA_DIR, UPLOADS_DIR, CHROMA_DIR, RENDERED_PAGES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

import os

def _get_ollama_url() -> str:
    try:
        import streamlit as st
        if "OLLAMA_URL" in st.secrets:
            return str(st.secrets["OLLAMA_URL"]).rstrip("/")
    except Exception:
        pass
    return os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")

OLLAMA_URL = _get_ollama_url()
LLM_MODEL = "gemma4:e2b-mlx"
EMBEDDING_MODEL = "nomic-embed-text:latest"
COLLECTION_NAME = "local_multimodal_rag"

# These are deliberately visible in the interface.  They are starting points,
# not universal truth: users can tune them for their own document collection.
DEFAULT_CHUNK_SIZE = 1_000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 3
DEFAULT_MIN_SIMILARITY = 0.35
RRF_K = 60

# Local Gemma settings. The context maximum was verified from the installed
# model metadata; the UI offers safe values no larger than this limit.
MODEL_MAX_CONTEXT_LENGTH = 131_072
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_OUTPUT_TOKENS = 512
DEFAULT_CONTEXT_LENGTH = 8_192

# Lesson 8 adaptations. OCR uses the locally installed Tesseract engine via
# Unstructured. Visual summaries use the already-installed Gemma vision model.
DEFAULT_OCR_ENABLED = True
DEFAULT_OCR_LANGUAGES = ("eng",)
DEFAULT_OCR_PAGE_LIMIT = 5
DEFAULT_VISUAL_SUMMARIES = False
DEFAULT_VISUAL_PAGE_LIMIT = 3
