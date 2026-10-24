"""Re-embed all documents into a separate model-isolated ChromaDB collection."""
import sys
from pathlib import Path
from typing import Optional

# Setup root path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import UPLOADS_DIR, CHROMA_DIR, MODEL_PROFILES, DEFAULT_PROFILE
from src.ingestion import load_any_document, split_documents
from src.vector_store import get_vector_store, add_documents_to_store, get_all_stored_documents


def reembed_all_documents(
    profile_key: str = "qwen3",
    source_dir: Optional[Path] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> int:
    """Re-embed all uploaded documents into the profile's dedicated ChromaDB collection."""
    profile = MODEL_PROFILES.get(profile_key, MODEL_PROFILES[DEFAULT_PROFILE])
    embedding_model = profile["embedding"]
    collection_name = profile["collection"]

    print(f"=== Re-embedding for Profile '{profile_key}' ===")
    print(f"Target Collection: {collection_name}")
    print(f"Embedding Model:   {embedding_model} ({profile['embedding_dim']} dimensions)")

    db = get_vector_store(
        persist_directory=CHROMA_DIR,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )

    existing_docs = get_all_stored_documents(db)
    print(f"Existing chunks in '{collection_name}': {len(existing_docs)}")

    folder = source_dir or UPLOADS_DIR
    files = [
        f for f in folder.iterdir()
        if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in [
            ".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg", ".webp"
        ]
    ]

    print(f"Found {len(files)} source documents in {folder}")
    total_added = 0

    for file_path in files:
        try:
            print(f"  Ingesting '{file_path.name}'...")
            docs = load_any_document(file_path)
            chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if chunks:
                add_documents_to_store(chunks, db)
                total_added += len(chunks)
                print(f"    -> Embedded {len(chunks)} chunks.")
        except Exception as e:
            print(f"    [Error] Failed to ingest {file_path.name}: {e}")

    updated_docs = get_all_stored_documents(db)
    print(f"Total chunks in '{collection_name}': {len(updated_docs)}")
    return total_added


if __name__ == "__main__":
    profile = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    reembed_all_documents(profile_key=profile)
