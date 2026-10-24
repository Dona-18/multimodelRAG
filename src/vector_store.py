"""ChromaDB vector store operations with isolated collection support."""
from pathlib import Path
from typing import List, Union, Optional
import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from src.config import CHROMA_DIR, EMBEDDING_MODEL, OLLAMA_URL, DEFAULT_COLLECTION_NAME


def get_collection_name_for_model(embedding_model: str) -> str:
    """Generate isolated collection name based on embedding model to avoid dimension collisions."""
    clean_name = embedding_model.replace(":", "_").replace("-", "_").replace(".", "_")
    return f"docs_{clean_name}"


def get_vector_store(
    persist_directory: Union[str, Path] = CHROMA_DIR,
    collection_name: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> Chroma:
    """Connect to or create a ChromaDB vector store with model-isolated collections."""
    model = embedding_model or EMBEDDING_MODEL
    coll_name = collection_name or get_collection_name_for_model(model)

    client = chromadb.PersistentClient(
        path=str(persist_directory),
        settings=Settings(anonymized_telemetry=False, is_persistent=True),
    )
    return Chroma(
        client=client,
        embedding_function=OllamaEmbeddings(
            model=model,
            base_url=OLLAMA_URL,
        ),
        collection_name=coll_name,
        collection_metadata={"hnsw:space": "cosine"},
    )


def add_documents_to_store(documents: List[Document], db: Chroma) -> int:
    """Add documents to ChromaDB."""
    if documents:
        db.add_documents(documents)
    return len(documents)


def get_all_stored_documents(db: Chroma) -> List[Document]:
    """Retrieve all stored documents from the active ChromaDB collection."""
    data = db._collection.get()
    contents = data.get("documents", [])
    metas = data.get("metadatas", [])
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(contents, metas)
    ]


def clear_vector_store(db: Chroma) -> None:
    """Delete all documents in the active ChromaDB collection."""
    ids = db._collection.get().get("ids", [])
    if ids:
        db._collection.delete(ids=ids)
