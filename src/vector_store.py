"""ChromaDB vector store operations."""
from pathlib import Path
from typing import List, Union
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from src.config import CHROMA_DIR, EMBEDDING_MODEL, OLLAMA_URL


def get_vector_store(
    persist_directory: Union[str, Path] = CHROMA_DIR,
    collection_name: str = "multimodal_documents",
) -> Chroma:
    """Connect to or create a ChromaDB vector store."""
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_URL,
        ),
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"},
    )


def add_documents_to_store(documents: List[Document], db: Chroma) -> int:
    """Add documents to ChromaDB."""
    if documents:
        db.add_documents(documents)
    return len(documents)


def get_all_stored_documents(db: Chroma) -> List[Document]:
    """Retrieve all stored documents from ChromaDB."""
    data = db._collection.get()
    contents = data.get("documents", [])
    metas = data.get("metadatas", [])
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(contents, metas)
    ]


def clear_vector_store(db: Chroma) -> None:
    """Delete all documents in the ChromaDB collection."""
    ids = db._collection.get().get("ids", [])
    if ids:
        db._collection.delete(ids=ids)
