"""Lesson-style LangChain Chroma storage with local Ollama embeddings."""
from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from src.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, OLLAMA_URL
from src.types import Chunk


class VectorStore:
    """Mirror lessons 1–2: OllamaEmbeddings + persistent cosine Chroma."""

    def __init__(self, persist_directory: Path = CHROMA_DIR, collection_name: str = COLLECTION_NAME) -> None:
        self.embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)
        self.db = Chroma(
            persist_directory=str(persist_directory),
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self.db._collection.count()

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 32) -> int:
        """Use ``Chroma.add_documents`` batches as in lesson 1."""
        if not chunks:
            return 0
        added = 0
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset : offset + batch_size]
            documents = [Document(page_content=chunk.text, metadata=chunk.chroma_metadata()) for chunk in batch]
            # Stable IDs keep re-indexing idempotent and retain exact citations.
            self.db.add_documents(documents, ids=[chunk.id for chunk in batch])
            added += len(batch)
        return added

    def remove_source(self, source_id: str) -> int:
        """Remove an older copy before a source is intentionally re-indexed."""
        existing = self.db._collection.get(where={"source_id": source_id})
        ids = existing.get("ids", [])
        if ids:
            self.db.delete(ids=ids)
        return len(ids)

    def all_chunks(self) -> list[Chunk]:
        data = self.db._collection.get(include=["documents", "metadatas"])
        chunks: list[Chunk] = []
        for chunk_id, text, metadata in zip(data.get("ids", []), data.get("documents", []), data.get("metadatas", [])):
            meta = metadata or {}
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=text,
                    source_name=str(meta.get("source_name", "Unknown")),
                    source_id=str(meta.get("source_id", "")),
                    page=int(meta["page"]) if meta.get("page_available") else None,
                    chunk_index=int(meta.get("chunk_index", 0)),
                    modality=str(meta.get("modality", "text")),
                    rendered_image=str(meta.get("rendered_image", "")) or None,
                )
            )
        return chunks

    def vector_search(self, query: str, k: int) -> list[Chunk]:
        """Use the reference code's ``db.as_retriever(...).invoke(query)`` path."""
        if self.count == 0:
            return []
        documents = self.db.as_retriever(search_kwargs={"k": min(k, self.count)}).invoke(query)
        return [self._chunk_from_document(document) for document in documents]

    def vector_similarity_scores(self, query: str, k: int) -> dict[str, float]:
        """Get relevance values solely for the evidence-sufficiency UI gate."""
        if self.count == 0:
            return {}
        pairs = self.db.similarity_search_with_relevance_scores(query, k=min(k, self.count))
        return {str(document.metadata["chunk_id"]): score for document, score in pairs}

    @staticmethod
    def _chunk_from_document(document: Document) -> Chunk:
        metadata = document.metadata
        return Chunk(
            id=str(metadata["chunk_id"]),
            text=document.page_content,
            source_name=str(metadata.get("source_name", "Unknown")),
            source_id=str(metadata.get("source_id", "")),
            page=int(metadata["page"]) if metadata.get("page_available") else None,
            chunk_index=int(metadata.get("chunk_index", 0)),
            modality=str(metadata.get("modality", "text")),
            rendered_image=str(metadata.get("rendered_image", "")) or None,
        )

    def clear(self) -> None:
        ids = self.db._collection.get().get("ids", [])
        if ids:
            self.db.delete(ids=ids)
