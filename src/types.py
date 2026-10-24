"""Plain data structures shared by the modular pipeline."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourcePage:
    """Extracted text from one logical page (or a non-paginated document)."""

    text: str
    source_name: str
    source_id: str
    page: int | None
    file_path: Path


@dataclass(frozen=True)
class Chunk:
    """A retrieval-ready chunk preserving the source and page reference."""

    id: str
    text: str
    source_name: str
    source_id: str
    page: int | None
    chunk_index: int
    modality: str = "text"
    rendered_image: str | None = None

    def chroma_metadata(self) -> dict[str, Any]:
        """Return Chroma-safe primitive metadata (Chroma does not accept None)."""
        return {
            "chunk_id": self.id,
            "source_name": self.source_name,
            "source_id": self.source_id,
            "page": self.page or 0,
            "page_available": self.page is not None,
            "chunk_index": self.chunk_index,
            "modality": self.modality,
            "rendered_image": self.rendered_image or "",
        }


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    vector_similarity: float | None
    bm25_score: float | None
    rrf_score: float


@dataclass
class RetrievalResult:
    hits: list[SearchHit] = field(default_factory=list)
    sufficient: bool = False
    reason: str = "No evidence was retrieved."


@dataclass(frozen=True)
class IngestReport:
    source_name: str
    source_id: str
    pages_or_sections: int
    chunks_added: int
