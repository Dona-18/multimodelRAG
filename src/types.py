"""Data structures and validated runtime settings shared by the modular pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_VISUAL_PAGE_LIMIT,
    DEFAULT_VISUAL_SUMMARIES,
    MODEL_MAX_CONTEXT_LENGTH,
)


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    context_length: int = DEFAULT_CONTEXT_LENGTH
    streaming: bool = False

    def validate(self) -> None:
        if not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2.")
        if not 0 < self.top_p <= 1:
            raise ValueError("Top-p must be greater than 0 and no more than 1.")
        if not 64 <= self.max_output_tokens <= 8_192:
            raise ValueError("Max output tokens must be between 64 and 8,192.")
        if not 512 <= self.context_length <= MODEL_MAX_CONTEXT_LENGTH:
            raise ValueError(f"Context length must be between 512 and {MODEL_MAX_CONTEXT_LENGTH:,}.")


@dataclass(frozen=True)
class RetrievalSettings:
    top_k: int = DEFAULT_TOP_K
    use_multi_query: bool = False
    use_hybrid_search: bool = True

    def validate(self) -> None:
        if not 1 <= self.top_k <= 20:
            raise ValueError("Retrieval top-k must be between 1 and 20.")


@dataclass(frozen=True)
class IndexSettings:
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    use_visual_summaries: bool = DEFAULT_VISUAL_SUMMARIES
    visual_page_limit: int = DEFAULT_VISUAL_PAGE_LIMIT

    def validate(self) -> None:
        if not 300 <= self.chunk_size <= 2_000:
            raise ValueError("Chunk size must be between 300 and 2,000 characters.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("Chunk overlap must be at least 0 and smaller than chunk size.")
        if not 1 <= self.visual_page_limit <= 10:
            raise ValueError("Visual page limit must be between 1 and 10.")


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
