"""Validated runtime settings shared by the Streamlit UI and backend."""
from __future__ import annotations

from dataclasses import dataclass

from src.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_OCR_ENABLED,
    DEFAULT_OCR_LANGUAGES,
    DEFAULT_OCR_PAGE_LIMIT,
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
    use_ocr: bool = DEFAULT_OCR_ENABLED
    ocr_languages: tuple[str, ...] = DEFAULT_OCR_LANGUAGES
    ocr_page_limit: int = DEFAULT_OCR_PAGE_LIMIT
    use_visual_summaries: bool = DEFAULT_VISUAL_SUMMARIES
    visual_page_limit: int = DEFAULT_VISUAL_PAGE_LIMIT

    def validate(self) -> None:
        if not 300 <= self.chunk_size <= 2_000:
            raise ValueError("Chunk size must be between 300 and 2,000 characters.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("Chunk overlap must be at least 0 and smaller than chunk size.")
        if self.use_ocr and not self.ocr_languages:
            raise ValueError("Choose at least one OCR language.")
        if not 1 <= self.ocr_page_limit <= 20:
            raise ValueError("OCR page limit must be between 1 and 20.")
        if not 1 <= self.visual_page_limit <= 10:
            raise ValueError("Visual page limit must be between 1 and 10.")
