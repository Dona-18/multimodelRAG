"""Remember the chunk settings used by the persisted local index."""
from __future__ import annotations

import json

from src.config import INDEX_SETTINGS_PATH
from src.settings import IndexSettings


def load_index_settings() -> IndexSettings | None:
    """Return None for older/unknown indexes so they are safely re-indexed."""
    if not INDEX_SETTINGS_PATH.is_file():
        return None
    try:
        data = json.loads(INDEX_SETTINGS_PATH.read_text(encoding="utf-8"))
        settings = IndexSettings(
            chunk_size=int(data["chunk_size"]),
            chunk_overlap=int(data["chunk_overlap"]),
            use_ocr=bool(data["use_ocr"]),
            ocr_languages=tuple(str(language) for language in data["ocr_languages"]),
            ocr_page_limit=int(data["ocr_page_limit"]),
            use_visual_summaries=bool(data["use_visual_summaries"]),
            visual_page_limit=int(data["visual_page_limit"]),
        )
        settings.validate()
        return settings
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_index_settings(settings: IndexSettings) -> None:
    settings.validate()
    INDEX_SETTINGS_PATH.write_text(
        json.dumps(
            {
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "use_ocr": settings.use_ocr,
                "ocr_languages": list(settings.ocr_languages),
                "ocr_page_limit": settings.ocr_page_limit,
                "use_visual_summaries": settings.use_visual_summaries,
                "visual_page_limit": settings.visual_page_limit,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def requires_reingestion(indexed_chunks: int, desired: IndexSettings) -> bool:
    """Chunk parameters are part of the index, so mixed settings are disallowed."""
    if indexed_chunks == 0:
        return False
    return load_index_settings() != desired
