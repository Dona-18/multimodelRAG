"""Single unified module for active document, index tracking, and UI state."""
from __future__ import annotations

import json
from pathlib import Path

from src.config import ACTIVE_SOURCE_PATH, INDEX_SETTINGS_PATH, UPLOADS_DIR
from src.types import IndexSettings


def chat_is_unlocked(indexed_chunks: int) -> bool:
    """Chat is available only after ingestion has produced local evidence."""
    return indexed_chunks > 0


def _is_project_upload(path: Path) -> bool:
    """Only accept files saved in this project's upload directory."""
    try:
        path.resolve().relative_to(UPLOADS_DIR.resolve())
    except ValueError:
        return False
    return True


def load_active_source() -> Path | None:
    """Return the current document, or None when removed or unknown."""
    if not ACTIVE_SOURCE_PATH.is_file():
        return None
    try:
        data = json.loads(ACTIVE_SOURCE_PATH.read_text(encoding="utf-8"))
        path = Path(data["path"]).resolve()
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return path if path.is_file() and _is_project_upload(path) else None


def save_active_source(path: Path) -> None:
    """Record the single successfully indexed project upload."""
    resolved = Path(path).resolve()
    if not resolved.is_file() or not _is_project_upload(resolved):
        raise ValueError("The active document must be a saved project upload.")
    ACTIVE_SOURCE_PATH.write_text(json.dumps({"path": str(resolved)}, indent=2), encoding="utf-8")


def clear_active_source() -> None:
    """Forget the current document without deleting the user's local upload."""
    try:
        ACTIVE_SOURCE_PATH.unlink()
    except FileNotFoundError:
        pass


def load_index_settings() -> IndexSettings | None:
    """Return None for older/unknown indexes so they are safely re-indexed."""
    if not INDEX_SETTINGS_PATH.is_file():
        return None
    try:
        data = json.loads(INDEX_SETTINGS_PATH.read_text(encoding="utf-8"))
        settings = IndexSettings(
            chunk_size=int(data["chunk_size"]),
            chunk_overlap=int(data["chunk_overlap"]),
            use_visual_summaries=bool(data.get("use_visual_summaries", False)),
            visual_page_limit=int(data.get("visual_page_limit", 3)),
        )
        settings.validate()
        return settings
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_index_settings(settings: IndexSettings) -> None:
    """Persist current index settings for change detection."""
    settings.validate()
    INDEX_SETTINGS_PATH.write_text(
        json.dumps(
            {
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
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
