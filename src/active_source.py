"""Persist the one local document that the assistant is allowed to use."""
from __future__ import annotations

import json
from pathlib import Path

from src.config import ACTIVE_SOURCE_PATH, UPLOADS_DIR


def _is_project_upload(path: Path) -> bool:
    """Only accept files saved in this project's upload directory."""
    try:
        path.resolve().relative_to(UPLOADS_DIR.resolve())
    except ValueError:
        return False
    return True


def load_active_source() -> Path | None:
    """Return the current document, or ``None`` when it was removed or is unknown."""
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
