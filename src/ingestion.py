"""Local document loading and lesson-style recursive character chunking."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from docx import Document as DocxDocument
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.config import UPLOADS_DIR
from src.types import Chunk, SourcePage

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".markdown"}


class DocumentLoadError(ValueError):
    """Raised when a user document cannot be safely loaded."""


def source_id_for_path(path: Path) -> str:
    """A content fingerprint makes re-indexing the exact upload idempotent."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_upload(file_name: str, content: bytes) -> Path:
    """Store an uploaded document only inside this project's local data folder."""
    safe_name = Path(file_name).name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise DocumentLoadError(f"Unsupported file type `{suffix or 'none'}`. Use: {allowed}.")
    fingerprint = hashlib.sha256(content).hexdigest()[:12]
    destination = UPLOADS_DIR / f"{fingerprint}_{safe_name}"
    destination.write_bytes(content)
    return destination


def _load_pdf(path: Path, source_id: str) -> list[SourcePage]:
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise DocumentLoadError(f"Could not open PDF `{path.name}`: {exc}") from exc

    pages: list[SourcePage] = []
    for number, pdf_page in enumerate(reader.pages, start=1):
        try:
            text = pdf_page.extract_text() or ""
        except Exception as exc:
            raise DocumentLoadError(
                f"Could not extract text from `{path.name}`, page {number}: {exc}"
            ) from exc
        # Preserve the original PDF page number for citations, including pages
        # without selectable text.
        pages.append(SourcePage(text, path.name, source_id, number, path))
    return pages


def _load_docx(path: Path, source_id: str) -> list[SourcePage]:
    try:
        document = DocxDocument(str(path))
    except Exception as exc:
        raise DocumentLoadError(f"Could not open DOCX `{path.name}`: {exc}") from exc

    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    tables: list[str] = []
    for table in document.tables:
        for row in table.rows:
            values = [cell.text.strip() for cell in row.cells]
            if any(values):
                tables.append(" | ".join(values))
    text = "\n\n".join(paragraphs + tables)
    return [SourcePage(text, path.name, source_id, None, path)]


def _load_plain_text(path: Path, source_id: str) -> list[SourcePage]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise DocumentLoadError(f"Could not read `{path.name}`: {exc}") from exc
    return [SourcePage(text, path.name, source_id, None, path)]


def load_document(path: Path) -> list[SourcePage]:
    """Extract PDF pages or a single TXT/Markdown/DOCX section locally."""
    path = Path(path)
    if not path.is_file():
        raise DocumentLoadError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DocumentLoadError(f"Unsupported file type: `{suffix or 'none'}`")
    source_id = source_id_for_path(path)
    if suffix == ".pdf":
        return _load_pdf(path, source_id)
    if suffix == ".docx":
        return _load_docx(path, source_id)
    return _load_plain_text(path, source_id)


def recursive_character_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Use the lesson's separator priority before cutting a long text window.

    The priority (paragraph, line, sentence, word, character) mirrors
    ``RecursiveCharacterTextSplitter`` without making LangChain a runtime
    requirement for this small local app.
    """
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")
    if not 0 <= overlap < chunk_size:
        raise ValueError("Overlap must be zero or greater and smaller than chunk size.")
    clean = re.sub(r"\r\n?", "\n", text).strip()
    if not clean:
        return []
    # This is the lesson's recursive splitter and separator ordering from
    # ``5_recursive_character_text_spliiter.py``, rather than a custom clone.
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    return splitter.split_text(clean)


def make_chunks(pages: list[SourcePage], chunk_size: int, overlap: int) -> list[Chunk]:
    """Convert extracted pages into page-preserving retrieval chunks."""
    # Convert to the lesson's Document shape before splitting. Metadata is
    # deliberately copied so chunks retain source filename and PDF page.
    documents = [
        Document(
            page_content=page.text,
            metadata={
                "source_name": page.source_name,
                "source_id": page.source_id,
                "page": page.page or 0,
                "page_available": page.page is not None,
            },
        )
        for page in pages
        if page.text.strip()
    ]
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    split_documents = splitter.split_documents(documents)
    chunks: list[Chunk] = []
    for chunk_index, document in enumerate(split_documents):
        metadata = document.metadata
        page = int(metadata["page"]) if metadata.get("page_available") else None
        raw_id = f"{metadata['source_id']}|{page}|{chunk_index}|{document.page_content}".encode("utf-8")
        chunks.append(
            Chunk(
                id=hashlib.sha256(raw_id).hexdigest(),
                text=document.page_content,
                source_name=str(metadata["source_name"]),
                source_id=str(metadata["source_id"]),
                page=page,
                chunk_index=chunk_index,
            )
        )
    return chunks

