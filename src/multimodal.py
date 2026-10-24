"""Lesson-8-style local OCR, title chunking, and visual page enrichment.

The lesson uses ``unstructured.partition.pdf`` and ``chunk_by_title`` for PDF
elements, then sends text plus images to ``ChatOllama``. This app keeps that
workflow local. It uses Unstructured's ``ocr_only`` strategy for scanned pages
so it can rely on the installed Tesseract engine without downloading the
lesson's optional high-resolution layout model.
"""
from __future__ import annotations

import base64
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.messages import HumanMessage
from pypdf import PdfReader, PdfWriter
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf

from src.config import RENDERED_PAGES_DIR
from src.ingestion import DocumentLoadError, load_document, make_chunks
from src.ollama_client import get_chat_model
from src.settings import GenerationSettings, IndexSettings
from src.types import Chunk, SourcePage

SCANNED_TEXT_THRESHOLD = 80


@dataclass(frozen=True)
class MultimodalReport:
    """Small, user-visible summary of what was added during PDF indexing."""

    scanned_pages_found: int = 0
    ocr_pages_indexed: int = 0
    ocr_pages_skipped: int = 0
    visual_pages_indexed: int = 0

    def message(self) -> str:
        details: list[str] = []
        if self.ocr_pages_indexed:
            details.append(f"OCR indexed {self.ocr_pages_indexed} scanned page(s)")
        if self.ocr_pages_skipped:
            details.append(f"{self.ocr_pages_skipped} scanned page(s) skipped by the OCR limit")
        if self.visual_pages_indexed:
            details.append(f"created {self.visual_pages_indexed} visual description(s)")
        return "; ".join(details) if details else "used selectable PDF text"


def _partition_single_page_with_ocr(path: Path, page_number: int, settings: IndexSettings) -> str:
    """Run the lesson's Unstructured OCR + title chunking on one original page."""
    reader = PdfReader(str(path))
    if page_number < 1 or page_number > len(reader.pages):
        raise DocumentLoadError(f"Page {page_number} is outside `{path.name}`.")

    with TemporaryDirectory(prefix="local-rag-ocr-") as temporary_directory:
        page_path = Path(temporary_directory) / "page.pdf"
        writer = PdfWriter()
        writer.add_page(reader.pages[page_number - 1])
        with page_path.open("wb") as output:
            writer.write(output)
        try:
            elements = partition_pdf(
                filename=str(page_path),
                strategy="ocr_only",
                languages=list(settings.ocr_languages),
            )
            title_chunks = chunk_by_title(
                elements,
                max_characters=settings.chunk_size,
                new_after_n_chars=max(300, settings.chunk_size - settings.chunk_overlap),
                combine_text_under_n_chars=min(500, settings.chunk_size // 2),
            )
        except Exception as exc:
            raise DocumentLoadError(
                f"Local OCR failed for `{path.name}`, page {page_number}. "
                "Confirm that Tesseract is installed and the selected language is available."
            ) from exc

    # ``chunk_by_title`` is the lesson's content-aware grouping technique.
    # Joining its chunks lets the app's page-aware splitter apply the selected
    # overlap while retaining the title boundaries in the source text.
    return "\n\n".join(chunk.text.strip() for chunk in title_chunks if chunk.text and chunk.text.strip())


def _render_pdf_pages(path: Path, pages: list[int], dpi: int = 144) -> dict[int, Path]:
    """Render selected PDF pages locally with Poppler for Gemma vision prompts."""
    if not pages:
        return {}
    page_count = len(PdfReader(str(path)).pages)
    requested = sorted(set(pages))
    if requested[0] < 1 or requested[-1] > page_count:
        raise DocumentLoadError(f"Choose PDF pages between 1 and {page_count}.")

    rendered: dict[int, Path] = {}
    source_key = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    for page_number in requested:
        prefix = RENDERED_PAGES_DIR / f"{source_key}_page_{page_number}"
        images = sorted(prefix.parent.glob(f"{prefix.name}-*.png"))
        if not images:
            try:
                subprocess.run(
                    [
                        "pdftoppm", "-f", str(page_number), "-l", str(page_number), "-r", str(dpi),
                        "-png", str(path), str(prefix),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError as exc:
                raise DocumentLoadError("PDF page rendering needs Poppler (`brew install poppler`).") from exc
            except subprocess.CalledProcessError as exc:
                raise DocumentLoadError(
                    f"Could not render `{path.name}`, page {page_number}: {exc.stderr.strip()}"
                ) from exc
            images = sorted(prefix.parent.glob(f"{prefix.name}-*.png"))
        if len(images) != 1:
            raise DocumentLoadError(f"Expected one rendered image for `{path.name}`, page {page_number}.")
        rendered[page_number] = images[0]
    return rendered


def _describe_page_image(
    image_path: Path,
    source_name: str,
    page_number: int,
    generation_settings: GenerationSettings,
) -> str:
    """Use lesson 8's local ``HumanMessage`` text-plus-image pattern."""
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    prompt = f"""Create a factual, searchable description of this document page.
Mention visible headings, labels, tables, chart axes, key numbers, diagrams,
and relationships. Do not infer facts that are not visible. Include alternative
search terms a user might use. Source: {source_name}; page: {page_number}.
Return only the searchable description."""
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ]
    )
    try:
        return str(get_chat_model(generation_settings).invoke([message]).content).strip()
    except Exception as exc:
        raise DocumentLoadError(
            f"Local visual analysis failed for `{source_name}`, page {page_number}."
        ) from exc


def _visual_chunk(page: SourcePage, description: str, image_path: Path, chunk_index: int) -> Chunk:
    raw_id = f"{page.source_id}|{page.page}|visual|{description}".encode("utf-8")
    return Chunk(
        id=hashlib.sha256(raw_id).hexdigest(),
        text=description,
        source_name=page.source_name,
        source_id=page.source_id,
        page=page.page,
        chunk_index=chunk_index,
        modality="visual_description",
        rendered_image=str(image_path),
    )


def build_pdf_chunks(
    path: Path,
    index_settings: IndexSettings,
    generation_settings: GenerationSettings,
) -> tuple[list[Chunk], MultimodalReport]:
    """Build page-cited PDF chunks with bounded local OCR and optional vision."""
    pages = load_document(path)
    scanned_pages = [page for page in pages if len(page.text.strip()) < SCANNED_TEXT_THRESHOLD]
    page_by_number = {page.page: page for page in pages}
    ocr_indexed = 0

    if index_settings.use_ocr:
        for page in scanned_pages[: index_settings.ocr_page_limit]:
            assert page.page is not None
            ocr_text = _partition_single_page_with_ocr(path, page.page, index_settings)
            if ocr_text:
                page_by_number[page.page] = SourcePage(
                    ocr_text, page.source_name, page.source_id, page.page, page.file_path
                )
                ocr_indexed += 1

    enriched_pages = [page_by_number[page.page] for page in pages]
    chunks = make_chunks(enriched_pages, index_settings.chunk_size, index_settings.chunk_overlap)
    visual_indexed = 0
    if index_settings.use_visual_summaries:
        selected_pages = list(range(1, min(len(pages), index_settings.visual_page_limit) + 1))
        rendered = _render_pdf_pages(path, selected_pages)
        for page_number, image_path in rendered.items():
            page = page_by_number[page_number]
            description = _describe_page_image(image_path, page.source_name, page_number, generation_settings)
            if description:
                chunks.append(_visual_chunk(page, description, image_path, len(chunks)))
                visual_indexed += 1

    report = MultimodalReport(
        scanned_pages_found=len(scanned_pages),
        ocr_pages_indexed=ocr_indexed,
        ocr_pages_skipped=max(0, len(scanned_pages) - ocr_indexed) if index_settings.use_ocr else 0,
        visual_pages_indexed=visual_indexed,
    )
    return chunks, report
