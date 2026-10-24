"""Visual page enrichment for PDFs using local Gemma vision."""
from __future__ import annotations

import base64
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage
from pypdf import PdfReader

from src.config import RENDERED_PAGES_DIR
from src.ingestion import DocumentLoadError, load_document, make_chunks
from src.ollama_client import get_chat_model
from src.types import Chunk, GenerationSettings, IndexSettings, SourcePage


@dataclass(frozen=True)
class MultimodalReport:
    """User-visible summary of what was added during PDF indexing."""

    visual_pages_indexed: int = 0

    def message(self) -> str:
        if self.visual_pages_indexed:
            return f"created {self.visual_pages_indexed} visual description(s)"
        return "used selectable PDF text"


def _render_pdf_pages(path: Path, pages: list[int], dpi: int = 144) -> dict[int, Path]:
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
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    prompt = f"""Create a factual, searchable description of this document page.
Mention visible headings, labels, tables, chart axes, key numbers, diagrams,
and relationships. Do not infer facts that are not visible. Source: {source_name}; page: {page_number}.
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
        raise DocumentLoadError(f"Local visual analysis failed for `{source_name}`, page {page_number}.") from exc


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
    """Build page-cited PDF chunks with optional Gemma vision descriptions."""
    pages = load_document(path)
    page_by_number = {page.page: page for page in pages}
    chunks = make_chunks(pages, index_settings.chunk_size, index_settings.chunk_overlap)
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

    return chunks, MultimodalReport(visual_pages_indexed=visual_indexed)
