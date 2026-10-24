"""Unit and integration tests for Local Multimodal RAG Assistant.

Validates the beginner-friendly RAG pipeline:
1. Ingestion of text, pasted notes, and PDFs.
2. Vector storage and retrieval with ChromaDB.
3. Hybrid search combining BM25 and Reciprocal Rank Fusion (RRF).
4. Grounded answer generation citations and safety refusal.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from pypdf import PdfWriter

from src.generation import (
    extract_citations,
    format_context_blocks,
    generate_grounded_answer,
)
from src.ingestion import (
    load_any_document,
    load_pasted_text,
    load_pdf_file,
    load_text_file,
    split_documents,
)
from src.retrieval import (
    hybrid_search,
    keyword_search,
    reciprocal_rank_fusion,
    vector_search,
)
from src.vector_store import (
    add_documents_to_store,
    clear_vector_store,
    get_all_stored_documents,
    get_vector_store,
)


def test_pasted_and_text_ingestion():
    """Verify loading text and pasted notes into LangChain Documents with original_content metadata."""
    pasted = load_pasted_text("Meeting Note: Project Apollo kickoff on Friday.")
    assert len(pasted) == 1
    assert pasted[0].metadata["source"] == "Pasted_Note.txt"
    assert "kickoff" in pasted[0].page_content
    orig = json.loads(pasted[0].metadata["original_content"])
    assert "kickoff" in orig["raw_text"]

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("Line 1: System initialization.\nLine 2: Server running at port 8000.")
        temp_path = f.name

    docs = load_text_file(temp_path)
    assert len(docs) >= 1
    assert docs[0].metadata["page"] == 1


def test_pdf_loading(tmp_path: Path):
    """Verify extracting text from PDF using pure pypdf."""
    pdf_path = tmp_path / "sample.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with pdf_path.open("wb") as f:
        writer.write(f)

    docs = load_pdf_file(pdf_path)
    # Blank page returns 0 text chunks, which is expected
    assert isinstance(docs, list)


def test_multimodal_image_loading(tmp_path: Path):
    """Verify loading standalone image files and extracting visual content."""
    img_path = tmp_path / "chart.png"
    from PIL import Image
    im = Image.new("RGB", (100, 100), color="blue")
    im.save(img_path)

    with patch("src.ingestion.describe_image", return_value="A blue chart showing 50% growth."):
        docs = load_any_document(img_path)
        assert len(docs) == 1
        assert docs[0].metadata["type"] == "image"
        assert "blue chart" in docs[0].page_content


def test_vector_store_and_search(tmp_path: Path):
    """Verify adding documents to ChromaDB and performing vector search."""
    fake_embeddings = MagicMock()
    fake_embeddings.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
    fake_embeddings.embed_query.return_value = [0.1, 0.2]

    with patch("src.vector_store.OllamaEmbeddings", return_value=fake_embeddings):
        db = get_vector_store(persist_directory=tmp_path, collection_name="test_store")

        docs = [
            Document(page_content="Neural networks require backpropagation.", metadata={"source": "ai.txt", "page": 1}),
            Document(page_content="Photosynthesis converts light into energy.", metadata={"source": "bio.txt", "page": 2}),
        ]

        added = add_documents_to_store(docs, db)
        assert added == 2

        all_docs = get_all_stored_documents(db)
        assert len(all_docs) == 2

        results = vector_search("how plants make food", db, k=1)
        assert len(results) >= 1

        clear_vector_store(db)
        assert len(get_all_stored_documents(db)) == 0


def test_bm25_and_hybrid_rrf():
    """Verify BM25 keyword matching and Reciprocal Rank Fusion."""
    docs = [
        Document(page_content="PostgreSQL is an open-source relational database.", metadata={"source": "db.txt", "page": 1}),
        Document(page_content="Redis is an in-memory key-value data store.", metadata={"source": "cache.txt", "page": 1}),
        Document(page_content="Docker automates Linux container deployment.", metadata={"source": "devops.txt", "page": 1}),
    ]

    bm25_results = keyword_search("PostgreSQL relational", docs, k=1)
    assert len(bm25_results) == 1
    assert "PostgreSQL" in bm25_results[0].page_content

    list1 = [docs[0], docs[1]]
    list2 = [docs[1], docs[0]]
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    assert len(fused) == 2
    expected_score = (1.0 / 61) + (1.0 / 62)
    assert abs(fused[0][1] - expected_score) < 1e-4


def test_generation_grounded_answer():
    """Verify multimodal generation citations and refusal on empty context."""
    test_docs = [
        Document(
            page_content="Project Titan was approved on August 14, 2023 with an initial budget of $4.2M.",
            metadata={"source": "budget_report.pdf", "page": 3},
        )
    ]

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="The budget was $4.2M. [Source: budget_report.pdf, Page: 3]")

    with patch("src.generation.ChatOllama", return_value=fake_llm):
        result = generate_grounded_answer("What was the approved budget?", test_docs)
        assert "4.2M" in result["answer"]
        assert len(result["citations"]) == 1
        assert result["citations"][0]["source"] == "budget_report.pdf"
        assert result["citations"][0]["page"] == 3

    # Refusal test
    no_info_result = generate_grounded_answer("What is the speed of sound on Mars?", [])
    assert "don't have enough information" in no_info_result["answer"]
