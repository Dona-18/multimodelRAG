---
title: Multimodal RAG Assistant
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
---

# Multimodal RAG Assistant

A local-first, multimodal document Q&A assistant for macOS Apple Silicon.
All vector embeddings, ChromaDB storage, vision captioning, and LLM inference run 100% locally on your Mac via Ollama without cloud APIs or data leakage.

## Architecture & Model Profiles

The system supports configurable model profiles with strict ChromaDB collection isolation:

| Profile | Answer LLM | Embedding Model | Vector Dims | ChromaDB Collection | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen3 (Recommended)** | `qwen3:latest` | `qwen3-embedding:0.6b` | **1024** | `docs_qwen3_embedding_0_6b` | **Default**: SOTA multilingual & reasoning |
| **Gemma 4 + Nomic (Fallback)** | `gemma4:e2b-mlx` | `nomic-embed-text:latest` | **768** | `docs_nomic_embed_text_latest` | Legacy fallback configuration |

> [!IMPORTANT]
> **Collection Isolation**: ChromaDB collections are strictly isolated per embedding model. Embeddings of different dimensions (1024 vs 768) are never mixed.

---

## Key Features

1. **Multimodal Ingestion**: Ingests PDFs, Word (`.docx`), plain text, Markdown, and images. Extracts figures and diagrams from PDF pages.
2. **Local Vision Captioning**: Analyzes extracted images with local VLM (`gemma4:e2b-mlx`) and embeds visual descriptions alongside text.
3. **Hybrid Search with RRF**: Combines dense vector similarity with sparse BM25 keyword matching fused via Reciprocal Rank Fusion ($1 / (60 + rank)$).
4. **Grounded Generation & Citations**: Synthesizes fact-checked answers citing specific document sources and pages, and renders original diagram images in the UI.
5. **Interactive Streamlit UI**: Document uploader embedded directly in the chat container, with support for direct chat bar file attachments (`st.chat_input(accept_file=True)`).

---

## Prerequisites & Ollama Models

Ensure the required local models are pulled via Ollama:

```bash
# Recommended default models
ollama pull qwen3
ollama pull qwen3-embedding:0.6b

# Fallback models
ollama pull gemma4:e2b-mlx
ollama pull nomic-embed-text:latest
```

---

## Setup & Running the App

```bash
cd /Users/dona/Projects/local-multimodal-rag-assistant

# Install dependencies
.venv/bin/pip install -r requirements.txt

# Run automated smoke test suite
.venv/bin/pytest tests/test_assistant.py

# Launch the Streamlit application
.venv/bin/streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Re-embedding Documents

To re-embed all documents in `data/uploads` into a specific model profile collection:

```bash
# Re-embed for Qwen3 (1024 dimensions)
.venv/bin/python src/reembed.py qwen3

# Re-embed for Gemma + Nomic (768 dimensions)
.venv/bin/python src/reembed.py gemma_nomic
```

---

## Benchmark Evaluation

Run the benchmark suite comparing Qwen3 and Gemma/Nomic across 5 ground-truth evaluation pairs:

```bash
.venv/bin/python evaluation/benchmark.py
```

### Measured Performance Summary

| Metric | Qwen3 + Qwen3-Embedding (New) | Gemma 4 + Nomic (Fallback) |
| :--- | :--- | :--- |
| **Retrieval Recall@5** | **100.0%** | 100.0% |
| **Answer Accuracy** | **63.3%** | 56.7% |
| **Faithfulness** | **80.0%** | 80.0% |
| **Citation Accuracy** | **90.0%** | 90.0% |
| **Avg Query Latency** | **17.83s** (with reasoning) | 5.85s |
| **Process Memory** | **158.1 MB** | 127.3 MB |

See `evaluation/benchmark_report.md` for full details.
