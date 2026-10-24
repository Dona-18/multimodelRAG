---
title: Multimodel RAG
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
---

# Multimodal RAG Assistant

A beginner-friendly, local-first document Q&A app for macOS Apple Silicon.
It keeps one current document, its Nomic embeddings, ChromaDB, and all Ollama
inference on your Mac. It does not use a cloud API.

## What was reused from `rag-for-beginners`

This project was intentionally adapted from the lesson material at
`/Users/dona/Desktop/rag-for-beginners`. The lesson files are unchanged.

| Lesson technique | Where it is used here |
| --- | --- |
| `1_ingestion_pipeline.py`: documents → chunks → Nomic → cosine Chroma | `src/ingestion.py`, `src/vector_store.py` with `OllamaEmbeddings` + `Chroma` |
| `5_recursive_character_text_spliiter.py`: paragraph/line/sentence/word split priority | `RecursiveCharacterTextSplitter` in `src/ingestion.py` |
| `2_retrieval_pipeline.py`: `db.as_retriever(...).invoke(query)` | `VectorStore.vector_search` |
| `3_answer_generation.py`: local Gemma answer using retrieved context only | `src/generation.py` |
| `12_hybrid_search.ipynb`: dense + BM25 retrieval | `src/retrieval.py` |
| `10_multi_query_retrieval.py`: JSON query variations using `ChatOllama` | `generate_query_variations` and the optional UI toggle |
| `11_reciprocal_rank_fusion.py`: RRF `1 / (60 + rank)` | `reciprocal_rank_fusion` |

The project uses the same core libraries as the lessons: `langchain-core`,
`langchain-text-splitters`, `langchain-ollama`, and `langchain-chroma`.
One deliberate adaptation is that RRF uses a stable chunk ID rather than chunk
text, so identical text on separate PDF pages remains separately citable.

## Prerequisites

The app expects these local Ollama models, without downloading anything:

```bash
ollama list
# Expect: gemma4:e2b-mlx and nomic-embed-text:latest
```

If Ollama is not already serving, run this in a separate terminal:

```bash
ollama serve
```

## Setup and run

```bash
cd /Users/dona/Projects/local-multimodal-rag-assistant
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest -q
.venv/bin/streamlit run app.py
```

Open the local URL Streamlit prints (normally `http://localhost:8501`).

## How to use the app

1. Confirm the sidebar reports both Ollama models as available.
2. Upload one PDF, DOCX, TXT, or Markdown document, then select **Index document**.
3. Ask questions. The app uses local Nomic vector retrieval and BM25 keyword
   search, fuses results with RRF, and asks local Gemma to answer only from
   that evidence. Enable multi-query retrieval when recall is more important
   than latency.
4. Open **Sources used** under an answer to inspect the chunks and page-aware
   citations. Uploading a replacement document creates its chunks first, then
   replaces the current local index and clears the chat history.

## Advanced settings

**Advanced Settings** controls temperature (default `0.2`), top-p, maximum
output tokens, context length, answer streaming, chunk size (default `1000`),
chunk overlap (default `150`), retrieval top-k (default `5`), multi-query
retrieval, and hybrid search. Generation values are passed to local
`ChatOllama`; retrieval values are passed to the search pipeline.

Chunk size and overlap define the stored index. Changing either locks chat and
requires re-indexing the current document, which prevents mixed chunk
boundaries. **Reset to Defaults** restores every advanced setting.

## Evidence safety

The retrieval gate compares the best cosine similarity to the visible threshold
(default `0.35`). If there is no sufficiently related evidence, the app returns:

> I don't have enough information to answer that question based on the provided documents.

Gemma is also instructed to give that exact response if the retrieved chunks do
not answer the question. This is a guardrail, not a guarantee that any model can
perfectly judge evidence. Review the source chunks for important work.

## Tests and measured retrieval check

`pytest` tests loaders, DOCX table extraction, PDF page metadata, recursive
chunking, one-active-document state, LangChain Chroma persistence, a real local
Nomic embedding/retrieval pass, BM25/RRF, multi-query parsing, source citations,
and the pre-generation refusal gate.

Run the small real-model retrieval check after setup:

```bash
.venv/bin/python evaluation/run_retrieval_eval.py
```

It indexes the tiny `sample_data` set in a temporary ChromaDB, calls the local
Nomic model, and reports recall@3 plus mean and P95 retrieval latency. It does
not generate answers, so its latency isolates embedding/retrieval work.

## Project map

```text
app.py                    Streamlit UI for one active document
src/active_source.py      Persistent current-document pointer
src/ingestion.py          Local file extraction and chunking
src/ollama_client.py      Local model availability and ChatOllama setup
src/vector_store.py       LangChain OllamaEmbeddings + persistent ChromaDB
src/retrieval.py          Vector/BM25 + multi-query + RRF + evidence threshold
src/generation.py         ChatOllama grounded prompts and citations
tests/                    Automated tests
evaluation/               Tiny retrieval test set and latency script
```

## Current scope

Implemented: one-document text ingestion, page-aware PDF citations,
DOCX/TXT/Markdown support, configurable recursive chunking and overlap, local
Nomic embeddings, persistent ChromaDB, hybrid BM25/vector RRF retrieval, local
Gemma answers, a refusal gate, optional multi-query expansion, and a compact
retrieval evaluation set.

Not implemented: OCR for scanned PDFs, page-image rendering or vision analysis,
table-layout reconstruction, semantic/agentic chunking, and reranking. These are
outside this text-focused one-document workflow.
