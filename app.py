"""A simple, local-first Streamlit interface for one-document RAG."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_VISUAL_PAGE_LIMIT,
    DEFAULT_VISUAL_SUMMARIES,
    EMBEDDING_MODEL,
    LLM_MODEL,
    MODEL_MAX_CONTEXT_LENGTH,
)
from src.generation import answer_question, make_answer_result, stream_answer
from src.ingestion import DocumentLoadError, load_document, make_chunks, save_upload
from src.multimodal import build_pdf_chunks
from src.ollama_client import LocalOllama, OllamaUnavailableError
from src.retrieval import hybrid_search
from src.types import GenerationSettings, IndexSettings, RetrievalSettings
from src.state import (
    chat_is_unlocked,
    clear_active_source,
    load_active_source,
    load_index_settings,
    requires_reingestion,
    save_active_source,
    save_index_settings,
)
from src.vector_store import VectorStore

st.set_page_config(page_title="Multimodal RAG Assistant", page_icon="📚", layout="centered")


@st.cache_resource
def get_services() -> tuple[VectorStore, LocalOllama]:
    return VectorStore(), LocalOllama()


def display_citation(hit) -> str:
    page = f", page {hit.chunk.page}" if hit.chunk.page is not None else ""
    return f"{hit.chunk.source_name}{page}"


def display_name(path: Path) -> str:
    """Hide the upload content fingerprint that prevents local filename clashes."""
    prefix, separator, name = path.name.partition("_")
    return name if separator and len(prefix) == 12 else path.name


def prepare_chunks(path: Path, index_settings: IndexSettings, generation_settings: GenerationSettings):
    """Validate and chunk a document before replacing any working index."""
    if path.suffix.lower() == ".pdf":
        chunks, report = build_pdf_chunks(path, index_settings, generation_settings)
        processing_message = report.message()
    else:
        pages = load_document(path)
        chunks = make_chunks(pages, index_settings.chunk_size, index_settings.chunk_overlap)
        processing_message = "used local text extraction"
    if not chunks:
        raise DocumentLoadError(
            f"`{display_name(path)}` has no indexable text. Enable local OCR for a scanned PDF."
        )
    return chunks, processing_message


def replace_current_document(
    path: Path, index_settings: IndexSettings, generation_settings: GenerationSettings
) -> tuple[int, str]:
    """Replace the one active index only after the new document was prepared safely."""
    if not models_ready:
        raise OllamaUnavailableError(model_error or "Start Ollama and install the required local models before indexing.")
    chunks, processing_message = prepare_chunks(path, index_settings, generation_settings)
    with st.spinner("Creating local embeddings and replacing the current document…"):
        store.clear()
        indexed = store.add_chunks(chunks)
    save_index_settings(index_settings)
    save_active_source(path)
    st.session_state.messages = []
    return indexed, processing_message


def render_upload_control(
    key: str, button_label: str, index_settings: IndexSettings, generation_settings: GenerationSettings
) -> None:
    """Show the app's single-document upload and replacement action."""
    upload = st.file_uploader(
        "Choose one PDF, DOCX, TXT, or Markdown document",
        type=["pdf", "docx", "txt", "md", "markdown"],
        accept_multiple_files=False,
        key=key,
        help="Only this document is processed. Selecting a new document replaces the current document.",
    )
    if st.button(button_label, type="primary", disabled=upload is None, key=f"{key}_index"):
        try:
            saved_path = save_upload(upload.name, upload.getvalue())
            indexed, processing_message = replace_current_document(saved_path, index_settings, generation_settings)
            st.session_state.notice = (
                f"Processed {indexed} chunks from {display_name(saved_path)}; {processing_message}. "
                "Chat is now unlocked."
            )
            st.rerun()
        except (DocumentLoadError, OllamaUnavailableError, ValueError) as exc:
            st.error(str(exc))


store, ollama = get_services()
ADVANCED_DEFAULTS = {
    "advanced_temperature": DEFAULT_TEMPERATURE,
    "advanced_max_output": DEFAULT_MAX_OUTPUT_TOKENS,
    "advanced_context_length": DEFAULT_CONTEXT_LENGTH,
    "advanced_streaming": False,
    "advanced_chunk_size": DEFAULT_CHUNK_SIZE,
    "advanced_chunk_overlap": DEFAULT_CHUNK_OVERLAP,
    "advanced_visual_summaries": DEFAULT_VISUAL_SUMMARIES,
    "advanced_visual_page_limit": DEFAULT_VISUAL_PAGE_LIMIT,
    "advanced_top_k": DEFAULT_TOP_K,
    "advanced_multi_query": False,
    "advanced_hybrid_search": True,
}
for state_key, default in (("messages", []), ("notice", ""), *ADVANCED_DEFAULTS.items()):
    if state_key not in st.session_state:
        st.session_state[state_key] = default

try:
    model_status = ollama.check_required_models()
    models_ready = all(state == "available" for state in model_status.values())
    model_error = ""
except OllamaUnavailableError as exc:
    model_status = {}
    models_ready = False
    model_error = str(exc)

with st.sidebar:
    st.header("Multimodal RAG")
    st.caption("Your document, embeddings, database, and answers stay on this Mac.")
    st.metric("Processed chunks", store.count)

    with st.expander("Advanced Settings"):
        if st.button(
            "Reset to Defaults",
            use_container_width=True,
            help="Restore every advanced setting to its recommended starting value.",
        ):
            st.session_state.update(ADVANCED_DEFAULTS)
            st.rerun()

        st.caption("Generation")
        temperature = st.slider(
            "Temperature", 0.0, 2.0, step=0.05, key="advanced_temperature",
            help="Lower values make answers more consistent. 0.2 is a grounded-answer starting point.",
        )
        max_output_tokens = st.number_input(
            "Max output tokens", min_value=64, max_value=8_192, step=64, key="advanced_max_output",
            help="Maximum length of Gemma's answer. A smaller limit is faster and more concise.",
        )
        context_length = st.select_slider(
            "Context length", options=[2_048, 4_096, 8_192, 16_384, 32_768, 65_536, MODEL_MAX_CONTEXT_LENGTH],
            key="advanced_context_length",
            help="Maximum tokens Gemma can hold for the prompt and answer. Larger values use more memory.",
        )
        streaming = st.checkbox(
            "Stream answer", key="advanced_streaming",
            help="Show Gemma's answer as it is generated instead of waiting for the full response.",
        )

        st.caption("Processing and retrieval")
        chunk_size = st.slider(
            "Chunk size", 300, 2_000, step=50, key="advanced_chunk_size",
            help="Characters per processed chunk. Changing this requires re-processing the current document.",
        )
        chunk_overlap = st.slider(
            "Chunk overlap", 0, min(500, chunk_size - 1), step=25, key="advanced_chunk_overlap",
            help="Characters shared by neighboring chunks. Changing this requires re-processing the current document.",
        )
        st.caption("Visual retrieval")
        use_visual_summaries = st.checkbox(
            "Add visual page descriptions", key="advanced_visual_summaries",
            help="Render the first PDF pages locally and ask installed Gemma vision for searchable descriptions of diagrams, tables, and charts.",
        )
        visual_page_limit = st.number_input(
            "Maximum pages for visual descriptions", min_value=1, max_value=10, step=1,
            key="advanced_visual_page_limit",
            help="Bounds local image rendering and Gemma vision time. Visual settings require re-processing.",
        )
        top_k = st.slider(
            "Retrieval top-k", 1, 20, key="advanced_top_k",
            help="Number of final evidence chunks available to the answer generator.",
        )
        use_hybrid_search = st.checkbox(
            "Use hybrid search", key="advanced_hybrid_search",
            help="Combine semantic vector search and BM25 keyword search with Reciprocal Rank Fusion.",
        )
        use_multi_query = st.checkbox(
            "Use multi-query retrieval", key="advanced_multi_query",
            help="Ask Gemma for three local query rephrasings before retrieval. This improves recall but adds latency.",
        )

    try:
        generation_settings = GenerationSettings(
            temperature=float(temperature), top_p=float(DEFAULT_TOP_P), max_output_tokens=int(max_output_tokens),
            context_length=int(context_length), streaming=streaming,
        )
        retrieval_settings = RetrievalSettings(
            top_k=int(top_k), use_multi_query=use_multi_query, use_hybrid_search=use_hybrid_search,
        )
        index_settings = IndexSettings(
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            use_visual_summaries=use_visual_summaries,
            visual_page_limit=int(visual_page_limit),
        )
        generation_settings.validate()
        retrieval_settings.validate()
        index_settings.validate()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    with st.expander("Local model status"):
        if model_error:
            st.error(model_error)
        else:
            for model, status in model_status.items():
                (st.success if status == "available" else st.error)(f"`{model}` · {status}")

    if load_active_source() and st.button("Clear current document", use_container_width=True):
        store.clear()
        clear_active_source()
        st.session_state.messages = []
        st.session_state.notice = "Cleared the current document. Your uploaded file remains stored locally."
        st.rerun()

st.title("📚 Multimodal RAG Assistant")
st.caption("Upload one document, then ask grounded questions with local citations.")

if st.session_state.notice:
    st.success(st.session_state.notice)
    st.session_state.notice = ""

active_source = load_active_source()
reingestion_required = requires_reingestion(store.count, index_settings)
chat_unlocked = chat_is_unlocked(store.count) and active_source is not None and not reingestion_required

if reingestion_required and active_source is not None:
    stored_index_settings = load_index_settings()
    if stored_index_settings is None:
        st.warning("This document was prepared before chunk-setting tracking. Re-process the current document once to continue.")
    else:
        st.warning("Chunking or PDF processing settings changed. Re-process the current document before asking questions.")
    if st.button("Re-process current document", type="primary"):
        try:
            indexed, processing_message = replace_current_document(active_source, index_settings, generation_settings)
            st.session_state.notice = f"Re-processed {indexed} chunks from {display_name(active_source)}; {processing_message}."
            st.rerun()
        except (DocumentLoadError, OllamaUnavailableError, ValueError) as exc:
            st.error(str(exc))
    st.chat_input("Re-process the document to unlock chat", disabled=True)
elif not chat_unlocked:
    st.subheader("Upload a document to unlock chat")
    if store.count and active_source is None:
        st.info("This older data is not associated with one active document. Upload a document to safely replace it.")
    else:
        st.info("The assistant answers only from the document you process here.")
    render_upload_control("document_upload", "Process document", index_settings, generation_settings)
    st.divider()
    st.chat_input("Upload a document above to unlock chat", disabled=True)
else:
    st.success(f"Ready to answer from {display_name(active_source)} ({store.count} processed chunks).")
    with st.expander("Replace current document"):
        st.caption("Processing a replacement clears the current document only after the new document is readable and chunked.")
        render_upload_control("replacement_upload", "Replace document", index_settings, generation_settings)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("hits"):
                with st.expander("Sources used"):
                    for index, hit in enumerate(message["hits"], start=1):
                        kind = "visual description" if hit.chunk.modality == "visual_description" else "text"
                        st.markdown(f"**{index}. {display_citation(hit)} · {kind}**")
                        st.caption(
                            f"Vector similarity: {hit.vector_similarity if hit.vector_similarity is not None else '—'} · "
                            f"RRF: {hit.rrf_score:.4f}"
                        )
                        st.write(hit.chunk.text)

    question = st.chat_input("Ask a question about the current document")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        try:
            with st.chat_message("assistant"), st.spinner("Searching your local document…"):
                result = hybrid_search(
                    question,
                    store,
                    top_k=retrieval_settings.top_k,
                    min_similarity=DEFAULT_MIN_SIMILARITY,
                    use_multi_query=retrieval_settings.use_multi_query,
                    use_hybrid_search=retrieval_settings.use_hybrid_search,
                    generation_settings=generation_settings,
                )
                if generation_settings.streaming:
                    answer_parts: list[str] = []

                    def stream_parts():
                        for part in stream_answer(question, result, generation_settings):
                            answer_parts.append(part)
                            yield part

                    answer_text = st.write_stream(stream_parts())
                    answer = make_answer_result(str(answer_text), result)
                else:
                    answer = answer_question(question, result, generation_settings)
                    st.markdown(answer.answer)
                st.caption(result.reason)
            st.session_state.messages.append({"role": "assistant", "content": answer.answer, "hits": answer.used_hits})
        except OllamaUnavailableError as exc:
            st.error(str(exc))
