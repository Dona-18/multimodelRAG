"""Streamlit Web Application for Local Multimodal RAG Assistant.

Simple, beginner-friendly RAG workflow:
1. Ingestion: Attach documents directly in the chat UI
2. Storage: ChromaDB vector store
3. Retrieval: Hybrid search (Vector + BM25 keyword + RRF)
4. Generation: Grounded answers with Ollama LLM and visual citations
"""
import sys
import json
from pathlib import Path
import urllib.request
import streamlit as st

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import UPLOADS_DIR, CHROMA_DIR, LLM_MODEL, EMBEDDING_MODEL, OLLAMA_URL
from src.ingestion import load_any_document, split_documents
from src.vector_store import (
    get_vector_store,
    add_documents_to_store,
    get_all_stored_documents,
    clear_vector_store,
)
from src.retrieval import hybrid_search
from src.generation import generate_grounded_answer

# Page config
st.set_page_config(page_title="Multimodal RAG Assistant", layout="centered")


def check_ollama_status() -> bool:
    """Check if local Ollama server is running."""
    try:
        url = f"{OLLAMA_URL}/api/tags"
        req = urllib.request.Request(url, headers={"User-Agent": "StreamlitRAG"})
        with urllib.request.urlopen(req, timeout=2) as res:
            return res.status == 200
    except Exception:
        return False


def get_installed_ollama_models() -> list[str]:
    """Fetch installed models from local Ollama instance."""
    try:
        url = f"{OLLAMA_URL}/api/tags"
        req = urllib.request.Request(url, headers={"User-Agent": "StreamlitRAG"})
        with urllib.request.urlopen(req, timeout=2) as res:
            if res.status == 200:
                data = json.loads(res.read().decode())
                return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []


# Cache vector store connection across reruns to avoid SQLite lock issues
@st.cache_resource
def load_vector_store(embedding_model: str = EMBEDDING_MODEL):
    return get_vector_store(CHROMA_DIR, embedding_model=embedding_model)


# State management
if "all_documents" not in st.session_state:
    st.session_state.all_documents = []

# --- SIDEBAR: STATUS, MODEL SELECTION & PARAMETERS ---
with st.sidebar:
    st.header("⚙️ Status")
    if check_ollama_status():
        st.success("Ollama: Online")
    else:
        st.error("Ollama: Offline (run `ollama serve`)")

    # Fetch installed models dynamically
    installed = get_installed_ollama_models()

    llm_candidates = [m for m in installed if "embed" not in m]
    fallback_llms = ["gemma4:e2b-mlx", "qwen2.5:7b", "llama3.1:8b", "mistral:latest"]
    llm_options = list(dict.fromkeys(llm_candidates + [LLM_MODEL] + fallback_llms))

    embed_candidates = [m for m in installed if "embed" in m or "bge" in m]
    fallback_embeds = ["nomic-embed-text:latest", "bge-m3:latest", "all-minilm:latest"]
    embedding_options = list(dict.fromkeys(embed_candidates + [EMBEDDING_MODEL] + fallback_embeds))

    selected_llm = st.selectbox(
        "LLM",
        options=llm_options,
        index=llm_options.index(LLM_MODEL) if LLM_MODEL in llm_options else 0,
        help="Select the LLM for grounded answer generation and vision descriptions",
    )

    selected_embedding = st.selectbox(
        "Embeddings",
        options=embedding_options,
        index=embedding_options.index(EMBEDDING_MODEL) if EMBEDDING_MODEL in embedding_options else 0,
        help="Select the embedding model for vector storage and retrieval",
    )

    chunks_placeholder = st.empty()

    st.markdown("---")
    st.header("⚙️ Parameters")
    top_k = st.slider(
        "Retrieved Chunks (k)",
        min_value=1,
        max_value=8,
        value=4,
        help="Number of evidence chunks retrieved for context",
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="Lower values give strictly factual answers",
    )

    with st.expander("Chunking Settings", expanded=False):
        chunk_size = st.slider(
            "Chunk Size",
            min_value=300,
            max_value=2000,
            value=1000,
            step=100,
            help="Character limit per text chunk",
        )
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=50,
            max_value=400,
            value=150,
            step=25,
            help="Character overlap between chunks",
        )

    st.markdown("---")
    if st.button("🗑️ Clear Conversation & Documents", width="stretch"):
        db_to_clear = load_vector_store(selected_embedding)
        clear_vector_store(db_to_clear)
        st.session_state.all_documents = []
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Cleared all documents and history. Attach a document to start again!",
            }
        ]
        st.success("Database cleared!")
        st.rerun()


# Load vector store using the user's selected embedding model
db = load_vector_store(selected_embedding)

# Load existing documents if not already loaded
if not st.session_state.all_documents:
    st.session_state.all_documents = get_all_stored_documents(db)

# Update chunks counter in sidebar
chunks_placeholder.caption(f"**Processed Chunks:** `{len(st.session_state.all_documents)}`")

# Initialize chat messages with a welcoming assistant message
if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Welcome to Multimodal RAG Assistant!**\n\n"
                "Upload a PDF, DOCX, TXT, or image in the box above or attach one "
                "in the chat bar below. I will extract text, diagrams, and images "
                "so you can ask questions with visual citations."
            ),
        }
    ]

has_docs = len(st.session_state.all_documents) > 0


def process_and_index_file(name: str, file_bytes: bytes, chunk_size: int, chunk_overlap: int):
    """Helper to save, process, chunk, and index an uploaded document."""
    save_path = UPLOADS_DIR / name
    save_path.write_bytes(file_bytes)
    docs = load_any_document(save_path)
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    add_documents_to_store(chunks, db)
    st.session_state.all_documents = get_all_stored_documents(db)
    img_count = sum(1 for d in docs if d.metadata.get("type") == "image")
    return len(chunks), img_count


# --- MAIN SCREEN: CHAT UI ---
st.title("Multimodal RAG Assistant")

# Document uploader container directly embedded in the Chat UI
with st.container(border=True):
    st.markdown("### 📁 Upload Document")
    st.caption("Upload PDF, DOCX, TXT, or Image")
    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, TXT, or Image",
        type=["pdf", "docx", "txt", "md", "png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
        key="main_doc_uploader",
    )

    if uploaded_file is not None:
        col_btn, col_meta = st.columns([1, 2], vertical_alignment="center")
        with col_btn:
            proc_btn = st.button("📥 Process Document", type="primary", width="stretch")
        with col_meta:
            size_kb = round(uploaded_file.size / 1024, 1)
            st.caption(f"**{uploaded_file.name}** ({size_kb} KB ready to index)")

        if proc_btn:
            with st.spinner(f"Processing `{uploaded_file.name}` and extracting visual diagrams..."):
                num_chunks, img_count = process_and_index_file(
                    uploaded_file.name,
                    uploaded_file.getbuffer(),
                    chunk_size,
                    chunk_overlap,
                )

            # Record confirmation into chat messages so conversation reflects the upload
            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    f"✅ Processed **`{uploaded_file.name}`** into {num_chunks} chunks "
                    f"(including {img_count} visual diagrams/images)!\n\n"
                    "You can now ask any question about this document below."
                ),
            })
            st.rerun()

if has_docs:
    sources = list(dict.fromkeys(d.metadata.get("source", "Doc") for d in st.session_state.all_documents))
    st.caption(f"🟢 **Ready to chat!** {len(st.session_state.all_documents)} chunks indexed from: `{'`, `'.join(sources)}`")

st.markdown("---")

# Render chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "evidence" in msg and msg["evidence"]:
            with st.expander("🔍 Citations & Evidence"):
                for i, doc in enumerate(msg["evidence"], 1):
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", 1)
                    st.markdown(f"**Evidence #{i} — `{source}` (Page {page})**")
                    image_path = doc.metadata.get("image_path")
                    if image_path and Path(image_path).exists():
                        st.image(
                            str(image_path),
                            caption=f"Extracted image from {source} (Page {page})",
                            alt=f"Visual evidence from {source} page {page}",
                            width="stretch",
                        )
                    st.write(doc.page_content)


# --- CHAT INPUT (Allows text and direct file attachments) ---
submission = st.chat_input(
    "Ask a question or attach a document...",
    accept_file=True,
    file_type=["pdf", "docx", "txt", "md", "png", "jpg", "jpeg", "webp"],
)

if submission:
    user_text = ""
    attached_files = []

    # Handle string or ChatInputValue object
    if hasattr(submission, "text"):
        user_text = (submission.text or "").strip()
        attached_files = getattr(submission, "files", []) or []
    elif isinstance(submission, str):
        user_text = submission.strip()

    # Process any attached files from the chat input
    if attached_files:
        for f in attached_files:
            with st.spinner(f"Processing attachment `{f.name}`..."):
                num_chunks, img_count = process_and_index_file(
                    f.name,
                    f.getbuffer(),
                    chunk_size,
                    chunk_overlap,
                )
            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    f"📎 Attached and processed **`{f.name}`** into {num_chunks} chunks "
                    f"(including {img_count} visual diagrams/images)!"
                ),
            })

    # Handle text question if provided
    if user_text:
        # Render and record user message
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        # Generate response
        with st.chat_message("assistant"):
            if not st.session_state.all_documents:
                no_doc_msg = "⚠️ No documents uploaded yet. Please upload or attach a document first so I have context to answer your question."
                st.markdown(no_doc_msg)
                st.session_state.messages.append({"role": "assistant", "content": no_doc_msg})
            else:
                with st.spinner("Searching document & generating grounded answer..."):
                    retrieved_docs = hybrid_search(
                        user_text,
                        db,
                        st.session_state.all_documents,
                        k=top_k,
                    )
                    result = generate_grounded_answer(
                        user_text,
                        retrieved_docs,
                        temperature=temperature,
                        model=selected_llm,
                    )
                    answer = result["answer"]

                st.markdown(answer)
                if retrieved_docs:
                    with st.expander("🔍 Citations & Evidence"):
                        for i, doc in enumerate(retrieved_docs, 1):
                            source = doc.metadata.get("source", "Unknown")
                            page = doc.metadata.get("page", 1)
                            st.markdown(f"**Evidence #{i} — `{source}` (Page {page})**")
                            image_path = doc.metadata.get("image_path")
                            if image_path and Path(image_path).exists():
                                st.image(
                                    str(image_path),
                                    caption=f"Extracted image from {source} (Page {page})",
                                    alt=f"Visual evidence from {source} page {page}",
                                    width="stretch",
                                )
                            st.write(doc.page_content)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "evidence": retrieved_docs,
                })

    st.rerun()
