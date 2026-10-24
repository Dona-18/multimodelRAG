"""Streamlit Web Application for Local Multimodal RAG Assistant.

Simple, beginner-friendly RAG workflow:
1. Ingestion: Upload documents (PDF, DOCX, TXT, images)
2. Storage: ChromaDB vector store
3. Retrieval: Hybrid search (Vector + BM25 keyword + RRF)
4. Generation: Grounded answers with Ollama LLM
"""
import sys
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


# Cache vector store connection across reruns to avoid SQLite lock issues
@st.cache_resource
def load_vector_store():
    return get_vector_store(CHROMA_DIR)


db = load_vector_store()

# State management
if "messages" not in st.session_state:
    st.session_state.messages = []
if "all_documents" not in st.session_state:
    st.session_state.all_documents = []

# Load existing documents if not already loaded
if not st.session_state.all_documents:
    st.session_state.all_documents = get_all_stored_documents(db)

has_docs = len(st.session_state.all_documents) > 0

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Status")
    if check_ollama_status():
        st.success("Ollama: Online")
    else:
        st.error("Ollama: Offline (run `ollama serve`)")

    st.caption(f"**LLM:** `{LLM_MODEL}`")
    st.caption(f"**Embeddings:** `{EMBEDDING_MODEL}`")
    st.caption(f"**Processed Chunks:** `{len(st.session_state.all_documents)}`")

    st.markdown("---")
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, TXT, or Image",
        type=["pdf", "docx", "txt", "md", "png", "jpg", "jpeg", "webp"],
    )
    if st.button("📥 Process Uploaded File", use_container_width=True):
        if uploaded_file:
            save_path = UPLOADS_DIR / uploaded_file.name
            save_path.write_bytes(uploaded_file.getbuffer())
            with st.spinner("Processing file and extracting visual content..."):
                docs = load_any_document(save_path)
                chunks = split_documents(docs)
                add_documents_to_store(chunks, db)
                st.session_state.all_documents = get_all_stored_documents(db)

            img_count = sum(1 for d in docs if d.metadata.get("type") == "image")
            if img_count > 0:
                st.success(f"Processed {len(chunks)} chunks (including {img_count} visual diagrams/images)!")
            else:
                st.success(f"Processed {len(chunks)} chunks!")
            st.rerun()
        else:
            st.warning("Please choose a file first.")

    st.markdown("---")
    if st.button("🗑️ Clear Processed Documents", use_container_width=True):
        clear_vector_store(db)
        st.session_state.all_documents = []
        st.session_state.messages = []
        st.success("Database cleared!")
        st.rerun()


# --- MAIN SCREEN ---
st.title("Multimodal RAG Assistant")

if has_docs:
    sources = list(dict.fromkeys(d.metadata.get("source", "Doc") for d in st.session_state.all_documents))
    st.caption(f"🟢 **Ready to chat!** {len(st.session_state.all_documents)} chunks processed from: `{'`, `'.join(sources)}`")

# Show chat messages
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
                        st.image(str(image_path), caption=f"Extracted image from {source} (Page {page})", use_container_width=True)
                    st.write(doc.page_content)


# --- CHAT INPUT ---
prompt_text = "Ask a question about your documents..." if has_docs else "Upload a document in the sidebar to begin..."
user_query = st.chat_input(prompt_text, disabled=not has_docs)

if user_query and has_docs:
    # 1. Add user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 2. Retrieve & generate answer
    with st.spinner("Searching documents & generating answer..."):
        retrieved_docs = hybrid_search(user_query, db, st.session_state.all_documents, k=4)
        result = generate_grounded_answer(user_query, retrieved_docs)
        answer = result["answer"]

    # 3. Add assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "evidence": retrieved_docs,
    })
    st.rerun()
