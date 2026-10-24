"""Streamlit Web Application for Local Multimodal RAG Assistant.

Simple, beginner-friendly RAG workflow:
1. Ingestion: Paste notes or upload documents (PDF, DOCX, TXT)
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
from src.ingestion import load_any_document, load_pasted_text, split_documents
from src.vector_store import (
    get_vector_store,
    add_documents_to_store,
    get_all_stored_documents,
    clear_vector_store,
)
from src.retrieval import hybrid_search
from src.generation import generate_grounded_answer

# Page config
st.set_page_config(page_title="Multimodal RAG Assistant", page_icon="🧠", layout="centered")


def check_ollama_status() -> bool:
    """Check if local Ollama server is running."""
    try:
        url = f"{OLLAMA_URL}/api/tags"
        req = urllib.request.Request(url, headers={"User-Agent": "StreamlitRAG"})
        with urllib.request.urlopen(req, timeout=2) as res:
            return res.status == 200
    except Exception:
        return False


# State management
if "messages" not in st.session_state:
    st.session_state.messages = []
if "all_documents" not in st.session_state:
    st.session_state.all_documents = []

@st.cache_resource
def load_vector_store():
    return get_vector_store(CHROMA_DIR)

db = load_vector_store()

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
    uploaded_file = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt", "md"])
    if st.button("📥 Process Uploaded File", use_container_width=True):
        if uploaded_file:
            save_path = UPLOADS_DIR / uploaded_file.name
            save_path.write_bytes(uploaded_file.getbuffer())
            with st.spinner("Processing file..."):
                docs = load_any_document(save_path)
                chunks = split_documents(docs)
                add_documents_to_store(chunks, db)
                st.session_state.all_documents = get_all_stored_documents(db)
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
st.title("🧠 Multimodal RAG Assistant")

if not has_docs:
    st.info("🔒 **Chat is locked.** Paste text or upload a document to unlock the assistant.")

    pasted_title = st.text_input("Document Title:", value="My_Notes.txt")
    pasted_text = st.text_area("Paste document text here:", height=180, placeholder="Type or paste notes here...")

    if st.button("🚀 Process Pasted Text & Unlock Chat", type="primary", use_container_width=True):
        if pasted_text.strip():
            with st.spinner("Processing notes..."):
                docs = load_pasted_text(pasted_text, source_name=pasted_title.strip() or "Notes.txt")
                chunks = split_documents(docs)
                add_documents_to_store(chunks, db)
                st.session_state.all_documents = get_all_stored_documents(db)
            st.success("Processed successfully! Chat is now ready.")
            st.rerun()
        else:
            st.warning("Please paste some text first.")
else:
    sources = list(dict.fromkeys(d.metadata.get("source", "Doc") for d in st.session_state.all_documents))
    st.success(f"🟢 **Ready to chat!** {len(st.session_state.all_documents)} chunks processed from: `{'`, `'.join(sources)}`")

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
                        st.write(doc.page_content)


# --- CHAT INPUT ---
prompt_text = "Ask a question about your documents..." if has_docs else "🔒 Paste a document above to unlock chat..."
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
