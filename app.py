"""Streamlit Web Application for Local Multimodal RAG & Agentic Assistant.

Workflow:
1. Ingestion: Attach documents directly in the chat UI
2. Storage: ChromaDB vector store with isolated collections per embedding model
3. Retrieval & Tooling: Hybrid search (Vector + BM25 keyword + RRF) bound as agent tools
4. Generation: Autonomous Agentic ReAct loop (Thought -> Action -> Observation) with visual citations
"""
import sys
from pathlib import Path
import urllib.request
import streamlit as st

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    UPLOADS_DIR,
    CHROMA_DIR,
    OLLAMA_URL,
    MODEL_PROFILES,
    DEFAULT_PROFILE,
)
from src.ingestion import load_any_document, split_documents
from src.vector_store import (
    get_vector_store,
    add_documents_to_store,
    get_all_stored_documents,
    clear_vector_store,
)
from src.retrieval import hybrid_search
from src.generation import generate_grounded_answer
from src.agent import run_agentic_rag

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


# Cache vector store connection per collection and embedding model to avoid SQLite lock issues
@st.cache_resource
def load_vector_store(collection_name: str, embedding_model: str):
    return get_vector_store(
        persist_directory=CHROMA_DIR,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )


# --- SIDEBAR: STATUS, MODEL PROFILES & PARAMETERS ---
with st.sidebar:
    st.header("⚙️ Status")
    if check_ollama_status():
        st.success("Service: Online")
    else:
        st.error("Service: Offline (run `ollama serve`)")

    st.markdown("---")
    st.header("🤖 Mode & Architecture")
    execution_mode = st.radio(
        "Workflow Engine",
        options=["Agentic RAG (ReAct Tool Loop)", "Standard Pipeline RAG"],
        index=0,
        help="Agentic RAG uses autonomous Thought -> Action -> Observation tool calling",
    )

    profile_keys = list(MODEL_PROFILES.keys())
    active_profile_key = st.selectbox(
        "Model Profile",
        options=profile_keys,
        format_func=lambda k: MODEL_PROFILES[k]["name"],
        index=0,
        help="Switch between Qwen3 (default) and Gemma 4 / Nomic (legacy fallback)",
    )
    active_profile = MODEL_PROFILES[active_profile_key]

    with st.expander("System Details", expanded=False):
        st.caption(f"**Engine:** `{execution_mode}`")
        st.caption(f"**LLM:** `{active_profile['llm']}`")
        st.caption(f"**Embeddings:** `{active_profile['embedding']}` ({active_profile['embedding_dim']} dims)")
        st.caption(f"**Vision Captioner:** `{active_profile['vision']}`")
        st.caption(f"**Collection:** `{active_profile['collection']}`")

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

# Load database specific to the selected profile collection
db = load_vector_store(
    collection_name=active_profile["collection"],
    embedding_model=active_profile["embedding"],
)

# State management
if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                f"👋 **Welcome to Multimodal RAG Assistant!**\n\n"
                f"Running in **{execution_mode}** with **{active_profile['name']}**.\n\n"
                "Upload a document above or ask questions directly below."
            ),
        }
    ]

# Fetch stored documents for active collection
all_documents = get_all_stored_documents(db)
has_docs = len(all_documents) > 0

with st.sidebar:
    st.caption(f"**Indexed Chunks:** `{len(all_documents)}`")
    st.markdown("---")
    if st.button("🗑️ Clear Current Collection & Chat", width="stretch"):
        clear_vector_store(db)
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": f"Cleared all documents in `{active_profile['collection']}`. Attach a document to start again!",
            }
        ]
        st.success(f"Collection `{active_profile['collection']}` cleared!")
        st.rerun()


def process_and_index_file(name: str, file_bytes: bytes, c_size: int, c_overlap: int):
    """Helper to save, process, chunk, and index an uploaded document."""
    save_path = UPLOADS_DIR / name
    save_path.write_bytes(file_bytes)
    docs = load_any_document(save_path)
    chunks = split_documents(docs, chunk_size=c_size, chunk_overlap=c_overlap)
    add_documents_to_store(chunks, db)
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
        key=f"uploader_{active_profile_key}",
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

            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    f"✅ Processed **`{uploaded_file.name}`** into {num_chunks} chunks "
                    f"(including {img_count} visual diagrams/images) using `{active_profile['embedding']}`!\n\n"
                    "You can now ask any question about this document below."
                ),
            })
            st.rerun()

if has_docs:
    sources = list(dict.fromkeys(d.metadata.get("source", "Doc") for d in all_documents))
    st.caption(f"🟢 **Ready to chat!** {len(all_documents)} chunks indexed in `{active_profile['collection']}` from: `{'`, `'.join(sources)}`")

st.markdown("---")

# Render chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Show agent tool execution steps if present
        if "tool_steps" in msg and msg["tool_steps"]:
            with st.status(f"🤖 Agent executed {len(msg['tool_steps'])} tool actions", state="complete"):
                for step in msg["tool_steps"]:
                    st.markdown(f"**Step {step['turn']}:** Invoked tool `{step['tool']}` with `{step['args']}`")
                    if step.get("output_snippet"):
                        st.caption(f"Observation: {step['output_snippet']}")

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

    if hasattr(submission, "text"):
        user_text = (submission.text or "").strip()
        attached_files = getattr(submission, "files", []) or []
    elif isinstance(submission, str):
        user_text = submission.strip()

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

    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        with st.chat_message("assistant"):
            current_docs = get_all_stored_documents(db)

            if "Agentic" in execution_mode:
                # --- AGENTIC RAG (Autonomous Tool Loop) ---
                with st.spinner(f"Agent reasoning with `{active_profile['llm']}`..."):
                    agent_res = run_agentic_rag(
                        query=user_text,
                        db=db,
                        all_documents=current_docs,
                        model=active_profile["llm"],
                        temperature=temperature,
                        top_k=top_k,
                    )
                    answer = agent_res["answer"]
                    tool_steps = agent_res.get("tool_steps", [])
                    evidence = agent_res.get("evidence", [])

                if tool_steps:
                    with st.status(f"🤖 Agent executed {len(tool_steps)} tool actions", state="complete"):
                        for step in tool_steps:
                            st.markdown(f"**Step {step['turn']}:** Invoked tool `{step['tool']}` with `{step['args']}`")
                            if step.get("output_snippet"):
                                st.caption(f"Observation: {step['output_snippet']}")

                st.markdown(answer)

                if evidence:
                    with st.expander("🔍 Citations & Evidence"):
                        for i, doc in enumerate(evidence, 1):
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
                    "tool_steps": tool_steps,
                    "evidence": evidence,
                })

            else:
                # --- STANDARD PIPELINE RAG ---
                if not current_docs:
                    no_doc_msg = "⚠️ No documents indexed in the current collection. Please upload a document first."
                    st.markdown(no_doc_msg)
                    st.session_state.messages.append({"role": "assistant", "content": no_doc_msg})
                else:
                    with st.spinner(f"Searching `{active_profile['name']}` & generating grounded answer..."):
                        retrieved_docs = hybrid_search(
                            user_text,
                            db,
                            current_docs,
                            k=top_k,
                        )
                        result = generate_grounded_answer(
                            user_text,
                            retrieved_docs,
                            temperature=temperature,
                            model=active_profile["llm"],
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
