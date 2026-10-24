"""Gradio Web Application for Local Multimodal RAG Assistant."""
import shutil
import urllib.request
from pathlib import Path
from typing import List

import gradio as gr

from src.config import CHROMA_DIR, EMBEDDING_MODEL, LLM_MODEL, OLLAMA_URL, UPLOADS_DIR
from src.generation import generate_grounded_answer
from src.ingestion import load_any_document, split_documents
from src.retrieval import hybrid_search
from src.vector_store import (
    add_documents_to_store,
    clear_vector_store,
    get_all_stored_documents,
    get_vector_store,
)

# Connect to ChromaDB
db = get_vector_store(CHROMA_DIR)
all_documents = get_all_stored_documents(db)


def check_ollama_status() -> str:
    """Check if local Ollama server is running."""
    try:
        url = f"{OLLAMA_URL}/api/tags"
        req = urllib.request.Request(url, headers={"User-Agent": "GradioRAG"})
        with urllib.request.urlopen(req, timeout=2) as res:
            if res.status == 200:
                return "🟢 **Ollama:** Online"
    except Exception:
        pass
    return "🔴 **Ollama:** Offline (`ollama serve` required)"


def get_current_chunk_status() -> str:
    """Return status line showing processed chunk count."""
    if not all_documents:
        return "**Processed Chunks:** 0"
    sources = list(dict.fromkeys(d.metadata.get("source", "Doc") for d in all_documents))
    return f"🟢 **Processed Chunks:** {len(all_documents)} (`{', '.join(sources)}`)"


def process_file(uploaded_file):
    """Process an uploaded document or image into ChromaDB."""
    global all_documents
    if not uploaded_file:
        return "⚠️ Please upload a document or image first.", get_current_chunk_status()

    file_path = Path(uploaded_file.name if hasattr(uploaded_file, "name") else uploaded_file)
    destination = UPLOADS_DIR / file_path.name
    shutil.copy(file_path, destination)

    docs = load_any_document(destination)
    if not docs:
        return f"⚠️ No text or visual content could be extracted from `{file_path.name}`.", get_current_chunk_status()

    chunks = split_documents(docs)
    add_documents_to_store(chunks, db)
    all_documents = get_all_stored_documents(db)

    img_count = sum(1 for d in docs if d.metadata.get("type") == "image")
    if img_count > 0:
        msg = f"✅ Processed `{file_path.name}`: {len(chunks)} chunks (including {img_count} visual diagrams/images)!"
    else:
        msg = f"✅ Processed `{file_path.name}`: {len(chunks)} chunks successfully!"

    return msg, get_current_chunk_status()


def clear_documents():
    """Clear all processed documents from ChromaDB."""
    global all_documents
    clear_vector_store(db)
    all_documents = []
    return "🗑️ All documents cleared from database.", "**Processed Chunks:** 0", []


def chat_response(message: str, history: List[dict]):
    """Retrieve evidence and generate grounded answer."""
    global all_documents
    clean_message = (message or "").strip()
    if not clean_message:
        return "", history

    if not all_documents:
        history.append({"role": "user", "content": clean_message})
        history.append({
            "role": "assistant",
            "content": "⚠️ No documents processed yet. Please upload a PDF, DOCX, TXT, or image in the left panel first.",
        })
        return "", history

    # 1. Retrieve evidence
    retrieved_docs = hybrid_search(clean_message, db, all_documents, k=4)
    result = generate_grounded_answer(clean_message, retrieved_docs)
    answer = result["answer"]

    # 2. Build citations & evidence collapsible section
    evidence_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 1)
        image_path = doc.metadata.get("image_path")

        part = f"**Evidence #{i} — `{source}` (Page {page})**\n\n"
        if image_path and Path(image_path).exists():
            part += f"![Extracted Image](/gradio_api/file={image_path})\n\n"
        part += f"{doc.page_content.strip()}"
        evidence_parts.append(part)

    if evidence_parts:
        evidence_text = "\n\n---\n\n".join(evidence_parts)
        full_reply = (
            f"{answer}\n\n"
            f"<details><summary>🔍 <b>Citations & Evidence ({len(retrieved_docs)} chunks)</b></summary>\n\n"
            f"{evidence_text}\n\n"
            f"</details>"
        )
    else:
        full_reply = answer

    history.append({"role": "user", "content": clean_message})
    history.append({"role": "assistant", "content": full_reply})
    return "", history


# Build Gradio UI
with gr.Blocks(title="Multimodal RAG Assistant") as demo:
    gr.Markdown("# Multimodal RAG Assistant")

    with gr.Row():
        # Left sidebar column
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ System Status")
            ollama_status = gr.Markdown(check_ollama_status())
            gr.Markdown(f"- **LLM:** `{LLM_MODEL}`\n- **Embeddings:** `{EMBEDDING_MODEL}`")
            chunk_status = gr.Markdown(get_current_chunk_status())

            gr.Markdown("---")
            gr.Markdown("### 📁 Upload Document")
            file_input = gr.File(
                label="Choose PDF, DOCX, TXT, or Image",
                file_types=[".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg", ".webp"],
            )
            process_btn = gr.Button("📥 Process Uploaded File", variant="primary")
            process_log = gr.Markdown("")

            gr.Markdown("---")
            clear_btn = gr.Button("🗑️ Clear Processed Documents", variant="secondary")

        # Right chat column
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Chat",
                height=520,
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask a question about your documents...",
                    show_label=False,
                    scale=5,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

    # Event handlers
    process_btn.click(
        fn=process_file,
        inputs=[file_input],
        outputs=[process_log, chunk_status],
    )

    clear_btn.click(
        fn=clear_documents,
        inputs=[],
        outputs=[process_log, chunk_status, chatbot],
    )

    send_btn.click(
        fn=chat_response,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot],
    )

    msg_input.submit(
        fn=chat_response,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot],
    )

if __name__ == "__main__":
    demo.launch()
