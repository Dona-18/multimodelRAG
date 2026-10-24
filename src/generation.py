"""Grounded answer generation using Ollama."""
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from src.config import LLM_MODEL, OLLAMA_URL


def format_context_blocks(relevant_docs: List[Document]) -> str:
    """Format retrieved documents as bullet points with source and page citations."""
    lines = []
    for doc in relevant_docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 1)
        lines.append(f"- [Source: {source}, Page: {page}]: {doc.page_content.strip()}")
    return "\n\n".join(lines)


def extract_citations(relevant_docs: List[Document]) -> List[Dict[str, Any]]:
    """Extract unique source filename and page numbers."""
    seen = set()
    citations = []
    for doc in relevant_docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 1)
        if (source, page) not in seen:
            seen.add((source, page))
            citations.append({"source": source, "page": page})
    return citations


def generate_grounded_answer(
    query: str,
    retrieved_documents: List[Document],
    temperature: float = 0.1,
    model: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Generate grounded answer using local Ollama model (defaulting to Qwen3)."""
    if not retrieved_documents:
        return {
            "answer": "I don't have enough information to answer that question based on the provided documents.",
            "citations": [],
            "documents": [],
        }

    active_model = model or LLM_MODEL
    context = format_context_blocks(retrieved_documents)
    prompt = f"""Based on the following documents, answer this question: {query}

Documents:
{context}

Provide a concise answer using only the documents above.
Cite the source and page for key facts, e.g. [Source: filename, Page: X].
If the answer is not in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""
    llm = ChatOllama(model=active_model, temperature=temperature, base_url=OLLAMA_URL)
    response = llm.invoke(prompt)

    # In case thinking/reasoning models output special tokens or structure
    answer_text = response.content.strip()

    return {
        "answer": answer_text,
        "citations": extract_citations(retrieved_documents),
        "documents": retrieved_documents,
    }