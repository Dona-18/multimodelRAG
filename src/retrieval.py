"""Simple retrieval: Vector search, BM25 keyword search, and RRF."""
import re
from collections import defaultdict
from typing import List, Tuple
from rank_bm25 import BM25Plus
from langchain_core.documents import Document
from langchain_chroma import Chroma


def vector_search(query: str, db: Chroma, k: int = 4) -> List[Document]:
    """Search ChromaDB using embeddings similarity."""
    return db.as_retriever(search_kwargs={"k": k}).invoke(query)


def keyword_search(query: str, documents: List[Document], k: int = 4) -> List[Document]:
    """Search documents by exact keyword matches using BM25."""
    if not documents:
        return []
    tokens = re.findall(r"\w+", query.lower())
    if not tokens:
        return documents[:k]

    corpus = [re.findall(r"\w+", doc.page_content.lower()) for doc in documents]
    bm25 = BM25Plus(corpus)
    scores = bm25.get_scores(tokens)

    ranked = sorted(zip(documents, scores), key=lambda item: item[1], reverse=True)
    results = [doc for doc, score in ranked if score > 0]
    return results[:k] if results else documents[:k]


def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    k: int = 60,
) -> List[Tuple[Document, float]]:
    """Combine multiple ranked search results using RRF score = 1 / (60 + rank)."""
    scores = defaultdict(float)
    unique_docs = {}

    for doc_list in ranked_lists:
        for rank, doc in enumerate(doc_list, 1):
            unique_docs[doc.page_content] = doc
            scores[doc.page_content] += 1.0 / (k + rank)

    return sorted(
        [(unique_docs[key], score) for key, score in scores.items()],
        key=lambda item: item[1],
        reverse=True,
    )


def hybrid_search(
    query: str,
    db: Chroma,
    all_documents: List[Document] = None,
    k: int = 4,
) -> List[Document]:
    """Run vector search and BM25 together, merging top results with RRF."""
    vector_results = vector_search(query, db, k=k * 2)
    if all_documents:
        bm25_results = keyword_search(query, all_documents, k=k * 2)
        fused = reciprocal_rank_fusion([vector_results, bm25_results], k=60)
        return [doc for doc, score in fused[:k]]
    return vector_results[:k]
