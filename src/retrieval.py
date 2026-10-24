"""Reference-style dense/BM25 retrieval, multi-query expansion, and RRF."""
from __future__ import annotations

import re
from collections import defaultdict

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from rank_bm25 import BM25Plus

from src.config import DEFAULT_MIN_SIMILARITY, RRF_K
from src.ollama_client import get_chat_model
from src.settings import GenerationSettings
from src.types import Chunk, RetrievalResult, SearchHit
from src.vector_store import VectorStore


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def keyword_search(query: str, chunks: list[Chunk], k: int) -> list[tuple[Chunk, float]]:
    """The same BM25 keyword-search complement demonstrated in lesson 12."""
    tokens = _tokens(query)
    if not chunks or not tokens:
        return []
    corpus = [_tokens(chunk.text) for chunk in chunks]
    scores = BM25Plus(corpus).get_scores(tokens)
    ranked = sorted(zip(chunks, scores), key=lambda pair: float(pair[1]), reverse=True)
    return [(chunk, float(score)) for chunk, score in ranked[:k] if score > 0]


class QueryVariations(BaseModel):
    """The Pydantic + JsonOutputParser shape used in lesson 10."""

    queries: list[str] = Field(description="Alternative retrieval queries")


def generate_query_variations(
    query: str,
    llm: ChatOllama | None = None,
    count: int = 3,
    generation_settings: GenerationSettings | None = None,
) -> list[str]:
    """Generate lesson-style query variations, with safe fallback to the original."""
    parser = JsonOutputParser(pydantic_object=QueryVariations)
    prompt = PromptTemplate(
        template=(
            "Generate {count} different variations of this query that would help retrieve "
            "relevant documents. Keep the original meaning and return only JSON.\n\n"
            "Original query: {original_query}\n{format_instructions}"
        ),
        input_variables=["original_query", "count"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    try:
        response = (prompt | (llm or get_chat_model(generation_settings)) | parser).invoke(
            {"original_query": query, "count": count}
        )
        candidates = response["queries"] if isinstance(response, dict) else response.queries
    except Exception:
        # Multi-query is optional; failed structured output must not block local QA.
        return [query]
    unique = [query]
    for candidate in candidates:
        cleaned = str(candidate).strip()
        if cleaned and cleaned.lower() not in {item.lower() for item in unique}:
            unique.append(cleaned)
    return unique[: count + 1]


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[Chunk, float]]], k: int = RRF_K
) -> list[tuple[Chunk, float]]:
    """Implement the lesson's score exactly: ``1 / (60 + rank)``.

    Chunk IDs are used as the key instead of raw text so identical sentences on
    different source pages remain separately citable.
    """
    scores: defaultdict[str, float] = defaultdict(float)
    chunks: dict[str, Chunk] = {}
    for ranked_list in ranked_lists:
        for rank, (chunk, _score) in enumerate(ranked_list, start=1):
            chunks[chunk.id] = chunk
            scores[chunk.id] += 1.0 / (k + rank)
    return sorted(((chunks[chunk_id], score) for chunk_id, score in scores.items()), key=lambda pair: pair[1], reverse=True)


def hybrid_search(
    query: str,
    store: VectorStore,
    top_k: int,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    use_multi_query: bool = False,
    use_hybrid_search: bool = True,
    generation_settings: GenerationSettings | None = None,
    llm: ChatOllama | None = None,
) -> RetrievalResult:
    """Use vector search alone or lesson 12 hybrid lists, with optional multi-query."""
    clean_query = query.strip()
    if not clean_query:
        return RetrievalResult(reason="Please enter a question.")
    all_chunks = store.all_chunks()
    if not all_chunks:
        return RetrievalResult(reason="No documents are indexed yet.")

    candidate_k = max(top_k * 2, 8)
    queries = (
        generate_query_variations(clean_query, llm=llm, generation_settings=generation_settings)
        if use_multi_query
        else [clean_query]
    )
    ranked_lists: list[list[tuple[Chunk, float]]] = []
    bm25_scores: dict[str, float] = {}
    for search_query in queries:
        vector_results = [(chunk, 1.0) for chunk in store.vector_search(search_query, candidate_k)]
        ranked_lists.append(vector_results)
        bm25_results = keyword_search(search_query, all_chunks, candidate_k) if use_hybrid_search else []
        if use_hybrid_search:
            ranked_lists.append(bm25_results)
        for chunk, score in bm25_results:
            bm25_scores[chunk.id] = max(score, bm25_scores.get(chunk.id, 0.0))
    fused = reciprocal_rank_fusion(ranked_lists)

    vector_scores = store.vector_similarity_scores(clean_query, candidate_k)
    hits = [
        SearchHit(
            chunk=chunk,
            vector_similarity=vector_scores.get(chunk.id),
            bm25_score=bm25_scores.get(chunk.id),
            rrf_score=rrf_score,
        )
        for chunk, rrf_score in fused[:top_k]
    ]
    best_similarity = max(vector_scores.values(), default=-1.0)
    if not hits:
        return RetrievalResult(hits=[], sufficient=False, reason="No relevant evidence was retrieved.")
    if best_similarity < min_similarity:
        return RetrievalResult(
            hits=hits,
            sufficient=False,
            reason=(
                f"The closest retrieved chunk scored {best_similarity:.2f}, below the "
                f"evidence threshold of {min_similarity:.2f}."
            ),
        )
    if use_multi_query and use_hybrid_search:
        method = "multi-query hybrid RRF"
    elif use_multi_query:
        method = "multi-query vector RRF"
    elif use_hybrid_search:
        method = "hybrid RRF"
    else:
        method = "vector retrieval"
    return RetrievalResult(hits=hits, sufficient=True, reason=f"Retrieved evidence passed the similarity threshold using {method}.")
