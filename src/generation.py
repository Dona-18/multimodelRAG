from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.ollama_client import get_chat_model
from src.types import GenerationSettings, RetrievalResult, SearchHit

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I don't have enough information to answer that question based on the provided documents."
)


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    citations: list[str]
    used_hits: list[SearchHit]
    sufficient: bool


def citation_for(hit: SearchHit) -> str:
    if hit.chunk.page is None:
        return hit.chunk.source_name
    return f"{hit.chunk.source_name}, p. {hit.chunk.page}"


def format_context(hits: list[SearchHit]) -> str:
    """Keep source and page adjacent to every evidence block for grounded QA."""
    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        kind = "visual description" if hit.chunk.modality == "visual_description" else "text"
        blocks.append(f"[Evidence {index} | {kind} | citation: [{citation_for(hit)}]]\n{hit.chunk.text}")
    return "\n\n".join(blocks)


def _evidence_images(hits: list[SearchHit]) -> list[dict[str, object]]:
    """Attach only retrieved local page images, following lesson 8's message form."""
    images: list[dict[str, object]] = []
    for hit in hits:
        image_path = Path(hit.chunk.rendered_image) if hit.chunk.rendered_image else None
        if image_path and image_path.is_file():
            image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
            images.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
    return images


def answer_messages(query: str, retrieval: RetrievalResult) -> list[SystemMessage | HumanMessage]:
    """Build the grounded lesson-style messages shared by streaming and normal QA."""
    prompt = f"""You are a careful local document assistant.
Answer the question using ONLY the evidence below. Do not use outside knowledge.
Every factual sentence must include one or more citations in the exact form
[filename, p. N] or [filename] when the source has no page number.
If the evidence does not answer the question, reply with exactly:
{INSUFFICIENT_EVIDENCE_MESSAGE}

Question: {query}

Evidence:
{format_context(retrieval.hits)}
"""
    message_content: str | list[dict[str, object]] = prompt
    images = _evidence_images(retrieval.hits)
    if images:
        message_content = [{"type": "text", "text": prompt}, *images]
    return [SystemMessage(content="You are a helpful, evidence-grounded assistant."), HumanMessage(content=message_content)]


def make_answer_result(answer: str, retrieval: RetrievalResult) -> AnswerResult:
    citations = list(dict.fromkeys(citation_for(hit) for hit in retrieval.hits))
    sufficient = INSUFFICIENT_EVIDENCE_MESSAGE.lower() not in answer.lower()
    return AnswerResult(answer, citations if sufficient else [], retrieval.hits, sufficient)


def answer_question(
    query: str,
    retrieval: RetrievalResult,
    generation_settings: GenerationSettings | None = None,
    llm: ChatOllama | None = None,
) -> AnswerResult:
    """Use only passed evidence; refuse before asking the model if it is weak."""
    if not retrieval.sufficient:
        return AnswerResult(INSUFFICIENT_EVIDENCE_MESSAGE, [], retrieval.hits, False)
    # This follows lesson 3's SystemMessage + HumanMessage ChatOllama call.
    response = (llm or get_chat_model(generation_settings)).invoke(answer_messages(query, retrieval))
    answer = str(response.content).strip()
    return make_answer_result(answer, retrieval)


def stream_answer(
    query: str,
    retrieval: RetrievalResult,
    generation_settings: GenerationSettings,
    llm: ChatOllama | None = None,
):
    """Yield real ``ChatOllama.stream`` text chunks when streaming is enabled."""
    if not retrieval.sufficient:
        yield INSUFFICIENT_EVIDENCE_MESSAGE
        return
    for response_chunk in (llm or get_chat_model(generation_settings)).stream(answer_messages(query, retrieval)):
        content = response_chunk.content
        if isinstance(content, str) and content:
            yield content