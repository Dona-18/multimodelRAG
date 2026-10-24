"""A tiny local Ollama client, keeping inference independent of cloud APIs."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
import ollama
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from src.config import EMBEDDING_MODEL, LLM_MODEL, OLLAMA_URL


class OllamaUnavailableError(RuntimeError):
    """Raised with a helpful message when Ollama cannot complete a request."""

class LocalOllama:
    """Use Ollama's Python client for embeddings, chat, and optional vision."""

    def __init__(self, host: str = "http://127.0.0.1:11434") -> None:
        self.client = ollama.Client(host=host)

    def list_models(self) -> set[str]:
        try:
            models = self.client.list().models
            return {model.model for model in models}
        except Exception as exc:  # connection errors differ across client versions
            raise OllamaUnavailableError(
                "Cannot reach Ollama. Start it with `ollama serve`, then retry."
            ) from exc

    def check_required_models(self) -> dict[str, str]:
        installed = self.list_models()
        return {
            EMBEDDING_MODEL: "available" if EMBEDDING_MODEL in installed else "missing",
            LLM_MODEL: "available" if LLM_MODEL in installed else "missing",
        }

    def supports_vision(self, model: str = LLM_MODEL) -> bool:
        try:
            details: Any = self.client.show(model)
            capabilities = details.get("capabilities", []) if isinstance(details, dict) else details.capabilities
            return "vision" in capabilities
        except Exception as exc:
            raise OllamaUnavailableError(f"Could not inspect model `{model}`.") from exc

    def describe_image(self, image_path: Path, page: int, source_name: str) -> str:
        """Ask the installed vision model for a retrieval-oriented page description."""
        if not self.supports_vision():
            raise OllamaUnavailableError(
                f"`{LLM_MODEL}` does not advertise vision support, so image analysis is unavailable."
            )
        try:
            image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
            prompt = (
                "Create a factual, searchable description of this document page. "
                "Mention visible headings, labels, chart axes, key numbers, diagrams, and relationships. "
                "Do not infer facts that are not visible. Return only the description. "
                f"Source: {source_name}; page: {page}."
            )
            # This matches lesson 8: a ChatOllama HumanMessage with text plus
            # a data-URL image, so the vision path uses the same library too.
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ]
            )
            return str(get_chat_model().invoke([message]).content).strip()
        except Exception as exc:
            raise OllamaUnavailableError(
                f"Vision analysis failed for {source_name}, page {page}."
            ) from exc


def get_chat_model() -> ChatOllama:
    """Return the lesson's local ChatOllama configuration for Gemma."""
    return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0, reasoning=False)
