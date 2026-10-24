"""A tiny local Ollama client, keeping inference independent of cloud APIs."""
from __future__ import annotations

import ollama
from langchain_ollama import ChatOllama

from src.config import EMBEDDING_MODEL, LLM_MODEL, OLLAMA_URL
from src.settings import GenerationSettings


class OllamaUnavailableError(RuntimeError):
    """Raised with a helpful message when Ollama cannot complete a request."""


class LocalOllama:
    """Use Ollama's Python client to check the local embedding and chat models."""

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

def get_chat_model(settings: GenerationSettings | None = None) -> ChatOllama:
    """Return the lesson's local ChatOllama configuration for Gemma."""
    active = settings or GenerationSettings()
    active.validate()
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        temperature=active.temperature,
        top_p=active.top_p,
        num_predict=active.max_output_tokens,
        num_ctx=active.context_length,
        reasoning=False,
    )
