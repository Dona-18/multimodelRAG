"""Tiny, testable UI state rules."""


def chat_is_unlocked(indexed_chunks: int) -> bool:
    """Chat is available only after ingestion has produced local evidence."""
    return indexed_chunks > 0
