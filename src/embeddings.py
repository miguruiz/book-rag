"""Embeddings abstraction layer.

This module provides a unified interface for generating text embeddings,
supporting multiple providers (Gemini, Ollama) via a single function.
"""

from .config import settings


def get_embedding(text: str) -> list[float]:
    """
    Generate an embedding vector for the given text.

    Args:
        text: The text to embed.

    Returns:
        A list of floats representing the embedding vector.

    Example:
        >>> embedding = get_embedding("Hello, world!")
        >>> len(embedding)  # Dimension varies by model
        768
    """
    if settings.llm_provider == "gemini":
        return _embed_with_gemini(text)
    else:
        return _embed_with_ollama(text)


def _embed_with_gemini(text: str) -> list[float]:
    """Generate embedding using Google Gemini API."""
    import google.generativeai as genai

    genai.configure(api_key=settings.google_api_key)

    response = genai.embed_content(
        model=f"models/{settings.gemini_embedding_model}",
        content=text
    )
    return response["embedding"]


def _embed_with_ollama(text: str) -> list[float]:
    """Generate embedding using local Ollama."""
    import ollama

    response = ollama.embeddings(
        model=settings.ollama_embedding_model,
        prompt=text
    )
    return response["embedding"]
