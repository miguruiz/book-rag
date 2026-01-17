"""LLM chat abstraction layer.

This module provides a unified interface for chat completion,
supporting multiple providers (Gemini, Ollama) via a single function.
"""

from .config import settings

# System prompt for RAG
RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use ONLY the information from the context to answer. If the context doesn't contain
enough information to answer the question, say so honestly.
Be concise and helpful."""


def chat(question: str, context: str) -> str:
    """
    Generate an answer to a question using the provided context.

    This is the core RAG generation step - given retrieved context and a user
    question, generate a grounded answer.

    Args:
        question: The user's question.
        context: Retrieved context from the vector database.

    Returns:
        The generated answer as a string.

    Example:
        >>> answer = chat("Who is the Queen?", "The Queen of Hearts ruled...")
        >>> print(answer)
        "The Queen of Hearts is the ruler who..."
    """
    if settings.llm_provider == "gemini":
        return _chat_with_gemini(question, context)
    else:
        return _chat_with_ollama(question, context)


def _chat_with_gemini(question: str, context: str) -> str:
    """Generate answer using Google Gemini API."""
    import google.generativeai as genai

    genai.configure(api_key=settings.google_api_key)

    model = genai.GenerativeModel(
        model_name=settings.gemini_chat_model,
        system_instruction=RAG_SYSTEM_PROMPT
    )

    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    response = model.generate_content(prompt)

    return response.text


def _chat_with_ollama(question: str, context: str) -> str:
    """Generate answer using local Ollama."""
    import ollama

    response = ollama.chat(
        model=settings.ollama_chat_model,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )

    return response["message"]["content"]
