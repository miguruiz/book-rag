"""Configuration management for Book RAG.

This module centralizes all configuration, making it easy to switch between
providers (Gemini vs Ollama) using environment variables.

Supports both:
- Environment variables (for Docker/Cloud Run)
- Streamlit secrets (for Streamlit Cloud)
"""

import os
from dataclasses import dataclass


def get_config(key: str, default: str = "") -> str:
    """Get config from environment or Streamlit secrets."""
    # First try environment
    value = os.getenv(key, "")
    if value:
        return value

    # Then try Streamlit secrets (if running in Streamlit)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    return default


@dataclass
class Settings:
    """Application settings loaded from environment or Streamlit secrets."""

    # Provider selection: "gemini" or "ollama"
    llm_provider: str = get_config("LLM_PROVIDER", "gemini")

    # Gemini settings
    google_api_key: str = get_config("GOOGLE_API_KEY", "")
    gemini_chat_model: str = get_config("GEMINI_CHAT_MODEL", "gemini-2.0-flash")
    gemini_embedding_model: str = get_config("GEMINI_EMBEDDING_MODEL", "text-embedding-004")

    # Ollama settings
    ollama_host: str = get_config("OLLAMA_HOST", "http://localhost:11434")
    ollama_chat_model: str = get_config("OLLAMA_CHAT_MODEL", "mannix/llama3.1-8b-abliterated")
    ollama_embedding_model: str = get_config("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

    # ChromaDB settings
    chroma_db_path: str = get_config("CHROMA_DB_PATH", "./chroma_db")
    collection_name: str = get_config("COLLECTION_NAME", "books")

    # Chunking settings
    chunk_size: int = int(get_config("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(get_config("CHUNK_OVERLAP", "100"))

    def validate(self):
        """Validate settings and raise errors for missing required values."""
        if self.llm_provider == "gemini" and not self.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required when using Gemini. "
                "Get one at https://aistudio.google.com/"
            )
        if self.llm_provider not in ("gemini", "ollama"):
            raise ValueError(f"LLM_PROVIDER must be 'gemini' or 'ollama', got '{self.llm_provider}'")


# Global settings instance
settings = Settings()
