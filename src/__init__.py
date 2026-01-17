"""Book RAG - A RAG chatbot for books."""

from .config import settings
from .db import get_collection, get_books, delete_book
from .embeddings import get_embedding
from .llm import chat
from .rag import ingest_book, query_books

__all__ = [
    "settings",
    "get_collection",
    "get_books",
    "delete_book",
    "get_embedding",
    "chat",
    "ingest_book",
    "query_books",
]
