"""Database abstraction layer for ChromaDB.

This module handles all interactions with the ChromaDB vector database,
providing a clean interface for storing and retrieving book chunks.
"""

import chromadb
from .config import settings

# Module-level client cache
_client = None


def get_client() -> chromadb.ClientAPI:
    """Get or create the ChromaDB client (singleton pattern)."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_db_path)
    return _client


def get_collection() -> chromadb.Collection:
    """Get or create the books collection."""
    client = get_client()
    return client.get_or_create_collection(name=settings.collection_name)


def get_books() -> list[str]:
    """
    Get list of unique book IDs in the collection.

    Returns:
        Sorted list of book IDs.

    Example:
        >>> get_books()
        ['alice', 'wizard-of-oz']
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    results = collection.get(include=["metadatas"])
    books = set(m.get("book") for m in results["metadatas"] if m.get("book"))
    return sorted(books)


def delete_book(book_id: str) -> None:
    """
    Delete all chunks for a specific book.

    Args:
        book_id: The ID of the book to delete.
    """
    collection = get_collection()
    collection.delete(where={"book": book_id})


def add_chunks(
    chunks: list[str],
    embeddings: list[list[float]],
    book_id: str,
    title: str
) -> None:
    """
    Add chunks to the collection with their embeddings.

    Args:
        chunks: List of text chunks.
        embeddings: List of embedding vectors (same length as chunks).
        book_id: Unique identifier for the book.
        title: Human-readable title.
    """
    collection = get_collection()

    collection.add(
        ids=[f"{book_id}_chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{"book": book_id, "title": title} for _ in chunks]
    )


def search(
    query_embedding: list[float],
    n_results: int = 3,
    book_id: str | None = None
) -> dict:
    """
    Search for similar chunks using a query embedding.

    Args:
        query_embedding: The embedding vector of the query.
        n_results: Number of results to return.
        book_id: Optional filter by book ID.

    Returns:
        ChromaDB query results with documents and metadatas.
    """
    collection = get_collection()

    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": n_results
    }
    if book_id:
        query_params["where"] = {"book": book_id}

    return collection.query(**query_params)
