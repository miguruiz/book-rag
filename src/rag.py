"""RAG (Retrieval-Augmented Generation) orchestration.

This module ties together embeddings, database, and LLM to provide
high-level RAG operations: ingesting books and answering questions.
"""

from .config import settings
from .db import get_books, delete_book, add_chunks, search, get_collection
from .embeddings import get_embedding
from .llm import chat


def chunk_text(text: str) -> list[str]:
    """
    Split text into fixed-size chunks with overlap.

    Overlap ensures that context isn't lost at chunk boundaries.
    For example, if a sentence spans two chunks, the overlap
    helps preserve that context in both.

    Args:
        text: The full text to chunk.

    Returns:
        List of text chunks.
    """
    chunks = []
    start = 0
    chunk_size = settings.chunk_size
    overlap = settings.chunk_overlap

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap

    return chunks


def clean_gutenberg_text(text: str) -> str:
    """
    Strip Project Gutenberg header/footer if present.

    Gutenberg texts have legal boilerplate at the start and end
    that we don't want polluting our RAG context.
    """
    if "*** START OF" in text:
        text = text.split("*** START OF")[1].split("***", 1)[1]
    if "*** END OF" in text:
        text = text.split("*** END OF")[0]
    return text


def ingest_book(
    text: str,
    book_id: str,
    title: str | None = None,
    progress_callback=None
) -> int:
    """
    Ingest a book into the vector database.

    This is the "indexing" phase of RAG:
    1. Clean the text
    2. Split into chunks
    3. Generate embeddings for each chunk
    4. Store in vector database

    Args:
        text: Raw text content of the book.
        book_id: Unique identifier (e.g., "alice").
        title: Human-readable title (e.g., "Alice in Wonderland").
        progress_callback: Optional callback(current, total) for progress.

    Returns:
        Number of chunks ingested.
    """
    # Clean and chunk
    clean_text = clean_gutenberg_text(text)
    chunks = chunk_text(clean_text)

    # Delete existing chunks for this book (allows re-ingestion)
    delete_book(book_id)

    # Generate embeddings
    embeddings = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        embeddings.append(embedding)

        if progress_callback:
            progress_callback(i + 1, len(chunks))

    # Store in database
    add_chunks(chunks, embeddings, book_id, title or book_id)

    return len(chunks)


def query_books(
    question: str,
    book_id: str | None = None,
    n_results: int = 3
) -> dict:
    """
    Query books using RAG.

    This is the "retrieval + generation" phase of RAG:
    1. Embed the question
    2. Search for similar chunks (retrieval)
    3. Generate answer using LLM (generation)

    Args:
        question: The user's question.
        book_id: Optional filter to specific book.
        n_results: Number of context chunks to retrieve.

    Returns:
        Dict with 'answer' and 'sources'.
    """
    collection = get_collection()
    if collection.count() == 0:
        raise ValueError("No books in database. Ingest a book first.")

    # Step 1: Embed the question
    query_embedding = get_embedding(question)

    # Step 2: Retrieve similar chunks
    results = search(query_embedding, n_results=n_results, book_id=book_id)
    contexts = results["documents"][0]
    metadatas = results["metadatas"][0]

    # Build context string for LLM
    context = "\n\n---\n\n".join(contexts)

    # Step 3: Generate answer
    answer = chat(question, context)

    # Build sources for transparency
    sources = [
        {"book": m.get("book", "unknown"), "text": ctx[:200]}
        for ctx, m in zip(contexts, metadatas)
    ]

    return {"answer": answer, "sources": sources}
