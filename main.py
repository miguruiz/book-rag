import ollama
import chromadb

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
DB_PATH = "./chroma_db"
COLLECTION_NAME = "books"


def get_client():
    """Get ChromaDB client."""
    return chromadb.PersistentClient(path=DB_PATH)


def get_collection():
    """Get or create the books collection."""
    client = get_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into fixed-size chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def clean_gutenberg_text(text):
    """Strip Project Gutenberg header/footer if present."""
    if "*** START OF" in text:
        text = text.split("*** START OF")[1].split("***", 1)[1]
    if "*** END OF" in text:
        text = text.split("*** END OF")[0]
    return text


def get_books():
    """Get list of unique book IDs in the collection."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    results = collection.get(include=["metadatas"])
    books = set(m.get("book") for m in results["metadatas"] if m.get("book"))
    return sorted(books)


def delete_book(book_id):
    """Delete all chunks for a specific book."""
    collection = get_collection()
    collection.delete(where={"book": book_id})


def ingest_book(text, book_id, title=None, progress_callback=None):
    """
    Ingest a book into the vector database.

    Args:
        text: Raw text content of the book
        book_id: Unique identifier for the book (e.g., "alice")
        title: Human-readable title (e.g., "Alice in Wonderland")
        progress_callback: Optional callback(current, total) for progress updates

    Returns:
        Number of chunks ingested
    """
    collection = get_collection()

    # Clean and chunk
    clean_text = clean_gutenberg_text(text)
    chunks = chunk_text(clean_text)

    # Delete existing chunks for this book (re-ingestion)
    delete_book(book_id)

    # Embed and store
    for i, chunk in enumerate(chunks):
        response = ollama.embeddings(model="nomic-embed-text", prompt=chunk)

        collection.add(
            ids=[f"{book_id}_chunk_{i}"],
            embeddings=[response["embedding"]],
            documents=[chunk],
            metadatas=[{"book": book_id, "title": title or book_id}]
        )

        if progress_callback:
            progress_callback(i + 1, len(chunks))

    return len(chunks)


# CLI entry point
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main.py <file.txt> [book_id] [title]")
        print("Example: python main.py data/alice.txt alice 'Alice in Wonderland'")
        sys.exit(1)

    file_path = sys.argv[1]
    book_id = sys.argv[2] if len(sys.argv) > 2 else file_path.split("/")[-1].replace(".txt", "")
    title = sys.argv[3] if len(sys.argv) > 3 else book_id.replace("-", " ").title()

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"📥 Ingesting '{title}' (id: {book_id})...")

    def show_progress(current, total):
        if current % 50 == 0 or current == total:
            print(f"  Processed {current}/{total} chunks...")

    num_chunks = ingest_book(text, book_id, title, progress_callback=show_progress)
    print(f"✅ Done! Added {num_chunks} chunks to knowledge base.")
