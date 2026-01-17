from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from main import get_collection, get_books, ingest_book, delete_book
import ollama

app = FastAPI(
    title="Book RAG API",
    description="Chat with your books using RAG",
    version="0.1.0"
)


# --- Models ---

class QueryRequest(BaseModel):
    question: str
    book_id: str | None = None
    n_results: int = 3


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


class BookInfo(BaseModel):
    book_id: str
    title: str


class IngestResponse(BaseModel):
    book_id: str
    title: str
    chunks: int


# --- Endpoints ---

@app.get("/books", response_model=list[str])
def list_books():
    """List all available books."""
    return get_books()


@app.post("/books", response_model=IngestResponse)
async def upload_book(
    file: UploadFile,
    book_id: str | None = None,
    title: str | None = None
):
    """Upload and ingest a new book."""
    if not file.filename.endswith(".txt"):
        raise HTTPException(400, "Only .txt files are supported")

    content = await file.read()
    text = content.decode("utf-8")

    # Generate defaults from filename
    if not book_id:
        book_id = file.filename.replace(".txt", "").lower().replace(" ", "-")
    if not title:
        title = book_id.replace("-", " ").title()

    num_chunks = ingest_book(text, book_id, title)

    return IngestResponse(book_id=book_id, title=title, chunks=num_chunks)


@app.delete("/books/{book_id}")
def remove_book(book_id: str):
    """Delete a book from the database."""
    if book_id not in get_books():
        raise HTTPException(404, f"Book '{book_id}' not found")

    delete_book(book_id)
    return {"status": "deleted", "book_id": book_id}


@app.post("/query", response_model=QueryResponse)
def query_books(req: QueryRequest):
    """Query books using RAG."""
    collection = get_collection()

    if collection.count() == 0:
        raise HTTPException(400, "No books in database. Upload a book first.")

    # Embed the question
    query_emb = ollama.embeddings(model="nomic-embed-text", prompt=req.question)["embedding"]

    # Query with optional book filter
    query_params = {
        "query_embeddings": [query_emb],
        "n_results": req.n_results
    }
    if req.book_id:
        query_params["where"] = {"book": req.book_id}

    results = collection.query(**query_params)
    contexts = results["documents"][0]
    metadatas = results["metadatas"][0]

    # Build context string
    context = "\n\n---\n\n".join(contexts)

    # Generate answer
    response = ollama.chat(model="mannix/llama3.1-8b-abliterated", messages=[
        {"role": "system", "content": "Answer using ONLY the provided context. Be helpful and concise."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.question}"}
    ])

    # Build sources
    sources = [
        {"book": m.get("book", "unknown"), "text": ctx[:200]}
        for ctx, m in zip(contexts, metadatas)
    ]

    return QueryResponse(answer=response["message"]["content"], sources=sources)


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
