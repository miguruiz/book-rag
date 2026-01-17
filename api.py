"""FastAPI REST API for Book RAG."""

from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from src import get_books, delete_book, ingest_book, query_books, settings

# Validate settings on startup
settings.validate()

app = FastAPI(
    title="Book RAG API",
    description="Chat with your books using RAG",
    version="0.2.0"
)


# --- Request/Response Models ---

class QueryRequest(BaseModel):
    question: str
    book_id: str | None = None
    n_results: int = 3


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


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
def query(req: QueryRequest):
    """Query books using RAG."""
    try:
        result = query_books(req.question, req.book_id, req.n_results)
        return QueryResponse(**result)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "provider": settings.llm_provider}
