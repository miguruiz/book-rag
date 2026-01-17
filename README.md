# Book RAG

A RAG (Retrieval-Augmented Generation) chatbot for books. Upload any book and chat with it using local LLMs.

## Architecture

```
┌─────────────────┐      ┌─────────────────────┐      ┌─────────────────┐
│    Streamlit    │ ───► │      FastAPI        │ ───► │    ChromaDB     │
│   (web.py)      │      │     (api.py)        │      │   (vectors)     │
│   Port 8501     │      │     Port 8000       │      │                 │
└─────────────────┘      └─────────────────────┘      └─────────────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │       Ollama        │
                         │  (embeddings + LLM) │
                         └─────────────────────┘
```

## Features

- Upload multiple books via the web UI
- Chat with a specific book or all books at once
- REST API for programmatic access
- OpenAPI docs at `/docs`

## Requirements

- Python 3.12+
- [Ollama](https://ollama.ai/) running locally
- Required Ollama models:
  - `nomic-embed-text` (embeddings)
  - `mannix/llama3.1-8b-abliterated` (chat)

## Installation

```bash
uv sync

ollama pull nomic-embed-text
ollama pull mannix/llama3.1-8b-abliterated
```

## Running Locally

Start both services (in separate terminals):

```bash
# Terminal 1: API
uv run uvicorn api:app --reload

# Terminal 2: Web UI
uv run streamlit run web.py
```

Then open http://localhost:8501

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/books` | List all books |
| POST | `/books` | Upload and ingest a book |
| DELETE | `/books/{id}` | Delete a book |
| POST | `/query` | Query books with RAG |
| GET | `/health` | Health check |
| GET | `/docs` | OpenAPI documentation |

### Example: Query via curl

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Who is the Queen?", "book_id": "alice"}'
```

## Project Structure

```
book-rag/
├── api.py           # FastAPI backend
├── web.py           # Streamlit frontend
├── main.py          # Core ingestion logic
├── query.py         # CLI query tool
├── data/            # Book files
└── chroma_db/       # Vector database
```
