# Book RAG

A RAG (Retrieval-Augmented Generation) chatbot for books. Upload any book and chat with it using AI.

## Features

- **Upload multiple books** via web UI or API
- **Chat with specific books** or search across all
- **Multiple LLM providers**: Google Gemini (cloud) or Ollama (local)
- **REST API** with OpenAPI docs
- **Multiple deployment targets**: Streamlit Cloud, Cloud Run

## Architecture

```
┌─────────────────┐      ┌─────────────────────┐      ┌─────────────────┐
│    Streamlit    │ ───► │      FastAPI        │ ───► │    ChromaDB     │
│    (web.py)     │      │     (api.py)        │      │   (vectors)     │
└─────────────────┘      └─────────────────────┘      └─────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────┐
                    │  LLM Provider (configurable)  │
                    │  - Google Gemini (cloud)      │
                    │  - Ollama (local)             │
                    └───────────────────────────────┘
```

## Quick Start

### Option 1: Streamlit Cloud (Easiest)

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from your fork
4. Add `GOOGLE_API_KEY` in app secrets ([get one free](https://aistudio.google.com/))

### Option 2: Local with Gemini

```bash
# Install dependencies
uv sync

# Set your API key
export GOOGLE_API_KEY="your-key-here"

# Run Streamlit directly (no API server needed)
uv run streamlit run web.py
```

### Option 3: Local with Ollama (fully local)

```bash
# Install and start Ollama
ollama pull nomic-embed-text
ollama pull mannix/llama3.1-8b-abliterated

# Set provider to Ollama
export LLM_PROVIDER=ollama

# Run
uv run streamlit run web.py
```

### Option 4: With FastAPI Backend

```bash
# Terminal 1: API server
uv run uvicorn api:app --reload

# Terminal 2: Streamlit (uses API)
USE_API=true uv run streamlit run web.py
```

## Project Structure

```
book-rag/
├── src/                    # Core library
│   ├── config.py           # Configuration management
│   ├── db.py               # ChromaDB operations
│   ├── embeddings.py       # Embedding abstraction
│   ├── llm.py              # LLM chat abstraction
│   └── rag.py              # RAG orchestration
│
├── api.py                  # FastAPI REST API
├── web.py                  # Streamlit web UI
├── main.py                 # CLI for ingestion
├── query.py                # CLI for querying
│
├── deploy/
│   ├── cloudrun-gemini/    # Cloud Run (Gemini) deployment
│   └── cloudrun-ollama/    # Cloud Run (Ollama) deployment
│
├── .github/workflows/      # CI/CD pipelines
├── requirements.txt        # For Streamlit Cloud
└── pyproject.toml          # For local dev (uv)
```

## Configuration

All configuration via environment variables (or Streamlit secrets):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `gemini` | `gemini` or `ollama` |
| `GOOGLE_API_KEY` | - | Required for Gemini |
| `GEMINI_CHAT_MODEL` | `gemini-2.0-flash` | Gemini model for chat |
| `GEMINI_EMBEDDING_MODEL` | `text-embedding-004` | Gemini model for embeddings |
| `OLLAMA_CHAT_MODEL` | `mannix/llama3.1-8b-abliterated` | Ollama model for chat |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Ollama model for embeddings |
| `USE_API` | `false` | Use FastAPI backend vs direct imports |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/books` | List all books |
| POST | `/books` | Upload and ingest a book |
| DELETE | `/books/{id}` | Delete a book |
| POST | `/query` | Query books with RAG |
| GET | `/health` | Health check |
| GET | `/docs` | OpenAPI documentation |

## Deployment

### Streamlit Cloud

1. Connect repo at [share.streamlit.io](https://share.streamlit.io)
2. Set main file: `web.py`
3. Add secret: `GOOGLE_API_KEY`

### Cloud Run (Gemini)

```bash
# Using Cloud Build
gcloud builds submit --config deploy/cloudrun-gemini/cloudbuild.yaml

# Or manual
docker build -f deploy/cloudrun-gemini/Dockerfile -t book-rag .
gcloud run deploy book-rag --image book-rag --set-secrets GOOGLE_API_KEY=google-api-key:latest
```

### CI/CD

- **Push to main** → Validates code, Streamlit Cloud auto-deploys
- **Tag v*-gcp** → Deploys to Cloud Run

## Development

```bash
# Install dependencies
uv sync

# Run tests (TODO)
uv run pytest

# Format code
uv run ruff format .
```

## License

MIT
