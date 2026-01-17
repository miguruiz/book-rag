# Alice Wonderland RAG

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about "Alice's Adventures in Wonderland" using local LLMs via Ollama and ChromaDB for vector storage.

## Requirements

- Python 3.12+
- [Ollama](https://ollama.ai/) running locally
- Required Ollama models:
  - `nomic-embed-text` (embeddings)
  - `mannix/llama3.1-8b-abliterated` (chat)

## Installation

```bash
# Install dependencies using uv
uv sync

# Pull the required Ollama models
ollama pull nomic-embed-text
ollama pull mannix/llama3.1-8b-abliterated
```

## Usage

### 1. Ingest the data

First, build the vector database from the source text:

```bash
uv run python main.py
```

This reads `data/alice.txt`, splits it into paragraphs, generates embeddings, and stores them in ChromaDB.

### 2. Query via CLI

Run a single query from the command line:

```bash
uv run python query.py
```

### 3. Web Interface

Launch the Streamlit chat interface:

```bash
uv run streamlit run web.py
```

Then open http://localhost:8501 in your browser.

## Project Structure

```
alice-wonderland/
├── main.py          # Data ingestion script
├── query.py         # CLI query interface
├── web.py           # Streamlit web UI
├── data/
│   └── alice.txt    # Source text (Alice in Wonderland)
├── pyproject.toml   # Project configuration
└── uv.lock          # Dependency lock file
```

## How It Works

1. **Ingestion**: Text is split into paragraphs and embedded using `nomic-embed-text`
2. **Storage**: Embeddings are stored in a local ChromaDB database
3. **Query**: User questions are embedded and matched against stored paragraphs
4. **Generation**: Retrieved context is passed to the LLM to generate grounded answers
