# Cloud Run Dockerfile - Ollama version (self-hosted LLM)
# Requires a GPU-enabled Cloud Run instance or separate Ollama server

FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency installation
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install all dependencies including ollama
RUN uv sync --no-dev

# Copy application code
COPY src/ ./src/
COPY api.py web.py ./

# Environment variables
ENV LLM_PROVIDER=ollama
ENV OLLAMA_HOST=http://ollama:11434
ENV USE_API=true
ENV API_URL=http://localhost:8000
ENV PORT=8080

# Expose port
EXPOSE 8080

# Start script
COPY deploy/cloudrun-ollama/start.sh ./
RUN chmod +x start.sh

CMD ["./start.sh"]
