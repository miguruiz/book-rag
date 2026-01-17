# Cloud Run Dockerfile - Gemini API version
# Runs FastAPI backend with Streamlit frontend

FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency installation
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (excluding ollama since we use Gemini)
RUN uv sync --no-dev

# Copy application code
COPY src/ ./src/
COPY api.py web.py ./

# Environment variables
ENV LLM_PROVIDER=gemini
ENV USE_API=true
ENV API_URL=http://localhost:8000
ENV PORT=8080

# Expose port
EXPOSE 8080

# Start both FastAPI and Streamlit using a simple script
COPY deploy/cloudrun-gemini/start.sh ./
RUN chmod +x start.sh

CMD ["./start.sh"]
