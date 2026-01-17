# Local development Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev

# Copy application
COPY src/ ./src/
COPY api.py web.py main.py query.py ./
COPY data/ ./data/

# Default to standalone mode (no FastAPI)
ENV LLM_PROVIDER=gemini
ENV STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "web.py", "--server.port=8501", "--server.address=0.0.0.0"]
