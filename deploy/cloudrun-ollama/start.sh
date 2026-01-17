#!/bin/bash
# Start both FastAPI and Streamlit (Ollama version)
# Note: Ollama must be running separately (e.g., as a sidecar or separate service)

# Start FastAPI in background
uvicorn api:app --host 0.0.0.0 --port 8000 &

# Wait for API to be ready
sleep 2

# Start Streamlit on the main port
streamlit run web.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true
