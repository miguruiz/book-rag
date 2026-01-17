#!/bin/bash
# Start both FastAPI and Streamlit

# Start FastAPI in background
uvicorn api:app --host 0.0.0.0 --port 8000 &

# Wait for API to be ready
sleep 2

# Start Streamlit on the main port (Cloud Run expects $PORT)
streamlit run web.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true
