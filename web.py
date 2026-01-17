"""Streamlit Web UI for Book RAG.

This can run in two modes:
1. Standalone (Streamlit Cloud): Imports src/ directly
2. With API (Cloud Run): Calls FastAPI backend via HTTP

Set USE_API=true to use API mode.
"""

import os
import streamlit as st

USE_API = os.getenv("USE_API", "false").lower() == "true"
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page Setup
st.set_page_config(page_title="Book RAG", page_icon="📚")


# --- API vs Direct Mode ---

if USE_API:
    import httpx

    def api_get(endpoint: str):
        try:
            resp = httpx.get(f"{API_URL}{endpoint}", timeout=30)
            resp.raise_for_status()
            return resp.json()
        except httpx.ConnectError:
            st.error("Cannot connect to API. Is it running?")
            st.stop()

    def api_post(endpoint: str, **kwargs):
        resp = httpx.post(f"{API_URL}{endpoint}", timeout=120, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def get_books_list():
        return api_get("/books")

    def upload_and_ingest(file, book_id, title):
        files = {"file": (file.name, file.getvalue(), "text/plain")}
        params = {"book_id": book_id, "title": title}
        return api_post("/books", files=files, params=params)

    def query_rag(question, book_id):
        result = api_post("/query", json={"question": question, "book_id": book_id, "n_results": 3})
        return result

else:
    # Direct mode - import from src/
    from src import get_books, ingest_book, query_books, settings

    # Validate on startup
    try:
        settings.validate()
    except ValueError as e:
        st.error(f"Configuration error: {e}")
        st.stop()

    def get_books_list():
        return get_books()

    def upload_and_ingest(file, book_id, title):
        text = file.getvalue().decode("utf-8")
        num_chunks = ingest_book(text, book_id, title)
        return {"title": title, "chunks": num_chunks}

    def query_rag(question, book_id):
        return query_books(question, book_id, n_results=3)


# --- Sidebar: Book Selection & Upload ---

with st.sidebar:
    st.header("📚 Library")

    books = get_books_list()

    if books:
        book_options = ["All Books"] + books
        selected = st.selectbox("Chat with:", book_options)
        selected_book = None if selected == "All Books" else selected
    else:
        st.info("No books yet. Upload one below!")
        selected_book = None

    st.divider()

    st.subheader("Add a Book")
    uploaded_file = st.file_uploader("Upload .txt file", type=["txt"])

    if uploaded_file:
        default_id = uploaded_file.name.replace(".txt", "").lower().replace(" ", "-")
        book_id = st.text_input("Book ID", value=default_id)
        book_title = st.text_input("Title", value=default_id.replace("-", " ").title())

        if st.button("Ingest Book", type="primary"):
            with st.spinner("Uploading and embedding..."):
                result = upload_and_ingest(uploaded_file, book_id, book_title)
            st.success(f"Added '{result['title']}' ({result['chunks']} chunks)")
            st.rerun()


# --- Main Chat Area ---

if selected_book:
    st.title(f"💬 Chat with {selected_book}")
else:
    st.title("💬 Chat with your Books")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not books:
    st.warning("Upload a book in the sidebar to get started!")
elif prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        result = query_rag(prompt, selected_book)

    with st.chat_message("assistant"):
        st.markdown(result["answer"])
        with st.expander("View sources"):
            for source in result["sources"]:
                st.caption(f"**[{source['book']}]** {source['text']}...")

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
