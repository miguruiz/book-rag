import streamlit as st
import httpx
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page Setup
st.set_page_config(page_title="Book RAG", page_icon="📚")


def api_get(endpoint: str):
    """GET request to API."""
    try:
        resp = httpx.get(f"{API_URL}{endpoint}", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        st.error("Cannot connect to API. Is it running?")
        st.code(f"uv run uvicorn api:app --reload")
        st.stop()


def api_post(endpoint: str, **kwargs):
    """POST request to API."""
    resp = httpx.post(f"{API_URL}{endpoint}", timeout=120, **kwargs)
    resp.raise_for_status()
    return resp.json()


# Sidebar - Book Selection & Upload
with st.sidebar:
    st.header("📚 Library")

    # Get available books
    books = api_get("/books")

    # Book selector
    if books:
        book_options = ["All Books"] + books
        selected = st.selectbox("Chat with:", book_options)
        selected_book = None if selected == "All Books" else selected
    else:
        st.info("No books yet. Upload one below!")
        selected_book = None

    st.divider()

    # Upload section
    st.subheader("Add a Book")
    uploaded_file = st.file_uploader("Upload .txt file", type=["txt"])

    if uploaded_file:
        # Book metadata inputs
        default_id = uploaded_file.name.replace(".txt", "").lower().replace(" ", "-")
        book_id = st.text_input("Book ID", value=default_id)
        book_title = st.text_input("Title", value=default_id.replace("-", " ").title())

        if st.button("Ingest Book", type="primary"):
            with st.spinner("Uploading and embedding..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/plain")}
                params = {"book_id": book_id, "title": book_title}
                result = api_post("/books", files=files, params=params)

            st.success(f"Added '{result['title']}' ({result['chunks']} chunks)")
            st.rerun()

# Main chat area
if selected_book:
    st.title(f"💬 Chat with {selected_book}")
else:
    st.title("💬 Chat with your Books")

# Chat History State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if not books:
    st.warning("Upload a book in the sidebar to get started!")
elif prompt := st.chat_input("Ask a question..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query API
    with st.spinner("Thinking..."):
        result = api_post("/query", json={
            "question": prompt,
            "book_id": selected_book,
            "n_results": 3
        })

    # Display Assistant message
    with st.chat_message("assistant"):
        st.markdown(result["answer"])
        with st.expander("View sources"):
            for source in result["sources"]:
                st.caption(f"**[{source['book']}]** {source['text']}...")

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
