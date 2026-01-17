import streamlit as st
import ollama
import chromadb

# Page Setup
st.set_page_config(page_title="Alice RAG", page_icon="🐇")
st.title("🤖 Chat with Alice in Wonderland")


# 1. Initialize DB Connection
@st.cache_resource
def get_db():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_collection(name="alice_collection")


collection = get_db()

# 2. Chat History State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Chat Input
if prompt := st.chat_input("Ask Alice something..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 4. RAG Logic
    with st.spinner("Thinking..."):
        # Get Embeddings & Search
        query_emb = ollama.embeddings(model="nomic-embed-text", prompt=prompt)["embedding"]
        results = collection.query(query_embeddings=[query_emb], n_results=1)
        context = results["documents"][0][0]

        # Generate Answer
        response = ollama.chat(model="mannix/llama3.1-8b-abliterated", messages=[
            {"role": "system", "content": "Answer using ONLY the context. Be brief."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
        ])

        full_response = response['message']['content']

    # 5. Display Assistant message
    with st.chat_message("assistant"):
        st.markdown(full_response)
        st.caption(f"Context used: {context[:100]}...")  # Show the 'evidence'

    st.session_state.messages.append({"role": "assistant", "content": full_response})