import ollama
import chromadb

# 1. Connect to our Alice DB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="alice_collection")

# 2. Your Question
query = "What is the last pharagrap that is actually part of the book, not information"

# 3. Embed the question
query_emb = ollama.embeddings(model="nomic-embed-text", prompt=query)["embedding"]

# 4. Find the most relevant paragraph (n_results=1 for specific answers)
results = collection.query(query_embeddings=[query_emb], n_results=1)
context = results["documents"][0][0]

# 5. Ask Llama to answer based on that context
response = ollama.chat(model="mannix/llama3.1-8b-abliterated", messages=[
    {"role": "system", "content": "Answer the question using ONLY the provided context from Alice in Wonderland. Be brief."},
    {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
])

print(f"\n🔍 FOUND CONTEXT:\n{context[:200]}...")
print(f"\n🤖 AI ANSWER:\n{response['message']['content']}")