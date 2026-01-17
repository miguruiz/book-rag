import sys
import ollama
from main import get_collection, get_books

# 1. Connect to our DB
collection = get_collection()

# 2. Parse arguments
if len(sys.argv) < 2:
    print("Usage: python query.py <question> [book_id]")
    print("Example: python query.py 'Who is the Queen?' alice")
    print(f"\nAvailable books: {', '.join(get_books()) or 'none'}")
    sys.exit(1)

query = sys.argv[1]
book_filter = sys.argv[2] if len(sys.argv) > 2 else None

# 3. Embed the question
query_emb = ollama.embeddings(model="nomic-embed-text", prompt=query)["embedding"]

# 4. Find the most relevant chunks
query_params = {
    "query_embeddings": [query_emb],
    "n_results": 3
}
if book_filter:
    query_params["where"] = {"book": book_filter}

results = collection.query(**query_params)
contexts = results["documents"][0]
context = "\n\n---\n\n".join(contexts)

# 5. Ask Llama to answer based on that context
response = ollama.chat(model="mannix/llama3.1-8b-abliterated", messages=[
    {"role": "system", "content": "Answer the question using ONLY the provided context. Be helpful and concise."},
    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
])

print(f"\n🔍 SOURCES:")
for i, ctx in enumerate(contexts, 1):
    book = results["metadatas"][0][i-1].get("book", "unknown")
    print(f"  [{book}] {ctx[:100]}...")

print(f"\n🤖 ANSWER:\n{response['message']['content']}")
