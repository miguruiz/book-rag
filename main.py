import ollama
import chromadb

# 1. Setup ChromaDB (Local storage)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="alice_collection")

# 2. Load the book
with open("data/alice.txt", "r", encoding="utf-8") as f:
    # We split by double newlines to get individual paragraphs
    paragraphs = [p.strip() for p in f.read().split('\n\n') if p.strip()]

print(f"📥 Processing {len(paragraphs)} paragraphs...")

# 3. Embed and Store
for i, para in enumerate(paragraphs):
    # Turn text into numbers using your NVIDIA GPU via Ollama
    response = ollama.embeddings(model="nomic-embed-text", prompt=para)

    collection.add(
        ids=[f"para_{i}"],
        embeddings=[response["embedding"]],
        documents=[para]
    )

    print(response)
    print(collection)
    break

print("✅ Knowledge base ready!")