import ollama
import chromadb

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into fixed-size chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


# 1. Setup ChromaDB (Local storage)
client = chromadb.PersistentClient(path="./chroma_db")
# Delete old collection if exists (to avoid mixing old/new chunks)
try:
    client.delete_collection(name="alice_collection")
except ValueError:
    pass
collection = client.create_collection(name="alice_collection")

# 2. Load and clean the book
with open("data/alice.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Strip Gutenberg header/footer
if "*** START OF" in raw_text:
    raw_text = raw_text.split("*** START OF")[1].split("***", 1)[1]
if "*** END OF" in raw_text:
    raw_text = raw_text.split("*** END OF")[0]

# 3. Chunk the text
chunks = chunk_text(raw_text)
print(f"📥 Processing {len(chunks)} chunks ({CHUNK_SIZE} chars, {CHUNK_OVERLAP} overlap)...")

# 4. Embed and Store
for i, chunk in enumerate(chunks):
    response = ollama.embeddings(model="nomic-embed-text", prompt=chunk)

    collection.add(
        ids=[f"chunk_{i}"],
        embeddings=[response["embedding"]],
        documents=[chunk]
    )

    if (i + 1) % 50 == 0:
        print(f"  Processed {i + 1}/{len(chunks)} chunks...")

print("✅ Knowledge base ready!")