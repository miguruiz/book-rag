#!/usr/bin/env python3
"""CLI tool for ingesting books into the RAG database."""

import sys
from src import ingest_book, settings


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <file.txt> [book_id] [title]")
        print("Example: python main.py data/alice.txt alice 'Alice in Wonderland'")
        print(f"\nCurrent provider: {settings.llm_provider}")
        sys.exit(1)

    # Validate settings
    settings.validate()

    file_path = sys.argv[1]
    book_id = sys.argv[2] if len(sys.argv) > 2 else file_path.split("/")[-1].replace(".txt", "")
    title = sys.argv[3] if len(sys.argv) > 3 else book_id.replace("-", " ").title()

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"📥 Ingesting '{title}' (id: {book_id}) using {settings.llm_provider}...")

    def show_progress(current, total):
        if current % 50 == 0 or current == total:
            print(f"  Processed {current}/{total} chunks...")

    num_chunks = ingest_book(text, book_id, title, progress_callback=show_progress)
    print(f"✅ Done! Added {num_chunks} chunks to knowledge base.")


if __name__ == "__main__":
    main()
