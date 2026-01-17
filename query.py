#!/usr/bin/env python3
"""CLI tool for querying the RAG database."""

import sys
from src import get_books, query_books, settings


def main():
    if len(sys.argv) < 2:
        print("Usage: python query.py <question> [book_id]")
        print("Example: python query.py 'Who is the Queen?' alice")
        print(f"\nAvailable books: {', '.join(get_books()) or 'none'}")
        print(f"Current provider: {settings.llm_provider}")
        sys.exit(1)

    # Validate settings
    settings.validate()

    question = sys.argv[1]
    book_id = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"🔍 Querying with {settings.llm_provider}...")

    result = query_books(question, book_id, n_results=3)

    print(f"\n📚 SOURCES:")
    for source in result["sources"]:
        print(f"  [{source['book']}] {source['text']}...")

    print(f"\n🤖 ANSWER:\n{result['answer']}")


if __name__ == "__main__":
    main()
