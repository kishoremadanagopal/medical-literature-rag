#!/usr/bin/env python3
"""
Query the Medical RAG system (Step 5).

Usage:
    python scripts/run_query.py "What are CRISPR off-target effects?"
    python scripts/run_query.py --interactive
    python scripts/run_query.py "cancer immunotherapy" --summarize
    python scripts/run_query.py "BRCA1 mutations" --journal "Nature"
    python scripts/run_query.py "drug resistance" --ensemble --top-k 10

Requires: Steps 2-4 completed (indices built).
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import settings
from config.logging_config import get_logger

logger = get_logger("run_query")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the Medical RAG system")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--summarize", action="store_true", help="Summarize a topic instead of Q&A")
    parser.add_argument("--ensemble", action="store_true", help="Use hybrid retrieval (embeddings + BM25 keyword search)")
    parser.add_argument("--top-k", type=int, default=None, help="Number of documents to retrieve")
    parser.add_argument("--journal", type=str, default=None, help="Filter by journal name")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    return parser.parse_args()


def init_chain():
    """Initialize the full RAG chain (loads models + indices)."""
    from src.processing.embedder import BioBERTEmbedder
    from src.retrieval.chroma_store import ChromaStore
    from src.retrieval.faiss_store import FAISSStore
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.generation.rag_chain import MedicalRAGChain

    print("  Loading embedding model...", end=" ", flush=True)
    embedder = BioBERTEmbedder()
    _ = embedder.model  # Force load
    print("✓")

    print("  Loading Chroma store...", end=" ", flush=True)
    chroma = ChromaStore()
    print(f"✓ ({chroma.count:,} vectors)")

    print("  Loading FAISS index...", end=" ", flush=True)
    faiss_store = FAISSStore()
    faiss_store.load()
    print(f"✓ ({faiss_store.count:,} vectors)")

    from src.retrieval.bm25_store import BM25Store
    print("  Building BM25 keyword index...", end=" ", flush=True)
    bm25 = BM25Store.from_faiss_store(faiss_store)
    print(f"✓ ({bm25.count:,} chunks)")

    retriever = HybridRetriever(chroma, faiss_store, bm25)
    chain = MedicalRAGChain(retriever, embedder)

    return chain


def print_result(result: dict, as_json: bool = False) -> None:
    """Pretty-print a query result."""
    if as_json:
        print(json.dumps(result, indent=2))
        return

    print(f"\n{'='*70}")
    print(f"  Query: {result['query']}")
    print(f"  Retrieval: {result['retrieval_time_ms']}ms | "
          f"Generation: {result['generation_time_ms']}ms | "
          f"Model: {result['model']}")
    print(f"{'='*70}\n")

    # Wrap answer text nicely
    answer = result["answer"]
    for line in answer.split("\n"):
        print(textwrap.fill(line, width=70, subsequent_indent="  "))
    print()

    # Sources
    if result["sources"]:
        print(f"  📚 Cited sources ({len(result['sources'])}):")
        for s in result["sources"]:
            print(f"     PMID {s['pmid']} — {s['title'][:55]}...")
            print(f"       {s['journal']} | relevance: {s['score']}")
        print()

    # Retrieved but not cited
    uncited = [
        d for d in result["all_retrieved"]
        if d["pmid"] not in result["cited_pmids"]
    ]
    if uncited:
        print(f"  📄 Retrieved but not cited ({len(uncited)}):")
        for s in uncited[:3]:
            print(f"     PMID {s['pmid']} — {s['title'][:55]}...")
        print()


def interactive_mode(chain) -> None:
    """Run an interactive query loop."""
    print("\n🧬 Medical Literature Q&A (type 'quit' to exit)\n")

    while True:
        try:
            question = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break

        # Handle special commands
        if question.startswith("/summarize "):
            topic = question[len("/summarize "):]
            print(f"\n  Summarizing: {topic}\n")
            result = chain.summarize_topic(topic)
            print(textwrap.fill(result["summary"], width=70))
            print(f"\n  (Cited {len(result['cited_pmids'])} papers from {result['total_sources']} retrieved)\n")
            continue

        result = chain.query(question)
        print_result(result)


def main() -> int:
    args = parse_args()
    settings.ensure_directories()

    if not args.question and not args.interactive:
        print("  Provide a question or use --interactive mode.")
        print("  Example: python scripts/run_query.py \"What is CRISPR?\"")
        return 1

    print("\n🧬 Medical RAG — Initializing\n")

    try:
        chain = init_chain()
    except FileNotFoundError as e:
        print(f"\n  ✗ {e}")
        print("  Run Steps 2-4 first to build the indices.")
        return 1
    except Exception as e:
        logger.error(f"Init failed: {e}", exc_info=True)
        print(f"\n  Error: {e}")
        return 1

    print("\n  ✓ System ready\n")

    if args.interactive:
        interactive_mode(chain)
        return 0

    # Single query
    where = {"journal": args.journal} if args.journal else None

    try:
        if args.summarize:
            result = chain.summarize_topic(args.question, top_k=args.top_k, where=where)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(f"\n  Topic: {result['topic']}\n")
                print(textwrap.fill(result["summary"], width=70))
                print(f"\n  Cited: {len(result['cited_pmids'])} papers\n")
        else:
            result = chain.query(
                args.question,
                top_k=args.top_k,
                where=where,
                ensemble=args.ensemble,
            )
            print_result(result, as_json=args.json)

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        print(f"\n  Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
