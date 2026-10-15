#!/usr/bin/env python3
"""
Build vector store indexes (Step 4).

Usage:
    python scripts/run_indexing.py              # Build both stores
    python scripts/run_indexing.py --chroma     # Chroma only
    python scripts/run_indexing.py --faiss      # FAISS only
    python scripts/run_indexing.py --benchmark  # Build + run latency benchmark

Requires: Step 3 output (data/embeddings/embeddings.npy + chunk_metadata.jsonl).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import settings
from config.logging_config import get_logger

logger = get_logger("run_indexing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chroma + FAISS vector indexes")
    parser.add_argument("--chroma", action="store_true", help="Build Chroma only")
    parser.add_argument("--faiss", action="store_true", help="Build FAISS only")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark after build")
    return parser.parse_args()


def build_chroma(embeddings_path: Path, metadata_path: Path) -> None:
    from src.retrieval.chroma_store import ChromaStore

    print("\n🗄️  Building Chroma index")
    print(f"   Persist dir: {settings.chroma_persist_dir}")

    start = time.time()
    store = ChromaStore()
    count = store.build_from_embeddings(embeddings_path, metadata_path)
    elapsed = time.time() - start

    print(f"   ✓ Indexed {count:,} vectors in {elapsed:.1f}s")


def build_faiss(embeddings_path: Path, metadata_path: Path) -> None:
    from src.retrieval.faiss_store import FAISSStore

    print(f"\n⚡ Building FAISS {settings.faiss_index_type} index")
    print(f"   Index dir: {settings.faiss_index_dir}")

    start = time.time()
    store = FAISSStore()
    count = store.build_from_embeddings(embeddings_path, metadata_path)
    store.save()
    elapsed = time.time() - start

    print(f"   ✓ Indexed {count:,} vectors in {elapsed:.1f}s")


def run_benchmark() -> None:
    import numpy as np
    from src.retrieval.chroma_store import ChromaStore
    from src.retrieval.faiss_store import FAISSStore

    print("\n📊 Latency benchmark (100 random queries, top_k=10)")

    embeddings = np.load(settings.embeddings_dir / "embeddings.npy")
    n_queries = min(100, len(embeddings))
    query_indices = np.random.choice(len(embeddings), n_queries, replace=False)
    query_vecs = embeddings[query_indices]

    # FAISS benchmark
    faiss_store = FAISSStore()
    faiss_store.load()
    faiss_bench = faiss_store.benchmark(query_vecs, top_k=10)

    # Chroma benchmark
    chroma_store = ChromaStore()
    import time as t
    chroma_latencies = []
    for vec in query_vecs:
        start = t.perf_counter()
        chroma_store.search(vec, top_k=10)
        chroma_latencies.append((t.perf_counter() - start) * 1000)
    chroma_latencies.sort()
    cn = len(chroma_latencies)

    print(f"\n   {'Metric':<12} {'FAISS':>10} {'Chroma':>10} {'Speedup':>10}")
    print(f"   {'─'*12} {'─'*10} {'─'*10} {'─'*10}")
    chroma_p50 = round(chroma_latencies[cn // 2], 2)
    chroma_p95 = round(chroma_latencies[int(cn * 0.95)], 2)
    print(f"   {'p50 (ms)':<12} {faiss_bench['p50_ms']:>10} {chroma_p50:>10} {chroma_p50 / max(faiss_bench['p50_ms'], 0.01):>9.1f}x")
    print(f"   {'p95 (ms)':<12} {faiss_bench['p95_ms']:>10} {chroma_p95:>10} {chroma_p95 / max(faiss_bench['p95_ms'], 0.01):>9.1f}x")


def main() -> int:
    args = parse_args()
    settings.ensure_directories()

    embeddings_path = settings.embeddings_dir / "embeddings.npy"
    metadata_path = settings.embeddings_dir / "chunk_metadata.jsonl"

    for path in [embeddings_path, metadata_path]:
        if not path.exists():
            print(f"\n  ✗ Missing: {path}")
            print("    Run Step 3 first: python scripts/run_processing.py")
            return 1

    build_both = not args.chroma and not args.faiss

    try:
        if build_both or args.chroma:
            build_chroma(embeddings_path, metadata_path)
        if build_both or args.faiss:
            build_faiss(embeddings_path, metadata_path)
        if args.benchmark:
            run_benchmark()
    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        print(f"\n  Error: {e}")
        return 1

    print(f"\n{'='*50}")
    print("  Step 4 complete — vector stores ready")
    print(f"{'='*50}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
