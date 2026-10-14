#!/usr/bin/env python3
"""
Run the evaluation suite (Step 6).

Usage:
    python scripts/run_evaluation.py                  # Full eval suite
    python scripts/run_evaluation.py --retrieval-only  # Just retrieval metrics
    python scripts/run_evaluation.py --latency-only    # Just latency benchmark
    python scripts/run_evaluation.py --classify-only   # Just domain-consistency eval
    python scripts/run_evaluation.py --n-queries 50    # Smaller eval set (faster)
    python scripts/run_evaluation.py --output results.json  # Save to JSON

Requires: Steps 2-4 completed (indices built).
Produces: The metrics reported in the README, saved to results/eval_results.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import settings
from config.logging_config import get_logger

logger = get_logger("run_evaluation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation suite")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--latency-only", action="store_true")
    parser.add_argument("--classify-only", action="store_true")
    parser.add_argument("--n-queries", type=int, default=200, help="Eval set size")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/eval_results.json"),
                        help="Where to save results JSON (default: results/eval_results.json)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings.ensure_directories()
    run_all = not (args.retrieval_only or args.latency_only or args.classify_only)

    chunks_path = settings.processed_data_dir / "chunks.jsonl"
    if not chunks_path.exists():
        print(f"\n  ✗ Chunks not found at {chunks_path}")
        print("    Run Steps 2-3 first.")
        return 1

    print("\n🧪 Medical RAG — Evaluation Suite\n")

    # ── Initialize stores ───────────────────────────────────────────
    from src.processing.embedder import BioBERTEmbedder
    from src.retrieval.bm25_store import BM25Store
    from src.retrieval.chroma_store import ChromaStore
    from src.retrieval.faiss_store import FAISSStore
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.evaluation.eval_harness import (
        benchmark_stores,
        build_eval_set,
        compare_retrieval_methods,
        evaluate_domain_consistency,
    )

    results = {"config": {
        "embedding_model": settings.embedding_model,
        "faiss_index_type": settings.faiss_index_type,
        "n_queries": args.n_queries,
        "top_k": args.top_k,
        "eval_method": "known-item search: article title as query, its PMID as the answer",
    }}

    print("  Loading components...", flush=True)
    embedder = BioBERTEmbedder()

    chroma = ChromaStore()
    faiss_store = FAISSStore()
    try:
        faiss_store.load()
    except FileNotFoundError:
        print("  ✗ FAISS index not found. Run Step 4 first.")
        return 1

    bm25 = BM25Store.from_faiss_store(faiss_store)
    retriever = HybridRetriever(chroma, faiss_store, bm25)
    print(f"  ✓ Loaded ({chroma.count:,} Chroma / {faiss_store.count:,} FAISS / "
          f"{bm25.count:,} BM25 chunks)")
    print(f"  ✓ Embedding model: {settings.embedding_model}\n")
    results["config"]["chunks_indexed"] = faiss_store.count

    # ── Build eval set ──────────────────────────────────────────────
    eval_set = build_eval_set(chunks_path, n_queries=args.n_queries)
    print(f"  ✓ {len(eval_set)} eval queries (title → own article)\n")

    # ── 1. Retrieval Evaluation ─────────────────────────────────────
    if run_all or args.retrieval_only:
        print("=" * 60)
        print("  📊 RETRIEVAL QUALITY  (BM25 baseline vs dense vs hybrid)")
        print("=" * 60)

        start = time.time()
        r = compare_retrieval_methods(
            retriever, embedder, eval_set, top_k=args.top_k, bm25_store=bm25,
        )
        k = args.top_k
        print(f"\n  {'Method':<10}{'NDCG@'+str(k):>10}{'Recall@'+str(k):>11}{'MRR':>8}{'ms/query':>10}")
        for name in ("bm25", "dense", "hybrid"):
            if name in r:
                m = r[name]
                print(f"  {name:<10}{m[f'mean_ndcg@{k}']:>10.4f}{m[f'mean_recall@{k}']:>11.4f}"
                      f"{m['mean_mrr']:>8.4f}{m['mean_latency_ms']:>10.1f}")
        c = r["comparison"]
        if "dense_vs_bm25_ndcg_lift_pct" in c:
            print(f"\n  → Dense  vs BM25 (NDCG): {c['dense_vs_bm25_ndcg_lift_pct']:+.1f}%")
            print(f"  → Hybrid vs BM25 (NDCG): {c['hybrid_vs_bm25_ndcg_lift_pct']:+.1f}%")
        print(f"  Completed in {time.time() - start:.1f}s\n")
        results["retrieval"] = r

    # ── 2. Domain Consistency ───────────────────────────────────────
    if run_all or args.classify_only:
        print("=" * 60)
        print("  🏷️  DOMAIN CONSISTENCY  (does the nearest other article share the domain?)")
        print("=" * 60)

        start = time.time()
        d = evaluate_domain_consistency(eval_set, retriever, embedder)
        print(f"\n  Agreement:          {d.get('accuracy', 0):.1%}")
        print(f"  Majority baseline:  {d.get('majority_baseline', 0):.1%}  (always guess the commonest domain)")
        print(f"  Macro F1:           {d.get('macro_f1', 0):.4f}")
        print(f"  Evaluated: {d['evaluated']}  |  Skipped (no domain match): {d['skipped_other']}")
        if "per_class" in d:
            print("\n  Per-domain:")
            for label, m in sorted(d["per_class"].items()):
                if m.get("support", 0) >= 3:
                    print(f"    {label:20s}  F1={m['f1']:.3f}  (n={m['support']})")
        print(f"\n  Completed in {time.time() - start:.1f}s\n")
        results["domain_consistency"] = d

    # ── 3. Latency Benchmark ────────────────────────────────────────
    if run_all or args.latency_only:
        print("=" * 60)
        print("  ⏱️  VECTOR SEARCH LATENCY  (FAISS vs Chroma, real query vectors)")
        print("=" * 60)

        start = time.time()
        query_vectors = [embedder.embed_single(e["query"]) for e in eval_set[:100]]
        lat = benchmark_stores(chroma, faiss_store, top_k=args.top_k, query_vectors=query_vectors)
        f, c = lat["faiss"], lat["chroma"]
        print(f"\n  FAISS ({settings.faiss_index_type}):  mean {f['mean_ms']:.3f} ms  |  p95 {f['p95_ms']:.3f} ms")
        print(f"  Chroma (HNSW):  mean {c['mean_ms']:.3f} ms  |  p95 {c['p95_ms']:.3f} ms")
        print(f"\n  → FAISS mean latency vs Chroma: {-lat['latency_reduction_pct']:+.1f}% "
              f"({lat['speedup_factor']:.2f}x speedup)")
        print(f"  Completed in {time.time() - start:.1f}s\n")
        results["latency"] = lat

    # ── Summary ─────────────────────────────────────────────────────
    print("=" * 60)
    print("  📋 SUMMARY")
    print("=" * 60)
    if "retrieval" in results:
        c = results["retrieval"]["comparison"]
        print(f"  NDCG@{args.top_k}: BM25 {c.get('bm25_ndcg', 0):.3f} | dense {c.get('dense_ndcg', 0):.3f}"
              f" | hybrid {c.get('hybrid_ndcg', 0):.3f}")
    if "domain_consistency" in results:
        d = results["domain_consistency"]
        print(f"  Domain consistency: {d.get('accuracy', 0):.1%} (baseline {d.get('majority_baseline', 0):.1%})")
    if "latency" in results:
        print(f"  FAISS latency vs Chroma: {-results['latency']['latency_reduction_pct']:+.1f}%")
    print()

    # ── Save results ────────────────────────────────────────────────
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Results saved to {args.output}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
