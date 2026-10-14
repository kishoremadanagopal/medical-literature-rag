"""
Evaluation metrics for the Medical RAG pipeline.

Three categories:

1. RETRIEVAL METRICS — Does the retriever find the right documents?
   - Precision@K: what fraction of retrieved docs are relevant?
   - Recall@K: what fraction of all relevant docs were retrieved?
   - NDCG@K: are relevant docs ranked higher? (position-aware)
   - MRR: how high is the FIRST relevant doc?

2. GENERATION METRICS — Does Claude's answer actually use the evidence?
   - Citation precision: what fraction of cited sources are relevant?
   - Citation recall: what fraction of relevant sources are cited?
   - Faithfulness: does the answer only use info from retrieved context?
     (approximated by checking if cited PMIDs appear in retrieved set)

3. CLASSIFICATION METRICS — Can the system tag articles by domain?
   - Accuracy, F1, precision, recall on domain classification

Where each README number comes from (see eval_harness.py):
   - Retrieval quality (NDCG/Recall/MRR vs BM25) → compare_retrieval_methods()
   - Domain consistency → evaluate_domain_consistency()
   - FAISS vs Chroma latency → benchmark_stores() / benchmark_latency()
"""

import time

import numpy as np

from config.logging_config import get_logger

logger = get_logger(__name__)


# ── Retrieval Metrics ───────────────────────────────────────────────

def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """
    Fraction of the top-K retrieved docs that are relevant.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevant_ids: Set of ground-truth relevant document IDs
        k: Cutoff position
    """
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of all relevant docs that appear in the top-K."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K.

    NDCG rewards placing relevant docs at higher positions.
    Score of 1.0 means perfect ranking; 0.0 means no relevant docs retrieved.

    With a single relevant article per query (our known-item eval set),
    NDCG@k = 1 / log2(rank + 1) if the article is found at that rank, else 0.
    """
    top_k = retrieved_ids[:k]
    if not top_k or not relevant_ids:
        return 0.0

    # DCG: sum of 1/log2(rank+1) for relevant docs
    dcg = 0.0
    for i, doc_id in enumerate(top_k):
        if doc_id in relevant_ids:
            dcg += 1.0 / np.log2(i + 2)  # +2 because rank starts at 1, log2(1)=0

    # Ideal DCG: if all relevant docs were at the top
    ideal_k = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_k))

    return dcg / idcg if idcg > 0 else 0.0


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """
    Mean Reciprocal Rank — 1/position of the first relevant result.
    MRR = 1.0 means the first result is relevant.
    """
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def compute_retrieval_metrics(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int = 10,
) -> dict:
    """Compute all retrieval metrics in one call."""
    return {
        f"precision@{k}": round(precision_at_k(retrieved_ids, relevant_ids, k), 4),
        f"recall@{k}": round(recall_at_k(retrieved_ids, relevant_ids, k), 4),
        f"ndcg@{k}": round(ndcg_at_k(retrieved_ids, relevant_ids, k), 4),
        "mrr": round(mrr(retrieved_ids, relevant_ids), 4),
    }


# ── Generation Metrics ──────────────────────────────────────────────

def citation_precision(cited_pmids: set[str], relevant_pmids: set[str]) -> float:
    """What fraction of cited sources are actually relevant?"""
    if not cited_pmids:
        return 0.0
    hits = len(cited_pmids & relevant_pmids)
    return hits / len(cited_pmids)


def citation_recall(cited_pmids: set[str], relevant_pmids: set[str]) -> float:
    """What fraction of relevant sources were cited?"""
    if not relevant_pmids:
        return 0.0
    hits = len(cited_pmids & relevant_pmids)
    return hits / len(relevant_pmids)


def faithfulness_score(cited_pmids: set[str], retrieved_pmids: set[str]) -> float:
    """
    Are all cited PMIDs from the retrieved set?
    A score < 1.0 means the model hallucinated a citation.
    """
    if not cited_pmids:
        return 1.0  # No citations = nothing to hallucinate
    grounded = len(cited_pmids & retrieved_pmids)
    return grounded / len(cited_pmids)


def compute_generation_metrics(
    cited_pmids: set[str],
    relevant_pmids: set[str],
    retrieved_pmids: set[str],
) -> dict:
    """Compute all generation metrics in one call."""
    return {
        "citation_precision": round(citation_precision(cited_pmids, relevant_pmids), 4),
        "citation_recall": round(citation_recall(cited_pmids, relevant_pmids), 4),
        "faithfulness": round(faithfulness_score(cited_pmids, retrieved_pmids), 4),
    }


# ── Classification Metrics ──────────────────────────────────────────

def classification_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """Simple accuracy: fraction of correct predictions."""
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def classification_report(y_true: list[str], y_pred: list[str]) -> dict:
    """
    Compute per-class precision, recall, F1 + overall accuracy.
    Uses sklearn if available, falls back to manual calculation.
    """
    try:
        from sklearn.metrics import classification_report as sk_report
        report = sk_report(y_true, y_pred, output_dict=True, zero_division=0)
        return {
            "accuracy": round(report["accuracy"], 4),
            "macro_f1": round(report["macro avg"]["f1-score"], 4),
            "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
            "per_class": {
                label: {
                    "precision": round(metrics["precision"], 4),
                    "recall": round(metrics["recall"], 4),
                    "f1": round(metrics["f1-score"], 4),
                    "support": int(metrics["support"]),
                }
                for label, metrics in report.items()
                if label not in ("accuracy", "macro avg", "weighted avg")
            },
        }
    except ImportError:
        return {"accuracy": round(classification_accuracy(y_true, y_pred), 4)}


# ── Latency Benchmarking ───────────────────────────────────────────

def benchmark_latency(
    search_fn,
    queries: list,
    runs: int = 3,
) -> dict:
    """
    Benchmark a search function over multiple queries and runs.
    Returns p50, p95, p99, mean latency in milliseconds.

    Used to compare FAISS vs Chroma search latency.
    """
    all_latencies = []

    for _ in range(runs):
        for query in queries:
            start = time.perf_counter()
            search_fn(query)
            elapsed_ms = (time.perf_counter() - start) * 1000
            all_latencies.append(elapsed_ms)

    all_latencies.sort()
    n = len(all_latencies)

    return {
        "total_queries": n,
        "p50_ms": round(all_latencies[n // 2], 3),
        "p95_ms": round(all_latencies[int(n * 0.95)], 3),
        "p99_ms": round(all_latencies[int(n * 0.99)], 3),
        "mean_ms": round(sum(all_latencies) / n, 3),
        "min_ms": round(all_latencies[0], 3),
        "max_ms": round(all_latencies[-1], 3),
    }
