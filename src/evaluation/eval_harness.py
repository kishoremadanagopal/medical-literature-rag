"""
Evaluation harness — produces the metrics reported in the README.

Three evaluation suites:

1. RETRIEVAL QUALITY
   Compares three retrievers on the same queries:
       - BM25 keyword search (the standard baseline)
       - Dense embedding search (FAISS)
       - Hybrid: dense + BM25 fused with Reciprocal Rank Fusion
   Reports NDCG@k, Recall@k and MRR, plus each method's lift over BM25.

2. DOMAIN CONSISTENCY
   For each query article, do the nearest *other* articles belong to the same
   broad biomedical domain (oncology, cardiology, ...)? Domains are derived
   from MeSH headings with keyword rules. This is a sanity check that the
   embedding space groups related research — it is not a trained classifier.

3. LATENCY
   Times FAISS vs Chroma vector search on real query embeddings.

Eval set (important caveat — say this out loud in interviews):
   There is no human-labelled query set. We use "known-item search": take an
   article's TITLE as the query, and count the search a success if that
   article comes back. This is cheap and objective, but titles share many
   exact words with their own abstracts, so it favours keyword search (BM25).
   A stronger evaluation would use a labelled biomedical benchmark such as
   BEIR's TREC-COVID, NFCorpus or SciFact.
"""

import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from config.logging_config import get_logger
from src.evaluation.metrics import (
    benchmark_latency,
    classification_report,
    compute_retrieval_metrics,
)

logger = get_logger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────

def as_term_list(value) -> list[str]:
    """
    Normalise MeSH terms to a list of strings.

    Chunk metadata stores them as a list, but Chroma flattens lists to a
    comma-separated string, so results from different stores differ.
    (The original code called .split() on the list form and crashed.)
    """
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(t).strip() for t in value if str(t).strip()]
    return [t.strip() for t in str(value).split(",") if t.strip()]


def unique_pmids(results: list[dict]) -> list[str]:
    """Ordered, de-duplicated PMIDs (an article can have several chunks)."""
    seen, out = set(), []
    for r in results:
        pmid = r.get("metadata", {}).get("pmid", "")
        if pmid and pmid not in seen:
            seen.add(pmid)
            out.append(pmid)
    return out


# ── Eval Dataset Construction ───────────────────────────────────────

def build_eval_set(
    chunks_path: Path,
    n_queries: int = 200,
    seed: int = 42,
) -> list[dict]:
    """
    Build a known-item eval set from the chunked corpus.

    For each sampled article: query = its title, relevant = its PMID.

    Returns:
        List of {query, relevant_pmids, mesh_terms, source_chunk_id}
    """
    rng = random.Random(seed)

    pmid_to_chunk: dict[str, dict] = {}
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)
            pmid = chunk["metadata"].get("pmid", "")
            if pmid and pmid not in pmid_to_chunk:
                pmid_to_chunk[pmid] = chunk

    candidates = [
        c for c in pmid_to_chunk.values()
        if len(c["metadata"].get("title", "")) >= 10
    ]
    rng.shuffle(candidates)

    eval_set = []
    for chunk in candidates[:n_queries]:
        meta = chunk["metadata"]
        eval_set.append({
            "query": meta["title"],
            "relevant_pmids": {meta["pmid"]},
            "mesh_terms": as_term_list(meta.get("mesh_terms")),
            "source_chunk_id": chunk["chunk_id"],
        })

    if len(eval_set) < n_queries:
        logger.warning(f"Only {len(eval_set)} usable articles for a {n_queries}-query eval set")
    logger.info(f"Built eval set: {len(eval_set)} queries from {len(pmid_to_chunk)} articles")
    return eval_set


# ── Retrieval Evaluation ────────────────────────────────────────────

def evaluate_retrieval(
    search_fn: Callable[[str], list[dict]],
    eval_set: list[dict],
    top_k: int = 10,
) -> dict:
    """
    Run every eval query through search_fn and score article-level results.

    Args:
        search_fn: query text -> ranked list of chunk result dicts. It should
                   over-fetch chunks (several per article are possible).
        eval_set: from build_eval_set
        top_k: cutoff for the metrics (counted in articles, not chunks)

    Returns:
        Mean and std of each metric, plus mean/p95 latency (ms) per query.
    """
    if not eval_set:
        raise ValueError("Empty eval set")

    all_metrics, latencies = [], []
    for i, example in enumerate(eval_set):
        start = time.perf_counter()
        results = search_fn(example["query"])
        latencies.append((time.perf_counter() - start) * 1000)

        retrieved = unique_pmids(results)[:top_k]
        all_metrics.append(compute_retrieval_metrics(
            retrieved_ids=retrieved,
            relevant_ids=example["relevant_pmids"],
            k=top_k,
        ))
        if (i + 1) % 50 == 0:
            logger.info(f"  Evaluated {i + 1}/{len(eval_set)} queries")

    agg = {}
    for key in all_metrics[0]:
        values = [m[key] for m in all_metrics]
        agg[f"mean_{key}"] = round(float(np.mean(values)), 4)
        agg[f"std_{key}"] = round(float(np.std(values)), 4)
    agg["mean_latency_ms"] = round(float(np.mean(latencies)), 2)
    agg["p95_latency_ms"] = round(float(np.percentile(latencies, 95)), 2)
    agg["num_queries"] = len(eval_set)
    return agg


def compare_retrieval_methods(
    retriever,
    embedder,
    eval_set: list[dict],
    top_k: int = 10,
    bm25_store=None,
) -> dict:
    """
    Compare BM25 (baseline), dense (FAISS) and hybrid (dense + BM25) retrieval.

    Lift is reported relative to BM25 — the honest baseline for any
    embedding-based retriever.
    """
    bm25_store = bm25_store or getattr(retriever, "bm25", None)
    fetch_k = top_k * 3  # chunks to fetch so we still have top_k distinct articles

    methods: dict[str, Callable[[str], list[dict]]] = {
        "dense": lambda q: retriever.faiss.search(embedder.embed_single(q), top_k=fetch_k),
    }
    if bm25_store is not None:
        methods = {
            "bm25": lambda q: bm25_store.search(q, top_k=fetch_k),
            **methods,
            "hybrid": lambda q: retriever.retrieve_ensemble(
                embedder.embed_single(q), top_k=fetch_k, query_text=q
            ),
        }
    else:
        logger.warning("No BM25 index supplied — reporting dense retrieval only")

    results = {}
    for name, fn in methods.items():
        logger.info(f"Evaluating {name} retrieval...")
        results[name] = evaluate_retrieval(fn, eval_set, top_k=top_k)

    ndcg_key = f"mean_ndcg@{top_k}"
    comparison = {f"{name}_ndcg": results[name][ndcg_key] for name in methods}
    if "bm25" in results:
        base = results["bm25"][ndcg_key]
        for name in ("dense", "hybrid"):
            lift = ((results[name][ndcg_key] - base) / base * 100) if base > 0 else 0.0
            comparison[f"{name}_vs_bm25_ndcg_lift_pct"] = round(lift, 1)
    results["comparison"] = comparison
    return results


# ── Domain Consistency ──────────────────────────────────────────────

# Keyword rules over MeSH headings, checked in order. Matching is on word
# starts, so "Breast Neoplasms" → oncology and "Heart Failure" → cardiology.
# (The original exact-match dictionary only knew top-level category names like
# "Neoplasms", which articles are rarely indexed with, so most were "other".)
DOMAIN_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("oncology", ("neoplasm", "cancer", "tumor", "tumour", "carcinoma", "lymphoma",
                  "leukemia", "melanoma", "sarcoma", "glioma", "glioblastoma", "oncolog",
                  "metasta")),
    ("cardiology", ("heart", "cardi", "coronary", "myocardial", "atrial", "hypertension",
                    "arrhythmi", "aortic", "vascular disease", "atherosclero")),
    ("neurology", ("brain", "neuro", "nervous system", "alzheimer", "parkinson", "stroke",
                   "epilep", "dementia", "multiple sclerosis", "cerebr", "spinal cord")),
    ("psychiatry", ("mental", "depress", "anxiety", "schizophreni", "bipolar", "psychiatr",
                    "substance-related", "stress disorders", "autis")),
    ("infectious_disease", ("infection", "virus", "viral", "bacteri", "covid", "sars-cov",
                            "hiv", "tuberculosis", "malaria", "sepsis", "influenza",
                            "communicable", "anti-bacterial")),
    ("immunology", ("immun", "autoimmun", "inflammat", "lupus", "arthritis, rheumatoid",
                    "allerg", "cytokine")),
    ("endocrinology", ("diabetes", "obesity", "metabolic", "insulin", "thyroid", "endocrin",
                       "glucose")),
    ("pulmonology", ("lung", "pulmonary", "respiratory", "asthma")),
    # "genes"/"gene expression" rather than bare "gene", which would match "General"
    ("genetics", ("genom", "genetic", "genes", "gene expression", "gene editing",
                  "mutation", "polymorphism", "crispr",
                  "sequence analysis", "dna", "rna", "proteom", "transcriptom")),
    ("pharmacology", ("drug", "pharmac", "pharmaceutical preparations")),
    ("bioinformatics", ("machine learning", "deep learning", "artificial intelligence",
                        "computational biology", "algorithms", "neural networks, computer")),
    ("orthopedics", ("musculoskeletal", "bone", "fracture", "osteo", "spine", "joint")),
]

_DOMAIN_PATTERNS = [
    (domain, re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keywords) + r")", re.I))
    for domain, keywords in DOMAIN_RULES
]


def classify_article(mesh_terms) -> str:
    """
    Assign a broad domain from MeSH headings by majority vote.

    Each heading votes for the first rule it matches; ties go to the rule
    listed first. Returns 'other' when nothing matches.
    """
    votes: Counter = Counter()
    for term in as_term_list(mesh_terms):
        for domain, pattern in _DOMAIN_PATTERNS:
            if pattern.search(term):
                votes[domain] += 1
                break
    if not votes:
        return "other"
    order = {d: i for i, (d, _) in enumerate(DOMAIN_RULES)}
    return max(votes, key=lambda d: (votes[d], -order[d]))


def evaluate_domain_consistency(
    eval_set: list[dict],
    retriever,
    embedder,
    top_k: int = 5,
) -> dict:
    """
    Does the nearest OTHER article share the query article's domain?

    For each query: embed the title, retrieve neighbours, skip the query's own
    article (otherwise this just grades an article against itself), and
    compare the domain of the first remaining article with the query's domain.

    Reported as accuracy / per-domain F1 via sklearn's classification_report,
    and as a majority-class baseline so the number has context.
    """
    y_true, y_pred = [], []
    skipped = 0

    for example in eval_set:
        true_domain = classify_article(example.get("mesh_terms", []))
        if true_domain == "other":
            skipped += 1
            continue

        own = example["relevant_pmids"]
        results = retriever.retrieve(embedder.embed_single(example["query"]), top_k=top_k * 3)
        neighbour = next(
            (r for r in results if r.get("metadata", {}).get("pmid") not in own), None
        )
        pred = classify_article(neighbour["metadata"].get("mesh_terms")) if neighbour else "other"

        y_true.append(true_domain)
        y_pred.append(pred)

    if not y_true:
        return {"accuracy": 0.0, "evaluated": 0, "skipped_other": skipped,
                "majority_baseline": 0.0}

    report = classification_report(y_true, y_pred)
    report["skipped_other"] = skipped
    report["evaluated"] = len(y_true)
    # What you'd score by always guessing the most common domain
    report["majority_baseline"] = round(Counter(y_true).most_common(1)[0][1] / len(y_true), 4)

    logger.info(
        f"Domain consistency: {report.get('accuracy', 0):.1%} on {len(y_true)} articles "
        f"(majority baseline {report['majority_baseline']:.1%})"
    )
    return report


# Backwards-compatible name
evaluate_classification = evaluate_domain_consistency


# ── Latency Benchmark ───────────────────────────────────────────────

def benchmark_stores(
    chroma_store,
    faiss_store,
    n_queries: int = 100,
    top_k: int = 10,
    seed: int = 42,
    query_vectors: Optional[list[np.ndarray]] = None,
) -> dict:
    """
    Compare vector-search latency between FAISS and Chroma.

    Uses real query embeddings when given (recommended); otherwise random
    unit vectors. This measures the vector database only — embedding the
    query usually costs far more than either search.
    """
    if query_vectors is None:
        rng = np.random.default_rng(seed)
        dim = faiss_store.index.d if faiss_store.index is not None else 768
        q = rng.standard_normal((n_queries, dim)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        query_vectors = list(q)
    else:
        query_vectors = list(query_vectors)[:n_queries]

    logger.info(f"Benchmarking {len(query_vectors)} queries, top_k={top_k}")

    faiss_results = benchmark_latency(
        lambda v: faiss_store.search(v, top_k=top_k), query_vectors, runs=3
    )
    chroma_results = benchmark_latency(
        lambda v: chroma_store.search(v, top_k=top_k), query_vectors, runs=3
    )

    speedup = chroma_results["mean_ms"] / max(faiss_results["mean_ms"], 0.001)
    return {
        "faiss": faiss_results,
        "chroma": chroma_results,
        "speedup_factor": round(speedup, 2),
        # Percent reduction in mean latency when using FAISS instead of Chroma
        "latency_reduction_pct": round(
            (1 - faiss_results["mean_ms"] / max(chroma_results["mean_ms"], 0.001)) * 100, 1
        ),
    }
