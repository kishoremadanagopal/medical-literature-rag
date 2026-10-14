"""
Tests for the evaluation, hybrid-retrieval and ingestion fixes.

Covers:
    - MeSH terms stored as lists (crashed run_evaluation.py before)
    - Domain rules on real-world MeSH headings
    - BM25 keyword index and Reciprocal Rank Fusion
    - Hybrid retrieval fusing dense + BM25 (not Chroma + FAISS)
    - Domain consistency skipping the query's own article
    - PubMed ID search staying under the 10,000-ID cap
    - Clear error when generating without an API key
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from config.config import settings
from src.evaluation.eval_harness import (
    as_term_list,
    build_eval_set,
    classify_article,
    compare_retrieval_methods,
    evaluate_domain_consistency,
    unique_pmids,
)
from src.retrieval.bm25_store import BM25Store, tokenize
from src.retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion


def _rec(cid, pmid, text, mesh=None):
    return {"chunk_id": cid, "text": text,
            "metadata": {"pmid": pmid, "title": text[:40], "mesh_terms": mesh or []}}


# ── MeSH term handling ──────────────────────────────────────────────

class TestMeshTerms:
    def test_list_passthrough(self):
        assert as_term_list(["Neoplasms", " Humans "]) == ["Neoplasms", "Humans"]

    def test_chroma_flattened_string(self):
        assert as_term_list("Neoplasms, Humans") == ["Neoplasms", "Humans"]

    def test_empty_values(self):
        assert as_term_list(None) == []
        assert as_term_list("") == []
        assert as_term_list([]) == []

    def test_build_eval_set_with_list_mesh(self, tmp_path):
        """Chunks store mesh_terms as a list — this used to raise AttributeError."""
        path = tmp_path / "chunks.jsonl"
        with open(path, "w") as f:
            for i in range(5):
                f.write(json.dumps({
                    "chunk_id": f"{i}_0",
                    "text": f"Article title number {i}. Body.",
                    "metadata": {"pmid": str(i), "title": f"Article title number {i}",
                                 "mesh_terms": ["Breast Neoplasms", "Humans"]},
                }) + "\n")
        eval_set = build_eval_set(path, n_queries=3)
        assert len(eval_set) == 3
        assert eval_set[0]["mesh_terms"] == ["Breast Neoplasms", "Humans"]
        assert eval_set[0]["relevant_pmids"] == {eval_set[0]["source_chunk_id"].split("_")[0]}

    def test_unique_pmids_keeps_order(self):
        results = [_rec("a", "1", "x"), _rec("b", "1", "y"), _rec("c", "2", "z")]
        assert unique_pmids(results) == ["1", "2"]


# ── Domain rules ────────────────────────────────────────────────────

class TestDomainRules:
    @pytest.mark.parametrize("heading,domain", [
        ("Breast Neoplasms", "oncology"),
        ("Heart Failure", "cardiology"),
        ("Alzheimer Disease", "neurology"),
        ("COVID-19", "infectious_disease"),
        ("Diabetes Mellitus, Type 2", "endocrinology"),
        ("Gene Expression Profiling", "genetics"),
        ("Deep Learning", "bioinformatics"),
    ])
    def test_specific_headings(self, heading, domain):
        assert classify_article([heading, "Humans"]) == domain

    def test_general_is_not_genetics(self):
        assert classify_article(["General Surgery"]) == "other"

    def test_majority_vote(self):
        terms = ["Lung Neoplasms", "Carcinoma, Non-Small-Cell Lung", "Genomics"]
        assert classify_article(terms) == "oncology"

    def test_accepts_chroma_string(self):
        assert classify_article("Heart Failure, Humans") == "cardiology"


# ── BM25 and fusion ─────────────────────────────────────────────────

CORPUS = [
    _rec("1_0", "1", "BRCA1 mutation carriers have elevated breast cancer risk"),
    _rec("2_0", "2", "Statin therapy lowers cardiovascular events in older adults"),
    _rec("3_0", "3", "Tau protein aggregation in Alzheimer disease progression"),
]


class TestBM25:
    def test_tokenize_drops_stopwords(self):
        assert tokenize("The BRCA1 mutation of the gene") == ["brca1", "mutation", "gene"]

    def test_exact_term_ranks_first(self):
        store = BM25Store(CORPUS)
        results = store.search("BRCA1 breast cancer", top_k=3)
        assert results[0]["metadata"]["pmid"] == "1"

    def test_no_match_returns_nothing(self):
        store = BM25Store(CORPUS)
        assert store.search("xylophone", top_k=3) == []

    def test_from_jsonl(self, tmp_path):
        path = tmp_path / "meta.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in CORPUS))
        assert BM25Store.from_jsonl(path).count == 3


class TestRRF:
    def test_item_in_both_lists_wins(self):
        a = [{"chunk_id": "x"}, {"chunk_id": "y"}]
        b = [{"chunk_id": "z"}, {"chunk_id": "y"}]
        fused = reciprocal_rank_fusion([(a, 0.5), (b, 0.5)], top_k=3)
        assert fused[0]["chunk_id"] == "y"
        assert {r["chunk_id"] for r in fused} == {"x", "y", "z"}

    def test_does_not_mutate_inputs(self):
        a = [{"chunk_id": "x"}]
        reciprocal_rank_fusion([(a, 1.0)], top_k=1)
        assert "rrf_score" not in a[0]


class TestHybridRetriever:
    def test_ensemble_fuses_dense_and_bm25(self):
        chroma, faiss_store = MagicMock(), MagicMock()
        faiss_store.search.return_value = [CORPUS[2], CORPUS[0]]
        retriever = HybridRetriever(chroma, faiss_store, BM25Store(CORPUS))

        results = retriever.retrieve_ensemble(
            np.zeros(768), top_k=2, query_text="BRCA1 breast cancer"
        )
        chroma.search.assert_not_called()  # dense side is FAISS when no filter
        assert results[0]["metadata"]["pmid"] == "1"  # in both lists → top
        assert all(r["source"] == "ensemble" for r in results)

    def test_filter_routes_dense_side_to_chroma(self):
        chroma, faiss_store = MagicMock(), MagicMock()
        chroma.search.return_value = [CORPUS[1]]
        retriever = HybridRetriever(chroma, faiss_store, BM25Store(CORPUS))
        retriever.retrieve_ensemble(np.zeros(768), top_k=2, where={"pmid": "2"},
                                    query_text="statin")
        chroma.search.assert_called_once()
        faiss_store.search.assert_not_called()

    def test_fallback_without_bm25(self):
        chroma, faiss_store = MagicMock(), MagicMock()
        chroma.search.return_value = [CORPUS[0]]
        faiss_store.search.return_value = [CORPUS[0]]
        retriever = HybridRetriever(chroma, faiss_store)
        assert len(retriever.retrieve_ensemble(np.zeros(768), top_k=2)) == 1


# ── Evaluation logic ────────────────────────────────────────────────

class TestEvaluation:
    def _embedder(self):
        emb = MagicMock()
        emb.embed_single.return_value = np.zeros(768)
        return emb

    def test_domain_consistency_skips_own_article(self):
        own = _rec("1_0", "1", "own", ["Breast Neoplasms"])
        other = _rec("9_0", "9", "other", ["Heart Failure"])
        retriever = MagicMock()
        retriever.retrieve.return_value = [own, other]
        eval_set = [{"query": "own", "relevant_pmids": {"1"},
                     "mesh_terms": ["Breast Neoplasms"]}]

        report = evaluate_domain_consistency(eval_set, retriever, self._embedder())
        # Neighbour is cardiology, query is oncology → disagreement, not a free match
        assert report["accuracy"] == 0.0
        assert report["evaluated"] == 1

    def test_compare_methods_reports_lift_vs_bm25(self):
        faiss_store = MagicMock()
        faiss_store.search.return_value = [CORPUS[2], CORPUS[1], CORPUS[0]]  # answer at rank 3
        retriever = HybridRetriever(MagicMock(), faiss_store, BM25Store(CORPUS))
        eval_set = [{"query": "BRCA1 mutation breast cancer", "relevant_pmids": {"1"},
                     "mesh_terms": []}]

        r = compare_retrieval_methods(retriever, self._embedder(), eval_set, top_k=3)
        assert set(r) >= {"bm25", "dense", "hybrid", "comparison"}
        assert r["bm25"]["mean_mrr"] == 1.0
        assert r["dense"]["mean_mrr"] == pytest.approx(1 / 3, abs=1e-3)
        assert r["comparison"]["dense_vs_bm25_ndcg_lift_pct"] < 0


# ── Ingestion: PubMed 10K cap ───────────────────────────────────────

class TestPubMedSlicing:
    def test_slices_by_year_under_cap(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "ncbi_email", "test@example.org")
        from src.ingestion import pubmed_fetcher as pf

        calls = []

        def fake_esearch(**kwargs):
            calls.append(kwargs)
            year = int(kwargs["mindate"])
            return [f"{year}{i:05d}" for i in range(kwargs["retmax"])]

        def fake_handle(**kw):
            handle = MagicMock()
            handle.kwargs = kw
            return handle

        with patch.object(pf.Entrez, "esearch", side_effect=fake_handle), \
             patch.object(pf.Entrez, "read",
                          side_effect=lambda h: {"IdList": fake_esearch(**h.kwargs)}), \
             patch.object(pf.time, "sleep"):
            fetcher = pf.PubMedFetcher(target_count=25_000, output_dir=tmp_path, start_year=2025)
            pmids = fetcher._get_pmids(resume=False)

        assert len(pmids) == 25_000
        assert len(set(pmids)) == 25_000
        assert all(c["retmax"] <= pf.MAX_IDS_PER_QUERY for c in calls)
        assert [c["mindate"] for c in calls] == ["2025", "2024", "2023"]

    def test_requires_email(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "ncbi_email", "")
        from src.ingestion.pubmed_fetcher import PubMedFetcher
        with pytest.raises(ValueError, match="NCBI_EMAIL"):
            PubMedFetcher(output_dir=tmp_path)


# ── Generation without API key ──────────────────────────────────────

def test_generation_without_key_has_clear_error(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    from src.generation.rag_chain import MedicalRAGChain
    chain = MedicalRAGChain(retriever=MagicMock(), embedder=MagicMock())
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        _ = chain.client
