"""
Tests for src/retrieval/ — Chroma, FAISS, and Hybrid Retriever.

Uses small synthetic datasets so tests run in seconds without the full pipeline.
Run: pytest tests/test_retrieval.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_data(tmp_path):
    """Create 100 synthetic embeddings + metadata for testing."""
    np.random.seed(42)
    n, dim = 100, 768

    # Create normalized random vectors (simulates real embeddings)
    embeddings = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    emb_path = tmp_path / "embeddings.npy"
    np.save(emb_path, embeddings)

    # Create metadata
    journals = ["Nature", "Science", "The Lancet", "NEJM", "BMJ"]
    meta_path = tmp_path / "chunk_metadata.jsonl"
    with open(meta_path, "w") as f:
        for i in range(n):
            record = {
                "chunk_id": f"PMID{i:04d}_chunk_0",
                "text": f"Article {i} about biomedical topic {i % 10}.",
                "metadata": {
                    "pmid": f"PMID{i:04d}",
                    "title": f"Article {i}",
                    "journal": journals[i % len(journals)],
                    "pub_date": f"202{i % 5}-01-01",
                    "mesh_terms": f"Term{i % 5}, Term{(i+1) % 5}",
                    "chunk_index": 0,
                    "token_estimate": 100,
                },
            }
            f.write(json.dumps(record) + "\n")

    return embeddings, emb_path, meta_path


@pytest.fixture
def query_vector(synthetic_data):
    """A query vector (first embedding, so top result should be itself)."""
    embeddings = synthetic_data[0]
    return embeddings[0]


# ── FAISS Tests ─────────────────────────────────────────────────────

class TestFAISSStore:
    def test_build_flat(self, synthetic_data, tmp_path):
        from src.retrieval.faiss_store import FAISSStore

        _, emb_path, meta_path = synthetic_data
        store = FAISSStore(index_dir=tmp_path / "faiss", index_type="Flat")
        count = store.build_from_embeddings(emb_path, meta_path)
        assert count == 100

    def test_build_ivf(self, synthetic_data, tmp_path):
        from src.retrieval.faiss_store import FAISSStore

        _, emb_path, meta_path = synthetic_data
        store = FAISSStore(index_dir=tmp_path / "faiss", index_type="IVF")
        count = store.build_from_embeddings(emb_path, meta_path, nlist=10)
        assert count == 100

    def test_search_returns_results(self, synthetic_data, query_vector, tmp_path):
        from src.retrieval.faiss_store import FAISSStore

        _, emb_path, meta_path = synthetic_data
        store = FAISSStore(index_dir=tmp_path / "faiss", index_type="Flat")
        store.build_from_embeddings(emb_path, meta_path)

        results = store.search(query_vector, top_k=5)
        assert len(results) == 5
        assert results[0]["chunk_id"] == "PMID0000_chunk_0"  # Most similar to itself
        assert results[0]["score"] > 0.99  # Cosine sim ≈ 1.0

    def test_save_and_load(self, synthetic_data, query_vector, tmp_path):
        from src.retrieval.faiss_store import FAISSStore

        _, emb_path, meta_path = synthetic_data
        store = FAISSStore(index_dir=tmp_path / "faiss", index_type="Flat")
        store.build_from_embeddings(emb_path, meta_path)
        store.save()

        loaded = FAISSStore(index_dir=tmp_path / "faiss")
        loaded.load()
        assert loaded.count == 100

        results = loaded.search(query_vector, top_k=3)
        assert len(results) == 3
        assert results[0]["chunk_id"] == "PMID0000_chunk_0"

    def test_benchmark(self, synthetic_data, tmp_path):
        from src.retrieval.faiss_store import FAISSStore

        embeddings, emb_path, meta_path = synthetic_data
        store = FAISSStore(index_dir=tmp_path / "faiss", index_type="Flat")
        store.build_from_embeddings(emb_path, meta_path)

        bench = store.benchmark(embeddings[:10], top_k=5)
        assert "p50_ms" in bench
        assert "p95_ms" in bench
        assert bench["queries"] == 10


# ── Chroma Tests ────────────────────────────────────────────────────

class TestChromaStore:
    def test_build(self, synthetic_data, tmp_path):
        from src.retrieval.chroma_store import ChromaStore

        _, emb_path, meta_path = synthetic_data
        store = ChromaStore(
            persist_dir=tmp_path / "chroma",
            collection_name="test_collection",
        )
        count = store.build_from_embeddings(emb_path, meta_path)
        assert count == 100

    def test_search(self, synthetic_data, query_vector, tmp_path):
        from src.retrieval.chroma_store import ChromaStore

        _, emb_path, meta_path = synthetic_data
        store = ChromaStore(
            persist_dir=tmp_path / "chroma",
            collection_name="test_search",
        )
        store.build_from_embeddings(emb_path, meta_path)

        results = store.search(query_vector, top_k=5)
        assert len(results) == 5
        assert results[0]["chunk_id"] == "PMID0000_chunk_0"
        assert results[0]["score"] > 0.9

    def test_metadata_filter(self, synthetic_data, query_vector, tmp_path):
        from src.retrieval.chroma_store import ChromaStore

        _, emb_path, meta_path = synthetic_data
        store = ChromaStore(
            persist_dir=tmp_path / "chroma",
            collection_name="test_filter",
        )
        store.build_from_embeddings(emb_path, meta_path)

        # Filter by journal
        results = store.search(
            query_vector, top_k=5, where={"journal": "Nature"}
        )
        assert len(results) > 0
        assert all(r["metadata"]["journal"] == "Nature" for r in results)

    def test_persistence(self, synthetic_data, query_vector, tmp_path):
        from src.retrieval.chroma_store import ChromaStore

        _, emb_path, meta_path = synthetic_data
        persist_dir = tmp_path / "chroma_persist"

        # Build
        store1 = ChromaStore(persist_dir=persist_dir, collection_name="persist_test")
        store1.build_from_embeddings(emb_path, meta_path)

        # Reload from disk
        store2 = ChromaStore(persist_dir=persist_dir, collection_name="persist_test")
        assert store2.count == 100

        results = store2.search(query_vector, top_k=3)
        assert len(results) == 3


# ── Hybrid Retriever Tests ──────────────────────────────────────────

class TestHybridRetriever:
    @pytest.fixture
    def stores(self, synthetic_data, tmp_path):
        from src.retrieval.chroma_store import ChromaStore
        from src.retrieval.faiss_store import FAISSStore

        _, emb_path, meta_path = synthetic_data
        chroma = ChromaStore(persist_dir=tmp_path / "chroma", collection_name="hybrid_test")
        chroma.build_from_embeddings(emb_path, meta_path)

        faiss_store = FAISSStore(index_dir=tmp_path / "faiss", index_type="Flat")
        faiss_store.build_from_embeddings(emb_path, meta_path)

        return chroma, faiss_store

    def test_routes_to_faiss_by_default(self, stores, query_vector):
        from src.retrieval.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever(*stores)
        results = retriever.retrieve(query_vector, top_k=5)
        assert len(results) == 5
        assert results[0]["source"] == "faiss"

    def test_routes_to_chroma_with_filter(self, stores, query_vector):
        from src.retrieval.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever(*stores)
        results = retriever.retrieve(
            query_vector, top_k=5, where={"journal": "Nature"}
        )
        assert all(r["source"] == "chroma" for r in results)
        assert all(r["metadata"]["journal"] == "Nature" for r in results)

    def test_ensemble_merges_and_deduplicates(self, stores, query_vector):
        from src.retrieval.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever(*stores)
        results = retriever.retrieve_ensemble(query_vector, top_k=5)
        assert len(results) == 5
        assert results[0]["source"] == "ensemble"
        assert "rrf_score" in results[0]

        # Should be deduplicated
        ids = [r["chunk_id"] for r in results]
        assert len(ids) == len(set(ids))
