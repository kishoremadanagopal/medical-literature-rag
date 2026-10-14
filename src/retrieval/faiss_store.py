"""
FAISS vector store — optimized for raw retrieval speed.

Why FAISS alongside Chroma?
    - Sub-millisecond search on 50K+ vectors (vs ~50-100ms for Chroma)
    - IVF (Inverted File Index) partitions vectors into clusters, then
      only searches the nearest clusters → O(sqrt(N)) instead of O(N)
    - Used as the "speed layer" in our hybrid retriever — when you don't
      need metadata filtering, route to FAISS (benchmarked against Chroma
      in scripts/run_evaluation.py)

Trade-offs vs Chroma:
    - No built-in metadata filtering (we maintain a side-car index)
    - No persistence by default (we handle save/load manually)
    - No document storage (just vectors + IDs)

Index types:
    - Flat: exact search, brute-force cosine. Best for < 10K vectors.
    - IVF:  approximate search with configurable recall. Best for 10K+.
      Uses nlist clusters and searches nprobe nearest clusters.
"""

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from config.config import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


# ── IVF tuning parameters ──────────────────────────────────────────
# Rule of thumb: nlist ≈ sqrt(N). For 50K vectors → nlist ≈ 224.
# nprobe controls speed/recall trade-off: higher = more accurate, slower.
DEFAULT_NLIST = 256
DEFAULT_NPROBE = 16  # Search 16 out of 256 clusters (~94% recall)


class FAISSStore:
    """
    FAISS vector index with sidecar metadata.

    Usage:
        store = FAISSStore()
        store.build_from_embeddings(embeddings_path, metadata_path)
        store.save()

        # Later:
        store = FAISSStore()
        store.load()
        results = store.search(query_vector, top_k=5)
    """

    def __init__(
        self,
        index_dir: Optional[Path] = None,
        index_type: Optional[str] = None,
    ):
        self.index_dir = Path(index_dir or settings.faiss_index_dir)
        self.index_type = index_type or settings.faiss_index_type
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self._index: Optional[faiss.Index] = None
        self._metadata: list[dict] = []     # Sidecar: row i → metadata for vector i
        self._chunk_ids: list[str] = []     # Sidecar: row i → chunk_id for vector i

    @property
    def index(self) -> Optional[faiss.Index]:
        return self._index

    @property
    def count(self) -> int:
        return self._index.ntotal if self._index else 0

    # ── Building the index ──────────────────────────────────────────

    def build_from_embeddings(
        self,
        embeddings_path: Path,
        metadata_path: Path,
        nlist: int = DEFAULT_NLIST,
        nprobe: int = DEFAULT_NPROBE,
    ) -> int:
        """
        Build a FAISS index from pre-computed embeddings.

        Args:
            embeddings_path: Path to embeddings.npy
            metadata_path: Path to chunk_metadata.jsonl
            nlist: Number of IVF clusters (only for IVF index)
            nprobe: Number of clusters to search (speed/recall trade-off)

        Returns:
            Number of vectors indexed.
        """
        logger.info(f"Building FAISS {self.index_type} index")

        embeddings = np.load(embeddings_path).astype(np.float32)
        dim = embeddings.shape[1]
        n = embeddings.shape[0]

        logger.info(f"Loaded {n} vectors of dimension {dim}")

        # Load sidecar metadata
        self._metadata = []
        self._chunk_ids = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                self._chunk_ids.append(record["chunk_id"])
                self._metadata.append(record)

        assert len(self._metadata) == n, (
            f"Mismatch: {n} vectors vs {len(self._metadata)} metadata"
        )

        # Build the index
        if self.index_type == "Flat":
            self._index = faiss.IndexFlatIP(dim)  # Inner product (= cosine for normalized vecs)
            self._index.add(embeddings)
        elif self.index_type == "IVF":
            # IVF needs at least nlist training vectors
            actual_nlist = min(nlist, n // 2)  # Safety: don't exceed half the dataset
            quantizer = faiss.IndexFlatIP(dim)
            self._index = faiss.IndexIVFFlat(quantizer, dim, actual_nlist, faiss.METRIC_INNER_PRODUCT)

            logger.info(f"Training IVF index with nlist={actual_nlist}")
            self._index.train(embeddings)
            self._index.add(embeddings)
            self._index.nprobe = nprobe
            logger.info(f"IVF index built: nprobe={nprobe}")
        else:
            raise ValueError(f"Unknown index type: {self.index_type}. Use 'Flat' or 'IVF'.")

        logger.info(f"FAISS index built: {self.count} vectors, type={self.index_type}")
        return self.count

    # ── Searching ───────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = None,
    ) -> list[dict]:
        """
        Find the top-k most similar vectors.

        Args:
            query_embedding: Query vector (768-dim numpy array)
            top_k: Number of results

        Returns:
            List of dicts with: chunk_id, text, metadata, score
        """
        if self._index is None:
            raise RuntimeError("Index not built or loaded. Call build_from_embeddings() or load().")

        top_k = top_k or settings.top_k
        top_k = min(top_k, self.count)  # Can't retrieve more than we have

        query = query_embedding.astype(np.float32).reshape(1, -1)
        scores, indices = self._index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            record = self._metadata[idx]
            results.append({
                "chunk_id": self._chunk_ids[idx],
                "text": record.get("text", ""),
                "metadata": record.get("metadata", {}),
                "score": float(score),
            })

        return results

    def search_by_text(
        self,
        query_text: str,
        embedder,
        top_k: int = None,
    ) -> list[dict]:
        """Convenience: embed query then search."""
        query_vec = embedder.embed_single(query_text)
        return self.search(query_vec, top_k=top_k)

    # ── Persistence ─────────────────────────────────────────────────

    def save(self) -> None:
        """Save the FAISS index + sidecar metadata to disk."""
        if self._index is None:
            raise RuntimeError("No index to save.")

        index_path = self.index_dir / "index.faiss"
        meta_path = self.index_dir / "sidecar_metadata.jsonl"
        config_path = self.index_dir / "index_config.json"

        faiss.write_index(self._index, str(index_path))

        with open(meta_path, "w", encoding="utf-8") as f:
            for record in self._metadata:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        config = {
            "index_type": self.index_type,
            "count": self.count,
            "dimension": self._index.d,
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"FAISS index saved to {self.index_dir}")

    def load(self) -> int:
        """Load a previously saved FAISS index + metadata."""
        index_path = self.index_dir / "index.faiss"
        meta_path = self.index_dir / "sidecar_metadata.jsonl"

        if not index_path.exists():
            raise FileNotFoundError(f"No FAISS index at {index_path}")

        self._index = faiss.read_index(str(index_path))

        self._metadata = []
        self._chunk_ids = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                self._chunk_ids.append(record["chunk_id"])
                self._metadata.append(record)

        # Restore nprobe for IVF indexes
        if hasattr(self._index, "nprobe"):
            self._index.nprobe = DEFAULT_NPROBE

        logger.info(f"FAISS index loaded: {self.count} vectors")
        return self.count

    # ── Benchmarking ────────────────────────────────────────────────

    def benchmark(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
    ) -> dict:
        """
        Benchmark search latency over a batch of queries.
        Returns dict with p50, p95, p99 latencies in milliseconds.
        """
        import time

        latencies = []
        for query in query_embeddings:
            start = time.perf_counter()
            self.search(query, top_k=top_k)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            latencies.append(elapsed)

        latencies = sorted(latencies)
        n = len(latencies)
        return {
            "queries": n,
            "p50_ms": round(latencies[n // 2], 2),
            "p95_ms": round(latencies[int(n * 0.95)], 2),
            "p99_ms": round(latencies[int(n * 0.99)], 2),
            "mean_ms": round(sum(latencies) / n, 2),
        }
