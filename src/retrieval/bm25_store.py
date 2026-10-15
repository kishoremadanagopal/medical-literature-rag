"""
BM25 keyword index — the sparse (lexical) half of hybrid retrieval.

Why have a keyword index next to the vector stores?
    - Dense embeddings capture meaning ("heart attack" ≈ "myocardial infarction")
      but can miss exact tokens: gene names (BRCA1), drug names, abbreviations.
    - BM25 is the standard keyword ranking function. It scores a chunk by how
      often the query's words appear in it, down-weighting common words and
      normalising for chunk length.
    - It is also the standard *baseline* for any retrieval system: if the
      embedding search can't beat BM25, the embeddings aren't earning their keep.

The index is built in memory from the same chunk records the vector stores
use, so all three stores return identical result dicts:
    {chunk_id, text, metadata, score}
"""

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from config.config import settings
from config.logging_config import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")

# Small English stopword list; kept local so there is no NLTK download step.
_STOPWORDS = frozenset("""
a an and are as at be by for from has have in is it its of on or that the
this to was were with we our these those their than then there which who
""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class BM25Store:
    """
    In-memory BM25 index over chunk texts.

    Usage:
        store = BM25Store.from_jsonl(settings.embeddings_dir / "chunk_metadata.jsonl")
        results = store.search("BRCA1 mutation breast cancer", top_k=10)
    """

    def __init__(self, records: list[dict], k1: float = 1.5, b: float = 0.75):
        if not records:
            raise ValueError("BM25Store needs at least one record")
        self._records = records
        self._bm25 = BM25Okapi([tokenize(r.get("text", "")) for r in records], k1=k1, b=b)
        logger.info(f"BM25 index built over {len(records):,} chunks")

    @classmethod
    def from_jsonl(cls, path: Path) -> "BM25Store":
        """Build from chunk_metadata.jsonl (or chunks.jsonl) — one record per line."""
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return cls(records)

    @classmethod
    def from_faiss_store(cls, faiss_store) -> "BM25Store":
        """Reuse the FAISS sidecar metadata so no extra file is needed."""
        return cls(list(faiss_store._metadata))

    @property
    def count(self) -> int:
        return len(self._records)

    def search(self, query_text: str, top_k: Optional[int] = None) -> list[dict]:
        top_k = min(top_k or settings.top_k, self.count)
        scores = self._bm25.get_scores(tokenize(query_text))
        # argpartition is O(N); only the top_k are fully sorted
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        results = []
        for idx in top_idx:
            if scores[idx] <= 0:
                continue  # no query word matched this chunk
            record = self._records[idx]
            results.append({
                "chunk_id": record["chunk_id"],
                "text": record.get("text", ""),
                "metadata": record.get("metadata", {}),
                "score": float(scores[idx]),
            })
        return results
