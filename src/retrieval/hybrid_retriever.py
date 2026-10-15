"""
Hybrid retriever — routes queries to the right vector store.

Strategy:
    - Query has metadata filters (year, journal, MeSH) → Chroma
    - Query is pure similarity search → FAISS (faster)
    - Hybrid (ensemble) mode fuses DENSE results (FAISS, or Chroma when a
      filter is set) with SPARSE keyword results (BM25) using Reciprocal Rank
      Fusion. Dense search finds paraphrases; BM25 finds exact terms such as
      gene and drug names. Fusing the two usually beats either alone.

Note: Chroma and FAISS hold the *same* embeddings, so fusing Chroma with FAISS
adds nothing — the two rankings are near-identical. That older behaviour is
kept only as a fallback when no BM25 index is supplied.

This is the component that LangChain's RAG chain will call in Step 5.
It implements a clean interface: query in → ranked documents out.
"""

from typing import Optional

import numpy as np

from config.config import settings
from config.logging_config import get_logger
from src.retrieval.chroma_store import ChromaStore
from src.retrieval.faiss_store import FAISSStore

logger = get_logger(__name__)


class HybridRetriever:
    """
    Routes retrieval queries to Chroma or FAISS based on query characteristics.

    Usage:
        retriever = HybridRetriever(chroma_store, faiss_store)

        # Pure similarity → FAISS (fast)
        results = retriever.retrieve(query_embedding)

        # With metadata filter → Chroma
        results = retriever.retrieve(query_embedding, where={"journal": "Nature"})

        # Ensemble: merge results from both stores
        results = retriever.retrieve_ensemble(query_embedding)
    """

    def __init__(
        self,
        chroma_store: ChromaStore,
        faiss_store: FAISSStore,
        bm25_store=None,
    ):
        self.chroma = chroma_store
        self.faiss = faiss_store
        self.bm25 = bm25_store  # Optional BM25Store for hybrid dense + keyword retrieval

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = None,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        force_store: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieve relevant documents, routing to the best store.

        Routing logic:
            1. If force_store is set → use that store
            2. If metadata filters are present → Chroma (only one that supports filtering)
            3. Otherwise → FAISS (faster for pure similarity)

        Args:
            query_embedding: 768-dim query vector
            top_k: Number of results
            where: Metadata filter dict (forces Chroma)
            where_document: Document content filter (forces Chroma)
            force_store: "chroma" or "faiss" to override routing

        Returns:
            List of result dicts: chunk_id, text, metadata, score, source
        """
        top_k = top_k or settings.top_k

        # Determine which store to use
        if force_store == "chroma":
            store_name = "chroma"
        elif force_store == "faiss":
            store_name = "faiss"
        elif where or where_document:
            store_name = "chroma"
        else:
            store_name = "faiss"

        logger.debug(f"Routing query to {store_name} (top_k={top_k})")

        if store_name == "chroma":
            results = self.chroma.search(
                query_embedding, top_k=top_k, where=where, where_document=where_document
            )
        else:
            results = self.faiss.search(query_embedding, top_k=top_k)

        # Tag results with their source
        for r in results:
            r["source"] = store_name

        return results

    def retrieve_ensemble(
        self,
        query_embedding: np.ndarray,
        top_k: int = None,
        where: Optional[dict] = None,
        weights: tuple[float, float] = (0.5, 0.5),
        query_text: Optional[str] = None,
    ) -> list[dict]:
        """
        Hybrid retrieval: merge two rankings with Reciprocal Rank Fusion (RRF).

        RRF ranks items by  sum(weight_i / (k + rank_in_list_i)),  k = 60.
        It only uses ranks, so the very different score scales of cosine
        similarity and BM25 never have to be normalised against each other.

        Lists fused:
            - With a BM25 index and query_text: dense (FAISS, or Chroma if a
              metadata filter is set) + BM25 keyword results.
            - Otherwise (fallback): Chroma + FAISS.

        Args:
            query_embedding: Query vector
            top_k: Final number of merged results
            where: Optional metadata filter (dense side goes through Chroma)
            weights: (dense_weight, sparse_weight) — or (chroma, faiss) in fallback
            query_text: Raw query text, needed for BM25

        Returns:
            Merged, deduplicated, re-ranked results.
        """
        top_k = top_k or settings.top_k
        fetch_k = top_k * 2  # Over-fetch from each list before merging

        if self.bm25 is not None and query_text:
            if where:
                dense = self.chroma.search(query_embedding, top_k=fetch_k, where=where)
                sparse = self._filter_by_metadata(
                    self.bm25.search(query_text, top_k=fetch_k * 5), where
                )[:fetch_k]
            else:
                dense = self.faiss.search(query_embedding, top_k=fetch_k)
                sparse = self.bm25.search(query_text, top_k=fetch_k)
            lists = [(dense, weights[0]), (sparse, weights[1])]
            label = "dense+bm25"
        else:
            chroma_results = self.chroma.search(query_embedding, top_k=fetch_k, where=where)
            faiss_results = self.faiss.search(query_embedding, top_k=fetch_k)
            lists = [(chroma_results, weights[0]), (faiss_results, weights[1])]
            label = "chroma+faiss"

        output = reciprocal_rank_fusion(lists, top_k=top_k)
        for r in output:
            r["source"] = "ensemble"

        logger.debug(
            f"Ensemble ({label}): " + " + ".join(str(len(l)) for l, _ in lists)
            + f" → {len(output)} merged"
        )
        return output

    @staticmethod
    def _filter_by_metadata(results: list[dict], where: dict) -> list[dict]:
        """Apply a simple {field: value} equality filter to BM25 results."""
        def keep(r):
            meta = r.get("metadata", {})
            return all(str(meta.get(k, "")) == str(v) for k, v in where.items()
                       if not k.startswith("$"))
        return [r for r in results if keep(r)]

    def retrieve_by_text(
        self,
        query_text: str,
        embedder,
        top_k: int = None,
        where: Optional[dict] = None,
        ensemble: bool = False,
    ) -> list[dict]:
        """
        Convenience: embed query text, then retrieve.
        This is the main entry point for the RAG chain in Step 5.
        """
        query_vec = embedder.embed_single(query_text)
        if ensemble:
            return self.retrieve_ensemble(
                query_vec, top_k=top_k, where=where, query_text=query_text
            )
        return self.retrieve(query_vec, top_k=top_k, where=where)


def reciprocal_rank_fusion(
    ranked_lists: list[tuple[list[dict], float]],
    top_k: int,
    k: int = 60,
) -> list[dict]:
    """
    Weighted Reciprocal Rank Fusion over result lists keyed by chunk_id.

    Args:
        ranked_lists: [(results, weight), ...], each results list best-first
        top_k: Number of fused results to return
        k: RRF damping constant (60 is the value from the original paper)
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, dict] = {}
    for results, weight in ranked_lists:
        for rank, result in enumerate(results):
            cid = result["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank + 1)
            first_seen.setdefault(cid, result)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    output = []
    for cid, score in ranked:
        result = dict(first_seen[cid])
        result["rrf_score"] = round(score, 6)
        output.append(result)
    return output
