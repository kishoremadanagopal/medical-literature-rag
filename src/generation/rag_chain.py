"""
RAG chain — connects the hybrid retriever to Claude for answer generation.

Architecture:
    Query → Embed → Retrieve (Hybrid) → Format Context → Claude → Cited Answer

Two modes:
    1. LangChain mode: Uses LangChain's LCEL (LangChain Expression Language)
       for composable, streamable chains. Best for the Streamlit demo.
    2. Direct mode: Calls the Anthropic SDK directly. Simpler, easier to debug,
       better for notebooks and testing.

Both modes produce identical output format:
    {
        "answer": "CRISPR-Cas9 has shown... [PMID:12345]",
        "sources": [{"pmid": "12345", "title": "...", "score": 0.92}],
        "query": "original question",
        "retrieval_time_ms": 45.2,
        "generation_time_ms": 1230.5,
    }
"""

import re
import time
from typing import Optional

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config.config import settings
from config.logging_config import get_logger
from src.generation.prompts import (
    QUERY_TEMPLATE,
    SUMMARY_TEMPLATE,
    SYSTEM_PROMPT,
    format_context,
)
from src.processing.embedder import BioBERTEmbedder
from src.retrieval.hybrid_retriever import HybridRetriever

logger = get_logger(__name__)


class MedicalRAGChain:
    """
    End-to-end RAG chain for medical literature Q&A.

    Usage:
        chain = MedicalRAGChain(retriever, embedder)
        result = chain.query("What are the latest treatments for glioblastoma?")
        print(result["answer"])
        print(result["sources"])
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        embedder: BioBERTEmbedder,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self.retriever = retriever
        self.embedder = embedder
        self.model = model or settings.claude_model
        self.temperature = temperature if temperature is not None else settings.claude_temperature
        self.max_tokens = max_tokens or settings.claude_max_tokens

        # Direct Anthropic client (for direct mode)
        self._client = None

        # LangChain model (for chain mode)
        self._llm = None

    @property
    def client(self) -> anthropic.Anthropic:
        """Lazy-load the Anthropic client."""
        if self._client is None:
            _require_api_key()
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        return self._client

    @property
    def llm(self) -> ChatAnthropic:
        """Lazy-load the LangChain ChatAnthropic model."""
        if self._llm is None:
            _require_api_key()
            self._llm = ChatAnthropic(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                anthropic_api_key=settings.anthropic_api_key,
            )
        return self._llm

    # ── Direct Mode (simple, for notebooks) ─────────────────────────

    def query(
        self,
        question: str,
        top_k: int = None,
        where: Optional[dict] = None,
        ensemble: bool = False,
    ) -> dict:
        """
        Answer a medical question using retrieve-then-generate.

        Args:
            question: Natural language question
            top_k: Number of documents to retrieve
            where: Metadata filter for Chroma (e.g. {"journal": "Nature"})
            ensemble: Use ensemble retrieval (both stores + RRF)

        Returns:
            Dict with 'answer', 'sources', 'query', timing info.
        """
        top_k = top_k or settings.top_k

        # ── Step 1: Retrieve ──
        retrieval_start = time.time()
        query_vec = self.embedder.embed_single(question)

        if ensemble:
            docs = self.retriever.retrieve_ensemble(
                query_vec, top_k=top_k, where=where, query_text=question
            )
        else:
            docs = self.retriever.retrieve(query_vec, top_k=top_k, where=where)

        retrieval_ms = (time.time() - retrieval_start) * 1000
        logger.info(f"Retrieved {len(docs)} docs in {retrieval_ms:.1f}ms")

        # ── Step 2: Format context ──
        context = format_context(docs)

        # ── Step 3: Generate answer ──
        generation_start = time.time()
        user_message = QUERY_TEMPLATE.format(context=context, question=question)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        answer = response.content[0].text
        generation_ms = (time.time() - generation_start) * 1000
        logger.info(f"Generated answer in {generation_ms:.1f}ms")

        # ── Step 4: Extract cited PMIDs ──
        cited_pmids = _extract_cited_pmids(answer)

        # Build source list (only sources that were actually cited)
        sources = []
        for doc in docs:
            pmid = doc.get("metadata", {}).get("pmid", "")
            if pmid in cited_pmids:
                sources.append({
                    "pmid": pmid,
                    "title": doc.get("metadata", {}).get("title", ""),
                    "journal": doc.get("metadata", {}).get("journal", ""),
                    "score": round(doc.get("score", 0), 4),
                })

        return {
            "answer": answer,
            "sources": sources,
            "all_retrieved": [
                {
                    "pmid": d.get("metadata", {}).get("pmid", ""),
                    "title": d.get("metadata", {}).get("title", ""),
                    "score": round(d.get("score", 0), 4),
                }
                for d in docs
            ],
            "query": question,
            "retrieval_time_ms": round(retrieval_ms, 1),
            "generation_time_ms": round(generation_ms, 1),
            "model": self.model,
            "cited_pmids": list(cited_pmids),
        }

    # ── LangChain Mode (for Streamlit streaming) ───────────────────

    def build_langchain_chain(self):
        """
        Build a LangChain LCEL chain for streaming responses.

        Returns a chain that accepts {"context": str, "question": str}
        and returns the answer string.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", QUERY_TEMPLATE),
        ])

        chain = prompt | self.llm | StrOutputParser()
        return chain

    def query_streaming(
        self,
        question: str,
        top_k: int = None,
        where: Optional[dict] = None,
        ensemble: bool = False,
    ):
        """
        Stream an answer token-by-token. Yields (chunk, metadata) tuples.

        The first yield is (None, metadata_dict) with retrieval info.
        Subsequent yields are (text_chunk, None) for each streamed token.

        Usage:
            for chunk, meta in chain.query_streaming("What is CRISPR?"):
                if meta:
                    st.write(f"Retrieved {len(meta['sources'])} sources")
                else:
                    st.write(chunk, end="")
        """
        top_k = top_k or settings.top_k

        # Retrieve
        retrieval_start = time.time()
        query_vec = self.embedder.embed_single(question)

        if ensemble:
            docs = self.retriever.retrieve_ensemble(
                query_vec, top_k=top_k, where=where, query_text=question
            )
        else:
            docs = self.retriever.retrieve(query_vec, top_k=top_k, where=where)

        retrieval_ms = (time.time() - retrieval_start) * 1000

        # Yield metadata first
        sources_meta = [
            {
                "pmid": d.get("metadata", {}).get("pmid", ""),
                "title": d.get("metadata", {}).get("title", ""),
                "score": round(d.get("score", 0), 4),
            }
            for d in docs
        ]
        yield None, {"sources": sources_meta, "retrieval_time_ms": round(retrieval_ms, 1)}

        # Stream generation
        context = format_context(docs)
        user_message = QUERY_TEMPLATE.format(context=context, question=question)

        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                yield text, None

    # ── Summarize Mode ──────────────────────────────────────────────

    def summarize_topic(
        self,
        topic: str,
        top_k: int = None,
        where: Optional[dict] = None,
    ) -> dict:
        """
        Generate a literature summary for a topic.
        Uses the SUMMARY_TEMPLATE instead of QUERY_TEMPLATE.
        """
        top_k = top_k or settings.top_k

        query_vec = self.embedder.embed_single(topic)
        docs = self.retriever.retrieve_ensemble(
            query_vec, top_k=top_k, where=where, query_text=topic
        )

        context = format_context(docs)
        user_message = SUMMARY_TEMPLATE.format(context=context)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        answer = response.content[0].text
        cited_pmids = _extract_cited_pmids(answer)

        return {
            "summary": answer,
            "topic": topic,
            "cited_pmids": list(cited_pmids),
            "total_sources": len(docs),
            "model": self.model,
        }


# ── Helpers ─────────────────────────────────────────────────────────

def _require_api_key() -> None:
    """Fail with a clear message if generation is attempted without a key."""
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env to generate answers "
            "(ingestion, indexing and evaluation do not need it)."
        )


def _extract_cited_pmids(text: str) -> set[str]:
    """
    Extract all PMIDs cited in an answer text.

    Handles formats:
        [PMID:12345]
        [PMID:12345, PMID:67890]
        [PMID: 12345]
    """
    pattern = r'PMID:\s*(\d+)'
    return set(re.findall(pattern, text))
