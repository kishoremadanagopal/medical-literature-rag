"""
Tests for src/generation/ — prompts, context formatting, citation extraction, RAG chain.

The Claude API is mocked so tests run without an API key.
Run: pytest tests/test_rag_chain.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.prompts import format_context, SYSTEM_PROMPT, QUERY_TEMPLATE
from src.generation.rag_chain import _extract_cited_pmids


# ── Prompt Tests ────────────────────────────────────────────────────

class TestPrompts:
    def test_system_prompt_has_citation_rules(self):
        assert "PMID" in SYSTEM_PROMPT
        assert "Cite" in SYSTEM_PROMPT or "cite" in SYSTEM_PROMPT

    def test_system_prompt_has_hallucination_guard(self):
        """System prompt must tell Claude to say 'I don't know'."""
        assert "cannot" in SYSTEM_PROMPT.lower() or "don't" in SYSTEM_PROMPT.lower()

    def test_query_template_has_placeholders(self):
        assert "{context}" in QUERY_TEMPLATE
        assert "{question}" in QUERY_TEMPLATE

    def test_query_template_renders(self):
        rendered = QUERY_TEMPLATE.format(
            context="Some context here", question="What is CRISPR?"
        )
        assert "Some context here" in rendered
        assert "What is CRISPR?" in rendered


# ── Context Formatting Tests ────────────────────────────────────────

class TestFormatContext:
    def test_empty_docs(self):
        result = format_context([])
        assert "No relevant documents" in result

    def test_single_doc(self):
        docs = [{
            "text": "CRISPR enables precise gene editing.",
            "metadata": {
                "pmid": "12345",
                "title": "CRISPR Review",
                "journal": "Nature",
                "pub_date": "2023-06",
            },
            "score": 0.95,
        }]
        result = format_context(docs)
        assert "PMID: 12345" in result
        assert "Nature" in result
        assert "CRISPR enables precise gene editing" in result
        assert "0.950" in result  # Relevance score

    def test_multiple_docs_separated(self):
        docs = [
            {
                "text": "First paper content.",
                "metadata": {"pmid": "111", "title": "Paper 1", "journal": "J1", "pub_date": "2023"},
                "score": 0.9,
            },
            {
                "text": "Second paper content.",
                "metadata": {"pmid": "222", "title": "Paper 2", "journal": "J2", "pub_date": "2024"},
                "score": 0.8,
            },
        ]
        result = format_context(docs)
        assert "Source 1" in result
        assert "Source 2" in result
        assert "---" in result  # Separator between docs

    def test_missing_metadata_handled(self):
        """Docs with missing fields shouldn't crash."""
        docs = [{"text": "Some content.", "metadata": {}, "score": 0.5}]
        result = format_context(docs)
        assert "Some content." in result


# ── Citation Extraction Tests ───────────────────────────────────────

class TestCitationExtraction:
    def test_single_citation(self):
        text = "CRISPR is effective [PMID:12345]."
        assert _extract_cited_pmids(text) == {"12345"}

    def test_multiple_citations(self):
        text = "Studies show [PMID:111] and [PMID:222] that..."
        assert _extract_cited_pmids(text) == {"111", "222"}

    def test_grouped_citations(self):
        text = "Multiple studies agree [PMID:111, PMID:222]."
        assert _extract_cited_pmids(text) == {"111", "222"}

    def test_citation_with_space(self):
        text = "As shown [PMID: 12345]."
        assert _extract_cited_pmids(text) == {"12345"}

    def test_no_citations(self):
        text = "This text has no citations."
        assert _extract_cited_pmids(text) == set()

    def test_duplicate_citations(self):
        text = "First [PMID:111] and again [PMID:111]."
        assert _extract_cited_pmids(text) == {"111"}


# ── RAG Chain Tests (mocked Claude) ─────────────────────────────────

class TestMedicalRAGChain:
    @pytest.fixture
    def mock_chain(self):
        """Create a RAG chain with mocked retriever, embedder, and Claude."""
        from src.generation.rag_chain import MedicalRAGChain

        # Mock embedder
        embedder = MagicMock()
        embedder.embed_single.return_value = [0.1] * 768  # Fake 768d vector

        # Mock retriever
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {
                "chunk_id": "PMD1234_chunk_0",
                "text": "CRISPR-Cas9 achieves 95% editing efficiency in T-cells.",
                "metadata": {
                    "pmid": "1234",
                    "title": "CRISPR T-Cell Therapy",
                    "journal": "Nature Medicine",
                    "pub_date": "2023-06",
                },
                "score": 0.92,
            },
            {
                "chunk_id": "PMD5678_chunk_0",
                "text": "Off-target effects remain a concern in clinical CRISPR applications.",
                "metadata": {
                    "pmid": "5678",
                    "title": "CRISPR Safety Profile",
                    "journal": "Science",
                    "pub_date": "2023-08",
                },
                "score": 0.87,
            },
        ]

        chain = MedicalRAGChain(retriever, embedder)
        return chain

    def test_query_calls_retriever_and_claude(self, mock_chain):
        """Test the full query flow with mocked Claude response."""
        # Mock Claude API response
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text=(
                "CRISPR-Cas9 has shown high editing efficiency in T-cells [PMID:1234]. "
                "However, off-target effects remain a concern [PMID:5678].\n\n"
                "Sources:\n"
                "- PMID:1234 - CRISPR T-Cell Therapy\n"
                "- PMID:5678 - CRISPR Safety Profile"
            ))
        ]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_chain._client = mock_client

        result = mock_chain.query("What are CRISPR off-target effects?")

        assert result["query"] == "What are CRISPR off-target effects?"
        assert "PMID:1234" in result["answer"]
        assert "PMID:5678" in result["answer"]
        assert {"1234", "5678"} == set(result["cited_pmids"])
        assert len(result["sources"]) == 2
        assert result["retrieval_time_ms"] >= 0
        assert result["generation_time_ms"] >= 0

    def test_query_filters_uncited_sources(self, mock_chain):
        """Sources retrieved but not cited should appear in all_retrieved, not sources."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text="Only one paper is relevant [PMID:1234].")
        ]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_chain._client = mock_client

        result = mock_chain.query("test")

        # Only PMID 1234 was cited
        assert len(result["sources"]) == 1
        assert result["sources"][0]["pmid"] == "1234"

        # But both were retrieved
        assert len(result["all_retrieved"]) == 2
