"""
PubMed article fetcher — pulls articles from NCBI via Entrez API.

Architecture:
    1. esearch  → get PMIDs matching query, one publication year at a time
                  (PubMed caps each query at 10,000 IDs)
    2. efetch   → download full XML records in batches
    3. parse    → extract structured fields (see parser.py)
    4. save     → append to JSONL file (one JSON object per line)

Key design decisions:
    - JSONL over JSON: Streaming writes. If the process crashes at 30K articles,
      you keep those 30K. A single JSON array would be corrupt.
    - Checkpoint file: Stores the last batch index so `--resume` picks up where
      it left off. Critical for a 50K ingestion that takes 2-4 hours.
    - Rate limiting: NCBI allows 3 req/sec (or 10 with API key). We respect this
      with time.sleep() rather than risking an IP ban.
"""

import json
import time
from datetime import date
from pathlib import Path
from typing import Optional

from Bio import Entrez

from config.config import settings
from config.logging_config import get_logger
from src.ingestion.parser import parse_pubmed_xml

logger = get_logger(__name__)


# ── NCBI rate limits ────────────────────────────────────────────────
# Without API key: 3 requests/second → sleep 0.34s between requests
# With API key:   10 requests/second → sleep 0.11s between requests
DELAY_WITH_KEY = 0.11
DELAY_WITHOUT_KEY = 0.34

# PubMed ESearch hard limit: only the first 10,000 results of a query are
# reachable, so each date-sliced query asks for at most this many IDs.
MAX_IDS_PER_QUERY = 9_999


class PubMedFetcher:
    """
    Fetches articles from PubMed and saves them as JSONL.

    Usage:
        fetcher = PubMedFetcher()
        fetcher.run()                    # Fresh start
        fetcher.run(resume=True)         # Resume from checkpoint
    """

    def __init__(
        self,
        target_count: Optional[int] = None,
        batch_size: Optional[int] = None,
        query: Optional[str] = None,
        output_dir: Optional[Path] = None,
        start_year: Optional[int] = None,
        min_year: int = 1990,
    ):
        if not settings.ncbi_email:
            raise ValueError("NCBI_EMAIL must be set in .env (required by NCBI terms of use).")

        self.target_count = target_count or settings.pubmed_target_count
        # Start from last year so the newest slice is a complete year.
        self.start_year = start_year or (date.today().year - 1)
        self.min_year = min_year
        self.batch_size = batch_size or settings.pubmed_batch_size
        self.query = query or settings.pubmed_query or self._default_query()
        self.output_dir = output_dir or settings.raw_data_dir

        # Configure Entrez
        Entrez.email = settings.ncbi_email
        if settings.ncbi_api_key:
            Entrez.api_key = settings.ncbi_api_key

        self.delay = DELAY_WITH_KEY if settings.ncbi_api_key else DELAY_WITHOUT_KEY

        # File paths
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / "pubmed_articles.jsonl"
        self.checkpoint_file = self.output_dir / "ingestion_checkpoint.json"
        self.pmids_file = self.output_dir / "pmids.json"

    @staticmethod
    def _default_query() -> str:
        """
        Broad biomedical query that returns millions of results.
        We filter for: has abstract, is in English, is a journal article.
        This gives us high-quality, text-rich records.
        """
        return (
            "(hasabstract[text] AND english[Language] AND "
            "journal article[Publication Type])"
        )

    # ── Public API ──────────────────────────────────────────────────

    def run(self, resume: bool = False) -> Path:
        """
        Execute the full ingestion pipeline.

        Args:
            resume: If True, continue from the last checkpoint.

        Returns:
            Path to the output JSONL file.
        """
        logger.info(f"Starting PubMed ingestion: target={self.target_count}, "
                     f"batch_size={self.batch_size}")
        logger.info(f"Query: {self.query}")

        # Step 1: Get PMIDs
        pmids = self._get_pmids(resume)
        total_pmids = len(pmids)
        logger.info(f"Total PMIDs to fetch: {total_pmids}")

        # Step 2: Determine starting batch
        start_batch = 0
        if resume:
            start_batch = self._load_checkpoint()
            if start_batch > 0:
                logger.info(f"Resuming from batch {start_batch}")

        # Step 3: Fetch in batches
        total_articles = 0
        total_batches = (total_pmids + self.batch_size - 1) // self.batch_size
        file_mode = "a" if resume and start_batch > 0 else "w"

        with open(self.output_file, file_mode, encoding="utf-8") as f:
            for batch_idx in range(start_batch, total_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, total_pmids)
                batch_pmids = pmids[batch_start:batch_end]

                # Fetch and parse this batch
                articles = self._fetch_batch(batch_pmids, batch_idx, total_batches)

                # Write to JSONL
                for article in articles:
                    f.write(json.dumps(article, ensure_ascii=False) + "\n")
                    total_articles += 1

                f.flush()  # Ensure data is on disk

                # Save checkpoint
                self._save_checkpoint(batch_idx + 1)

                # Progress log every 10 batches
                if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
                    logger.info(
                        f"Progress: batch {batch_idx + 1}/{total_batches} "
                        f"({total_articles} articles saved)"
                    )

        logger.info(f"Ingestion complete: {total_articles} articles → {self.output_file}")
        return self.output_file

    # ── PMID Retrieval ──────────────────────────────────────────────

    def _get_pmids(self, resume: bool) -> list[str]:
        """
        Get PMIDs matching the query. Uses cached PMIDs if resuming.
        """
        if resume and self.pmids_file.exists():
            logger.info("Loading cached PMIDs from previous run")
            with open(self.pmids_file) as f:
                return json.load(f)

        logger.info("Searching PubMed for matching article IDs...")

        # PubMed's ESearch only returns the first 10,000 IDs of any query
        # (retstart + retmax <= 10,000). To collect more, we split the query
        # into publication-year slices, newest first, and take up to
        # MAX_IDS_PER_QUERY from each slice until the target is reached.
        pmids: list[str] = []
        seen: set[str] = set()
        year = self.start_year

        while len(pmids) < self.target_count and year >= self.min_year:
            remaining = self.target_count - len(pmids)
            slice_ids = self._search_ids(
                retmax=min(MAX_IDS_PER_QUERY, remaining),
                mindate=str(year),
                maxdate=str(year),
            )
            new_ids = [p for p in slice_ids if p not in seen]
            seen.update(new_ids)
            pmids.extend(new_ids)
            logger.info(f"  {year}: +{len(new_ids)} PMIDs (total {len(pmids)})")
            year -= 1

        pmids = pmids[: self.target_count]
        logger.info(f"Found {len(pmids)} PMIDs (requested {self.target_count})")

        # Cache PMIDs so resume doesn't re-query
        with open(self.pmids_file, "w") as f:
            json.dump(pmids, f)

        return pmids

    def _search_ids(self, retmax: int, mindate: str, maxdate: str) -> list[str]:
        """One ESearch call restricted to a publication-date range."""
        handle = Entrez.esearch(
            db="pubmed",
            term=self.query,
            retmax=retmax,
            datetype="pdat",
            mindate=mindate,
            maxdate=maxdate,
        )
        results = Entrez.read(handle)
        handle.close()
        time.sleep(self.delay)
        return list(results.get("IdList", []))

    # ── Batch Fetching ──────────────────────────────────────────────

    def _fetch_batch(
        self, pmids: list[str], batch_idx: int, total_batches: int
    ) -> list[dict]:
        """
        Fetch a single batch of articles by PMID.
        Retries up to 3 times on failure.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                handle = Entrez.efetch(
                    db="pubmed",
                    id=",".join(pmids),
                    rettype="xml",
                    retmode="xml",
                )
                xml_data = handle.read()
                handle.close()

                # Respect rate limit
                time.sleep(self.delay)

                # Parse XML → list of dicts
                if isinstance(xml_data, bytes):
                    xml_data = xml_data.decode("utf-8")

                articles = parse_pubmed_xml(xml_data)
                return articles

            except Exception as e:
                wait = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                logger.warning(
                    f"Batch {batch_idx + 1}/{total_batches} attempt "
                    f"{attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)

        logger.error(
            f"Batch {batch_idx + 1}/{total_batches} failed after "
            f"{max_retries} retries. Skipping {len(pmids)} articles."
        )
        return []

    # ── Checkpoint Management ───────────────────────────────────────

    def _save_checkpoint(self, batch_idx: int) -> None:
        """Save progress so ingestion can resume."""
        checkpoint = {
            "last_completed_batch": batch_idx,
            "target_count": self.target_count,
            "batch_size": self.batch_size,
            "query": self.query,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def _load_checkpoint(self) -> int:
        """Load the last completed batch index."""
        if not self.checkpoint_file.exists():
            return 0
        try:
            with open(self.checkpoint_file) as f:
                checkpoint = json.load(f)
            return checkpoint.get("last_completed_batch", 0)
        except (json.JSONDecodeError, KeyError):
            logger.warning("Corrupt checkpoint file. Starting from scratch.")
            return 0
