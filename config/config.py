"""
Central configuration for the Medical Literature Retrieval System.

Loads values from environment variables (via .env), validates them with Pydantic,
and exposes a single `settings` object imported throughout the codebase.

Usage:
    from config.config import settings
    print(settings.claude_model)
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root (resolves to medical-rag/ regardless of where script is run)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Validated application settings loaded from .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------- Anthropic Claude API --------
    # Optional at load time so ingestion, indexing and evaluation run without a
    # key. Only answer generation (rag_chain) needs it, and it checks there.
    anthropic_api_key: str = Field(default="", description="Claude API key")
    claude_model: str = Field(default="claude-sonnet-4-5")
    claude_max_tokens: int = Field(default=2048, ge=1, le=8192)
    claude_temperature: float = Field(default=0.1, ge=0.0, le=1.0)

    # -------- NCBI / PubMed --------
    # Required by NCBI terms of use for ingestion; checked by the fetcher.
    ncbi_email: str = Field(default="", description="Required by NCBI ToS")
    ncbi_api_key: Optional[str] = Field(default=None)

    # -------- Ingestion --------
    pubmed_target_count: int = Field(default=50_000, ge=1)
    pubmed_batch_size: int = Field(default=200, ge=1, le=10_000)
    pubmed_query: str = Field(default="")

    # -------- Embeddings --------
    # Default: PubMedBERT fine-tuned for retrieval on MS MARCO (768-dim).
    # Raw BioBERT (dmis-lab/biobert-v1.1) also works but was never trained for
    # sentence similarity, so it ranks noticeably worse for search.
    embedding_model: str = Field(default="pritamdeka/S-PubMedBert-MS-MARCO")
    embedding_batch_size: int = Field(default=32, ge=1)
    embedding_device: Literal["cpu", "cuda", "mps"] = Field(default="cpu")

    # -------- Chunking --------
    chunk_size: int = Field(default=512, ge=64, le=2048)
    chunk_overlap: int = Field(default=64, ge=0)

    # -------- Vector Stores --------
    chroma_persist_dir: Path = Field(default=PROJECT_ROOT / "data" / "chroma_db")
    chroma_collection_name: str = Field(default="pubmed_biobert")
    faiss_index_dir: Path = Field(default=PROJECT_ROOT / "data" / "faiss_index")
    faiss_index_type: Literal["Flat", "IVF"] = Field(default="IVF")

    # -------- Retrieval --------
    top_k: int = Field(default=8, ge=1, le=100)
    rerank_top_k: int = Field(default=4, ge=1, le=100)

    # -------- Logging --------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    log_file: Path = Field(default=PROJECT_ROOT / "logs" / "medical_rag.log")

    # -------- Derived paths (not from env) --------
    raw_data_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw")
    processed_data_dir: Path = Field(default=PROJECT_ROOT / "data" / "processed")
    embeddings_dir: Path = Field(default=PROJECT_ROOT / "data" / "embeddings")

    # -------- Validators --------
    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError(f"chunk_overlap ({v}) must be < chunk_size ({chunk_size})")
        return v

    @field_validator("rerank_top_k")
    @classmethod
    def rerank_less_than_top_k(cls, v: int, info) -> int:
        top_k = info.data.get("top_k", 8)
        if v > top_k:
            raise ValueError(f"rerank_top_k ({v}) must be <= top_k ({top_k})")
        return v

    def ensure_directories(self) -> None:
        """Create all required directories. Call once at app startup."""
        for path in [
            self.raw_data_dir,
            self.processed_data_dir,
            self.embeddings_dir,
            self.chroma_persist_dir,
            self.faiss_index_dir,
            self.log_file.parent,
        ]:
            path.mkdir(parents=True, exist_ok=True)


# Singleton — import this everywhere
settings = Settings()
