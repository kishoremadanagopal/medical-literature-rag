"""
Embedding pipeline (sentence-transformers).

Default model: pritamdeka/S-PubMedBert-MS-MARCO
    - PubMedBERT (pre-trained on PubMed abstracts + full text), then
      fine-tuned on MS MARCO for search: query → relevant passage.
    - 768-dim vectors, runs locally — no API costs, no rate limits.

Why not raw BioBERT (dmis-lab/biobert-v1.1)?
    - BioBERT knows biomedical vocabulary, but it was trained for masked-word
      prediction, not for sentence similarity. Loading it in
      sentence-transformers silently adds mean pooling, and the resulting
      vectors rank search results poorly. A model fine-tuned for retrieval
      fixes that. Any sentence-transformers model can be set via
      EMBEDDING_MODEL in .env (BioBERT included) and compared with
      scripts/run_evaluation.py.

We use sentence-transformers as the wrapper because it handles:
    - Pooling (mean of token embeddings)
    - Normalization (unit vectors for cosine similarity)
    - Batching with progress bars
    - GPU/CPU device management
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

from config.config import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


class BioBERTEmbedder:
    """
    Embeds text chunks with a biomedical sentence-transformers model.
    (Class name kept from the original BioBERT version for compatibility.)

    Lazy-loads the model on first use (~500MB download on first run,
    ~2 seconds to load from cache after that).

    Usage:
        embedder = BioBERTEmbedder()
        vectors = embedder.embed_texts(["protein folding", "drug resistance"])
        # vectors.shape = (2, 768)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self.batch_size = batch_size or settings.embedding_batch_size
        self._model = None  # Lazy load

    @property
    def model(self):
        """Lazy-load the model (first call downloads ~500MB)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(
                f"Model loaded. Embedding dimension: {self._model.get_sentence_embedding_dimension()}"
            )
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding vector dimension (768 for BERT-base models)."""
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """
        Embed a list of texts into vectors.

        Args:
            texts: List of text strings to embed
            show_progress: Show tqdm progress bar

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        logger.info(f"Embedding {len(texts)} texts in batches of {self.batch_size}")

        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # Unit vectors → cosine sim = dot product
            convert_to_numpy=True,
        )

        logger.info(f"Embedding complete: shape={vectors.shape}")
        return vectors

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text (for query-time use)."""
        return self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]


def embed_chunks_from_jsonl(
    chunks_path: Path,
    output_dir: Optional[Path] = None,
    batch_size: int = None,
) -> tuple[Path, Path]:
    """
    Read chunks JSONL, embed all texts, save vectors + metadata.

    Saves two files:
        - embeddings.npy:   numpy array of shape (N, 768)
        - chunk_metadata.jsonl: metadata for each vector (same row order)

    These two files are loaded by both Chroma and FAISS in Step 4.

    Args:
        chunks_path: Path to chunks.jsonl from the chunker
        output_dir: Where to save embeddings (default: data/embeddings/)
        batch_size: Override embedding batch size

    Returns:
        Tuple of (embeddings_path, metadata_path)
    """
    output_dir = output_dir or settings.embeddings_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_dir / "embeddings.npy"
    metadata_path = output_dir / "chunk_metadata.jsonl"

    # ---- 1. Load all chunks ----
    logger.info(f"Loading chunks from {chunks_path}")
    texts = []
    metadata_records = []

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line.strip())
            texts.append(chunk["text"])
            metadata_records.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"],
            })

    logger.info(f"Loaded {len(texts)} chunks")

    if not texts:
        logger.warning("No chunks to embed!")
        return embeddings_path, metadata_path

    # ---- 2. Embed in batches ----
    embedder = BioBERTEmbedder(batch_size=batch_size)

    # For very large corpora, process in mega-batches to manage RAM
    MEGA_BATCH = 10_000
    all_embeddings = []

    for start in range(0, len(texts), MEGA_BATCH):
        end = min(start + MEGA_BATCH, len(texts))
        logger.info(f"Embedding mega-batch {start}–{end} of {len(texts)}")
        batch_vectors = embedder.embed_texts(texts[start:end])
        all_embeddings.append(batch_vectors)

    embeddings = np.vstack(all_embeddings)
    logger.info(f"Final embeddings shape: {embeddings.shape}")

    # ---- 3. Save to disk ----
    np.save(embeddings_path, embeddings)
    logger.info(f"Saved embeddings → {embeddings_path} ({embeddings.nbytes / 1e6:.1f} MB)")

    with open(metadata_path, "w", encoding="utf-8") as f:
        for record in metadata_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"Saved metadata → {metadata_path}")

    return embeddings_path, metadata_path
