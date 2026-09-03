"""Embedding model wrapper.

A thin wrapper around sentence-transformers so the ingestion CLI and the
API load the model the exact same way (same normalization, same model
name from config) — divergence here would silently break similarity
scores between index-build time and query time.
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    # Cached process-wide: loading the model is the slow part (reads
    # weights from disk / downloads them on first run), so both the
    # ingestion CLI (one-shot) and the API server (long-lived) only pay
    # that cost once per process.
    return SentenceTransformer(settings.embedding_model_name)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts as L2-normalized float32 vectors.

    Normalizing at embedding time lets us use a plain inner-product FAISS
    index and get cosine similarity for free, instead of maintaining a
    separate normalization step at search time.
    """
    model = get_embedder()
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vectors.astype("float32")


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]
