"""Central configuration.

Everything here can be overridden with an environment variable of the same
name (case-insensitive), which is what lets the ingestion CLI and the API
server share defaults while still being independently configurable inside
Docker (e.g. INDEX_DIR=/data/index).
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Where the FAISS index + chunk metadata are persisted by the ingestion
    # program and later loaded by the API. This is the contract between the
    # two halves of the system.
    index_dir: Path = Path("data/index")

    # Multilingual so it works on the French source documents without a
    # dedicated French-only model; small enough to embed on CPU in
    # reasonable time.
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Word-count based chunking (see app/pdf_processing.py for why words
    # rather than characters or tokens).
    #
    # Sized against the embedding model's own limit, not chosen arbitrarily:
    # paraphrase-multilingual-MiniLM-L12-v2 has max_seq_length=128 tokens,
    # and SentenceTransformer.encode() silently truncates anything longer
    # rather than erroring, so an oversized chunk doesn't fail - it just
    # gets embedded from a truncated prefix without any signal that it
    # happened. Measured with the model's own tokenizer against the real
    # supplied corpus: at the previous default (200 words), 92% of chunks
    # exceeded 128 tokens (mean 281). At 50 words, 99.6% fit fully within
    # budget (mean ~81 tokens, comfortable headroom); going lower to chase
    # the last <1% would shrink chunks past the point of being a
    # "meaningful chunk" for a human reading the search result. Overlap
    # kept at the same ~20% proportion as before.
    chunk_size_words: int = 50
    chunk_overlap_words: int = 10

    default_top_k: int = 5
    max_top_k: int = 50

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
