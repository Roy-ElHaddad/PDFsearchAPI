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
    chunk_size_words: int = 200
    chunk_overlap_words: int = 40

    default_top_k: int = 5
    max_top_k: int = 50

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
