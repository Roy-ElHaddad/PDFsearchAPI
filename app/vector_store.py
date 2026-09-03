"""FAISS index + chunk metadata persistence.

FAISS only stores vectors and hands back integer positions on search — it
has no concept of documents, pages or text. So we keep a parallel Python
list of Chunk metadata whose ordering matches the vectors added to the
index (position i in the index <-> chunks[i]), and persist both together
as the on-disk contract between the ingestion CLI and the API server:

    <index_dir>/index.faiss     - the FAISS index
    <index_dir>/metadata.json   - chunk metadata + the embedding model name

For the handful-of-PDFs scale this exercise targets (thousands of chunks,
not millions), an exact IndexFlatIP is the right choice over an
approximate index (HNSW, IVF, ...): it needs no training step, has no
recall/speed tuning to get wrong, and brute-force search over a few
thousand vectors is still sub-millisecond. Vectors are expected to already
be L2-normalized (see app/embeddings.py), which turns inner product into
cosine similarity.
"""

import json
from pathlib import Path

import faiss
import numpy as np

from app.config import settings
from app.models import Chunk

INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.json"


class VectorStore:
    def __init__(self, index: faiss.Index, chunks: list[Chunk]):
        self.index = index
        self.chunks = chunks

    def __len__(self) -> int:
        return len(self.chunks)

    @classmethod
    def build(cls, chunks: list[Chunk], vectors: np.ndarray) -> "VectorStore":
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"chunk/vector count mismatch: {len(chunks)} chunks vs {vectors.shape[0]} vectors"
            )
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        return cls(index, chunks)

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / INDEX_FILENAME))

        payload = {
            "embedding_model": settings.embedding_model_name,
            "chunk_count": len(self.chunks),
            "chunks": [c.to_dict() for c in self.chunks],
        }
        with open(index_dir / METADATA_FILENAME, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        index_path = index_dir / INDEX_FILENAME
        metadata_path = index_dir / METADATA_FILENAME
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"No index found at {index_dir}. Run the ingestion CLI first: "
                f"python -m app.ingest <pdf_folder>"
            )

        index = faiss.read_index(str(index_path))
        with open(metadata_path, encoding="utf-8") as f:
            payload = json.load(f)

        stored_model = payload.get("embedding_model")
        if stored_model and stored_model != settings.embedding_model_name:
            raise RuntimeError(
                f"Index was built with embedding model '{stored_model}' but the API is "
                f"configured for '{settings.embedding_model_name}'. Rebuild the index or "
                f"align EMBEDDING_MODEL_NAME."
            )

        chunks = [Chunk.from_dict(d) for d in payload["chunks"]]
        return cls(index, chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        if len(self.chunks) == 0:
            return []
        top_k = min(top_k, len(self.chunks))
        query = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results
