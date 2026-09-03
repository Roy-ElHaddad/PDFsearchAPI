import numpy as np
import pytest

from app.models import Chunk
from app.vector_store import VectorStore


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(document_name="rapport_a.pdf", page_number=1, chunk_index=0, text="Le budget de l'Etat pour 2024."),
        Chunk(document_name="rapport_a.pdf", page_number=2, chunk_index=1, text="Les politiques publiques de sante."),
        Chunk(document_name="rapport_b.pdf", page_number=1, chunk_index=0, text="Compte rendu de la seance parlementaire."),
    ]


@pytest.fixture
def sample_vectors() -> np.ndarray:
    # Small orthogonal-ish vectors so nearest-neighbour search is
    # deterministic and easy to reason about in assertions, without
    # needing the real embedding model in the test suite.
    vectors = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype="float32",
    )
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


@pytest.fixture
def store(sample_chunks, sample_vectors) -> VectorStore:
    return VectorStore.build(sample_chunks, sample_vectors)
