import json

import numpy as np
import pytest

from app.vector_store import VectorStore


def test_search_returns_closest_chunk_first(store, sample_chunks):
    # Query vector identical to the second sample vector's direction.
    query = np.array([0.0, 1.0, 0.0, 0.0], dtype="float32")

    results = store.search(query, top_k=2)

    assert len(results) == 2
    top_chunk, top_score = results[0]
    assert top_chunk.text == sample_chunks[1].text
    assert top_score > results[1][1]  # best match scores highest


def test_search_top_k_capped_at_available_chunks(store):
    results = store.search(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"), top_k=1000)
    assert len(results) == 3  # only 3 chunks were indexed


def test_save_and_load_round_trip(tmp_path, store, sample_chunks):
    index_dir = tmp_path / "index"
    store.save(index_dir)

    loaded = VectorStore.load(index_dir)

    assert len(loaded) == len(sample_chunks)
    assert [c.document_name for c in loaded.chunks] == [c.document_name for c in sample_chunks]
    assert [c.text for c in loaded.chunks] == [c.text for c in sample_chunks]

    query = np.array([0.0, 0.0, 1.0, 0.0], dtype="float32")
    results = loaded.search(query, top_k=1)
    assert results[0][0].text == sample_chunks[2].text


def test_load_missing_index_raises_file_not_found(tmp_path):
    try:
        VectorStore.load(tmp_path / "does_not_exist")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as exc:
        assert "ingest" in str(exc)


def test_load_rejects_index_metadata_count_mismatch(tmp_path, store):
    # Simulates a torn write (e.g. metadata.json rewritten but index.faiss
    # left stale) — position i in the FAISS index no longer lines up with
    # chunks[i], which load() must refuse rather than serve mismatched
    # results.
    index_dir = tmp_path / "index"
    store.save(index_dir)

    metadata_path = index_dir / "metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["chunks"].pop()
    payload["chunk_count"] = len(payload["chunks"])
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Corrupt index"):
        VectorStore.load(index_dir)
