import numpy as np
from fastapi.testclient import TestClient

from app.api import main


def test_search_returns_results_for_loaded_index(monkeypatch, tmp_path, store):
    index_dir = tmp_path / "index"
    store.save(index_dir)
    monkeypatch.setattr(main.settings, "index_dir", index_dir)

    # Skip the real (slow, network-dependent) embedding model in tests:
    # a no-op loader, and a query embedder that returns a fixed vector
    # matching the "rapport_b.pdf" sample chunk's direction.
    monkeypatch.setattr(main, "get_embedder", lambda: None)
    monkeypatch.setattr(main, "embed_query", lambda q: np.array([0.0, 0.0, 1.0, 0.0], dtype="float32"))

    with TestClient(main.app) as client:
        response = client.post("/search", json={"query": "seance parlementaire", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "seance parlementaire"
    assert len(body["results"]) == 2
    top = body["results"][0]
    assert top["document_name"] == "rapport_b.pdf"
    assert top["page_number"] == 1
    assert top["chunk_index"] == 0
    assert "score" in top and "text" in top


def test_search_returns_503_when_index_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(main.settings, "index_dir", tmp_path / "no_such_index")
    monkeypatch.setattr(main, "get_embedder", lambda: None)

    with TestClient(main.app) as client:
        response = client.post("/search", json={"query": "anything"})

    assert response.status_code == 503


def test_search_rejects_empty_query(monkeypatch, tmp_path, store):
    index_dir = tmp_path / "index"
    store.save(index_dir)
    monkeypatch.setattr(main.settings, "index_dir", index_dir)
    monkeypatch.setattr(main, "get_embedder", lambda: None)

    with TestClient(main.app) as client:
        response = client.post("/search", json={"query": ""})

    assert response.status_code == 422


def test_health_reports_chunk_count(monkeypatch, tmp_path, store):
    index_dir = tmp_path / "index"
    store.save(index_dir)
    monkeypatch.setattr(main.settings, "index_dir", index_dir)
    monkeypatch.setattr(main, "get_embedder", lambda: None)

    with TestClient(main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "chunk_count": 3, "detail": None}
