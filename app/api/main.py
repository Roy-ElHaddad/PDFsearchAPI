"""FastAPI application serving search results from the index built by the
ingestion CLI (app/ingest.py).

The index is loaded once at startup, not per-request — with a
few-thousand-chunk FAISS index that's a few tens of MB at most, so keeping
it resident in memory for the process lifetime is the obvious choice over
re-reading it from disk on every /search call.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.api.schemas import SearchRequest, SearchResponse, SearchResult
from app.config import settings
from app.embeddings import embed_query, get_embedder
from app.vector_store import VectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the embedding model eagerly (rather than on first request) so
    # that the first real search isn't the one paying a multi-second
    # model-load penalty.
    get_embedder()

    try:
        app.state.store = VectorStore.load(settings.index_dir)
        app.state.store_error = None
    except (FileNotFoundError, RuntimeError) as exc:
        # Don't crash the server if ingestion hasn't run yet / index is
        # stale — surface it through /health and /search instead, so the
        # container stays up and the error is actionable from the API
        # itself rather than only in logs.
        app.state.store = None
        app.state.store_error = str(exc)
        print(f"[startup] index not loaded: {exc}")

    yield


app = FastAPI(
    title="PDF Search API",
    description="Similarity search over locally ingested PDF documents.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    store = app.state.store
    return {
        "status": "ok" if store is not None else "index_missing",
        "chunk_count": len(store) if store is not None else 0,
        "detail": app.state.store_error,
    }


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    store = app.state.store
    if store is None:
        raise HTTPException(
            status_code=503,
            detail=app.state.store_error
            or "Index not loaded. Run the ingestion CLI first: python -m app.ingest <pdf_folder>",
        )

    query_vector = embed_query(request.query)
    hits = store.search(query_vector, request.top_k)

    results = [
        SearchResult(
            document_name=chunk.document_name,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            score=score,
            text=chunk.text,
        )
        for chunk, score in hits
    ]
    return SearchResponse(query=request.query, results=results)
