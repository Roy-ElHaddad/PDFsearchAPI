# PDF Search API

A small backend that lets you semantically search a local collection of PDF
documents (built against real French municipal/public documents — council
deliberations, agendas, agreements). It has two parts:

1. **Ingestion CLI** (`app/ingest.py`) — reads PDFs from a folder, extracts
   text, chunks it, embeds the chunks locally on CPU, and persists a FAISS
   index + chunk metadata to disk.
2. **API server** (`app/api/main.py`) — a FastAPI app that loads that index
   and exposes `POST /search` for similarity search over the chunks.

Everything runs locally: no paid APIs, no managed cloud services. The only
network access is the one-time download of the embedding model's weights
from Hugging Face on first run.

## Contents

- [Quickstart (Docker + Makefile)](#quickstart-docker--makefile)
- [API reference](#api-reference)
- [Running without Docker](#running-without-docker)
- [Configuration](#configuration)
- [Running tests](#running-tests)
- [Design decisions & trade-offs](#design-decisions--trade-offs)
- [Solution review](#solution-review)

## Quickstart (Docker + Makefile)

Requires Docker (and Docker Compose, bundled with modern Docker Desktop /
`docker-compose-plugin` — used by `make up`/`down`/`logs`). Commands assume
you're in the repo root. Run `make help` at any point for the full target
list.

#### 1. Place PDFs locally

Download the PDFs from the shared folder anywhere on your machine. The
default is `data/raw_pdfs/`, but this is not a hardcoded requirement — see
below.

```bash
mkdir -p data/raw_pdfs
cp /path/to/downloaded/*.pdf data/raw_pdfs/
```

#### 2. Build the image

```bash
make build
```

#### 3. Run ingestion

```bash
make ingest
```

This builds the image if needed, then runs `python -m app.ingest <folder>`
inside a container, mounting `data/raw_pdfs` (read-only) and `data/index`
(read-write) so the CLI's own folder-path argument reads your PDFs from
the host and the resulting index is written back to the host too.

**PDFs somewhere other than `data/raw_pdfs`?** Point at any local folder
directly — no need to copy files into the repo first:

```bash
make ingest PDF_DIR=/Users/you/Downloads/datapolitics_pdfs
```

#### 4. Verify the index was created

```bash
ls data/index
# index.faiss  metadata.json
```

#### 5. Start the API server

```bash
make up
```

Runs detached (`docker compose up -d`), so the command returns immediately;
`make logs` tails it from there. Port 8000 already taken? `make up
API_PORT=9000` binds a different host port without touching the container.

#### 6. Call `/search`

```bash
curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Quelle est la position du document sur les politiques publiques ?", "top_k": 5}'
```

Or open `http://localhost:8000/docs` for interactive Swagger UI.

#### 7. Rebuild the index if the PDF folder changes

Ingestion always does a full rebuild from whatever `PDF_DIR` currently
contains — add/remove/replace files there and re-run step 3:

```bash
make ingest
```

The running API process won't pick up a rebuilt index automatically (it's
loaded once into memory at startup); restart it:

```bash
make restart
```

`make down` stops the server; `make logs` tails it while it's running;
`make clean` deletes the built index if you want to confirm ingestion is
required from a clean state. Every target is a thin wrapper around a single
`docker build` / `docker run` / `docker compose` call — see `Makefile` for
the exact underlying commands if you'd rather run them by hand.

## API reference

### `POST /search`

Request:

```json
{
  "query": "Quelle est la position du document sur les politiques publiques ?",
  "top_k": 5
}
```

`top_k` is optional (defaults to 5, capped at 50).

Response:

```json
{
  "query": "Quelle est la position du document sur les politiques publiques ?",
  "results": [
    {
      "document_name": "example.pdf",
      "page_number": 3,
      "chunk_index": 12,
      "score": 0.82,
      "text": "Contenu du passage correspondant..."
    }
  ]
}
```

`score` is cosine similarity in `[-1, 1]` (in practice close to `[0, 1]` for
this kind of text), higher is more relevant.

Returns `503` if the index hasn't been built yet (ingestion not run).

### `GET /health`

Not required by the brief, but cheap and useful for confirming the server
actually loaded an index before you start querying it:

```json
{"status": "ok", "chunk_count": 542, "detail": null}
```

## Running without Docker

Requires Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m app.ingest data/raw_pdfs
uvicorn app.api.main:app --reload
```

## Configuration

Everything is a `Settings` field in `app/config.py`, overridable via
environment variable (or a `.env` file) of the same name:

| Variable | Default | Purpose |
|---|---|---|
| `INDEX_DIR` | `data/index` | Where the CLI writes / the API reads `index.faiss` + `metadata.json`. |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Local CPU embedding model, shared by ingestion and the API. |
| `CHUNK_SIZE_WORDS` | `50` | Target chunk size, in words - sized to the embedding model's 128-token limit, see below. |
| `CHUNK_OVERLAP_WORDS` | `10` | Overlap between consecutive chunks, in words. |
| `DEFAULT_TOP_K` | `5` | Default `top_k` when omitted from a search request. |
| `MAX_TOP_K` | `50` | Upper bound on `top_k`, enforced by request validation. |

Chunk size/overlap can also be overridden per ingestion run via CLI flags
(going much above the 50-word default reintroduces the embedding-model
truncation described above — see the chunking bullet in the next section):

```bash
python -m app.ingest data/raw_pdfs --chunk-size 80 --chunk-overlap 15
```

The API refuses to load an index that was built with a different embedding
model than it's currently configured for (checked via `metadata.json`),
rather than silently serving nonsense similarity scores.

## Running tests

```bash
pip install -r requirements-dev.txt
make test   # or: python -m pytest
```

Tests cover chunking logic, the FAISS store's save/load/search round-trip,
and the API's request handling — all with the embedding model mocked out,
so the suite runs in a few seconds without downloading any weights.

## Design decisions & trade-offs

- **PDF extraction — `pypdf`.** Pure-Python, no compiled dependencies (no
  extra apt packages, smaller/simpler Docker image than e.g. PyMuPDF), and
  MIT-licensed. Trade-off: weaker at complex layouts and multi-column text
  than layout-aware alternatives like `pdfplumber` or PyMuPDF. Given the
  brief explicitly deprioritizes extraction perfection, this felt like the
  right place to spend the least effort.

- **Chunking — fixed word-count sliding window, per page, sized to the
  embedding model's context window.** Chunking is done independently per
  page (not across the whole document) so that `page_number` on every
  chunk is unambiguous — a chunk spanning two pages would otherwise have
  to be assigned to one page arbitrarily. The chunk size itself (50 words,
  10 overlap) isn't an arbitrary round number: `paraphrase-multilingual-
  MiniLM-L12-v2` has a hard `max_seq_length` of 128 tokens, and
  `SentenceTransformer.encode()` truncates anything longer *silently* —
  no error, no warning at inference time, just a chunk embedded from a
  truncated prefix. Measured with the model's own tokenizer against the
  real supplied corpus: the original 200-word default put 92% of chunks
  over that limit (mean 281 tokens — more than double budget); 50 words
  gets 99.6% fully within it (mean ~81 tokens). Word count is still only a
  proxy for token count (French runs ~1.7 tokens/word with this
  tokenizer, with a long tail from numbers and legal citations), so a
  residual <1% of chunks still overflow slightly — going smaller to chase
  literal zero would shrink chunks past the point of being a "meaningful
  chunk" a person can read as a passage, which is the actual requirement.
  Overlap stays at the same ~20% proportion as before, so a sentence that
  lands on a chunk boundary is still fully readable in at least one chunk.

- **`chunk_index` is a running counter per document** (not reset per page),
  so results from the same document sort into a meaningful reading order
  even across page boundaries.

- **Embeddings — `paraphrase-multilingual-MiniLM-L12-v2`.** Matches the
  brief's suggestion: multilingual (works on French without a
  French-specific model), small enough (~120M params) to embed on CPU in
  seconds for a corpus this size, no GPU required.

- **Vector store — FAISS `IndexFlatIP` over normalized vectors** (i.e.
  cosine similarity via brute-force inner product), not an approximate
  index like HNSW/IVF. At the scale this exercise targets (hundreds to low
  thousands of chunks — the supplied corpus produced 542), brute-force
  search is sub-millisecond and exact, and needs no training step or
  recall/speed tuning. FAISS has no notion of documents/pages/scores
  itself, so chunk metadata is kept as a parallel Python list persisted
  alongside the index (`metadata.json`), aligned by position.

- **Metadata format — plain JSON, not a database.** For a few hundred
  chunks this is simpler to inspect, version, and reason about than
  standing up SQLite/Postgres, and it's exactly what "persist the index
  and metadata to disk" calls for. It would stop being the right choice
  well before the vector count did (see [Production readiness](#production-readiness)).

- **API — one endpoint plus `/health`.** `POST /search` matches the brief's
  schema exactly. `/health` was added because loading a FAISS index at
  startup is the one thing that can silently fail (index not built yet,
  model mismatch), and a missing-index server should stay up and report
  `503` with an actionable message on `/search` rather than crash-loop —
  easier to debug than a container that won't start.

- **Docker image — CPU-only PyTorch installed explicitly.** `torch` is a
  transitive dependency of `sentence-transformers`; installing it straight
  from PyPI resolved a CUDA-enabled build even on this non-GPU, non-x86_64
  image, pulling in several GB of NVIDIA driver packages that would never
  be used (image ballooned to ~9.6GB in initial testing). Installing the
  official CPU wheel first (`--index-url https://download.pytorch.org/whl/cpu`)
  before the rest of `requirements.txt` brought that down to ~1.9GB — worth
  calling out since it's the kind of thing that's easy to miss until you
  actually build the image.

- **Makefile as the single entry point, Compose only for the long-running
  service.** `make ingest` runs a plain `docker run` rather than a Compose
  service, specifically so the PDF folder stays a real argument
  (`PDF_DIR=...`) instead of a path baked into a static Compose volume
  mount — the brief's "the local folder path must be provided as a
  command-line argument" requirement is honored end-to-end, not just at
  the `app.ingest` layer. `docker compose` is used only for `api`, since
  that's the one genuinely long-running service where Compose's declarative
  `up`/`down`/`logs` model is a real fit. Both paths share a named
  `pdfsearch_model_cache` volume so the ~470MB of model weights are only
  ever downloaded once, regardless of which target triggers it first.

## Solution review

### Main limitations

- No OCR: PDFs with no embedded text layer (scanned documents/images)
  extract zero text and are silently unsearchable. This isn't hypothetical
  — one of the seven PDFs supplied for this exercise
  (`AFF-2026.06.11-DP-26A0019-19-PLACE-DES-HAUTS-TAILLIS-ACCORD.pdf`) is
  exactly this case: `pypdf` returns empty text for both its pages.
- No cleanup of repeated headers/footers, page numbers, or other
  boilerplate — they end up embedded like any other text and can dilute or
  duplicate results.
- Chunking is naive fixed-word-count, not sentence- or layout-aware; it can
  split mid-sentence or mid-table-row.
- Pure dense-vector search, no lexical/keyword component — no BM25 or
  exact-match fallback.
- Every ingestion run does a full rebuild of the index; there's no
  incremental update, document diffing, or delete-by-filename.
- No auth, rate limiting, or structured application logging on the API
  (uvicorn's default access log is the only thing recording requests) —
  it's built for local evaluation, not for being exposed anywhere.

### Situations where search quality may be poor

- **Scanned/image-only PDFs** (see above) — zero recall, not just poor
  recall, since there's no extracted text to embed at all.
- **Table-heavy documents** (e.g. the subsidy table and deliberation table
  in this corpus) — `pypdf` linearizes rows/columns into a single text
  stream, so the resulting chunks can be structurally garbled even though
  the underlying content is intact.
- **Exact-match queries**: reference numbers, dates, SIRET/legal
  identifiers, proper nouns. Dense embedding similarity is comparatively
  weak at this compared to keyword/lexical search — a hybrid approach
  would help here specifically.
- **Small corpora relative to `top_k`**: with 542 chunks total in the
  supplied corpus, asking for `top_k=50` (the API's max) will return
  low-similarity "least-bad" matches well past the point of actual
  relevance for a narrow query, since the index doesn't discard
  nothing-like-this results — it just returns them.
- **Domain-specific French administrative jargon/acronyms** not well
  represented in the multilingual model's training data, versus a
  French-specific or larger model.

### Assumptions

- PDFs are downloaded locally by the user before ingestion runs, per the
  brief — the ingestion program never touches Google Drive or any network
  source for the documents themselves.
- The input folder is flat (no recursive subfolder scanning) — "read all
  PDF files from the input folder" is read literally.
- "Meaningful chunks" is interpreted as fixed-size word windows with
  overlap rather than semantic/sentence-aware segmentation, in line with
  the brief's explicit steer not to over-invest in extraction/chunking
  perfection.
- Re-running ingestion fully replaces the index rather than merging into
  it — matches "rebuild the index if the PDF folder changes" as a
  from-scratch operation, not an incremental one.
- `score` in the API response is raw cosine similarity, not a calibrated
  confidence/probability — documented as such rather than implying more
  precision than it has.
- The document corpus is small (a handful of PDFs, as stated in the brief)
  — this justifies an in-memory flat FAISS index and a single-process API
  instead of anything built for horizontal scale.

### What I'd improve with more time

- **Extraction**: add a layout-aware extractor (`pdfplumber`/PyMuPDF) and
  an OCR fallback (e.g. Tesseract) for pages with no text layer — directly
  motivated by the one scanned PDF in the actual supplied corpus being
  completely unsearchable right now.
- **Chunking**: sentence/paragraph-aware splitting instead of raw word
  windows, and detection + suppression of repeated headers/footers across
  a document's pages.
- **Hybrid retrieval**: combine the FAISS dense search with a lexical
  index (e.g. `rank_bm25`) and merge/re-rank, specifically to cover the
  exact-match weakness noted above.
- **Incremental ingestion**: hash each PDF's content and skip/update only
  what changed, instead of a full rebuild every run.
- **A filesystem watcher for the PDF folder** (e.g. via the `watchdog`
  library) that detects new/changed/removed files and triggers ingestion
  automatically, instead of requiring a manual `make ingest` after every
  change. This only pays off once incremental ingestion (above) exists —
  wiring a watcher to the *current* full-rebuild ingestion would mean every
  dropped file re-embeds the entire corpus, which is worse than doing it
  manually when you know the corpus changed. It would also need a debounce
  window before triggering, since a bulk copy fires one filesystem event
  per file and a naive watcher could start ingesting a PDF mid-write.
  Deliberately not built for this exercise — the brief frames ingestion as
  an operator-invoked CLI step ("run the ingestion command", "rebuild the
  index if the PDF folder changes") — but it's a natural next step once
  ingestion stops being a one-off task.
- **A small evaluation set**: a handful of hand-labeled query → relevant
  chunk pairs, so changes to chunk size/overlap/model can be judged by a
  number instead of by eye.
- **Better observability**: structured logs, and an endpoint to list what's
  currently indexed (`GET /documents`) for debugging what did or didn't
  make it into the index.

### Production readiness

- Replace the in-process flat FAISS index with a dedicated vector database
  (Qdrant, pgvector, Milvus, ...) once corpus size, query volume, or the
  need for incremental upserts/durability outgrows a single process with
  an in-memory index rebuilt from scratch each time.
- Move ingestion from a manually-invoked CLI to an event-driven
  pipeline/worker, with retries and per-document ingestion status tracking.
  Concretely: source PDFs directly from object storage (S3/GCS/Azure Blob)
  instead of a pre-downloaded local folder, triggered by bucket event
  notifications rather than a full-bucket poll (same reasoning as the
  filesystem-watcher point above — an event per new/changed object, not a
  rescan-everything trigger), so a new document becomes searchable without
  anyone manually downloading a folder and re-running a CLI. This is the
  natural evolution of that same idea once the corpus is something
  analysts continuously add to rather than a fixed handful of files handed
  over once. Deliberately not built here — the brief is explicit that
  ingestion runs against PDFs "after they have been downloaded locally"
  with the folder path as a CLI argument, so this would be solving a
  problem outside this exercise's scope, not a missed requirement.
- Add authentication/authorization, rate limiting, structured logging,
  metrics and tracing to the API.
- Turn extraction-quality issues (scanned pages, empty documents, garbled
  tables) into a monitored pipeline stage with per-document quality
  signals, instead of the documents just silently contributing fewer/no
  chunks.
- CI: automated tests + linting on every change; treat the index as a
  versioned artifact (a manifest of source PDF hashes + embedding model
  version that produced it) for reproducibility and rollback.
- Optionally add an LLM-based answer-synthesis layer on top of retrieved
  chunks with citations back to source passages — explicitly out of scope
  for this exercise, but the natural next step for the "research tool for
  analysts" framing in the brief. To keep it consistent with the rest of
  the solution (local, no paid APIs), this would run through a self-hosted
  model via Ollama rather than a hosted LLM API.
