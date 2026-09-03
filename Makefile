.PHONY: help build ingest up down restart logs test clean

IMAGE      ?= pdfsearch-api:latest
# Folder of PDFs to ingest. Override per-run: make ingest PDF_DIR=/path/to/other/pdfs
# Defaults to data/raw_pdfs to match the Quickstart in the README.
PDF_DIR    ?= data/raw_pdfs
INDEX_DIR  ?= data/index
API_PORT   ?= 8000

help:
	@echo "make build              build the Docker image"
	@echo "make ingest             ingest PDFs from PDF_DIR (default: data/raw_pdfs) into INDEX_DIR"
	@echo "make ingest PDF_DIR=... ingest PDFs from any local folder, not just data/raw_pdfs"
	@echo "make up                 start the API server on API_PORT (default: 8000)"
	@echo "make down               stop the API server"
	@echo "make restart            rebuild the index picked up by a running API (down + up)"
	@echo "make logs               tail the API server logs"
	@echo "make test               run the test suite locally (requires requirements-dev.txt)"
	@echo "make clean              delete the built index (forces a fresh ingest)"

build:
	docker build -t $(IMAGE) .

# PDF_DIR is mounted read-only into its own container path, independent of
# where INDEX_DIR lives, so this works whether the PDFs are already inside
# ./data/raw_pdfs or sitting anywhere else on your machine — the brief's
# "folder path is a CLI argument" requirement, honored end-to-end even
# when running through Docker rather than only in the bare `app.ingest`
# entrypoint.
ingest: build
	mkdir -p "$(INDEX_DIR)"
	docker run --rm \
		-v "$(abspath $(PDF_DIR)):/pdfs:ro" \
		-v "$(abspath $(INDEX_DIR)):/app/data/index" \
		-v pdfsearch_model_cache:/root/.cache \
		$(IMAGE) python -m app.ingest /pdfs --output-dir data/index

up: build
	docker compose up api

down:
	docker compose down

restart: down up

logs:
	docker compose logs -f api

# Runs against the local venv, not Docker: the test suite mocks out the
# embedding model, so it's fast and doesn't need a container — installing
# requirements-dev.txt into the runtime image would only bloat it for no
# benefit at request-serving time.
test:
	python -m pytest

clean:
	rm -f $(INDEX_DIR)/index.faiss $(INDEX_DIR)/metadata.json
