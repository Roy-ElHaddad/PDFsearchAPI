"""Ingestion CLI: read PDFs from a folder, chunk them, embed the chunks and
persist a FAISS index + metadata to disk for the API to load.

Usage:
    python -m app.ingest /path/to/pdf_folder
    python -m app.ingest /path/to/pdf_folder --output-dir data/index --chunk-size 200 --chunk-overlap 40
"""

import argparse
import sys
import time
from pathlib import Path

from app.config import settings
from app.embeddings import embed_texts
from app.models import Chunk
from app.pdf_processing import process_pdf
from app.vector_store import VectorStore


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.ingest",
        description="Extract, chunk and embed PDFs into a local FAISS index.",
    )
    parser.add_argument(
        "pdf_folder",
        type=Path,
        help="Local folder containing the PDF files to ingest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.index_dir,
        help=f"Where to write index.faiss + metadata.json (default: {settings.index_dir}).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size_words,
        help=f"Chunk size in words (default: {settings.chunk_size_words}).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.chunk_overlap_words,
        help=f"Overlap between consecutive chunks, in words (default: {settings.chunk_overlap_words}).",
    )
    return parser.parse_args(argv)


def find_pdfs(pdf_folder: Path) -> list[Path]:
    if not pdf_folder.is_dir():
        raise NotADirectoryError(f"{pdf_folder} is not a directory")
    # Case-insensitive match without listing the directory twice.
    return sorted(p for p in pdf_folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")


def run(args: argparse.Namespace) -> None:
    pdf_paths = find_pdfs(args.pdf_folder)
    if not pdf_paths:
        print(f"No PDF files found in {args.pdf_folder}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pdf_paths)} PDF file(s) in {args.pdf_folder}")

    all_chunks: list[Chunk] = []
    for pdf_path in pdf_paths:
        t0 = time.time()
        try:
            doc_chunks = process_pdf(pdf_path, args.chunk_size, args.chunk_overlap)
        except Exception as exc:
            # One malformed PDF shouldn't abort ingestion of the rest.
            print(f"  [skip] {pdf_path.name}: failed to process ({exc})", file=sys.stderr)
            continue
        all_chunks.extend(doc_chunks)
        print(f"  {pdf_path.name}: {len(doc_chunks)} chunks ({time.time() - t0:.1f}s)")

    if not all_chunks:
        print("No text could be extracted from any PDF — nothing to index.", file=sys.stderr)
        sys.exit(1)

    print(f"Embedding {len(all_chunks)} chunks with '{settings.embedding_model_name}' ...")
    t0 = time.time()
    vectors = embed_texts([c.text for c in all_chunks])
    print(f"  done in {time.time() - t0:.1f}s")

    store = VectorStore.build(all_chunks, vectors)
    store.save(args.output_dir)

    print(
        f"Indexed {len(all_chunks)} chunks from {len(pdf_paths)} document(s) "
        f"-> {args.output_dir}/index.faiss + metadata.json"
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
