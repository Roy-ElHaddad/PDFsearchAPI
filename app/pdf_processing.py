"""PDF text extraction and chunking.

Design notes (see README for the full write-up):

- Extraction is per-page via pypdf. Real-world PDFs commonly break words
  across lines, repeat running headers/footers on every page, and sometimes
  extract table layouts as jumbled text. We deliberately do *not* try to
  fix this: the brief explicitly says not to over-invest in cleaning, and a
  chunk-level embedding is fairly tolerant of local noise. We only do the
  cheap, safe normalization (collapsing whitespace) that has no risk of
  destroying real content.

- Chunking is word-count based with overlap, computed independently per
  page. That keeps "page_number" on each chunk unambiguous and correct —
  the alternative (chunking across the whole document and letting a chunk
  span two pages) would force us to either duplicate the chunk under two
  page numbers or pick one arbitrarily. Word count is a simple, model-
  agnostic proxy for token count; it's not exact for the embedding model's
  actual tokenizer, but it's dependency-free and close enough for chunks
  this small.
"""

import re
from pathlib import Path

from pypdf import PdfReader

from app.models import Chunk

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def extract_pages(pdf_path: Path) -> list[str]:
    """Return one normalized text string per page, in page order.

    A page that fails to extract (corrupt content stream, scanned image
    with no text layer, etc.) yields an empty string rather than raising,
    so one bad page doesn't take down the whole document.
    """
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""
        pages.append(_normalize_whitespace(raw_text))
    return pages


def chunk_page_text(
    text: str,
    chunk_size_words: int,
    overlap_words: int,
) -> list[str]:
    """Split one page's text into overlapping, word-count-bounded chunks.

    Overlap exists so that a sentence spanning a chunk boundary is still
    fully present in at least one chunk, at the cost of a bit of index
    redundancy.
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size_words:
        return [text]

    step = max(chunk_size_words - overlap_words, 1)
    chunks = []
    start = 0
    while start < len(words):
        window = words[start : start + chunk_size_words]
        chunks.append(" ".join(window))
        if start + chunk_size_words >= len(words):
            break
        start += step
    return chunks


def process_pdf(
    pdf_path: Path,
    chunk_size_words: int,
    overlap_words: int,
) -> list[Chunk]:
    """Extract + chunk a single PDF into Chunk records.

    chunk_index is a running counter across the whole document (not reset
    per page), so it doubles as a stable position/order indicator when
    displaying results from the same document.
    """
    document_name = pdf_path.name
    chunks: list[Chunk] = []
    running_index = 0

    for page_number, page_text in enumerate(extract_pages(pdf_path), start=1):
        for piece in chunk_page_text(page_text, chunk_size_words, overlap_words):
            if not piece.strip():
                continue
            chunks.append(
                Chunk(
                    document_name=document_name,
                    page_number=page_number,
                    chunk_index=running_index,
                    text=piece,
                )
            )
            running_index += 1

    return chunks
