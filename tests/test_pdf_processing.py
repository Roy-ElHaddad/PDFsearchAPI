from app.pdf_processing import _normalize_whitespace, chunk_page_text


def test_normalize_whitespace_collapses_newlines_and_spaces():
    raw = "Bonjour   le\nmonde\n\n  la\t Republique"
    assert _normalize_whitespace(raw) == "Bonjour le monde la Republique"


def test_chunk_page_text_short_text_returns_single_chunk():
    text = "Une phrase courte."
    chunks = chunk_page_text(text, chunk_size_words=200, overlap_words=40)
    assert chunks == [text]


def test_chunk_page_text_empty_text_returns_no_chunks():
    assert chunk_page_text("", chunk_size_words=200, overlap_words=40) == []
    assert chunk_page_text("   ", chunk_size_words=200, overlap_words=40) == []


def test_chunk_page_text_splits_long_text_with_overlap():
    words = [f"mot{i}" for i in range(100)]
    text = " ".join(words)

    chunks = chunk_page_text(text, chunk_size_words=30, overlap_words=10)

    assert len(chunks) > 1
    # Every chunk stays within the requested size.
    assert all(len(c.split()) <= 30 for c in chunks)
    # Consecutive chunks overlap: the tail of one reappears at the head of the next.
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    assert first_words[-10:] == second_words[:10]
    # No word from the source text is lost.
    covered = set(" ".join(chunks).split())
    assert covered == set(words)


def test_chunk_page_text_zero_overlap_step_still_terminates():
    text = " ".join(f"w{i}" for i in range(50))
    chunks = chunk_page_text(text, chunk_size_words=10, overlap_words=10)
    # overlap == chunk_size would make the step 0; we clamp step to >=1
    # so this must still terminate instead of looping forever.
    assert len(chunks) > 0
