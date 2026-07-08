import pytest
from app.chunking import chunk_text

def test_chunk_text_basic():
    text = "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5. Sentence 6."
    chunks = chunk_text(text, chunk_size=3, overlap=1)

    # Expected:
    # Chunk 1: sentences 0, 1, 2
    # Chunk 2: sentences 2, 3, 4
    # Chunk 3: sentences 4, 5

    assert len(chunks) >= 3
    assert "Sentence 1." in chunks[0]
    assert "Sentence 3." in chunks[0]
    assert "Sentence 3." in chunks[1]
    assert "Sentence 5." in chunks[1]
    assert "Sentence 5." in chunks[2]
    assert "Sentence 6." in chunks[2]

def test_chunk_text_no_overlap():
    text = "Sentence 1. Sentence 2. Sentence 3. Sentence 4."
    chunks = chunk_text(text, chunk_size=2, overlap=0)

    assert len(chunks) == 2
    assert "Sentence 2." in chunks[0]
    assert "Sentence 3." in chunks[1]
    assert "Sentence 2." not in chunks[1]

def test_chunk_text_empty():
    assert chunk_text("") == []
