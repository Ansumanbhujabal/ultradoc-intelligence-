"""Tests for section-based chunker."""
from app.pipeline.chunker import chunk_document

def test_chunk_basic_sections():
    text = """Carrier Details
Carrier: SWIFT SHIFT LOGISTICS LLC
MC: 1685682

Stops
Pickup: Los Angeles
Drop: Fontana

Rate Breakdown
Total: $400.00 USD"""
    chunks = chunk_document(text)
    assert len(chunks) >= 3
    assert all("text" in c and "section" in c and "index" in c for c in chunks)

def test_chunk_preserves_tables():
    text = """Rate Breakdown
Carrier Pay | Total
Flatbed:$ 400.00 USD | 400.00 USD"""
    chunks = chunk_document(text)
    rate_chunk = [c for c in chunks if "400" in c["text"]]
    assert len(rate_chunk) >= 1
    assert "Flatbed" in rate_chunk[0]["text"]

def test_chunk_single_section():
    text = "Simple document with no clear sections or headers."
    chunks = chunk_document(text)
    assert len(chunks) >= 1
    assert chunks[0]["section"] == "content"

def test_chunk_empty():
    chunks = chunk_document("")
    assert len(chunks) == 0
