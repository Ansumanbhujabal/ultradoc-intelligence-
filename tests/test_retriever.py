"""Tests for hybrid retriever and RRF fusion."""
from app.pipeline.retriever import rrf_fuse, bm25_search, metadata_filter

def test_rrf_fuse_basic():
    list_a = ["chunk_1", "chunk_2", "chunk_3"]
    list_b = ["chunk_2", "chunk_1", "chunk_4"]
    result = rrf_fuse([list_a, list_b], k=60)
    assert "chunk_1" in result[:2]
    assert "chunk_2" in result[:2]

def test_rrf_fuse_single_list():
    result = rrf_fuse([["a", "b", "c"]], k=60)
    assert result == ["a", "b", "c"]

def test_rrf_fuse_empty():
    result = rrf_fuse([], k=60)
    assert result == []

def test_bm25_search():
    chunks = [
        {"text": "Carrier SWIFT SHIFT LOGISTICS LLC", "section": "carrier", "index": 0},
        {"text": "Pickup from Los Angeles Airport", "section": "stops", "index": 1},
        {"text": "Rate Breakdown Total 400 USD", "section": "rate", "index": 2},
    ]
    results = bm25_search("carrier rate", chunks, top_k=2)
    assert len(results) <= 2
    texts = [r["text"] for r in results]
    assert any("Carrier" in t or "Rate" in t for t in texts)

def test_metadata_filter():
    chunks = [
        {"text": "SWIFT SHIFT", "section": "carrier_details", "index": 0},
        {"text": "Los Angeles", "section": "stops", "index": 1},
        {"text": "400 USD", "section": "rate_breakdown", "index": 2},
    ]
    result = metadata_filter("What is the rate?", chunks)
    assert len(result) >= 1
    assert any(c["section"] == "rate_breakdown" for c in result)
