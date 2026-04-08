"""Hybrid retrieval: vector search + BM25 + metadata filter + RRF fusion."""

from collections import defaultdict
from rank_bm25 import BM25Okapi
from app.config import settings
from app.storage.vector_store import VectorStore
from app.storage.cache import DocumentCache
from app.observability.tracer import Tracer
from app.observability.logger import get_logger

logger = get_logger("retriever")

SECTION_KEYWORDS = {
    "rate": ["rate", "charge", "cost", "price", "pay", "total", "amount"],
    "carrier": ["carrier", "mc", "dot", "scac", "trucking", "logistics"],
    "stop": ["pickup", "delivery", "drop", "stop", "ship", "appointment"],
    "shipper": ["shipper", "sender", "origin", "from"],
    "consignee": ["consignee", "receiver", "destination", "to", "deliver"],
    "driver": ["driver", "truck", "trailer", "cell"],
    "instruction": ["instruction", "special", "standing", "note"],
    "commodity": ["commodity", "weight", "units", "quantity", "description"],
}


def retrieve(question: str, doc_id: str, tracer: Tracer, mode: str = "hybrid", top_k: int = None) -> list[dict]:
    top_k = top_k or settings.final_top_k
    cache = DocumentCache.get_instance()
    record = cache.get(doc_id)
    if not record:
        return []

    with tracer.span("retrieval") as span:
        span.metadata["mode"] = mode

        if mode == "vector":
            results = _vector_search(question, doc_id, settings.retrieval_top_k)
            span.metadata["vector_results"] = len(results)
        elif mode == "bm25":
            results = bm25_search(question, record.chunks, settings.retrieval_top_k)
            span.metadata["bm25_results"] = len(results)
        else:
            vector_results = _vector_search(question, doc_id, settings.retrieval_top_k)
            bm25_results = bm25_search(question, record.chunks, settings.retrieval_top_k)
            meta_results = metadata_filter(question, record.chunks)
            span.metadata["vector_results"] = len(vector_results)
            span.metadata["bm25_results"] = len(bm25_results)
            span.metadata["metadata_results"] = len(meta_results)
            results = _hybrid_fuse(vector_results, bm25_results, meta_results, record.chunks)

        results = results[:top_k]
        span.metadata["final_results"] = len(results)
        if results:
            span.metadata["best_similarity"] = results[0].get("similarity", 0)

    top_score = results[0].get("similarity", 0) if results else 0
    logger.info("retrieval_complete", extra={"extra_data": {
        "mode": mode, "chunks_retrieved": len(results), "top_score": top_score,
    }})
    return results


def _vector_search(question: str, doc_id: str, top_k: int) -> list[dict]:
    store = VectorStore.get_instance()
    return store.query(question, doc_id, top_k)


def bm25_search(question: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    if not chunks:
        return []
    corpus = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(corpus)
    query_tokens = question.lower().split()
    scores = bm25.get_scores(query_tokens)
    scored = list(zip(chunks, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    results = []
    for chunk, score in scored[:top_k]:
        if score > 0:
            results.append({**chunk, "similarity": min(score / 10.0, 1.0)})
    return results


def metadata_filter(question: str, chunks: list[dict]) -> list[dict]:
    question_lower = question.lower()
    matched = []
    for chunk in chunks:
        section = chunk.get("section", "").lower()
        for section_key, keywords in SECTION_KEYWORDS.items():
            if any(kw in question_lower for kw in keywords):
                if section_key in section or any(kw in section for kw in keywords):
                    matched.append({**chunk, "similarity": 0.7})
                    break
    return matched


def rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    if not ranked_lists:
        return []
    scores = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, item_id in enumerate(ranked_list):
            scores[item_id] += 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


def _hybrid_fuse(vector_results, bm25_results, meta_results, all_chunks) -> list[dict]:
    def get_indices(results):
        return [r["index"] for r in results]

    ranked_lists = []
    if vector_results:
        ranked_lists.append(get_indices(vector_results))
    if bm25_results:
        ranked_lists.append(get_indices(bm25_results))
    if meta_results:
        ranked_lists.append(get_indices(meta_results))

    if not ranked_lists:
        return []

    fused_indices = rrf_fuse(ranked_lists)

    sim_lookup = {}
    for results in [vector_results, bm25_results, meta_results]:
        for r in results:
            idx = r["index"]
            if idx not in sim_lookup or r.get("similarity", 0) > sim_lookup[idx]:
                sim_lookup[idx] = r.get("similarity", 0)

    chunk_lookup = {c["index"]: c for c in all_chunks}
    results = []
    for idx in fused_indices:
        if idx in chunk_lookup:
            chunk = chunk_lookup[idx].copy()
            chunk["similarity"] = sim_lookup.get(idx, 0.5)
            results.append(chunk)
    return results
