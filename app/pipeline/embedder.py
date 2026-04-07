"""Embed document chunks and store in vector database."""

from app.llm.provider import get_provider
from app.storage.vector_store import VectorStore
from app.observability.tracer import Tracer


def embed_and_store(doc_id: str, chunks: list[dict], tracer: Tracer) -> int:
    """Embed chunks and store in ChromaDB. Returns number of chunks stored."""
    if not chunks:
        tracer.skip("embedding", "no chunks to embed")
        return 0

    provider = get_provider()
    store = VectorStore.get_instance()

    with tracer.span("embedding") as span:
        texts = [c["text"] for c in chunks]
        embeddings = provider.embed(texts)
        store.add_chunks(doc_id, chunks, embeddings)
        span.metadata = {
            "chunk_count": len(chunks),
            "embedding_dimensions": len(embeddings[0]) if embeddings else 0,
        }

    return len(chunks)
