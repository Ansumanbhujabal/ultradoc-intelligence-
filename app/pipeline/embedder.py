"""Store document chunks in vector database (ChromaDB handles embedding internally)."""

from app.storage.vector_store import VectorStore
from app.observability.tracer import Tracer


def embed_and_store(doc_id: str, chunks: list[dict], tracer: Tracer) -> int:
    """Store chunks in ChromaDB. ChromaDB embeds using its default local model.
    Returns number of chunks stored.
    """
    if not chunks:
        tracer.skip("embedding", "no chunks to embed")
        return 0

    store = VectorStore.get_instance()

    with tracer.span("embedding") as span:
        store.add_chunks(doc_id, chunks)
        span.metadata = {
            "chunk_count": len(chunks),
            "embedding_model": "chromadb-default (all-MiniLM-L6-v2)",
        }

    return len(chunks)
