"""ChromaDB wrapper for vector storage and retrieval."""
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import settings
from app.observability.logger import get_logger

logger = get_logger("vector_store")


class VectorStore:
    _instance = None

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )

    @classmethod
    def get_instance(cls) -> "VectorStore":
        if cls._instance is None:
            cls._instance = VectorStore()
        return cls._instance

    def add_chunks(self, doc_id: str, chunks: list[dict], embeddings: list[list[float]]):
        ids = [f"{doc_id}_chunk_{c['index']}" for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {"doc_id": doc_id, "section": c["section"], "index": c["index"]}
            for c in chunks
        ]
        self.collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        logger.info(f"Stored {len(chunks)} chunks for doc {doc_id}")

    def query(self, query_embedding: list[float], doc_id: str, top_k: int = 5) -> list[dict]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            where={"doc_id": doc_id},
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][i]
                similarity = 1 - distance
                chunks.append({
                    "text": doc,
                    "section": results["metadatas"][0][i].get("section", ""),
                    "index": results["metadatas"][0][i].get("index", 0),
                    "similarity": similarity,
                })
        return chunks

    def delete_document(self, doc_id: str):
        self.collection.delete(where={"doc_id": doc_id})

    def has_document(self, doc_id: str) -> bool:
        results = self.collection.get(where={"doc_id": doc_id}, limit=1)
        return len(results["ids"]) > 0
