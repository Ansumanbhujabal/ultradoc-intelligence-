"""In-memory document cache for full text, chunks, and metadata."""
from typing import Optional
from app.models.schemas import DocType


class DocumentRecord:
    def __init__(self, doc_id: str, file_name: str, full_text: str, doc_type: DocType, chunks: list[dict], page_count: int):
        self.doc_id = doc_id
        self.file_name = file_name
        self.full_text = full_text
        self.doc_type = doc_type
        self.chunks = chunks
        self.page_count = page_count
        self.content_hash: Optional[str] = None  # SHA256 hash for dedup
        self.extraction_result: Optional[dict] = None  # cached extraction


class DocumentCache:
    _instance = None

    def __init__(self):
        self._cache: dict[str, DocumentRecord] = {}

    @classmethod
    def get_instance(cls) -> "DocumentCache":
        if cls._instance is None:
            cls._instance = DocumentCache()
        return cls._instance

    def store(self, record: DocumentRecord):
        self._cache[record.doc_id] = record

    def get(self, doc_id: str) -> Optional[DocumentRecord]:
        return self._cache.get(doc_id)

    def exists(self, doc_id: str) -> bool:
        return doc_id in self._cache

    def list_documents(self) -> list[dict]:
        return [
            {"doc_id": r.doc_id, "file_name": r.file_name, "doc_type": r.doc_type.value}
            for r in self._cache.values()
        ]

    def delete(self, doc_id: str):
        self._cache.pop(doc_id, None)
