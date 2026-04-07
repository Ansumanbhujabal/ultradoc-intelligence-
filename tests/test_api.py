"""API integration tests using FastAPI TestClient."""
import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.llm.provider import LLMResponse
from app.models.schemas import DocType, GuardrailStatus
from app.storage.cache import DocumentCache, DocumentRecord


@pytest.fixture(autouse=True)
def _clear_cache():
    """Reset the DocumentCache singleton between tests."""
    cache = DocumentCache.get_instance()
    cache._cache.clear()
    yield
    cache._cache.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_provider():
    """Mock the LLM provider to avoid real API calls."""
    mock = MagicMock()
    mock.fast_model = "gpt-4.1-mini"
    mock.default_model = "gpt-4o"

    # Default generate response (classification)
    classify_response = LLMResponse(
        content="rate_confirmation", tokens_in=10, tokens_out=1,
        latency_ms=100, model_name="gpt-4.1-mini",
    )
    # Q&A response
    qa_response = LLMResponse(
        content="SECTION: Rate Breakdown\nANSWER: The carrier rate is $400.00 USD\nCONFIDENCE: HIGH",
        tokens_in=100, tokens_out=50, latency_ms=500, model_name="gpt-4o",
    )
    # Extraction response
    extract_response = LLMResponse(
        content='{"shipment_id": "LD53657", "shipper": "Test Shipper", "consignee": "Test Consignee", '
                '"pickup_datetime": null, "delivery_datetime": null, "equipment_type": "Flatbed", '
                '"mode": "FTL", "rate": 400.0, "currency": "USD", "weight": "56000 lbs", '
                '"carrier_name": "SWIFT SHIFT"}',
        tokens_in=200, tokens_out=100, latency_ms=800, model_name="gpt-4o",
    )

    mock.generate.side_effect = [classify_response, qa_response]
    mock.generate_structured.return_value = extract_response

    return mock


def _seed_document(doc_id: str = "test-doc-001") -> DocumentRecord:
    """Helper: seed a document into the cache for endpoints that need it."""
    record = DocumentRecord(
        doc_id=doc_id,
        file_name="test.txt",
        full_text="Shipment ID: LD53657\nCarrier: SWIFT SHIFT\nRate: $400.00 USD\nEquipment: Flatbed",
        doc_type=DocType.RATE_CONFIRMATION,
        chunks=[
            {"text": "Shipment ID: LD53657", "index": 0},
            {"text": "Carrier: SWIFT SHIFT", "index": 1},
            {"text": "Rate: $400.00 USD", "index": 2},
        ],
        page_count=1,
    )
    record.content_hash = "abc123"
    DocumentCache.get_instance().store(record)
    return record


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
class TestDocumentsEndpoint:
    def test_list_empty(self, client):
        resp = client.get("/documents")
        assert resp.status_code == 200
        assert resp.json()["documents"] == []

    def test_list_with_documents(self, client):
        _seed_document("doc-a")
        _seed_document("doc-b")
        resp = client.get("/documents")
        assert resp.status_code == 200
        docs = resp.json()["documents"]
        assert len(docs) == 2
        assert all("doc_id" in d for d in docs)


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------
class TestTracesEndpoint:
    def test_traces(self, client):
        resp = client.get("/traces")
        assert resp.status_code == 200
        assert "traces" in resp.json()


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
class TestUploadEndpoint:
    def test_upload_unsupported_format(self, client):
        resp = client.post(
            "/upload",
            files={"file": ("test.xyz", b"data", "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "Unsupported" in resp.json()["detail"]

    def test_upload_no_file(self, client):
        resp = client.post("/upload")
        assert resp.status_code == 422  # FastAPI validation error

    @patch("app.main.embed_and_store")
    @patch("app.main.chunk_document")
    @patch("app.main.classify_document")
    def test_upload_txt(self, mock_classify, mock_chunk, mock_embed, client):
        mock_classify.return_value = DocType.UNKNOWN
        mock_chunk.return_value = [
            {"text": "Shipment ID: LD99999", "index": 0},
            {"text": "Carrier: Test Carrier", "index": 1},
        ]
        mock_embed.return_value = 2

        content = b"Shipment ID: LD99999\nCarrier: Test Carrier\nRate: $500"
        resp = client.post(
            "/upload",
            files={"file": ("test.txt", content, "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "doc_id" in data
        assert data["chunk_count"] == 2
        assert data["doc_type"] == "unknown"
        assert data["status"] == "success"

    @patch("app.main.embed_and_store")
    @patch("app.main.chunk_document")
    @patch("app.main.classify_document")
    def test_upload_dedup(self, mock_classify, mock_chunk, mock_embed, client):
        """Uploading identical content should return the original doc_id."""
        mock_classify.return_value = DocType.RATE_CONFIRMATION
        mock_chunk.return_value = [{"text": "chunk", "index": 0}]
        mock_embed.return_value = 1

        content = b"Identical content for dedup test"
        # First upload
        resp1 = client.post(
            "/upload", files={"file": ("a.txt", content, "text/plain")},
        )
        assert resp1.status_code == 200
        doc_id_1 = resp1.json()["doc_id"]

        # Second upload with same content
        resp2 = client.post(
            "/upload", files={"file": ("b.txt", content, "text/plain")},
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["doc_id"] == doc_id_1
        assert data2["status"] == "already_uploaded"

    @patch("app.main.embed_and_store")
    @patch("app.main.chunk_document")
    @patch("app.main.classify_document")
    @patch("app.main.parse_document")
    def test_upload_parse_failure(self, mock_parse, mock_classify, mock_chunk, mock_embed, client):
        mock_parse.return_value = {"status": "error", "error": "corrupt file"}
        content = b"bad data"
        resp = client.post(
            "/upload", files={"file": ("bad.txt", content, "text/plain")},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------
class TestAskEndpoint:
    def test_ask_missing_doc(self, client):
        resp = client.post("/ask", json={
            "doc_id": "nonexistent",
            "question": "What is the rate?",
        })
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_ask_missing_question(self, client):
        resp = client.post("/ask", json={"doc_id": "some-id"})
        assert resp.status_code == 422  # pydantic validation

    @patch("app.main.generate_answer")
    @patch("app.main.retrieve")
    @patch("app.main.check_scope")
    def test_ask_out_of_scope(self, mock_scope, mock_retrieve, mock_gen, client):
        _seed_document("scoped-doc")
        mock_scope.return_value = GuardrailStatus.OUT_OF_SCOPE

        resp = client.post("/ask", json={
            "doc_id": "scoped-doc",
            "question": "What is the weather?",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["guardrail_status"] == "out_of_scope"
        # generate and retrieve should not be called for out-of-scope
        mock_retrieve.assert_not_called()
        mock_gen.assert_not_called()

    @patch("app.main.check_grounding")
    @patch("app.main.generate_answer")
    @patch("app.main.retrieve")
    @patch("app.main.check_scope")
    def test_ask_success(self, mock_scope, mock_retrieve, mock_gen, mock_grounding, client):
        _seed_document("ask-doc")
        mock_scope.return_value = GuardrailStatus.PASSED
        mock_retrieve.return_value = [
            {"text": "Rate: $400.00 USD", "similarity": 0.92, "index": 0},
        ]
        mock_gen.return_value = {
            "answer": "The carrier rate is $400.00 USD",
            "source_text": "Rate: $400.00 USD",
            "llm_confidence": "HIGH",
        }
        mock_grounding.return_value = (GuardrailStatus.PASSED, 0.85)

        resp = client.post("/ask", json={
            "doc_id": "ask-doc",
            "question": "What is the rate?",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert data["guardrail_status"] == "passed"
        assert data["confidence"]["score"] > 0

    @patch("app.main.check_retrieval_threshold")
    @patch("app.main.retrieve")
    @patch("app.main.check_scope")
    def test_ask_not_found(self, mock_scope, mock_retrieve, mock_threshold, client):
        _seed_document("nf-doc")
        mock_scope.return_value = GuardrailStatus.PASSED
        mock_retrieve.return_value = [
            {"text": "irrelevant chunk", "similarity": 0.05, "index": 0},
        ]
        mock_threshold.return_value = GuardrailStatus.NOT_FOUND

        resp = client.post("/ask", json={
            "doc_id": "nf-doc",
            "question": "What is the quantum state?",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["guardrail_status"] == "not_found"
        assert "not found" in data["answer"].lower() or "Not found" in data["answer"]


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------
class TestExtractEndpoint:
    def test_extract_missing_doc(self, client):
        resp = client.post("/extract", json={"doc_id": "nonexistent"})
        assert resp.status_code == 404

    @patch("app.main.extract_shipment_data")
    def test_extract_success(self, mock_extract, client):
        from app.models.schemas import DocType
        _seed_document("ext-doc")

        # Mock the extractor pipeline
        mock_shipment = MagicMock()
        mock_shipment.model_dump.return_value = {
            "shipment_id": "LD53657",
            "shipper": "Test Shipper",
            "consignee": "Test Consignee",
            "pickup_datetime": None,
            "delivery_datetime": None,
            "equipment_type": "Flatbed",
            "mode": "FTL",
            "rate": 400.0,
            "currency": "USD",
            "weight": "56000 lbs",
            "carrier_name": "SWIFT SHIFT",
        }
        mock_extract.return_value = {
            "shipment_data": mock_shipment,
            "completeness_score": 0.8,
        }

        resp = client.post("/extract", json={"doc_id": "ext-doc"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["extracted_data"]["shipment_id"] == "LD53657"
        assert data["completeness_score"] == 0.8
        assert data["doc_type"] == "rate_confirmation"

    @patch("app.main.extract_shipment_data")
    def test_extract_cached(self, mock_extract, client):
        """Second extraction should return cached result without calling extractor."""
        record = _seed_document("cache-ext-doc")
        record.extraction_result = {
            "extracted_data": {"shipment_id": "CACHED"},
            "completeness_score": 0.9,
            "doc_type": DocType.RATE_CONFIRMATION,
        }

        resp = client.post("/extract", json={"doc_id": "cache-ext-doc"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["extracted_data"]["shipment_id"] == "CACHED"
        mock_extract.assert_not_called()


# ---------------------------------------------------------------------------
# Gradio mount
# ---------------------------------------------------------------------------
class TestGradioMount:
    def test_ui_route_exists(self):
        """Verify Gradio can be mounted on FastAPI."""
        import gradio as gr
        from ui.gradio_app import build_ui
        demo = build_ui()
        mounted_app = gr.mount_gradio_app(app, demo, path="/ui")
        test_client = TestClient(mounted_app)
        # The /ui path should not 404
        resp = test_client.get("/ui/")
        assert resp.status_code != 404


# ---------------------------------------------------------------------------
# Swagger / OpenAPI docs
# ---------------------------------------------------------------------------
class TestSwaggerDocs:
    def test_docs_accessible(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_openapi_json(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        paths = list(data.get("paths", {}).keys())
        assert "/upload" in paths
        assert "/ask" in paths
        assert "/extract" in paths
        assert "/health" in paths
        assert "/documents" in paths
        assert "/traces" in paths
