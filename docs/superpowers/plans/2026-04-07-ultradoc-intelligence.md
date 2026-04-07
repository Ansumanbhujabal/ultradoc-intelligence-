# Ultra Doc-Intelligence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a POC AI system for logistics document Q&A with RAG, guardrails, structured extraction, and observability.

**Architecture:** FastAPI backend with a deterministic pipeline (parse → classify → chunk → embed → retrieve → generate → validate). Provider-agnostic LLM layer with Azure OpenAI primary. Hybrid retrieval (vector + BM25 + metadata + RRF). 3-layer deterministic guardrails. Composite confidence scoring. Gradio UI with 4 tabs. Docker + HF Spaces deployment.

**Tech Stack:** Python 3.11, FastAPI, pdfplumber, python-docx, ChromaDB, OpenAI SDK, rank-bm25, Pydantic v2, Gradio

**Spec:** `docs/superpowers/specs/2026-04-07-ultradoc-intelligence-design.md`

---

## File Map

| File | Responsibility | Created in Task |
|------|---------------|----------------|
| `app/__init__.py` | Package init | 1 |
| `app/config.py` | Settings from .env via pydantic-settings | 1 |
| `app/models/__init__.py` | Package init | 2 |
| `app/models/schemas.py` | Request/Response Pydantic models | 2 |
| `app/models/extraction.py` | Shipment extraction Pydantic schema | 2 |
| `app/observability/__init__.py` | Package init | 3 |
| `app/observability/tracer.py` | Request tracing with per-stage spans | 3 |
| `app/observability/logger.py` | Structured JSON logger | 3 |
| `app/llm/__init__.py` | Package init | 4 |
| `app/llm/provider.py` | Provider-agnostic LLM abstraction | 4 |
| `app/llm/prompts/__init__.py` | Package init | 5 |
| `app/llm/prompts/registry.py` | Prompt registry (name + version) | 5 |
| `app/llm/prompts/system.py` | Base system prompts | 5 |
| `app/llm/prompts/qa.py` | Q&A prompt templates | 5 |
| `app/llm/prompts/extraction/bol.py` | BOL extraction prompt | 5 |
| `app/llm/prompts/extraction/rate_confirm.py` | RC extraction prompt | 5 |
| `app/llm/prompts/extraction/generic.py` | Generic extraction prompt | 5 |
| `app/llm/prompts/guardrails.py` | Out-of-scope keywords | 5 |
| `app/pipeline/__init__.py` | Package init | 6 |
| `app/pipeline/parser.py` | PDF/DOCX/TXT parsing + LLM vision fallback | 6 |
| `app/pipeline/classifier.py` | Doc type detection | 7 |
| `app/pipeline/chunker.py` | Section-based chunking | 8 |
| `app/storage/__init__.py` | Package init | 9 |
| `app/storage/vector_store.py` | ChromaDB wrapper | 9 |
| `app/storage/cache.py` | Document + embedding cache | 9 |
| `app/pipeline/embedder.py` | Embed chunks + store in ChromaDB | 10 |
| `app/pipeline/retriever.py` | Hybrid retrieval (vector + BM25 + metadata + RRF) | 11 |
| `app/pipeline/guardrails.py` | 3-layer guardrail system | 12 |
| `app/pipeline/generator.py` | LLM Q&A generation with hidden CoT | 13 |
| `app/pipeline/extractor.py` | Structured extraction (Pydantic + function calling) | 14 |
| `app/main.py` | FastAPI app + 3 endpoints | 15 |
| `ui/__init__.py` | Package init | 16 |
| `ui/gradio_app.py` | Gradio UI with 4 tabs | 16 |
| `run.py` | Single entrypoint (FastAPI + Gradio) | 17 |
| `requirements.txt` | Dependencies | 1 |
| `Dockerfile` | Container image | 17 |
| `docker-compose.yml` | Local orchestration | 17 |
| `eval/ground_truth.json` | Test cases from sample docs | 18 |
| `eval/run_eval.py` | Automated eval runner | 18 |
| `eval/report.py` | Metrics computation + output | 18 |
| `tests/test_guardrails.py` | Guardrail unit tests | 12 |
| `tests/test_retriever.py` | Retriever + RRF unit tests | 11 |
| `tests/test_confidence.py` | Confidence scoring tests | 13 |
| `tests/test_parser.py` | Parser tests | 6 |
| `tests/test_chunker.py` | Chunker tests | 8 |
| `tests/conftest.py` | Shared fixtures | 6 |

---

### Task 1: Project Setup + Config + Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `app/__init__.py`
- Create: `app/config.py`
- Modify: `.env` (add embedding model config)

- [ ] **Step 1: Create requirements.txt**

```txt
fastapi==0.115.0
uvicorn[standard]==0.30.0
pdfplumber==0.11.0
python-docx==1.1.0
chromadb==0.5.23
openai==1.75.0
rank-bm25==0.2.2
pydantic>=2.0
pydantic-settings>=2.0
gradio==5.20.0
httpx==0.28.0
python-multipart==0.0.18
numpy>=1.26.0
pytest==8.3.0
pytest-asyncio==0.24.0
```

- [ ] **Step 2: Create app/__init__.py**

```python
"""Ultra Doc-Intelligence — AI-powered logistics document Q&A system."""
```

- [ ] **Step 3: Update .env with full config**

Add these lines to the existing `.env`:
```
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-3-small
AZURE_OPENAI_FAST_MODEL=gpt-4.1-mini
LLM_PROVIDER=azure_openai
CHROMA_PERSIST_DIR=./data/chroma
UPLOAD_DIR=./data/uploads
LOG_LEVEL=INFO
```

- [ ] **Step 4: Create app/config.py**

```python
"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Azure OpenAI
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_api_version: str = "2024-02-15-preview"
    azure_openai_model: str = "gpt-4o"
    azure_openai_fast_model: str = "gpt-4.1-mini"
    azure_openai_embedding_model: str = "text-embedding-3-small"

    # Provider
    llm_provider: str = "azure_openai"

    # Storage
    chroma_persist_dir: str = "./data/chroma"
    upload_dir: str = "./data/uploads"

    # Pipeline defaults
    default_confidence_threshold: float = 0.3
    default_retrieval_mode: str = "hybrid"
    grounding_overlap_threshold: float = 0.4
    retrieval_top_k: int = 5
    final_top_k: int = 3

    # Logging
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
```

- [ ] **Step 5: Install dependencies and verify**

Run: `cd /opt/CodeRepo/ultradoc-intelligence && pip install -r requirements.txt`
Expected: All packages install successfully.

Run: `python -c "from app.config import settings; print(settings.llm_provider)"`
Expected: `azure_openai`

- [ ] **Step 6: Initialize git and commit**

```bash
git init
echo "__pycache__/\n*.pyc\n.env\ndata/\n*.egg-info/\n.pytest_cache/" > .gitignore
git add .gitignore requirements.txt app/__init__.py app/config.py ultradoc-requirements.md ultradoc_sample_test_data/
git commit -m "feat: project setup with config and dependencies"
```

---

### Task 2: Pydantic Models (Schemas + Extraction)

**Files:**
- Create: `app/models/__init__.py`
- Create: `app/models/schemas.py`
- Create: `app/models/extraction.py`

- [ ] **Step 1: Create app/models/__init__.py**

```python
"""Pydantic models for request/response schemas and extraction."""

from app.models.schemas import (
    UploadResponse,
    AskRequest,
    AskResponse,
    ExtractRequest,
    ExtractResponse,
    ConfidenceResult,
)
from app.models.extraction import ShipmentData
```

- [ ] **Step 2: Create app/models/schemas.py**

```python
"""Request and response schemas for API endpoints."""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class DocType(str, Enum):
    BOL = "bill_of_lading"
    RATE_CONFIRMATION = "rate_confirmation"
    INVOICE = "invoice"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class GuardrailStatus(str, Enum):
    PASSED = "passed"
    LOW_GROUNDING = "low_grounding"
    NOT_FOUND = "not_found"
    OUT_OF_SCOPE = "out_of_scope"


class ConfidenceBreakdown(BaseModel):
    retrieval: float = Field(description="Hybrid retrieval similarity score")
    grounding: float = Field(description="Answer-source token overlap ratio")
    llm_assessment: float = Field(description="LLM self-assessed confidence")


class ConfidenceResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    level: ConfidenceLevel
    breakdown: ConfidenceBreakdown


# Upload
class UploadResponse(BaseModel):
    doc_id: str
    doc_type: DocType
    page_count: int
    chunk_count: int
    status: str = "success"


# Ask
class AskRequest(BaseModel):
    doc_id: str
    question: str
    enable_query_rewrite: bool = False
    retrieval_mode: str = "hybrid"
    confidence_threshold: float = 0.3


class AskResponse(BaseModel):
    answer: str
    source_text: str
    confidence: ConfidenceResult
    guardrail_status: GuardrailStatus
    query_rewritten: bool = False
    retrieval_mode: str = "hybrid"


# Extract
class ExtractRequest(BaseModel):
    doc_id: str


class ExtractResponse(BaseModel):
    extracted_data: dict
    completeness_score: float = Field(ge=0.0, le=1.0)
    doc_type: DocType
```

- [ ] **Step 3: Create app/models/extraction.py**

```python
"""Shipment data extraction schema for structured output."""

from pydantic import BaseModel, Field
from typing import Optional


class ShipmentData(BaseModel):
    """Structured shipment data extracted from logistics documents.
    Fields are Optional — null when not found in the document.
    """

    shipment_id: Optional[str] = Field(None, description="Load ID or shipment reference number")
    shipper: Optional[str] = Field(None, description="Shipper name and address")
    consignee: Optional[str] = Field(None, description="Consignee/receiver name and address")
    pickup_datetime: Optional[str] = Field(None, description="Pickup date and time in ISO format")
    delivery_datetime: Optional[str] = Field(None, description="Delivery date and time in ISO format")
    equipment_type: Optional[str] = Field(None, description="Equipment type (e.g., Flatbed, Dry Van)")
    mode: Optional[str] = Field(None, description="Shipping mode (e.g., FTL, LTL)")
    rate: Optional[float] = Field(None, description="Rate/charge amount as a number")
    currency: Optional[str] = Field(None, description="Currency code (e.g., USD)")
    weight: Optional[str] = Field(None, description="Weight with unit (e.g., '56000 lbs')")
    carrier_name: Optional[str] = Field(None, description="Carrier company name")

    def completeness_score(self) -> float:
        """Percentage of non-null fields."""
        fields = self.model_fields.keys()
        non_null = sum(1 for f in fields if getattr(self, f) is not None)
        return round(non_null / len(fields), 2)
```

- [ ] **Step 4: Verify models load correctly**

Run: `python -c "from app.models import ShipmentData, AskRequest; print(ShipmentData.model_json_schema()['properties'].keys())"`
Expected: All 11 field names printed.

- [ ] **Step 5: Commit**

```bash
git add app/models/
git commit -m "feat: add Pydantic models for API schemas and shipment extraction"
```

---

### Task 3: Observability — Tracer + Logger

**Files:**
- Create: `app/observability/__init__.py`
- Create: `app/observability/tracer.py`
- Create: `app/observability/logger.py`

- [ ] **Step 1: Create app/observability/__init__.py**

```python
"""Observability — structured tracing and logging for the pipeline."""

from app.observability.tracer import Tracer
from app.observability.logger import get_logger
```

- [ ] **Step 2: Create app/observability/logger.py**

```python
"""Structured JSON logger."""

import logging
import json
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)
        return json.dumps(log_entry)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```

- [ ] **Step 3: Create app/observability/tracer.py**

```python
"""Request tracing with per-stage spans."""

import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

from app.observability.logger import get_logger

logger = get_logger("tracer")


class Span:
    def __init__(self, stage: str):
        self.stage = stage
        self.status = "in_progress"
        self.latency_ms = 0
        self.metadata: dict = {}
        self.error: Optional[str] = None

    def to_dict(self) -> dict:
        result = {
            "stage": self.stage,
            "status": self.status,
            "latency_ms": self.latency_ms,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        if self.error:
            result["error"] = self.error
        return result


class Tracer:
    """Traces a single request through the pipeline."""

    # Class-level storage for recent traces (simple in-memory)
    _recent_traces: list[dict] = []
    MAX_TRACES = 50

    def __init__(self, endpoint: str, doc_id: str = ""):
        self.trace_id = f"tr_{uuid.uuid4().hex[:12]}"
        self.endpoint = endpoint
        self.doc_id = doc_id
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.spans: list[Span] = []
        self._start_time = time.perf_counter()

    @contextmanager
    def span(self, stage: str):
        """Context manager for timing a pipeline stage."""
        s = Span(stage)
        start = time.perf_counter()
        try:
            yield s
            s.status = "success"
        except Exception as e:
            s.status = "error"
            s.error = str(e)
            raise
        finally:
            s.latency_ms = round((time.perf_counter() - start) * 1000)
            self.spans.append(s)

    def skip(self, stage: str, note: str = ""):
        """Record a skipped stage."""
        s = Span(stage)
        s.status = "skipped"
        if note:
            s.metadata["note"] = note
        self.spans.append(s)

    def to_dict(self) -> dict:
        total_latency = round((time.perf_counter() - self._start_time) * 1000)
        total_tokens = sum(
            s.metadata.get("tokens_in", 0) + s.metadata.get("tokens_out", 0)
            for s in self.spans
        )
        return {
            "trace_id": self.trace_id,
            "doc_id": self.doc_id,
            "endpoint": self.endpoint,
            "timestamp": self.timestamp,
            "total_latency_ms": total_latency,
            "total_tokens": total_tokens,
            "spans": [s.to_dict() for s in self.spans],
        }

    def finish(self):
        """Finalize trace and store it."""
        trace_dict = self.to_dict()
        Tracer._recent_traces.append(trace_dict)
        if len(Tracer._recent_traces) > Tracer.MAX_TRACES:
            Tracer._recent_traces.pop(0)
        logger.info("trace_complete", extra={"extra_data": trace_dict})
        return trace_dict

    @classmethod
    def get_recent_traces(cls) -> list[dict]:
        return list(reversed(cls._recent_traces))
```

- [ ] **Step 4: Verify tracer works**

Run:
```python
python -c "
from app.observability.tracer import Tracer
t = Tracer('/ask', 'doc_123')
with t.span('test_stage') as s:
    s.metadata['key'] = 'value'
result = t.finish()
print(result['trace_id'], len(result['spans']), result['spans'][0]['status'])
"
```
Expected: `tr_<hex> 1 success`

- [ ] **Step 5: Commit**

```bash
git add app/observability/
git commit -m "feat: add structured tracing and JSON logging"
```

---

### Task 4: LLM Abstraction Layer

**Files:**
- Create: `app/llm/__init__.py`
- Create: `app/llm/provider.py`

- [ ] **Step 1: Create app/llm/__init__.py**

```python
"""Provider-agnostic LLM abstraction layer."""

from app.llm.provider import get_provider, LLMProvider, LLMResponse
```

- [ ] **Step 2: Create app/llm/provider.py**

```python
"""Provider-agnostic LLM abstraction with Azure OpenAI primary."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from openai import AzureOpenAI

from app.config import settings


@dataclass
class LLMResponse:
    content: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    model_name: str = ""


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, messages: list[dict], model: Optional[str] = None, **kwargs) -> LLMResponse:
        ...

    @abstractmethod
    def generate_structured(
        self, messages: list[dict], schema: dict, model: Optional[str] = None, **kwargs
    ) -> LLMResponse:
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class AzureOpenAIProvider(LLMProvider):
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
        self.default_model = settings.azure_openai_model
        self.fast_model = settings.azure_openai_fast_model
        self.embedding_model = settings.azure_openai_embedding_model

    def generate(self, messages: list[dict], model: Optional[str] = None, **kwargs) -> LLMResponse:
        model = model or self.default_model
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 2000),
        )
        latency = round((time.perf_counter() - start) * 1000)
        usage = response.usage
        return LLMResponse(
            content=response.choices[0].message.content or "",
            tokens_in=usage.prompt_tokens if usage else 0,
            tokens_out=usage.completion_tokens if usage else 0,
            latency_ms=latency,
            model_name=model,
        )

    def generate_structured(
        self, messages: list[dict], schema: dict, model: Optional[str] = None, **kwargs
    ) -> LLMResponse:
        model = model or self.default_model
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=kwargs.get("max_tokens", 2000),
            response_format={"type": "json_object"},
        )
        latency = round((time.perf_counter() - start) * 1000)
        usage = response.usage
        return LLMResponse(
            content=response.choices[0].message.content or "",
            tokens_in=usage.prompt_tokens if usage else 0,
            tokens_out=usage.completion_tokens if usage else 0,
            latency_ms=latency,
            model_name=model,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]


_provider_instance: Optional[LLMProvider] = None


def get_provider() -> LLMProvider:
    global _provider_instance
    if _provider_instance is None:
        if settings.llm_provider == "azure_openai":
            _provider_instance = AzureOpenAIProvider()
        else:
            raise ValueError(f"Unsupported provider: {settings.llm_provider}")
    return _provider_instance
```

- [ ] **Step 3: Verify provider initializes**

Run: `python -c "from app.llm.provider import get_provider; p = get_provider(); print(type(p).__name__)"`
Expected: `AzureOpenAIProvider`

- [ ] **Step 4: Commit**

```bash
git add app/llm/
git commit -m "feat: add provider-agnostic LLM abstraction layer"
```

---

### Task 5: Prompt Registry + Templates

**Files:**
- Create: `app/llm/prompts/__init__.py`
- Create: `app/llm/prompts/registry.py`
- Create: `app/llm/prompts/system.py`
- Create: `app/llm/prompts/qa.py`
- Create: `app/llm/prompts/extraction/__init__.py`
- Create: `app/llm/prompts/extraction/bol.py`
- Create: `app/llm/prompts/extraction/rate_confirm.py`
- Create: `app/llm/prompts/extraction/generic.py`
- Create: `app/llm/prompts/guardrails.py`

- [ ] **Step 1: Create prompt registry**

Create `app/llm/prompts/__init__.py`:
```python
"""Prompt templates and registry for the pipeline."""

from app.llm.prompts.registry import PromptRegistry, registry
```

Create `app/llm/prompts/registry.py`:
```python
"""Prompt registry — load templates by name and version."""

from typing import Optional


class PromptTemplate:
    def __init__(self, name: str, template: str, version: str = "v1"):
        self.name = name
        self.template = template
        self.version = version

    def render(self, **kwargs) -> str:
        return self.template.format(**kwargs)


class PromptRegistry:
    def __init__(self):
        self._templates: dict[str, dict[str, PromptTemplate]] = {}

    def register(self, name: str, template: str, version: str = "v1"):
        if name not in self._templates:
            self._templates[name] = {}
        self._templates[name][version] = PromptTemplate(name, template, version)

    def get(self, name: str, version: str = "v1") -> PromptTemplate:
        if name not in self._templates:
            raise KeyError(f"Prompt template '{name}' not found")
        versions = self._templates[name]
        if version not in versions:
            raise KeyError(f"Version '{version}' not found for prompt '{name}'")
        return versions[version]

    def list_templates(self) -> list[dict]:
        result = []
        for name, versions in self._templates.items():
            for version in versions:
                result.append({"name": name, "version": version})
        return result


registry = PromptRegistry()
```

- [ ] **Step 2: Create system prompts**

Create `app/llm/prompts/system.py`:
```python
"""Base system prompts used across all LLM calls."""

from app.llm.prompts.registry import registry

SYSTEM_PROMPT = """You are a logistics document analyst for a Transportation Management System (TMS).

Rules:
- Answer ONLY from the provided document content. Never use external knowledge.
- If the information is not found in the document, respond with: "Not found in document."
- Never infer, assume, or fabricate information.
- Be precise with numbers, dates, names, and addresses — copy them exactly from the document.
- When quoting from the document, preserve the original formatting."""

CLASSIFICATION_PROMPT = """Classify this logistics document into one of these types:
- bill_of_lading: A Bill of Lading (BOL) document with shipper/consignee, commodity, and shipping details
- rate_confirmation: A Rate Confirmation (RC) document with carrier details, stops, rates, and instructions
- invoice: An invoice or billing document with charges and payment details
- unknown: Cannot determine the document type

Document content:
{document_text}

Respond with ONLY the document type (bill_of_lading, rate_confirmation, invoice, or unknown). No explanation."""

QUERY_REWRITE_PROMPT = """Rewrite this user question to be more specific and retrieval-friendly for searching a logistics document.
Keep the same intent, but use precise logistics terminology.

Original question: {question}

Rewritten question:"""

registry.register("system", SYSTEM_PROMPT)
registry.register("classification", CLASSIFICATION_PROMPT)
registry.register("query_rewrite", QUERY_REWRITE_PROMPT)
```

- [ ] **Step 3: Create Q&A prompt**

Create `app/llm/prompts/qa.py`:
```python
"""Q&A prompt templates with hidden Chain-of-Thought."""

from app.llm.prompts.registry import registry

QA_PROMPT = """Based on the following logistics document, answer the user's question.

<document>
{document_text}
</document>

<relevant_sections>
{source_chunks}
</relevant_sections>

<question>
{question}
</question>

Instructions:
1. First, identify which section of the document contains the answer.
2. Then, provide a clear and concise answer.
3. Assess your confidence: HIGH (answer is clearly stated in document), MEDIUM (answer requires some interpretation), LOW (answer is uncertain or partially found).

Respond in this exact format:
SECTION: [section name or heading where you found the answer]
ANSWER: [your answer]
CONFIDENCE: [HIGH/MEDIUM/LOW]"""

registry.register("qa_with_cot", QA_PROMPT)
```

- [ ] **Step 4: Create extraction prompts**

Create `app/llm/prompts/extraction/__init__.py`:
```python
"""Document-type-specific extraction prompts."""
```

Create `app/llm/prompts/extraction/bol.py`:
```python
"""Bill of Lading extraction prompt."""

from app.llm.prompts.registry import registry

BOL_EXTRACTION_PROMPT = """Extract structured shipment data from this Bill of Lading document.

<document>
{document_text}
</document>

Extract the following fields into a JSON object. Use null for any field not found in the document.

Fields:
- shipment_id: The Load ID, reference number, or BOL number
- shipper: Full shipper name and address
- consignee: Full consignee/receiver name and address
- pickup_datetime: Pickup or ship date/time in ISO format (YYYY-MM-DDTHH:MM:SS). If only date is available, use T00:00:00
- delivery_datetime: Delivery date/time in ISO format. If only date is available, use T00:00:00
- equipment_type: Equipment type (e.g., Flatbed, Dry Van, Reefer)
- mode: Shipping mode (e.g., FTL, LTL)
- rate: Numeric rate/charge amount (number only, no currency symbol)
- currency: Currency code (e.g., USD, CAD)
- weight: Weight with unit (e.g., "56000 lbs")
- carrier_name: Carrier or transportation company name

Example output:
{{"shipment_id": "LD12345", "shipper": "ABC Corp, 123 Main St, City, ST 12345", "consignee": "XYZ Inc, 456 Oak Ave, Town, ST 67890", "pickup_datetime": "2026-01-15T09:00:00", "delivery_datetime": "2026-01-16T14:00:00", "equipment_type": "Dry Van", "mode": "FTL", "rate": 1500.00, "currency": "USD", "weight": "42000 lbs", "carrier_name": "Fast Freight LLC"}}

Respond with ONLY the JSON object. No explanation or markdown formatting."""

registry.register("extraction_bol", BOL_EXTRACTION_PROMPT)
```

Create `app/llm/prompts/extraction/rate_confirm.py`:
```python
"""Rate Confirmation extraction prompt."""

from app.llm.prompts.registry import registry

RC_EXTRACTION_PROMPT = """Extract structured shipment data from this Rate Confirmation document.

<document>
{document_text}
</document>

Extract the following fields into a JSON object. Use null for any field not found in the document.

Rate Confirmations typically contain:
- Reference/Load ID in the header section
- Carrier details section with carrier name, MC number, equipment
- Stops section with pickup (shipper) and drop (consignee) locations, dates, times
- Rate Breakdown section with charges and totals
- Customer or shipper details

Fields:
- shipment_id: The Reference ID, Load ID, or booking reference
- shipper: Full shipper/pickup location name and address (from the Pickup stop)
- consignee: Full consignee/drop location name and address (from the Drop stop)
- pickup_datetime: Shipping/pickup date and time in ISO format (YYYY-MM-DDTHH:MM:SS). Combine Shipping Date + Shipping Time or Appointment time
- delivery_datetime: Delivery date and time in ISO format. Combine Delivery Date + Delivery Time
- equipment_type: Equipment type from Carrier Details (e.g., Flatbed, Dry Van)
- mode: Load Type (e.g., FTL, LTL)
- rate: Total rate/agreed amount as a number (from Rate Breakdown total or Agreed Amount)
- currency: Currency code (e.g., USD)
- weight: Weight with unit from commodity details (e.g., "56000 lbs")
- carrier_name: Carrier company name from Carrier Details section

Example output:
{{"shipment_id": "LD53657", "shipper": "AAA, Los Angeles International Airport (LAX), World Way, Los Angeles, CA, USA", "consignee": "xyz, 7470 Cherry Avenue, Fontana, CA 92336, USA", "pickup_datetime": "2026-02-08T09:00:00", "delivery_datetime": "2026-02-08T09:00:00", "equipment_type": "Flatbed", "mode": "FTL", "rate": 400.00, "currency": "USD", "weight": "56000.00 lbs", "carrier_name": "SWIFT SHIFT LOGISTICS LLC"}}

Respond with ONLY the JSON object. No explanation or markdown formatting."""

registry.register("extraction_rc", RC_EXTRACTION_PROMPT)
```

Create `app/llm/prompts/extraction/generic.py`:
```python
"""Generic extraction prompt for unknown document types."""

from app.llm.prompts.registry import registry

GENERIC_EXTRACTION_PROMPT = """Extract structured shipment data from this logistics document.

<document>
{document_text}
</document>

Extract the following fields into a JSON object. Use null for any field not found in the document.

Fields:
- shipment_id: Any reference number, load ID, BOL number, or tracking number
- shipper: Shipper/sender name and address
- consignee: Consignee/receiver name and address
- pickup_datetime: Pickup date/time in ISO format (YYYY-MM-DDTHH:MM:SS), null if not found
- delivery_datetime: Delivery date/time in ISO format, null if not found
- equipment_type: Equipment or trailer type, null if not found
- mode: Shipping mode (FTL, LTL, etc.), null if not found
- rate: Rate or charge amount as a number, null if not found
- currency: Currency code, null if not found
- weight: Weight with unit, null if not found
- carrier_name: Carrier or transportation company name, null if not found

Respond with ONLY the JSON object. No explanation or markdown formatting."""

registry.register("extraction_generic", GENERIC_EXTRACTION_PROMPT)
```

- [ ] **Step 5: Create guardrail keywords**

Create `app/llm/prompts/guardrails.py`:
```python
"""Out-of-scope detection — keyword-based, no LLM call needed."""

# Logistics and document-related terms. If a question contains NONE of these,
# it's likely out-of-scope.
LOGISTICS_KEYWORDS = {
    # Document terms
    "document", "doc", "bol", "bill of lading", "rate confirmation", "invoice",
    "shipment", "load", "order", "reference",
    # Entities
    "shipper", "consignee", "carrier", "driver", "dispatcher", "receiver",
    "sender", "customer", "vendor",
    # Logistics terms
    "pickup", "delivery", "drop", "stop", "route", "transit",
    "freight", "cargo", "commodity", "weight", "units", "quantity",
    "equipment", "trailer", "truck", "flatbed", "dry van", "reefer",
    "ftl", "ltl", "mode",
    # Financial
    "rate", "charge", "cost", "price", "pay", "total", "amount",
    "currency", "usd", "invoice", "billing", "cod",
    # Dates/times
    "date", "time", "when", "schedule", "appointment", "deadline",
    # Addresses
    "address", "location", "city", "state", "zip", "where",
    # Document queries
    "what", "who", "how much", "how many", "which", "list", "show",
    "find", "tell", "extract", "id", "number", "name", "po",
    # General document interaction
    "page", "section", "field", "detail", "information", "data",
    "mention", "say", "state", "contain", "include", "describe",
}


def is_in_scope(question: str) -> bool:
    """Check if a question is related to logistics/document content."""
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in LOGISTICS_KEYWORDS)
```

- [ ] **Step 6: Verify prompts load**

Run:
```python
python -c "
from app.llm.prompts.system import *
from app.llm.prompts.qa import *
from app.llm.prompts.extraction.bol import *
from app.llm.prompts.extraction.rate_confirm import *
from app.llm.prompts.extraction.generic import *
from app.llm.prompts.registry import registry
print([t['name'] for t in registry.list_templates()])
"
```
Expected: `['system', 'classification', 'query_rewrite', 'qa_with_cot', 'extraction_bol', 'extraction_rc', 'extraction_generic']`

- [ ] **Step 7: Commit**

```bash
git add app/llm/prompts/
git commit -m "feat: add prompt registry with versioned templates for Q&A, extraction, and guardrails"
```

---

### Task 6: Document Parser

**Files:**
- Create: `app/pipeline/__init__.py`
- Create: `app/pipeline/parser.py`
- Create: `tests/conftest.py`
- Create: `tests/test_parser.py`

- [ ] **Step 1: Create tests/conftest.py with shared fixtures**

```python
"""Shared test fixtures."""

import os
import pytest

SAMPLE_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "ultradoc_sample_test_data"
)


@pytest.fixture
def sample_bol_path():
    return os.path.join(SAMPLE_DATA_DIR, "BOL53657_billoflading.pdf")


@pytest.fixture
def sample_carrier_rc_path():
    return os.path.join(SAMPLE_DATA_DIR, "LD53657-Carrier-RC.pdf")


@pytest.fixture
def sample_shipper_rc_path():
    return os.path.join(SAMPLE_DATA_DIR, "LD53657-Shipper-RC.pdf")
```

- [ ] **Step 2: Write parser tests**

Create `tests/test_parser.py`:
```python
"""Tests for document parser."""

import pytest
from app.pipeline.parser import parse_document


def test_parse_pdf(sample_bol_path):
    result = parse_document(sample_bol_path)
    assert result["status"] == "success"
    assert result["page_count"] >= 1
    assert len(result["text"]) > 100
    assert "LD53657" in result["text"]


def test_parse_pdf_extracts_key_fields(sample_carrier_rc_path):
    result = parse_document(sample_carrier_rc_path)
    assert "SWIFT SHIFT LOGISTICS" in result["text"]
    assert "400" in result["text"]


def test_parse_txt(tmp_path):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Shipment ID: LD99999\nCarrier: Test Carrier")
    result = parse_document(str(txt_file))
    assert result["status"] == "success"
    assert "LD99999" in result["text"]
    assert result["page_count"] == 1


def test_parse_unsupported_format(tmp_path):
    bad_file = tmp_path / "test.xyz"
    bad_file.write_text("data")
    result = parse_document(str(bad_file))
    assert result["status"] == "error"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /opt/CodeRepo/ultradoc-intelligence && python -m pytest tests/test_parser.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipeline.parser'`

- [ ] **Step 4: Create app/pipeline/__init__.py**

```python
"""Document processing pipeline — parse, classify, chunk, retrieve, generate."""
```

- [ ] **Step 5: Create app/pipeline/parser.py**

```python
"""Document parser — PDF (pdfplumber), DOCX (python-docx), TXT."""

import os
from typing import Optional

import pdfplumber

from app.observability.logger import get_logger

logger = get_logger("parser")


def parse_document(file_path: str) -> dict:
    """Parse a document and return extracted text.

    Returns:
        dict with keys: text, page_count, status, error (if any)
    """
    ext = os.path.splitext(file_path)[1].lower()

    parsers = {
        ".pdf": _parse_pdf,
        ".docx": _parse_docx,
        ".txt": _parse_txt,
    }

    parser = parsers.get(ext)
    if parser is None:
        return {
            "text": "",
            "page_count": 0,
            "status": "error",
            "error": f"Unsupported file format: {ext}",
        }

    try:
        return parser(file_path)
    except Exception as e:
        logger.error(f"Parse failed for {file_path}: {e}")
        return {
            "text": "",
            "page_count": 0,
            "status": "error",
            "error": str(e),
        }


def _parse_pdf(file_path: str) -> dict:
    """Parse PDF using pdfplumber with table-aware extraction."""
    all_text_parts = []

    with pdfplumber.open(file_path) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            page_text_parts = []

            # Extract tables first
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    table_text = _format_table(table)
                    if table_text.strip():
                        page_text_parts.append(table_text)

            # Extract remaining text (non-table areas)
            text = page.extract_text() or ""
            if text.strip():
                page_text_parts.append(text)

            all_text_parts.append("\n".join(page_text_parts))

    full_text = "\n\n--- Page Break ---\n\n".join(all_text_parts)

    return {
        "text": full_text,
        "page_count": page_count,
        "status": "success",
    }


def _format_table(table: list[list]) -> str:
    """Format a table as readable text with key-value alignment."""
    if not table:
        return ""

    rows = []
    for row in table:
        cells = [str(cell).strip() if cell else "" for cell in row]
        rows.append(" | ".join(cells))

    return "\n".join(rows)


def _parse_docx(file_path: str) -> dict:
    """Parse DOCX using python-docx."""
    from docx import Document

    doc = Document(file_path)
    paragraphs = []

    for para in doc.paragraphs:
        if para.text.strip():
            paragraphs.append(para.text)

    # Extract tables
    for table in doc.tables:
        table_rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            table_rows.append(" | ".join(cells))
        paragraphs.append("\n".join(table_rows))

    return {
        "text": "\n\n".join(paragraphs),
        "page_count": 1,  # DOCX doesn't have page concept in python-docx
        "status": "success",
    }


def _parse_txt(file_path: str) -> dict:
    """Parse plain text file."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    return {
        "text": text,
        "page_count": 1,
        "status": "success",
    }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_parser.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add app/pipeline/__init__.py app/pipeline/parser.py tests/
git commit -m "feat: add document parser with PDF table extraction, DOCX, and TXT support"
```

---

### Task 7: Document Classifier

**Files:**
- Create: `app/pipeline/classifier.py`

- [ ] **Step 1: Create app/pipeline/classifier.py**

```python
"""Document type classification using LLM."""

from app.llm.provider import get_provider
from app.llm.prompts.system import *  # registers prompts
from app.llm.prompts.registry import registry
from app.models.schemas import DocType
from app.observability.tracer import Tracer


def classify_document(text: str, tracer: Tracer) -> DocType:
    """Classify a logistics document by type using LLM."""
    # Use first 2000 chars — enough to identify doc type without burning tokens
    text_preview = text[:2000]

    prompt = registry.get("classification")
    rendered = prompt.render(document_text=text_preview)

    provider = get_provider()

    with tracer.span("classification") as span:
        response = provider.generate(
            messages=[{"role": "user", "content": rendered}],
            model=provider.fast_model if hasattr(provider, "fast_model") else None,
            max_tokens=50,
        )
        span.metadata = {
            "tokens_in": response.tokens_in,
            "tokens_out": response.tokens_out,
            "model": response.model_name,
        }

    result = response.content.strip().lower()

    type_map = {
        "bill_of_lading": DocType.BOL,
        "rate_confirmation": DocType.RATE_CONFIRMATION,
        "invoice": DocType.INVOICE,
        "unknown": DocType.UNKNOWN,
    }

    return type_map.get(result, DocType.UNKNOWN)
```

- [ ] **Step 2: Commit**

```bash
git add app/pipeline/classifier.py
git commit -m "feat: add LLM-based document type classifier"
```

---

### Task 8: Section-Based Chunker

**Files:**
- Create: `app/pipeline/chunker.py`
- Create: `tests/test_chunker.py`

- [ ] **Step 1: Write chunker tests**

Create `tests/test_chunker.py`:
```python
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
    # Table row should not be split across chunks
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_chunker.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Create app/pipeline/chunker.py**

```python
"""Section-based document chunker that keeps tables intact."""

import re


# Common section headers in logistics documents
SECTION_PATTERNS = [
    r"^(Bill of Lading|BOL)\s*$",
    r"^(Carrier|Customer)\s+(Details|Information)",
    r"^(Carrier|Customer)\s+Rate\s+and\s+Load\s+Confirmation",
    r"^(Shipper|Consignee|3rd Party Billing|Transportation Company)",
    r"^(Stops|Stop\s+\d+|Pickup|Drop|Delivery)",
    r"^(Rate\s+Breakdown|Carrier\s+Pay|Rate\s+Details)",
    r"^(Standing\s+Instructions|Special\s+Instructions)",
    r"^(Shipper\s*&?\s*Carrier\s+Instructions)",
    r"^(Driver\s+Details|Driver\s+Information)",
    r"^(Notes|Comments|Description|Commodity)",
    r"^(#\s+Of\s+Units|Weight|COD\s+Value)",
    r"^(Consignor|Consignee)\s+name",
    r"^(Test\s+RC\s+Instructions)",
    r"^---\s*Page\s*Break\s*---$",
]

SECTION_REGEX = re.compile("|".join(SECTION_PATTERNS), re.IGNORECASE | re.MULTILINE)


def chunk_document(text: str, max_chunk_size: int = 1000) -> list[dict]:
    """Split document into section-based chunks.

    Each chunk has:
        - text: the chunk content
        - section: detected section name (or 'content')
        - index: chunk position index
    """
    if not text.strip():
        return []

    sections = _split_into_sections(text)

    chunks = []
    for section_name, section_text in sections:
        if not section_text.strip():
            continue

        # If section is small enough, keep it as one chunk
        if len(section_text) <= max_chunk_size:
            chunks.append({
                "text": section_text.strip(),
                "section": section_name,
                "index": len(chunks),
            })
        else:
            # Split large sections by paragraph, keeping tables intact
            sub_chunks = _split_section(section_text, max_chunk_size)
            for sub in sub_chunks:
                if sub.strip():
                    chunks.append({
                        "text": sub.strip(),
                        "section": section_name,
                        "index": len(chunks),
                    })

    return chunks


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split text into (section_name, section_text) pairs."""
    lines = text.split("\n")
    sections = []
    current_section = "content"
    current_lines = []

    for line in lines:
        # Check if this line is a section header
        match = SECTION_REGEX.search(line.strip())
        if match and len(line.strip()) < 80:  # Headers are typically short
            # Save previous section
            if current_lines:
                sections.append((current_section, "\n".join(current_lines)))
                current_lines = []
            current_section = _normalize_section_name(line.strip())
            current_lines.append(line)
        else:
            current_lines.append(line)

    # Don't forget the last section
    if current_lines:
        sections.append((current_section, "\n".join(current_lines)))

    return sections


def _normalize_section_name(header: str) -> str:
    """Normalize section header to a clean name."""
    # Remove special characters and lowercase
    name = re.sub(r"[^a-zA-Z0-9\s]", "", header).strip().lower()
    # Collapse whitespace
    name = re.sub(r"\s+", "_", name)
    return name or "content"


def _split_section(text: str, max_size: int) -> list[str]:
    """Split a large section into smaller chunks at paragraph boundaries."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_size:
            current = current + "\n\n" + para if current else para
        else:
            if current:
                chunks.append(current)
            current = para

    if current:
        chunks.append(current)

    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_chunker.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/chunker.py tests/test_chunker.py
git commit -m "feat: add section-based chunker that preserves table integrity"
```

---

### Task 9: Storage — ChromaDB Wrapper + Cache

**Files:**
- Create: `app/storage/__init__.py`
- Create: `app/storage/vector_store.py`
- Create: `app/storage/cache.py`

- [ ] **Step 1: Create app/storage/__init__.py**

```python
"""Storage layer — ChromaDB vector store and document cache."""

from app.storage.vector_store import VectorStore
from app.storage.cache import DocumentCache
```

- [ ] **Step 2: Create app/storage/vector_store.py**

```python
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

    def add_chunks(
        self,
        doc_id: str,
        chunks: list[dict],
        embeddings: list[list[float]],
    ):
        """Store chunks with their embeddings, keyed by doc_id."""
        ids = [f"{doc_id}_chunk_{c['index']}" for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {"doc_id": doc_id, "section": c["section"], "index": c["index"]}
            for c in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(f"Stored {len(chunks)} chunks for doc {doc_id}")

    def query(
        self,
        query_embedding: list[float],
        doc_id: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Query chunks for a specific document."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            where={"doc_id": doc_id},
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                # ChromaDB returns distances; convert to similarity for cosine
                distance = results["distances"][0][i]
                similarity = 1 - distance  # cosine distance to similarity
                chunks.append({
                    "text": doc,
                    "section": results["metadatas"][0][i].get("section", ""),
                    "index": results["metadatas"][0][i].get("index", 0),
                    "similarity": similarity,
                })

        return chunks

    def delete_document(self, doc_id: str):
        """Delete all chunks for a document."""
        self.collection.delete(where={"doc_id": doc_id})

    def has_document(self, doc_id: str) -> bool:
        """Check if a document exists in the store."""
        results = self.collection.get(where={"doc_id": doc_id}, limit=1)
        return len(results["ids"]) > 0
```

- [ ] **Step 3: Create app/storage/cache.py**

```python
"""In-memory document cache for full text, chunks, and metadata."""

from typing import Optional
from app.models.schemas import DocType


class DocumentRecord:
    def __init__(
        self,
        doc_id: str,
        file_name: str,
        full_text: str,
        doc_type: DocType,
        chunks: list[dict],
        page_count: int,
    ):
        self.doc_id = doc_id
        self.file_name = file_name
        self.full_text = full_text
        self.doc_type = doc_type
        self.chunks = chunks
        self.page_count = page_count


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
            {
                "doc_id": r.doc_id,
                "file_name": r.file_name,
                "doc_type": r.doc_type.value,
            }
            for r in self._cache.values()
        ]

    def delete(self, doc_id: str):
        self._cache.pop(doc_id, None)
```

- [ ] **Step 4: Verify storage modules load**

Run: `python -c "from app.storage import VectorStore, DocumentCache; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add app/storage/
git commit -m "feat: add ChromaDB vector store wrapper and in-memory document cache"
```

---

### Task 10: Embedder

**Files:**
- Create: `app/pipeline/embedder.py`

- [ ] **Step 1: Create app/pipeline/embedder.py**

```python
"""Embed document chunks and store in vector database."""

from app.llm.provider import get_provider
from app.storage.vector_store import VectorStore
from app.observability.tracer import Tracer


def embed_and_store(doc_id: str, chunks: list[dict], tracer: Tracer) -> int:
    """Embed chunks and store in ChromaDB.

    Returns: number of chunks stored.
    """
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
```

- [ ] **Step 2: Commit**

```bash
git add app/pipeline/embedder.py
git commit -m "feat: add embedder to vectorize and store document chunks"
```

---

### Task 11: Hybrid Retriever (Vector + BM25 + Metadata + RRF)

**Files:**
- Create: `app/pipeline/retriever.py`
- Create: `tests/test_retriever.py`

- [ ] **Step 1: Write retriever tests**

Create `tests/test_retriever.py`:
```python
"""Tests for hybrid retriever and RRF fusion."""

from app.pipeline.retriever import rrf_fuse, bm25_search, metadata_filter


def test_rrf_fuse_basic():
    list_a = ["chunk_1", "chunk_2", "chunk_3"]
    list_b = ["chunk_2", "chunk_1", "chunk_4"]
    result = rrf_fuse([list_a, list_b], k=60)
    # chunk_1 and chunk_2 should be top since they appear in both
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
    # Should return chunks related to carrier or rate
    texts = [r["text"] for r in results]
    assert any("Carrier" in t or "Rate" in t for t in texts)


def test_metadata_filter():
    chunks = [
        {"text": "SWIFT SHIFT", "section": "carrier_details", "index": 0},
        {"text": "Los Angeles", "section": "stops", "index": 1},
        {"text": "400 USD", "section": "rate_breakdown", "index": 2},
    ]
    result = metadata_filter("What is the rate?", chunks)
    # Should match rate_breakdown section
    assert len(result) >= 1
    assert any(c["section"] == "rate_breakdown" for c in result)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_retriever.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Create app/pipeline/retriever.py**

```python
"""Hybrid retrieval: vector search + BM25 + metadata filter + RRF fusion."""

from collections import defaultdict
from typing import Optional

from rank_bm25 import BM25Okapi

from app.config import settings
from app.llm.provider import get_provider
from app.storage.vector_store import VectorStore
from app.storage.cache import DocumentCache
from app.observability.tracer import Tracer


# Section keywords for metadata filtering
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


def retrieve(
    question: str,
    doc_id: str,
    tracer: Tracer,
    mode: str = "hybrid",
    top_k: int = None,
) -> list[dict]:
    """Retrieve relevant chunks using hybrid retrieval.

    Args:
        question: user question
        doc_id: document ID
        tracer: request tracer
        mode: "hybrid", "vector", or "bm25"
        top_k: number of final results (default from config)

    Returns:
        List of chunks with similarity scores, sorted by relevance.
    """
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
            # Hybrid: all three signals + RRF
            vector_results = _vector_search(question, doc_id, settings.retrieval_top_k)
            bm25_results = bm25_search(question, record.chunks, settings.retrieval_top_k)
            meta_results = metadata_filter(question, record.chunks)

            span.metadata["vector_results"] = len(vector_results)
            span.metadata["bm25_results"] = len(bm25_results)
            span.metadata["metadata_results"] = len(meta_results)

            results = _hybrid_fuse(
                vector_results, bm25_results, meta_results, record.chunks
            )

        # Take top-k and compute best similarity
        results = results[:top_k]
        span.metadata["final_results"] = len(results)
        if results:
            span.metadata["best_similarity"] = results[0].get("similarity", 0)

    return results


def _vector_search(question: str, doc_id: str, top_k: int) -> list[dict]:
    """Search ChromaDB for semantically similar chunks."""
    provider = get_provider()
    store = VectorStore.get_instance()

    query_embedding = provider.embed([question])[0]
    return store.query(query_embedding, doc_id, top_k)


def bm25_search(question: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """BM25 keyword search over chunks."""
    if not chunks:
        return []

    corpus = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(corpus)
    query_tokens = question.lower().split()
    scores = bm25.get_scores(query_tokens)

    # Pair chunks with scores and sort
    scored = list(zip(chunks, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for chunk, score in scored[:top_k]:
        if score > 0:
            results.append({
                **chunk,
                "similarity": min(score / 10.0, 1.0),  # Normalize BM25 score
            })

    return results


def metadata_filter(question: str, chunks: list[dict]) -> list[dict]:
    """Filter chunks by matching section headers to question keywords."""
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
    """Reciprocal Rank Fusion to merge multiple ranked lists.

    Args:
        ranked_lists: list of ranked ID lists
        k: RRF constant (default 60)

    Returns:
        Merged ranked list of IDs.
    """
    if not ranked_lists:
        return []

    scores = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, item_id in enumerate(ranked_list):
            scores[item_id] += 1.0 / (k + rank + 1)

    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


def _hybrid_fuse(
    vector_results: list[dict],
    bm25_results: list[dict],
    meta_results: list[dict],
    all_chunks: list[dict],
) -> list[dict]:
    """Fuse results from all three retrieval signals using RRF."""
    # Build ranked lists of chunk indices
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

    # Build index -> best similarity lookup
    sim_lookup = {}
    for results in [vector_results, bm25_results, meta_results]:
        for r in results:
            idx = r["index"]
            if idx not in sim_lookup or r.get("similarity", 0) > sim_lookup[idx]:
                sim_lookup[idx] = r.get("similarity", 0)

    # Build final results
    chunk_lookup = {c["index"]: c for c in all_chunks}
    results = []
    for idx in fused_indices:
        if idx in chunk_lookup:
            chunk = chunk_lookup[idx].copy()
            chunk["similarity"] = sim_lookup.get(idx, 0.5)
            results.append(chunk)

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_retriever.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/retriever.py tests/test_retriever.py
git commit -m "feat: add hybrid retriever with vector, BM25, metadata filter, and RRF fusion"
```

---

### Task 12: Guardrails (3-Layer System)

**Files:**
- Create: `app/pipeline/guardrails.py`
- Create: `tests/test_guardrails.py`

- [ ] **Step 1: Write guardrail tests**

Create `tests/test_guardrails.py`:
```python
"""Tests for the 3-layer guardrail system."""

from app.pipeline.guardrails import (
    check_scope,
    check_retrieval_threshold,
    check_grounding,
)
from app.models.schemas import GuardrailStatus


def test_scope_in_scope():
    assert check_scope("What is the carrier rate?") == GuardrailStatus.PASSED


def test_scope_out_of_scope():
    assert check_scope("What is the weather today?") == GuardrailStatus.OUT_OF_SCOPE


def test_scope_edge_case_empty():
    assert check_scope("") == GuardrailStatus.OUT_OF_SCOPE


def test_retrieval_threshold_passes():
    chunks = [{"text": "rate is 400", "similarity": 0.8}]
    assert check_retrieval_threshold(chunks, 0.3) == GuardrailStatus.PASSED


def test_retrieval_threshold_fails():
    chunks = [{"text": "rate is 400", "similarity": 0.1}]
    assert check_retrieval_threshold(chunks, 0.3) == GuardrailStatus.NOT_FOUND


def test_retrieval_threshold_empty():
    assert check_retrieval_threshold([], 0.3) == GuardrailStatus.NOT_FOUND


def test_grounding_high_overlap():
    answer = "The carrier rate is 400 USD for flatbed"
    source = "Carrier Pay Flatbed: $400.00 USD Total: 400.00 USD"
    status, ratio = check_grounding(answer, source, 0.4)
    assert status == GuardrailStatus.PASSED
    assert ratio > 0.4


def test_grounding_low_overlap():
    answer = "The shipment will arrive on Mars next Tuesday via teleportation"
    source = "Pickup from Los Angeles Airport"
    status, ratio = check_grounding(answer, source, 0.4)
    assert status == GuardrailStatus.LOW_GROUNDING
    assert ratio < 0.4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_guardrails.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Create app/pipeline/guardrails.py**

```python
"""3-layer deterministic guardrail system."""

import re
from app.models.schemas import GuardrailStatus
from app.llm.prompts.guardrails import is_in_scope


def check_scope(question: str) -> GuardrailStatus:
    """Layer 1: Pre-retrieval out-of-scope check (no LLM call)."""
    if not question.strip():
        return GuardrailStatus.OUT_OF_SCOPE

    if is_in_scope(question):
        return GuardrailStatus.PASSED

    return GuardrailStatus.OUT_OF_SCOPE


def check_retrieval_threshold(
    chunks: list[dict], threshold: float
) -> GuardrailStatus:
    """Layer 2: Post-retrieval — check if best chunk meets similarity threshold."""
    if not chunks:
        return GuardrailStatus.NOT_FOUND

    best_similarity = max(c.get("similarity", 0) for c in chunks)
    if best_similarity < threshold:
        return GuardrailStatus.NOT_FOUND

    return GuardrailStatus.PASSED


def check_grounding(
    answer: str, source_text: str, threshold: float = 0.4
) -> tuple[GuardrailStatus, float]:
    """Layer 3: Post-generation — check answer-source token overlap.

    Returns:
        (status, overlap_ratio)
    """
    if not answer or not source_text:
        return GuardrailStatus.LOW_GROUNDING, 0.0

    # Tokenize: lowercase, split on non-alphanumeric, remove short tokens
    def tokenize(text):
        tokens = re.findall(r"\b\w+\b", text.lower())
        return set(t for t in tokens if len(t) > 2)

    answer_tokens = tokenize(answer)
    source_tokens = tokenize(source_text)

    if not answer_tokens:
        return GuardrailStatus.LOW_GROUNDING, 0.0

    overlap = answer_tokens & source_tokens
    ratio = len(overlap) / len(answer_tokens)

    if ratio < threshold:
        return GuardrailStatus.LOW_GROUNDING, round(ratio, 2)

    return GuardrailStatus.PASSED, round(ratio, 2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_guardrails.py -v`
Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/guardrails.py tests/test_guardrails.py
git commit -m "feat: add 3-layer deterministic guardrail system"
```

---

### Task 13: Q&A Generator + Confidence Scorer

**Files:**
- Create: `app/pipeline/generator.py`
- Create: `tests/test_confidence.py`

- [ ] **Step 1: Write confidence scoring tests**

Create `tests/test_confidence.py`:
```python
"""Tests for confidence scoring."""

from app.pipeline.generator import compute_confidence
from app.models.schemas import ConfidenceLevel


def test_confidence_high():
    result = compute_confidence(
        retrieval_score=0.9,
        grounding_ratio=0.8,
        llm_assessment="HIGH",
    )
    assert result.score > 0.7
    assert result.level == ConfidenceLevel.HIGH


def test_confidence_low():
    result = compute_confidence(
        retrieval_score=0.2,
        grounding_ratio=0.1,
        llm_assessment="LOW",
    )
    assert result.score < 0.4
    assert result.level == ConfidenceLevel.LOW


def test_confidence_medium():
    result = compute_confidence(
        retrieval_score=0.5,
        grounding_ratio=0.5,
        llm_assessment="MEDIUM",
    )
    assert 0.4 <= result.score <= 0.7
    assert result.level == ConfidenceLevel.MEDIUM


def test_confidence_breakdown_present():
    result = compute_confidence(0.9, 0.8, "HIGH")
    assert result.breakdown.retrieval > 0
    assert result.breakdown.grounding > 0
    assert result.breakdown.llm_assessment > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_confidence.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Create app/pipeline/generator.py**

```python
"""Q&A generation with hidden Chain-of-Thought and confidence scoring."""

import re
from typing import Optional

from app.config import settings
from app.llm.provider import get_provider, LLMResponse
from app.llm.prompts.system import *  # register prompts
from app.llm.prompts.qa import *  # register prompts
from app.llm.prompts.registry import registry
from app.models.schemas import (
    ConfidenceResult,
    ConfidenceBreakdown,
    ConfidenceLevel,
)
from app.observability.tracer import Tracer


def generate_answer(
    question: str,
    full_text: str,
    source_chunks: list[dict],
    tracer: Tracer,
) -> dict:
    """Generate an answer using full document context + retrieved chunks.

    Returns:
        dict with: answer, source_text, llm_confidence, tokens_in, tokens_out
    """
    prompt = registry.get("qa_with_cot")
    source_text = "\n\n".join(c["text"] for c in source_chunks[:3])

    rendered = prompt.render(
        document_text=full_text,
        source_chunks=source_text,
        question=question,
    )

    system_prompt = registry.get("system").template
    provider = get_provider()

    with tracer.span("generation") as span:
        response = provider.generate(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": rendered},
            ],
        )
        span.metadata = {
            "tokens_in": response.tokens_in,
            "tokens_out": response.tokens_out,
            "model": response.model_name,
        }

    # Parse the structured response
    parsed = _parse_cot_response(response.content)

    return {
        "answer": parsed["answer"],
        "source_text": source_text,
        "llm_confidence": parsed["confidence"],
        "section": parsed["section"],
        "tokens_in": response.tokens_in,
        "tokens_out": response.tokens_out,
    }


def rewrite_query(question: str, tracer: Tracer) -> str:
    """Rewrite a question to be more retrieval-friendly using fast model."""
    prompt = registry.get("query_rewrite")
    rendered = prompt.render(question=question)

    provider = get_provider()
    fast_model = getattr(provider, "fast_model", None)

    with tracer.span("query_rewrite") as span:
        response = provider.generate(
            messages=[{"role": "user", "content": rendered}],
            model=fast_model,
            max_tokens=200,
        )
        span.metadata = {
            "original": question,
            "rewritten": response.content.strip(),
            "tokens_in": response.tokens_in,
            "tokens_out": response.tokens_out,
            "model": response.model_name,
        }

    return response.content.strip()


def compute_confidence(
    retrieval_score: float,
    grounding_ratio: float,
    llm_assessment: str,
) -> ConfidenceResult:
    """Compute composite confidence from 3 signals."""
    # Map LLM assessment to numeric
    assessment_map = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.2}
    llm_score = assessment_map.get(llm_assessment.upper(), 0.4)

    # Weights: retrieval 40%, grounding 35%, LLM 25%
    composite = (
        0.40 * min(retrieval_score, 1.0)
        + 0.35 * min(grounding_ratio, 1.0)
        + 0.25 * llm_score
    )
    composite = round(composite, 2)

    # Determine level
    if composite > 0.7:
        level = ConfidenceLevel.HIGH
    elif composite >= 0.4:
        level = ConfidenceLevel.MEDIUM
    else:
        level = ConfidenceLevel.LOW

    return ConfidenceResult(
        score=composite,
        level=level,
        breakdown=ConfidenceBreakdown(
            retrieval=round(min(retrieval_score, 1.0), 2),
            grounding=round(min(grounding_ratio, 1.0), 2),
            llm_assessment=round(llm_score, 2),
        ),
    )


def _parse_cot_response(response_text: str) -> dict:
    """Parse the structured CoT response from the LLM."""
    section = ""
    answer = response_text  # fallback to full response
    confidence = "MEDIUM"  # default

    # Try to parse structured format
    section_match = re.search(r"SECTION:\s*(.+?)(?:\n|$)", response_text)
    answer_match = re.search(r"ANSWER:\s*(.+?)(?:\nCONFIDENCE:|$)", response_text, re.DOTALL)
    confidence_match = re.search(r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)", response_text, re.IGNORECASE)

    if section_match:
        section = section_match.group(1).strip()
    if answer_match:
        answer = answer_match.group(1).strip()
    if confidence_match:
        confidence = confidence_match.group(1).strip().upper()

    return {
        "section": section,
        "answer": answer,
        "confidence": confidence,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_confidence.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/generator.py tests/test_confidence.py
git commit -m "feat: add Q&A generator with hidden CoT and composite confidence scoring"
```

---

### Task 14: Structured Extractor

**Files:**
- Create: `app/pipeline/extractor.py`

- [ ] **Step 1: Create app/pipeline/extractor.py**

```python
"""Structured extraction using Pydantic schema + LLM function calling."""

import json

from app.llm.provider import get_provider
from app.llm.prompts.system import *  # register prompts
from app.llm.prompts.extraction.bol import *  # register prompts
from app.llm.prompts.extraction.rate_confirm import *  # register prompts
from app.llm.prompts.extraction.generic import *  # register prompts
from app.llm.prompts.registry import registry
from app.models.schemas import DocType
from app.models.extraction import ShipmentData
from app.observability.tracer import Tracer
from app.observability.logger import get_logger

logger = get_logger("extractor")

# Map doc types to prompt template names
DOC_TYPE_PROMPT_MAP = {
    DocType.BOL: "extraction_bol",
    DocType.RATE_CONFIRMATION: "extraction_rc",
    DocType.INVOICE: "extraction_generic",
    DocType.UNKNOWN: "extraction_generic",
}


def extract_shipment_data(
    full_text: str,
    doc_type: DocType,
    tracer: Tracer,
) -> dict:
    """Extract structured shipment data from document text.

    Returns:
        dict with: shipment_data (ShipmentData), completeness_score, tokens_in, tokens_out
    """
    # Select prompt based on doc type
    prompt_name = DOC_TYPE_PROMPT_MAP.get(doc_type, "extraction_generic")
    prompt = registry.get(prompt_name)
    rendered = prompt.render(document_text=full_text)

    system_prompt = registry.get("system").template
    provider = get_provider()

    with tracer.span("extraction") as span:
        response = provider.generate_structured(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": rendered},
            ],
            schema=ShipmentData.model_json_schema(),
        )
        span.metadata = {
            "tokens_in": response.tokens_in,
            "tokens_out": response.tokens_out,
            "model": response.model_name,
            "doc_type": doc_type.value,
            "prompt_template": prompt_name,
        }

    # Parse and validate with Pydantic
    try:
        raw = json.loads(response.content)
        shipment = ShipmentData.model_validate(raw)
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Extraction parse failed: {e}")
        shipment = ShipmentData()  # all nulls

    completeness = shipment.completeness_score()

    with tracer.span("extraction_validation") as span:
        null_fields = [
            f for f in shipment.model_fields
            if getattr(shipment, f) is None
        ]
        span.metadata = {
            "completeness_score": completeness,
            "null_fields": null_fields,
            "non_null_count": len(shipment.model_fields) - len(null_fields),
        }

    return {
        "shipment_data": shipment,
        "completeness_score": completeness,
        "tokens_in": response.tokens_in,
        "tokens_out": response.tokens_out,
    }
```

- [ ] **Step 2: Commit**

```bash
git add app/pipeline/extractor.py
git commit -m "feat: add structured extractor with Pydantic schema and doc-type-specific prompts"
```

---

### Task 15: FastAPI Endpoints

**Files:**
- Create: `app/main.py`

- [ ] **Step 1: Create app/main.py**

```python
"""FastAPI application with /upload, /ask, and /extract endpoints."""

import os
import uuid
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models.schemas import (
    AskRequest,
    AskResponse,
    ExtractRequest,
    ExtractResponse,
    UploadResponse,
    GuardrailStatus,
    DocType,
)
from app.pipeline.parser import parse_document
from app.pipeline.classifier import classify_document
from app.pipeline.chunker import chunk_document
from app.pipeline.embedder import embed_and_store
from app.pipeline.retriever import retrieve
from app.pipeline.generator import generate_answer, rewrite_query, compute_confidence
from app.pipeline.extractor import extract_shipment_data
from app.pipeline.guardrails import check_scope, check_retrieval_threshold, check_grounding
from app.storage.cache import DocumentCache, DocumentRecord
from app.observability.tracer import Tracer
from app.observability.logger import get_logger

logger = get_logger("main")

app = FastAPI(
    title="Ultra Doc-Intelligence",
    description="AI-powered logistics document Q&A system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a logistics document."""
    doc_id = str(uuid.uuid4())
    tracer = Tracer("/upload", doc_id)

    # Validate file type
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(400, f"Unsupported file type: {ext}. Use PDF, DOCX, or TXT.")

    # Save file
    os.makedirs(settings.upload_dir, exist_ok=True)
    file_path = os.path.join(settings.upload_dir, f"{doc_id}{ext}")
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Parse
    with tracer.span("parsing") as span:
        parse_result = parse_document(file_path)
        if parse_result["status"] != "success":
            raise HTTPException(422, f"Failed to parse document: {parse_result.get('error', 'unknown')}")
        span.metadata = {"page_count": parse_result["page_count"]}

    # Classify
    doc_type = classify_document(parse_result["text"], tracer)

    # Chunk
    with tracer.span("chunking") as span:
        chunks = chunk_document(parse_result["text"])
        span.metadata = {"chunk_count": len(chunks)}

    # Embed and store
    embed_and_store(doc_id, chunks, tracer)

    # Cache
    cache = DocumentCache.get_instance()
    cache.store(DocumentRecord(
        doc_id=doc_id,
        file_name=file.filename or "unknown",
        full_text=parse_result["text"],
        doc_type=doc_type,
        chunks=chunks,
        page_count=parse_result["page_count"],
    ))

    tracer.finish()

    return UploadResponse(
        doc_id=doc_id,
        doc_type=doc_type,
        page_count=parse_result["page_count"],
        chunk_count=len(chunks),
    )


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Ask a question about an uploaded document."""
    tracer = Tracer("/ask", request.doc_id)

    # Check document exists
    cache = DocumentCache.get_instance()
    record = cache.get(request.doc_id)
    if not record:
        raise HTTPException(404, f"Document {request.doc_id} not found. Upload it first.")

    # Guardrail 1: scope check
    with tracer.span("guardrail_scope") as span:
        scope_status = check_scope(request.question)
        span.metadata = {"status": scope_status.value}

    if scope_status == GuardrailStatus.OUT_OF_SCOPE:
        tracer.finish()
        return AskResponse(
            answer="This question does not appear to be related to the uploaded document.",
            source_text="",
            confidence=compute_confidence(0.0, 0.0, "LOW"),
            guardrail_status=GuardrailStatus.OUT_OF_SCOPE,
            query_rewritten=False,
            retrieval_mode=request.retrieval_mode,
        )

    # Optional query rewrite
    question = request.question
    query_rewritten = False
    if request.enable_query_rewrite:
        question = rewrite_query(request.question, tracer)
        query_rewritten = True
    else:
        tracer.skip("query_rewrite", "enable_query_rewrite=false")

    # Retrieve
    chunks = retrieve(
        question=question,
        doc_id=request.doc_id,
        tracer=tracer,
        mode=request.retrieval_mode,
    )

    # Guardrail 2: retrieval threshold
    with tracer.span("guardrail_threshold") as span:
        threshold_status = check_retrieval_threshold(chunks, request.confidence_threshold)
        span.metadata = {
            "status": threshold_status.value,
            "threshold": request.confidence_threshold,
            "best_similarity": chunks[0]["similarity"] if chunks else 0,
        }

    if threshold_status == GuardrailStatus.NOT_FOUND:
        tracer.finish()
        return AskResponse(
            answer="Not found in document. The information you're looking for does not appear to be in this document.",
            source_text="",
            confidence=compute_confidence(
                chunks[0]["similarity"] if chunks else 0.0, 0.0, "LOW"
            ),
            guardrail_status=GuardrailStatus.NOT_FOUND,
            query_rewritten=query_rewritten,
            retrieval_mode=request.retrieval_mode,
        )

    # Generate answer
    gen_result = generate_answer(
        question=question,
        full_text=record.full_text,
        source_chunks=chunks,
        tracer=tracer,
    )

    # Guardrail 3: grounding check
    with tracer.span("guardrail_grounding") as span:
        grounding_status, grounding_ratio = check_grounding(
            gen_result["answer"],
            gen_result["source_text"],
            settings.grounding_overlap_threshold,
        )
        span.metadata = {
            "status": grounding_status.value,
            "overlap_ratio": grounding_ratio,
        }

    # Compute confidence
    retrieval_score = chunks[0]["similarity"] if chunks else 0.0
    with tracer.span("confidence_scoring") as span:
        confidence = compute_confidence(
            retrieval_score=retrieval_score,
            grounding_ratio=grounding_ratio,
            llm_assessment=gen_result["llm_confidence"],
        )
        span.metadata = {
            "score": confidence.score,
            "level": confidence.level.value,
        }

    # Use worst guardrail status
    final_status = grounding_status if grounding_status != GuardrailStatus.PASSED else GuardrailStatus.PASSED

    tracer.finish()

    return AskResponse(
        answer=gen_result["answer"],
        source_text=gen_result["source_text"],
        confidence=confidence,
        guardrail_status=final_status,
        query_rewritten=query_rewritten,
        retrieval_mode=request.retrieval_mode,
    )


@app.post("/extract", response_model=ExtractResponse)
async def extract_data(request: ExtractRequest):
    """Extract structured shipment data from an uploaded document."""
    tracer = Tracer("/extract", request.doc_id)

    cache = DocumentCache.get_instance()
    record = cache.get(request.doc_id)
    if not record:
        raise HTTPException(404, f"Document {request.doc_id} not found. Upload it first.")

    result = extract_shipment_data(
        full_text=record.full_text,
        doc_type=record.doc_type,
        tracer=tracer,
    )

    tracer.finish()

    return ExtractResponse(
        extracted_data=result["shipment_data"].model_dump(),
        completeness_score=result["completeness_score"],
        doc_type=record.doc_type,
    )


@app.get("/documents")
def list_documents():
    """List all uploaded documents."""
    cache = DocumentCache.get_instance()
    return {"documents": cache.list_documents()}


@app.get("/traces")
def get_traces():
    """Get recent request traces."""
    return {"traces": Tracer.get_recent_traces()}
```

- [ ] **Step 2: Verify the app starts**

Run: `cd /opt/CodeRepo/ultradoc-intelligence && python -c "from app.main import app; print(app.title)"`
Expected: `Ultra Doc-Intelligence`

- [ ] **Step 3: Commit**

```bash
git add app/main.py
git commit -m "feat: add FastAPI endpoints for upload, ask, extract, documents, and traces"
```

---

### Task 16: Gradio UI

**Files:**
- Create: `ui/__init__.py`
- Create: `ui/gradio_app.py`

- [ ] **Step 1: Create ui/__init__.py**

```python
"""Gradio UI for Ultra Doc-Intelligence."""
```

- [ ] **Step 2: Create ui/gradio_app.py**

```python
"""Gradio UI with 4 tabs: Upload, Ask, Extract, Traces."""

import json
import gradio as gr
import httpx

API_BASE = "http://localhost:8000"


def get_doc_choices():
    """Fetch uploaded documents for dropdown."""
    try:
        resp = httpx.get(f"{API_BASE}/documents", timeout=5)
        docs = resp.json().get("documents", [])
        return {f"{d['file_name']} ({d['doc_id'][:8]}...)": d["doc_id"] for d in docs}
    except Exception:
        return {}


def upload_file(file):
    """Upload a document via API."""
    if file is None:
        return "No file selected.", gr.update(choices=[])

    try:
        with open(file.name, "rb") as f:
            resp = httpx.post(
                f"{API_BASE}/upload",
                files={"file": (file.name.split("/")[-1], f)},
                timeout=60,
            )

        if resp.status_code != 200:
            return f"Upload failed: {resp.text}", gr.update()

        data = resp.json()
        result = (
            f"**Upload Successful**\n\n"
            f"- **Doc ID:** `{data['doc_id']}`\n"
            f"- **Type:** {data['doc_type']}\n"
            f"- **Pages:** {data['page_count']}\n"
            f"- **Chunks:** {data['chunk_count']}"
        )

        # Refresh doc choices
        choices = get_doc_choices()
        choice_list = list(choices.keys())
        return result, gr.update(choices=choice_list, value=choice_list[-1] if choice_list else None)

    except Exception as e:
        return f"Error: {str(e)}", gr.update()


def ask_question(doc_label, question, enable_rewrite, retrieval_mode, threshold):
    """Ask a question about a document."""
    if not doc_label or not question:
        return "Please select a document and enter a question."

    choices = get_doc_choices()
    doc_id = choices.get(doc_label)
    if not doc_id:
        return "Document not found. Please re-upload."

    try:
        resp = httpx.post(
            f"{API_BASE}/ask",
            json={
                "doc_id": doc_id,
                "question": question,
                "enable_query_rewrite": enable_rewrite,
                "retrieval_mode": retrieval_mode,
                "confidence_threshold": threshold,
            },
            timeout=60,
        )

        if resp.status_code != 200:
            return f"Error: {resp.text}"

        data = resp.json()
        conf = data["confidence"]

        # Color-coded confidence
        level = conf["level"]
        color = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}.get(level, "gray")
        badge = f'<span style="color:{color};font-weight:bold">{level} ({conf["score"]})</span>'

        result = f"### Answer\n\n{data['answer']}\n\n"
        result += f"### Confidence: {badge}\n\n"
        result += f"| Signal | Score |\n|--------|-------|\n"
        result += f"| Retrieval | {conf['breakdown']['retrieval']} |\n"
        result += f"| Grounding | {conf['breakdown']['grounding']} |\n"
        result += f"| LLM Assessment | {conf['breakdown']['llm_assessment']} |\n\n"
        result += f"### Guardrail: `{data['guardrail_status']}`\n\n"

        if data.get("query_rewritten"):
            result += f"*Query was rewritten for better retrieval*\n\n"

        if data.get("source_text"):
            result += f"<details><summary>Source Text</summary>\n\n```\n{data['source_text']}\n```\n</details>"

        return result

    except Exception as e:
        return f"Error: {str(e)}"


def extract_data(doc_label):
    """Extract structured data from a document."""
    if not doc_label:
        return "Please select a document.", ""

    choices = get_doc_choices()
    doc_id = choices.get(doc_label)
    if not doc_id:
        return "Document not found.", ""

    try:
        resp = httpx.post(
            f"{API_BASE}/extract",
            json={"doc_id": doc_id},
            timeout=60,
        )

        if resp.status_code != 200:
            return f"Error: {resp.text}", ""

        data = resp.json()
        summary = (
            f"**Document Type:** {data['doc_type']}\n\n"
            f"**Completeness:** {data['completeness_score'] * 100:.0f}% fields extracted"
        )
        return summary, json.dumps(data["extracted_data"], indent=2)

    except Exception as e:
        return f"Error: {str(e)}", ""


def get_traces():
    """Fetch recent traces."""
    try:
        resp = httpx.get(f"{API_BASE}/traces", timeout=5)
        traces = resp.json().get("traces", [])
        if not traces:
            return "No traces yet. Upload a document or ask a question first."

        output = ""
        for trace in traces[:10]:
            output += f"### {trace['trace_id']} | `{trace['endpoint']}` | {trace['total_latency_ms']}ms | {trace['total_tokens']} tokens\n\n"
            for span in trace.get("spans", []):
                status_icon = {"success": "OK", "error": "ERR", "skipped": "SKIP"}.get(span["status"], "?")
                output += f"- **{span['stage']}**: {status_icon} ({span['latency_ms']}ms)"
                if span.get("metadata"):
                    meta_str = ", ".join(f"{k}={v}" for k, v in span["metadata"].items())
                    output += f" — {meta_str}"
                output += "\n"
            output += "\n---\n\n"

        return output

    except Exception as e:
        return f"Error fetching traces: {str(e)}"


def build_ui() -> gr.Blocks:
    """Build the Gradio UI."""
    with gr.Blocks(title="Ultra Doc-Intelligence", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Ultra Doc-Intelligence\n*AI-powered logistics document Q&A system*")

        with gr.Tabs():
            # Tab 1: Upload
            with gr.Tab("Upload"):
                file_input = gr.File(label="Upload Document (PDF, DOCX, TXT)")
                upload_btn = gr.Button("Upload & Process", variant="primary")
                upload_output = gr.Markdown()

            # Tab 2: Ask
            with gr.Tab("Ask"):
                with gr.Row():
                    doc_dropdown = gr.Dropdown(
                        label="Select Document",
                        choices=[],
                        interactive=True,
                    )
                    refresh_btn = gr.Button("Refresh", size="sm")

                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., What is the carrier rate?",
                )

                with gr.Row():
                    rewrite_toggle = gr.Checkbox(label="Enable Query Rewrite", value=False)
                    retrieval_dropdown = gr.Dropdown(
                        label="Retrieval Mode",
                        choices=["hybrid", "vector", "bm25"],
                        value="hybrid",
                    )
                    threshold_slider = gr.Slider(
                        label="Confidence Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.05,
                    )

                ask_btn = gr.Button("Ask", variant="primary")
                ask_output = gr.Markdown()

            # Tab 3: Extract
            with gr.Tab("Extract"):
                with gr.Row():
                    extract_doc_dropdown = gr.Dropdown(
                        label="Select Document",
                        choices=[],
                        interactive=True,
                    )
                    extract_refresh_btn = gr.Button("Refresh", size="sm")

                extract_btn = gr.Button("Run Extraction", variant="primary")
                extract_summary = gr.Markdown()
                extract_json = gr.Code(label="Extracted Data (JSON)", language="json")

            # Tab 4: Traces
            with gr.Tab("Traces"):
                traces_btn = gr.Button("Refresh Traces", variant="secondary")
                traces_output = gr.Markdown()

        # Wire up events
        upload_btn.click(
            fn=upload_file,
            inputs=[file_input],
            outputs=[upload_output, doc_dropdown],
        )

        def refresh_docs():
            choices = list(get_doc_choices().keys())
            return gr.update(choices=choices)

        refresh_btn.click(fn=refresh_docs, outputs=[doc_dropdown])
        extract_refresh_btn.click(fn=refresh_docs, outputs=[extract_doc_dropdown])

        ask_btn.click(
            fn=ask_question,
            inputs=[doc_dropdown, question_input, rewrite_toggle, retrieval_dropdown, threshold_slider],
            outputs=[ask_output],
        )

        extract_btn.click(
            fn=extract_data,
            inputs=[extract_doc_dropdown],
            outputs=[extract_summary, extract_json],
        )

        traces_btn.click(fn=get_traces, outputs=[traces_output])

    return demo
```

- [ ] **Step 3: Commit**

```bash
git add ui/
git commit -m "feat: add Gradio UI with Upload, Ask, Extract, and Traces tabs"
```

---

### Task 17: Entrypoint + Docker

**Files:**
- Create: `run.py`
- Create: `Dockerfile`
- Create: `docker-compose.yml`

- [ ] **Step 1: Create run.py**

```python
"""Single entrypoint — starts FastAPI + Gradio."""

import threading
import uvicorn
from app.main import app as fastapi_app
from ui.gradio_app import build_ui


def start_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="info")


def start_gradio():
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    # Start FastAPI in a background thread
    api_thread = threading.Thread(target=start_fastapi, daemon=True)
    api_thread.start()

    # Start Gradio in the main thread (it handles signals)
    start_gradio()
```

- [ ] **Step 2: Create Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/chroma data/uploads

# Expose ports
EXPOSE 8000 7860

# Start the application
CMD ["python", "run.py"]
```

- [ ] **Step 3: Create docker-compose.yml**

```yaml
version: "3.8"

services:
  app:
    build: .
    ports:
      - "8000:8000"
      - "7860:7860"
    env_file: .env
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

- [ ] **Step 4: Verify run.py imports work**

Run: `python -c "from app.main import app; from ui.gradio_app import build_ui; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add run.py Dockerfile docker-compose.yml
git commit -m "feat: add entrypoint, Dockerfile, and docker-compose for local deployment"
```

---

### Task 18: Eval Framework

**Files:**
- Create: `eval/ground_truth.json`
- Create: `eval/run_eval.py`
- Create: `eval/report.py`

- [ ] **Step 1: Create eval/ground_truth.json**

```json
[
  {
    "doc_file": "BOL53657_billoflading.pdf",
    "type": "qa",
    "question": "What is the load ID?",
    "expected_answer": "LD53657",
    "expected_source_contains": "Load ID"
  },
  {
    "doc_file": "BOL53657_billoflading.pdf",
    "type": "qa",
    "question": "Who is the shipper?",
    "expected_answer": "AAA",
    "expected_source_contains": "Shipper"
  },
  {
    "doc_file": "BOL53657_billoflading.pdf",
    "type": "qa",
    "question": "Who is the consignee?",
    "expected_answer": "xyz",
    "expected_source_contains": "Consignee"
  },
  {
    "doc_file": "BOL53657_billoflading.pdf",
    "type": "qa",
    "question": "What is the ship date?",
    "expected_answer": "02-08-2026",
    "expected_source_contains": "Ship Date"
  },
  {
    "doc_file": "BOL53657_billoflading.pdf",
    "type": "qa",
    "question": "What is the weight of the commodity?",
    "expected_answer": "56000",
    "expected_source_contains": "Weight"
  },
  {
    "doc_file": "BOL53657_billoflading.pdf",
    "type": "qa",
    "question": "What is the COD value?",
    "expected_answer": "64000",
    "expected_source_contains": "COD"
  },
  {
    "doc_file": "LD53657-Carrier-RC.pdf",
    "type": "qa",
    "question": "Who is the carrier?",
    "expected_answer": "SWIFT SHIFT LOGISTICS LLC",
    "expected_source_contains": "Carrier"
  },
  {
    "doc_file": "LD53657-Carrier-RC.pdf",
    "type": "qa",
    "question": "What is the carrier rate?",
    "expected_answer": "400",
    "expected_source_contains": "400"
  },
  {
    "doc_file": "LD53657-Carrier-RC.pdf",
    "type": "qa",
    "question": "What equipment type is being used?",
    "expected_answer": "Flatbed",
    "expected_source_contains": "Flatbed"
  },
  {
    "doc_file": "LD53657-Carrier-RC.pdf",
    "type": "qa",
    "question": "Who is the driver?",
    "expected_answer": "John Doe",
    "expected_source_contains": "Driver"
  },
  {
    "doc_file": "LD53657-Shipper-RC.pdf",
    "type": "qa",
    "question": "Who is the customer?",
    "expected_answer": "Test ABC",
    "expected_source_contains": "Customer"
  },
  {
    "doc_file": "LD53657-Shipper-RC.pdf",
    "type": "qa",
    "question": "What is the total shipper rate?",
    "expected_answer": "1000",
    "expected_source_contains": "1000"
  },
  {
    "doc_file": "BOL53657_billoflading.pdf",
    "type": "qa",
    "question": "What is the weather today?",
    "expected_answer": "__OUT_OF_SCOPE__",
    "expected_source_contains": ""
  },
  {
    "doc_file": "BOL53657_billoflading.pdf",
    "type": "qa",
    "question": "What is the insurance policy number?",
    "expected_answer": "__NOT_FOUND__",
    "expected_source_contains": ""
  },
  {
    "doc_file": "LD53657-Carrier-RC.pdf",
    "type": "extraction",
    "expected_fields": {
      "shipment_id": "LD53657",
      "shipper": "AAA",
      "consignee": "xyz",
      "equipment_type": "Flatbed",
      "mode": "FTL",
      "rate": 400.0,
      "currency": "USD",
      "carrier_name": "SWIFT SHIFT LOGISTICS LLC"
    }
  },
  {
    "doc_file": "BOL53657_billoflading.pdf",
    "type": "extraction",
    "expected_fields": {
      "shipment_id": "LD53657",
      "shipper": "AAA",
      "consignee": "xyz",
      "weight": "56000"
    }
  }
]
```

- [ ] **Step 2: Create eval/report.py**

```python
"""Evaluation metrics computation."""

from dataclasses import dataclass, field


@dataclass
class EvalReport:
    # Q&A
    total_qa: int = 0
    correct_answers: int = 0
    source_hits: int = 0
    confidence_sum: float = 0.0
    correct_refusals: int = 0
    false_refusals: int = 0
    false_acceptances: int = 0

    # Extraction
    total_extraction: int = 0
    field_matches: int = 0
    total_fields_checked: int = 0

    # Performance
    latencies: list = field(default_factory=list)
    token_counts: list = field(default_factory=list)

    @property
    def answer_accuracy(self) -> float:
        return self.correct_answers / self.total_qa if self.total_qa else 0.0

    @property
    def source_hit_rate(self) -> float:
        return self.source_hits / self.total_qa if self.total_qa else 0.0

    @property
    def avg_confidence(self) -> float:
        return self.confidence_sum / self.total_qa if self.total_qa else 0.0

    @property
    def field_accuracy(self) -> float:
        return self.field_matches / self.total_fields_checked if self.total_fields_checked else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    def to_dict(self) -> dict:
        return {
            "qa": {
                "total": self.total_qa,
                "answer_accuracy": round(self.answer_accuracy, 3),
                "source_hit_rate": round(self.source_hit_rate, 3),
                "avg_confidence": round(self.avg_confidence, 3),
                "correct_refusals": self.correct_refusals,
                "false_refusals": self.false_refusals,
                "false_acceptances": self.false_acceptances,
            },
            "extraction": {
                "total": self.total_extraction,
                "field_accuracy": round(self.field_accuracy, 3),
                "fields_checked": self.total_fields_checked,
                "fields_matched": self.field_matches,
            },
            "performance": {
                "avg_latency_ms": round(self.avg_latency_ms, 1),
                "total_requests": len(self.latencies),
            },
        }

    def print_report(self):
        d = self.to_dict()
        print("\n" + "=" * 50)
        print("EVALUATION REPORT")
        print("=" * 50)
        print(f"\nQ&A ({d['qa']['total']} questions):")
        print(f"  Answer Accuracy:   {d['qa']['answer_accuracy']:.1%}")
        print(f"  Source Hit Rate:   {d['qa']['source_hit_rate']:.1%}")
        print(f"  Avg Confidence:    {d['qa']['avg_confidence']:.2f}")
        print(f"  Correct Refusals:  {d['qa']['correct_refusals']}")
        print(f"  False Refusals:    {d['qa']['false_refusals']}")
        print(f"\nExtraction ({d['extraction']['total']} documents):")
        print(f"  Field Accuracy:    {d['extraction']['field_accuracy']:.1%}")
        print(f"  Fields Checked:    {d['extraction']['fields_checked']}")
        print(f"\nPerformance:")
        print(f"  Avg Latency:       {d['performance']['avg_latency_ms']:.0f}ms")
        print(f"  Total Requests:    {d['performance']['total_requests']}")
        print("=" * 50)
```

- [ ] **Step 3: Create eval/run_eval.py**

```python
"""Automated evaluation runner."""

import json
import os
import sys
import time
import argparse

import httpx

from eval.report import EvalReport

API_BASE = "http://localhost:8000"
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ultradoc_sample_test_data")


def run_evaluation(ground_truth_path: str, flags: dict = None):
    flags = flags or {}
    report = EvalReport()

    with open(ground_truth_path) as f:
        test_cases = json.load(f)

    # Upload all unique documents first
    doc_map = {}  # file_name -> doc_id
    unique_files = set(tc["doc_file"] for tc in test_cases)

    print(f"Uploading {len(unique_files)} documents...")
    for file_name in unique_files:
        file_path = os.path.join(SAMPLE_DIR, file_name)
        if not os.path.exists(file_path):
            print(f"  SKIP: {file_name} not found")
            continue

        with open(file_path, "rb") as f:
            resp = httpx.post(
                f"{API_BASE}/upload",
                files={"file": (file_name, f)},
                timeout=60,
            )

        if resp.status_code == 200:
            doc_map[file_name] = resp.json()["doc_id"]
            print(f"  OK: {file_name} -> {doc_map[file_name][:8]}...")
        else:
            print(f"  FAIL: {file_name}: {resp.text}")

    # Run test cases
    print(f"\nRunning {len(test_cases)} test cases...")
    for i, tc in enumerate(test_cases):
        doc_id = doc_map.get(tc["doc_file"])
        if not doc_id:
            print(f"  [{i+1}] SKIP: doc not uploaded")
            continue

        if tc["type"] == "qa":
            _run_qa_case(tc, doc_id, report, flags, i + 1)
        elif tc["type"] == "extraction":
            _run_extraction_case(tc, doc_id, report, i + 1)

    report.print_report()
    return report


def _run_qa_case(tc: dict, doc_id: str, report: EvalReport, flags: dict, idx: int):
    report.total_qa += 1
    start = time.perf_counter()

    try:
        resp = httpx.post(
            f"{API_BASE}/ask",
            json={
                "doc_id": doc_id,
                "question": tc["question"],
                "enable_query_rewrite": flags.get("enable_query_rewrite", False),
                "retrieval_mode": flags.get("retrieval_mode", "hybrid"),
            },
            timeout=60,
        )
        latency = (time.perf_counter() - start) * 1000
        report.latencies.append(latency)

        if resp.status_code != 200:
            print(f"  [{idx}] ERROR: {resp.status_code}")
            return

        data = resp.json()
        answer = data["answer"]
        expected = tc["expected_answer"]
        guardrail = data["guardrail_status"]

        # Check special cases
        if expected == "__OUT_OF_SCOPE__":
            if guardrail == "out_of_scope":
                report.correct_refusals += 1
                report.correct_answers += 1
                print(f"  [{idx}] PASS (correct refusal): {tc['question'][:40]}...")
            else:
                report.false_acceptances += 1
                print(f"  [{idx}] FAIL (should have refused): {tc['question'][:40]}...")
            return

        if expected == "__NOT_FOUND__":
            if guardrail == "not_found" or "not found" in answer.lower():
                report.correct_refusals += 1
                report.correct_answers += 1
                print(f"  [{idx}] PASS (correct not-found): {tc['question'][:40]}...")
            else:
                report.false_acceptances += 1
                print(f"  [{idx}] FAIL (should have said not found): {tc['question'][:40]}...")
            return

        # Check answer contains expected
        if expected.lower() in answer.lower():
            report.correct_answers += 1
            print(f"  [{idx}] PASS: {tc['question'][:40]}...")
        else:
            print(f"  [{idx}] FAIL: expected '{expected}' in answer '{answer[:60]}...'")

        # Check source text
        source_keyword = tc.get("expected_source_contains", "")
        if source_keyword and source_keyword.lower() in data.get("source_text", "").lower():
            report.source_hits += 1

        # Track confidence
        report.confidence_sum += data["confidence"]["score"]

    except Exception as e:
        print(f"  [{idx}] ERROR: {e}")


def _run_extraction_case(tc: dict, doc_id: str, report: EvalReport, idx: int):
    report.total_extraction += 1

    try:
        resp = httpx.post(
            f"{API_BASE}/extract",
            json={"doc_id": doc_id},
            timeout=60,
        )
        report.latencies.append(0)  # extraction latency not critical for eval

        if resp.status_code != 200:
            print(f"  [{idx}] EXTRACT ERROR: {resp.status_code}")
            return

        extracted = resp.json()["extracted_data"]
        expected = tc["expected_fields"]

        for field_name, expected_val in expected.items():
            report.total_fields_checked += 1
            actual = extracted.get(field_name)

            if actual is None:
                print(f"  [{idx}] EXTRACT MISS: {field_name} (got null)")
                continue

            # Fuzzy match: check if expected value is contained in actual
            actual_str = str(actual).lower()
            expected_str = str(expected_val).lower()

            if expected_str in actual_str:
                report.field_matches += 1
                print(f"  [{idx}] EXTRACT PASS: {field_name}")
            else:
                print(f"  [{idx}] EXTRACT FAIL: {field_name} expected='{expected_val}' got='{actual}'")

    except Exception as e:
        print(f"  [{idx}] EXTRACT ERROR: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ultra Doc-Intelligence evaluation")
    parser.add_argument(
        "--ground-truth",
        default="eval/ground_truth.json",
        help="Path to ground truth JSON",
    )
    parser.add_argument(
        "--flags",
        default="{}",
        help='JSON string of flags (e.g., \'{"enable_query_rewrite": true}\')',
    )
    args = parser.parse_args()

    flags = json.loads(args.flags)
    run_evaluation(args.ground_truth, flags)
```

- [ ] **Step 4: Commit**

```bash
git add eval/
git commit -m "feat: add evaluation framework with ground truth, runner, and metrics report"
```

---

### Task 19: Integration Test — End-to-End Smoke Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Create tests/test_integration.py**

```python
"""End-to-end integration test — starts server, uploads doc, asks question, extracts."""

import time
import threading
import pytest
import httpx
import uvicorn
import os

from app.main import app

SAMPLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "ultradoc_sample_test_data"
)
BASE_URL = "http://localhost:8765"


@pytest.fixture(scope="module")
def server():
    """Start FastAPI server in background."""
    config = uvicorn.Config(app, host="127.0.0.1", port=8765, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # Wait for server to be ready
    for _ in range(20):
        try:
            httpx.get(f"{BASE_URL}/health", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    yield
    server.should_exit = True


@pytest.fixture(scope="module")
def uploaded_doc_id(server):
    """Upload the carrier RC and return doc_id."""
    file_path = os.path.join(SAMPLE_DIR, "LD53657-Carrier-RC.pdf")
    with open(file_path, "rb") as f:
        resp = httpx.post(
            f"{BASE_URL}/upload",
            files={"file": ("LD53657-Carrier-RC.pdf", f)},
            timeout=60,
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["chunk_count"] > 0
    return data["doc_id"]


def test_health(server):
    resp = httpx.get(f"{BASE_URL}/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_upload(uploaded_doc_id):
    assert uploaded_doc_id is not None


def test_ask_question(uploaded_doc_id):
    resp = httpx.post(
        f"{BASE_URL}/ask",
        json={
            "doc_id": uploaded_doc_id,
            "question": "Who is the carrier?",
        },
        timeout=60,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "SWIFT SHIFT" in data["answer"].upper()
    assert data["confidence"]["score"] > 0
    assert data["guardrail_status"] == "passed"


def test_ask_out_of_scope(uploaded_doc_id):
    resp = httpx.post(
        f"{BASE_URL}/ask",
        json={
            "doc_id": uploaded_doc_id,
            "question": "What is the weather today?",
        },
        timeout=60,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["guardrail_status"] == "out_of_scope"


def test_extract(uploaded_doc_id):
    resp = httpx.post(
        f"{BASE_URL}/extract",
        json={"doc_id": uploaded_doc_id},
        timeout=60,
    )
    assert resp.status_code == 200
    data = resp.json()
    extracted = data["extracted_data"]
    assert extracted["shipment_id"] is not None
    assert data["completeness_score"] > 0.5


def test_traces(server):
    resp = httpx.get(f"{BASE_URL}/traces")
    assert resp.status_code == 200
    traces = resp.json()["traces"]
    assert len(traces) > 0
```

- [ ] **Step 2: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: add end-to-end integration tests"
```

---

### Task 20: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create README.md**

```markdown
# Ultra Doc-Intelligence

AI-powered logistics document Q&A system that allows users to upload logistics documents and interact with them using natural language. Built as a TMS (Transportation Management System) AI assistant.

## Features

- **Document Upload & Processing** — PDF, DOCX, TXT with table-aware parsing
- **Natural Language Q&A** — Ask questions, get grounded answers with source text and confidence scores
- **Structured Extraction** — Extract shipment data (shipper, consignee, rate, dates, etc.) as JSON
- **3-Layer Guardrails** — Deterministic hallucination prevention
- **Hybrid Retrieval** — Vector search + BM25 + metadata filtering with RRF fusion
- **Observability** — Structured traces with per-stage latency and token tracking
- **Evaluation Framework** — Ground truth testing with accuracy metrics

## Architecture

```
User → Gradio UI → FastAPI API → Pipeline Orchestrator
                                   ├── Parser (pdfplumber + fallback)
                                   ├── Classifier (LLM-based doc type detection)
                                   ├── Chunker (section-based, tables intact)
                                   ├── Embedder (Azure OpenAI → ChromaDB)
                                   ├── Retriever (hybrid: vector + BM25 + metadata + RRF)
                                   ├── Generator (full-context LLM + hidden CoT)
                                   ├── Extractor (Pydantic + function calling)
                                   └── Guardrails (3 deterministic layers)
```

### Chunking Strategy

Section-based chunking that detects logistics document headers (Carrier Details, Stops, Rate Breakdown, etc.) and keeps tables intact within chunks. This preserves tabular structure that naive character-splitting destroys.

### Retrieval Method

**Hybrid retrieval** combining three signals:
1. **Vector search** (ChromaDB cosine similarity) — semantic matching
2. **BM25 keyword search** — exact term matching for IDs, names, numbers
3. **Metadata filtering** — section header matching for field-specific questions

Results merged via **Reciprocal Rank Fusion (RRF)** for robust ranking.

Optional **query rewriting** (A/B testable flag) reformulates casual questions for better retrieval.

### Guardrails Approach

Three deterministic layers — no extra LLM calls:
1. **Pre-retrieval**: Keyword-based out-of-scope detection
2. **Post-retrieval**: Similarity threshold check (configurable)
3. **Post-generation**: Token overlap grounding check

### Confidence Scoring

Composite score from 3 signals:
| Signal | Weight | Type |
|--------|--------|------|
| Retrieval similarity | 40% | Deterministic |
| Answer-source token overlap | 35% | Deterministic |
| LLM self-assessment | 25% | LLM-derived |

Each response includes the full breakdown for debuggability.

### Failure Cases

- Scanned/image PDFs with no text layer (fallback: LLM vision)
- Documents > 50 pages (context window limit)
- Multi-document queries not supported
- Questions requiring cross-document reasoning

### Improvement Ideas

- Langfuse/Langsmith for production prompt management and tracing
- Reranker model (Cohere/cross-encoder) after hybrid retrieval
- Multi-document queries and cross-doc comparison
- Fine-tuned extraction model for logistics fields
- OCR pipeline for scanned documents
- GraphRAG for entity relationship extraction

## Quick Start

### Local (Docker)

```bash
# Clone and configure
git clone <repo-url>
cd ultradoc-intelligence
cp .env.example .env  # Add your Azure OpenAI credentials

# Run with Docker
docker-compose up --build

# Access
# API: http://localhost:8000
# UI:  http://localhost:7860
```

### Local (Python)

```bash
pip install -r requirements.txt
python run.py
```

### API Endpoints

```bash
# Upload a document
curl -X POST http://localhost:8000/upload -F "file=@document.pdf"

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "...", "question": "What is the carrier rate?"}'

# Extract structured data
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "..."}'
```

### Run Evaluation

```bash
# Start the server first, then:
python -m eval.run_eval

# Compare with query rewriting:
python -m eval.run_eval --flags '{"enable_query_rewrite": true}'
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI, Python 3.11 |
| LLM | Azure OpenAI (gpt-4o, gpt-4.1-mini) |
| Embeddings | text-embedding-3-small |
| Vector Store | ChromaDB |
| PDF Parsing | pdfplumber |
| BM25 Search | rank_bm25 |
| Validation | Pydantic v2 |
| UI | Gradio |
| Deployment | Docker, HuggingFace Spaces |
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add comprehensive README with architecture, strategies, and usage"
```

---

## Task Dependency Graph

```
Task 1 (Setup)
  └── Task 2 (Models)
  └── Task 3 (Observability)
  └── Task 4 (LLM Provider)
       └── Task 5 (Prompts)
            └── Task 7 (Classifier)
            └── Task 13 (Generator)
            └── Task 14 (Extractor)
  └── Task 6 (Parser)
  └── Task 8 (Chunker)
  └── Task 9 (Storage)
       └── Task 10 (Embedder)
       └── Task 11 (Retriever)
  └── Task 12 (Guardrails)

Tasks 2-14 → Task 15 (FastAPI Endpoints)
Task 15 → Task 16 (Gradio UI)
Task 15 + 16 → Task 17 (Entrypoint + Docker)
Task 15 → Task 18 (Eval Framework)
Task 15 → Task 19 (Integration Tests)
Task 17 → Task 20 (README)
```

## Parallel Execution Opportunities

These task groups can be built in parallel by separate agents:

- **Group A (Tasks 2, 3)**: Models + Observability — no dependencies on each other
- **Group B (Tasks 6, 8, 12)**: Parser + Chunker + Guardrails — independent modules
- **Group C (Tasks 9, 10, 11)**: Storage + Embedder + Retriever — sequential but isolated from Groups A/B
- **Group D (Tasks 18, 20)**: Eval + README — can be written after Task 15
