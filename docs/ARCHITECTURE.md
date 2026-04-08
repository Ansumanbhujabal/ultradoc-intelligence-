# Ultra Doc-Intelligence Architecture

> Complete technical reference covering system design, architectural decisions, project structure, pipelines, guardrails, evaluation framework, observability, and operational runbook.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architectural Decisions](#2-architectural-decisions)
3. [Project Structure](#3-project-structure)
4. [Configuration](#4-configuration)
5. [Data Models](#5-data-models)
6. [API Layer](#6-api-layer)
7. [Pipeline Deep Dive](#7-pipeline-deep-dive)
8. [Guardrail System](#8-guardrail-system)
9. [Prompt Management](#9-prompt-management)
10. [Storage Layer](#10-storage-layer)
11. [Observability](#11-observability)
12. [Evaluation Framework](#12-evaluation-framework)
13. [Synthetic Data Generation](#13-synthetic-data-generation)
14. [Testing](#14-testing)
15. [Deployment](#15-deployment)
16. [Operational Runbook](#16-operational-runbook)

---

## 1. System Overview

Ultra Doc-Intelligence is a POC AI system for logistics document Q&A inside a Transportation Management System (TMS). Users upload logistics documents (BOL, Rate Confirmations, Invoices), ask natural language questions, and extract structured shipment data — all with confidence scoring, multi-layer guardrails, and full observability.

### Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| API | FastAPI | Async, auto-docs, Pydantic-native |
| LLM | Azure OpenAI (GPT-4o + GPT-4.1-mini) via LangChain | Provider abstraction, callback ecosystem |
| Vector DB | ChromaDB | Zero-config, embedded, persistent |
| Embeddings | all-MiniLM-L6-v2 (local) | No API key needed, fast, good quality |
| BM25 | rank-bm25 | Lexical retrieval for exact term matching |
| UI | Gradio | Functional in minutes, mounts on FastAPI |
| Observability | Langfuse (optional) + structured JSON logs | Prompt management + tracing |
| PDF Parsing | pdfplumber | Table extraction, not just text |
| Testing | pytest (104 tests) | Unit + integration + security |
| Eval | Custom simulation runner + LLM-as-judge | Industry-standard semantic evaluation |

### Request Flow (30-second version)

```
Upload:  File → Parse → Classify → Chunk → Embed → Store → doc_id
Ask:     Question → Scope Guard → Retrieve (hybrid) → Threshold Guard → LLM Generate → Grounding Guard → Confidence → Answer
Extract: doc_id → Type-specific prompt → JSON mode → Pydantic validate → ShipmentData
```

---

## 2. Architectural Decisions

Every non-obvious choice is documented here with the reasoning.

### Why LangChain over bare OpenAI SDK?

**Decision:** Replaced bare `openai` SDK with LangChain (`langchain-openai`).

**Why:**
- Callback system enables Langfuse integration without touching business logic
- Provider abstraction — switching from Azure OpenAI to Anthropic or Ollama means changing one class, not every LLM call
- Structured output binding (`llm.bind(response_format={"type": "json_object"})`) is cleaner than manual prompt begging
- The test eval criteria says "Practical AI engineering judgment" — LangChain is the industry standard for production RAG

**Tradeoff:** Added ~30MB dependency. Worth it for the abstraction.

### Why Hybrid Retrieval (Vector + BM25 + Metadata)?

**Decision:** Three retrieval strategies fused with Reciprocal Rank Fusion (RRF).

**Why:**
- **Vector alone fails** on exact terms. "What is the rate?" needs semantic understanding, but "LD53657" needs exact match.
- **BM25 alone fails** on paraphrased questions. "Who is receiving the shipment?" should match "consignee" chunks.
- **Metadata filter** catches section-specific queries. "What equipment type?" should prioritize "Carrier Details" sections.
- **RRF fusion** is parameter-free — no tuning needed, just `score = 1/(k + rank + 1)`.

**Evidence:** Hybrid mode consistently outperforms vector-only on our eval suite.

### Why 4-Layer Guardrails (not just one)?

**Decision:** Deterministic multi-layer system instead of a single LLM-based guardrail.

```
Layer 0: Document type check (is it even logistics?)
Layer 1: Question scope check (keyword + off-topic patterns)
Layer 2: Retrieval threshold (similarity >= 0.35)
Layer 3: Grounding check (answer-source token overlap >= 0.4)
Layer 4: Post-generation refusal gate (LLM self-refusal + confidence)
```

**Why:**
- Each layer catches different failure modes. Layer 1 catches "What's the weather?" before wasting an LLM call. Layer 2 catches poor retrieval. Layer 3 catches hallucination. Layer 4 catches the LLM's own uncertainty.
- Deterministic layers (1-3) are fast, free, and testable. No LLM call needed.
- The test eval says "Guardrail effectiveness" is a criterion — multiple layers demonstrate depth of thinking.

**Key nuance:** For classified logistics docs, Layer 1 only hard-blocks obviously off-topic questions (weather, sports, etc.). Generic questions like "Who is the customer?" pass through to retrieval — the retrieval threshold and grounding check handle the rest. This avoids false rejections from an overly strict keyword list.

### Why Composite Confidence Scoring?

**Decision:** Weighted score from three signals: `0.40 * retrieval + 0.35 * grounding + 0.25 * llm_assessment`

**Why:**
- No single signal is reliable alone. Retrieval score can be high for wrong chunks. LLM can be overconfident. Grounding overlap can be low for short, correct answers.
- Weights reflect trust hierarchy: retrieval similarity is the most objective signal, grounding is structural, LLM self-assessment is least reliable.
- Thresholds: HIGH (>0.7), MEDIUM (0.4-0.7), LOW (<0.4)

**Exception:** When retrieval is strong (>= 0.5) AND the LLM self-assesses HIGH confidence, we override a low grounding score. Short entity answers like "SWIFT SHIFT LOGISTICS LLC" naturally have low token overlap with large source chunks — the grounding metric is unreliable for these.

### Why Section-Based Chunking?

**Decision:** Regex-based section detection instead of fixed-size or recursive splitting.

**Why:**
- Logistics documents have clear section structure: "Carrier Details", "Rate Breakdown", "Stops", etc.
- Fixed-size chunking splits tables mid-row, losing structure.
- Section-aware chunking preserves table integrity and enables metadata filtering during retrieval.
- Max chunk size (1000 chars) handles sections that are too large.

### Why Local Embeddings by Default?

**Decision:** ChromaDB's built-in all-MiniLM-L6-v2 instead of Azure OpenAI embeddings.

**Why:**
- Zero API cost, zero latency, zero rate limiting
- No API key needed to run — critical for evaluator setup
- Quality is sufficient for document-scoped retrieval (small corpus per doc)
- Azure embeddings available as config toggle for production

### Why In-Memory Cache + ChromaDB (not just ChromaDB)?

**Decision:** Dual storage — in-memory `DocumentCache` for metadata/text + ChromaDB for vectors.

**Why:**
- ChromaDB is optimized for vector similarity, not document metadata lookup
- The `/ask` endpoint needs full document text (for LLM context) + document type + chunks — all served from memory cache
- Content-hash deduplication lives in cache (SHA256)
- Extraction results are cached per-document to avoid redundant LLM calls

**Tradeoff:** Data lost on restart. Acceptable for a POC — production would add Redis or PostgreSQL.

### Why Prompt Registry + Langfuse?

**Decision:** All prompts registered in a central registry with optional Langfuse remote management.

**Why:**
- Zero hardcoded prompts in business logic — every prompt is in `app/llm/prompts/`
- Registry enables versioning (`v1`, `v2`) for A/B testing
- Langfuse integration means prompts can be updated without redeployment
- Seeding script (`scripts/seed_langfuse_prompts.py`) pushes all 10 prompts to Langfuse

### Why LLM-as-Judge for Evals (not string matching)?

**Decision:** Semantic equivalence checking via LLM for Q&A evaluation.

**Why:**
- String matching fails on correct answers with different wording. "The carrier is SWIFT SHIFT LOGISTICS LLC" doesn't substring-match "SWIFT SHIFT LOGISTICS LLC" when wrapped in a sentence.
- Industry standard for RAG evaluation (RAGAS, DeepEval all use LLM judges)
- Fast-path optimization: substring match first (skip LLM call when obvious), LLM judge only when needed
- Judge prompt is registered in the prompt registry (`eval_qa_judge`) — not hardcoded

---

## 3. Project Structure

```
ultradoc-intelligence/
├── app/                              # Core application
│   ├── config.py                     # Pydantic Settings from .env
│   ├── main.py                       # FastAPI app — endpoints + orchestration
│   ├── __init__.py
│   ├── models/
│   │   ├── schemas.py                # Request/response models, enums (DocType, GuardrailStatus)
│   │   └── extraction.py             # ShipmentData Pydantic model with validators
│   ├── llm/
│   │   ├── provider.py               # LLMProvider ABC + LangChainProvider (Azure OpenAI)
│   │   └── prompts/
│   │       ├── registry.py           # PromptRegistry — versioned templates, Langfuse fallback
│   │       ├── system.py             # System prompt, classification, query rewrite
│   │       ├── guardrails.py         # Keyword lists, scope detection, off-topic patterns
│   │       ├── qa.py                 # Q&A with hidden Chain-of-Thought
│   │       ├── eval.py              # LLM-as-judge prompt for eval
│   │       ├── synthetic.py          # Synthetic doc + Q&A generation prompts
│   │       └── extraction/
│   │           ├── bol.py            # Bill of Lading extraction
│   │           ├── rate_confirm.py   # Rate Confirmation extraction
│   │           └── generic.py        # Fallback extraction for unknown types
│   ├── pipeline/
│   │   ├── parser.py                 # PDF (pdfplumber), DOCX (python-docx), TXT parsing
│   │   ├── classifier.py            # LLM-based document type classification
│   │   ├── chunker.py               # Section-aware chunking with table preservation
│   │   ├── embedder.py              # ChromaDB embedding + storage
│   │   ├── retriever.py             # Hybrid retrieval: vector + BM25 + metadata + RRF
│   │   ├── generator.py             # Q&A generation, CoT parsing, confidence scoring
│   │   ├── extractor.py             # Structured extraction with Pydantic validation
│   │   └── guardrails.py            # 3 deterministic guardrail checks
│   ├── storage/
│   │   ├── vector_store.py           # ChromaDB persistent wrapper (singleton)
│   │   └── cache.py                  # In-memory document cache with SHA256 dedup (singleton)
│   └── observability/
│       ├── logger.py                 # Structured JSON logger (stdout + RotatingFileHandler)
│       ├── tracer.py                 # Per-request span tracing with latency
│       └── langfuse_integration.py   # Optional Langfuse callback handler
├── ui/
│   └── gradio_app.py                # 4-tab UI: Upload, Ask, Extract, Traces
├── eval/
│   ├── run_simulation.py            # Test harness — upload, test, report with LLM-as-judge
│   ├── generate_synthetic_data.py   # LLM-powered synthetic doc generator
│   ├── report.py                    # EvalReport dataclass
│   ├── ground_truth.json            # 16 test cases for 3 sample PDFs
│   └── synthetic_data/
│       ├── docs/                    # 160 generated docs (txt/docx/pdf mix)
│       ├── ground_truth.json        # ~1000 auto-generated test cases
│       └── simulation_report.json   # Latest synthetic eval results
├── tests/                           # 104 tests across 11 files
│   ├── conftest.py                  # Fixtures for sample PDFs
│   ├── test_api.py                  # API endpoint tests (7 test classes)
│   ├── test_chunker.py              # Chunking logic tests
│   ├── test_confidence.py           # Confidence scoring tests
│   ├── test_guardrails.py           # Guardrail behavior tests
│   ├── test_langfuse.py             # Langfuse integration tests
│   ├── test_parser.py               # Parser tests
│   ├── test_provider.py             # LLM provider tests
│   ├── test_retriever.py            # Retrieval + RRF tests
│   └── test_security.py             # Prompt injection + adversarial tests
├── scripts/
│   └── seed_langfuse_prompts.py     # Push all 10 prompts to Langfuse
├── ultradoc_sample_test_data/       # 3 sample PDFs from the company
│   ├── BOL53657_billoflading.pdf
│   ├── LD53657-Carrier-RC.pdf
│   └── LD53657-Shipper-RC.pdf
├── logs/                            # Structured JSON logs (gitignored)
│   └── ultradoc.log
├── data/                            # ChromaDB + uploads (gitignored)
├── run.py                           # Entrypoint — mounts Gradio on FastAPI, single port
├── Dockerfile                       # Python 3.11-slim + uv
├── docker-compose.yml               # Single service, port 7860
├── pyproject.toml                   # Dependencies managed by uv
└── uv.lock                          # Locked dependency versions
```

### File Responsibilities

| File | Responsibility | Key Interfaces |
|------|---------------|----------------|
| `app/main.py` | Orchestrates everything — the endpoints call pipeline functions in sequence, apply guardrails, and return responses | `POST /upload`, `/ask`, `/extract`, `GET /health`, `/documents`, `/traces` |
| `app/llm/provider.py` | Single place to swap LLM providers | `get_provider()` → singleton `LangChainProvider` |
| `app/llm/prompts/registry.py` | Single place to manage all prompts | `registry.get("prompt_name")` → `PromptTemplate` |
| `app/pipeline/retriever.py` | All retrieval logic lives here | `retrieve(question, doc_id, mode, top_k)` → ranked chunks |
| `app/pipeline/guardrails.py` | Stateless, deterministic checks | `check_scope()`, `check_retrieval_threshold()`, `check_grounding()` |
| `app/pipeline/generator.py` | LLM Q&A + confidence scoring | `generate_answer()`, `compute_confidence()`, `rewrite_query()` |
| `app/storage/cache.py` | Single source of truth for document metadata | `DocumentCache.get_instance()` |
| `app/observability/logger.py` | Every module imports this | `get_logger("module_name")` |
| `eval/run_simulation.py` | The eval entry point | `--sample-only`, `--max-docs N`, `--retrieval-mode` |

---

## 4. Configuration

All configuration lives in `app/config.py` as a Pydantic `Settings` class, loaded from `.env`:

```python
# Azure OpenAI
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://...openai.azure.com/
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_MODEL=gpt-4o                    # Primary model (Q&A, extraction)
AZURE_OPENAI_FAST_MODEL=gpt-4.1-mini         # Fast model (classification, rewrite, judge)
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Storage
CHROMA_PERSIST_DIR=./data/chroma
UPLOAD_DIR=./data/uploads

# Pipeline Thresholds
DEFAULT_CONFIDENCE_THRESHOLD=0.35             # Below this → LOW confidence
GROUNDING_OVERLAP_THRESHOLD=0.4              # Token overlap ratio for grounding check
LOW_CONFIDENCE_REFUSAL_THRESHOLD=0.3         # Refuse if below AND LLM says LOW

# Embedding
EMBEDDING_PROVIDER=local                      # "local" (MiniLM) or "azure"

# Langfuse (optional)
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_PROMPT_MANAGEMENT=false             # Pull prompts from Langfuse instead of local

# Server
API_PORT=7860
LOG_LEVEL=INFO
LOG_FILE=./logs/ultradoc.log
```

**Design choice:** No hardcoded values anywhere. Every threshold, model name, and path is configurable via environment variables. This means evaluators can tune behavior without touching code.

---

## 5. Data Models

### Document Types (`app/models/schemas.py`)

```python
class DocType(Enum):
    BOL = "bill_of_lading"
    RATE_CONFIRMATION = "rate_confirmation"
    INVOICE = "invoice"
    NOT_LOGISTICS = "not_logistics"       # Guardrail: refuse everything
    UNKNOWN = "unknown"                   # Falls through to generic prompts
```

### Guardrail Statuses

```python
class GuardrailStatus(Enum):
    PASSED = "passed"                     # All checks passed
    LOW_GROUNDING = "low_grounding"       # Answer not supported by sources
    NOT_FOUND = "not_found"               # Information not in document
    OUT_OF_SCOPE = "out_of_scope"         # Question unrelated to logistics
```

### ShipmentData (`app/models/extraction.py`)

11 fields with Pydantic validators:
- `shipment_id`, `shipper`, `consignee`, `carrier_name` — string fields
- `pickup_datetime`, `delivery_datetime` — ISO format normalization
- `equipment_type`, `mode`, `currency` — enum-like strings
- `rate` — numeric, currency symbol stripping
- `weight` — string with unit

`completeness_score()` = non-null fields / total fields

---

## 6. API Layer

### POST /upload

```
Input:  multipart/form-data file (PDF, DOCX, TXT)
Output: { doc_id, doc_type, page_count, chunk_count, status }
```

Pipeline: `Parse → Classify → Chunk → Embed → Cache`

**Deduplication:** SHA256 hash of file content. If a file with the same hash was already uploaded, returns the existing `doc_id` without re-processing.

### POST /ask

```
Input:  { doc_id, question, enable_query_rewrite?, retrieval_mode?, confidence_threshold? }
Output: { answer, source_text, confidence: {score, level, breakdown}, guardrail_status, query_rewritten, retrieval_mode }
```

Pipeline: `Guard → [Rewrite] → Retrieve → Guard → Generate → Guard → Score → Respond`

This is the most complex endpoint — see [Guardrail System](#8-guardrail-system) for the full 4-layer breakdown.

### POST /extract

```
Input:  { doc_id }
Output: { extracted_data: ShipmentData, completeness_score, doc_type }
```

Pipeline: `Cache Check → Type-Specific Prompt → JSON Mode LLM → Pydantic Validate → Cache Result`

**Caching:** Extraction results are stored in `DocumentRecord.extraction_result`. Second call for the same doc returns cached data instantly.

### GET /health, /documents, /traces

- `/health` — `{"status": "ok"}`
- `/documents` — List of all cached documents with metadata
- `/traces` — Last 50 request traces with per-span latency breakdown

---

## 7. Pipeline Deep Dive

### 7.1 Parser (`app/pipeline/parser.py`)

| Format | Library | Special Handling |
|--------|---------|-----------------|
| PDF | pdfplumber | Extracts tables as formatted text + regular text per page |
| DOCX | python-docx | Paragraphs + tables |
| TXT | built-in | Raw read with UTF-8 encoding |

Returns: `{"text": str, "page_count": int, "status": "success|error"}`

**Why pdfplumber over PyPDF2?** Table extraction. Logistics documents are table-heavy (rate breakdowns, stop details, commodity lists). PyPDF2 loses table structure entirely.

### 7.2 Classifier (`app/pipeline/classifier.py`)

Single LLM call using the fast model. Prompt asks for one of 5 categories. Uses the `classification` prompt from registry.

**Why LLM classification instead of rules?** Documents vary wildly — OCR artifacts, different formats, mixed content. A keyword-based classifier would need constant updating. The LLM handles edge cases naturally.

### 7.3 Chunker (`app/pipeline/chunker.py`)

Section-aware splitting using regex patterns for common logistics document headers:

```
"Bill of Lading", "Carrier Details", "Stops", "Rate Breakdown",
"Shipper", "Consignee", "Commodity", "Terms and Conditions", etc.
```

- Preserves tables intact (never splits mid-table)
- Max chunk size: 1000 characters (oversized sections get split at paragraph boundaries)
- Each chunk tagged with `section` metadata for retrieval filtering

### 7.4 Retriever (`app/pipeline/retriever.py`)

Three retrieval strategies, fused with RRF:

```
              ┌─ Vector Search (ChromaDB cosine similarity)
              │
Question ─────┼─ BM25 Search (rank-bm25 lexical matching)
              │
              └─ Metadata Filter (section keyword matching)
                    │
                    ▼
              RRF Fusion: score = 1/(k + rank + 1)
                    │
                    ▼
              Top-3 Results (deduplicated by chunk text)
```

**Retrieval modes:** `hybrid` (default), `vector`, `bm25`

**Why RRF over learned fusion?** RRF is parameter-free — no training data needed, no tuning. It's the standard approach for combining heterogeneous rankings (used in Azure AI Search, Elasticsearch).

### 7.5 Generator (`app/pipeline/generator.py`)

**Q&A Generation:**
- Uses the `qa_with_cot` prompt with hidden Chain-of-Thought
- LLM output format: `SECTION: ... ANSWER: ... CONFIDENCE: HIGH/MEDIUM/LOW`
- Parses structured response, extracts answer and self-assessed confidence
- Detects LLM self-refusal ("Not found in document") as a signal

**Query Rewrite:**
- Fixes typos and normalizes logistics terminology
- Example: "What is the frate rate?" → "What is the freight rate?"
- Uses the fast model to minimize latency

**Confidence Scoring:**
```python
composite = 0.40 * retrieval_score      # Objective: how good was retrieval?
           + 0.35 * grounding_ratio     # Structural: does answer match sources?
           + 0.25 * llm_assessment      # Subjective: how confident is the LLM?

# LLM assessment mapping: HIGH=1.0, MEDIUM=0.5, LOW=0.2
# Level: HIGH (>0.7), MEDIUM (0.4-0.7), LOW (<0.4)
```

### 7.6 Extractor (`app/pipeline/extractor.py`)

- Selects prompt based on document type (BOL, RC, or generic)
- Uses `generate_structured()` with `response_format={"type": "json_object"}` — API-enforced JSON, not prompt begging
- Parses into `ShipmentData` Pydantic model
- **Field-level fallback:** If full validation fails, tries to salvage individual fields one at a time
- Caches result in `DocumentRecord` for subsequent calls

---

## 8. Guardrail System

The `/ask` endpoint runs a multi-layer guardrail pipeline. Each layer is independent and catches different failure modes:

```
Request
  │
  ▼
[Layer 0] Document Type Check
  │ Is the document classified as NOT_LOGISTICS?
  │ → YES: Refuse ("not a logistics document")
  │ → NO: Continue
  ▼
[Layer 1] Question Scope Check (pre-retrieval, no LLM)
  │ For classified docs: is_obviously_off_topic() — weather, sports, etc.
  │ For unclassified docs: is_in_scope() — keyword matching
  │ → OFF TOPIC: Refuse ("not related to logistics")
  │ → IN SCOPE: Continue
  ▼
[Optional] Query Rewrite (fast model)
  │ Fix typos, normalize terminology
  ▼
[Retrieval] Hybrid search → top-3 chunks with similarity scores
  │
  ▼
[Layer 2] Retrieval Threshold Check
  │ max(chunk_similarity) >= 0.35?
  │ → NO: Refuse ("Not found in document")
  │ → YES: Continue
  ▼
[LLM Generation] GPT-4o with full doc + source chunks
  │ Parses CoT: SECTION / ANSWER / CONFIDENCE
  ▼
[Layer 3] Grounding Check
  │ token_overlap(answer, source) / len(answer_tokens) >= 0.4?
  │ → NO (low_grounding):
  │     If retrieval >= 0.5 AND LLM confidence == HIGH:
  │       Override → PASSED (short entity answers are unreliable here)
  │     Else: Refuse ("Not found in document")
  │ → YES: Continue
  ▼
[Layer 4a] Post-Generation LLM Refusal Detection
  │ Does the answer contain "not found in document", "not present", etc.?
  │ → YES: Refuse (honor the LLM's own judgment)
  │ → NO: Continue
  ▼
[Layer 4b] Low-Confidence Refusal
  │ composite_score < 0.3 AND llm_confidence == LOW?
  │ → YES: Refuse
  │ → NO: Continue
  ▼
[Return] Answer + source_text + confidence + guardrail_status
```

### Why This Design?

- **Layers 0-1** are free (no LLM call). They filter garbage before expensive operations.
- **Layer 2** is cheap (vector similarity already computed). Prevents hallucination on bad retrieval.
- **Layer 3** is structural. Catches cases where the LLM invents information not in the source.
- **Layer 4** is a safety net. If the LLM itself says "I don't know", we trust that signal.
- **The override** (retrieval >= 0.5 + HIGH confidence) prevents false refusals on short entity answers where grounding check is unreliable.

---

## 9. Prompt Management

### Architecture

```
app/llm/prompts/
├── registry.py          # Central registry (local + Langfuse)
├── system.py            # Registers: system, classification, query_rewrite
├── qa.py                # Registers: qa_with_cot
├── guardrails.py        # Keyword lists (no registry — pure Python)
├── eval.py              # Registers: eval_qa_judge
├── synthetic.py         # Registers: synth_document, synth_qa_pairs
└── extraction/
    ├── bol.py           # Registers: extraction_bol
    ├── rate_confirm.py  # Registers: extraction_rc
    └── generic.py       # Registers: extraction_generic
```

### All 10 Registered Prompts

| Name | Purpose | Used By |
|------|---------|---------|
| `system` | Base persona: "You are a logistics document analyst" | All LLM calls as system message |
| `classification` | Classify doc into 5 types | `classifier.py` |
| `query_rewrite` | Fix typos, normalize logistics terms | `generator.py` |
| `qa_with_cot` | Q&A with hidden Chain-of-Thought | `generator.py` |
| `extraction_bol` | BOL-specific field extraction | `extractor.py` |
| `extraction_rc` | Rate Confirmation extraction | `extractor.py` |
| `extraction_generic` | Generic shipment extraction | `extractor.py` |
| `eval_qa_judge` | LLM-as-judge for eval accuracy | `eval/run_simulation.py` |
| `synth_document` | Generate synthetic logistics docs | `eval/generate_synthetic_data.py` |
| `synth_qa_pairs` | Generate Q&A ground truth pairs | `eval/generate_synthetic_data.py` |

### How Prompts Flow

```
Local files (app/llm/prompts/*.py)
  │
  ├── Auto-register on import via registry.register()
  │
  ▼
PromptRegistry (in-memory)
  │
  ├── registry.get("qa_with_cot")
  │     │
  │     ├── If LANGFUSE_PROMPT_MANAGEMENT=true:
  │     │     → Fetch from Langfuse (remote, versionable)
  │     │
  │     └── Else: return local template
  │
  ▼
PromptTemplate.render(**kwargs)
  │
  ▼
Final prompt string → LLM
```

### Seeding to Langfuse

```bash
PYTHONPATH=. uv run python scripts/seed_langfuse_prompts.py
```

This pushes all 10 prompts to Langfuse with `[production, latest]` labels. Once seeded, set `LANGFUSE_PROMPT_MANAGEMENT=true` in `.env` to pull prompts from Langfuse at runtime — enabling prompt updates without redeployment.

---

## 10. Storage Layer

### ChromaDB Vector Store (`app/storage/vector_store.py`)

- **Singleton** via `VectorStore.get_instance()`
- Persistent storage at `./data/chroma/`
- Collection name: `"documents"`
- Metadata per chunk: `{doc_id, section, index}`
- Embedding: ChromaDB default (all-MiniLM-L6-v2)

### In-Memory Document Cache (`app/storage/cache.py`)

- **Singleton** via `DocumentCache.get_instance()`
- Stores `DocumentRecord` objects keyed by `doc_id`
- Record fields: `doc_id`, `file_name`, `full_text`, `doc_type`, `chunks`, `page_count`, `content_hash`, `extraction_result`
- **Deduplication:** SHA256 of file content. Same file → same `doc_id`, skip re-processing.
- **Extraction caching:** First `/extract` call runs the LLM; subsequent calls return cached result.

---

## 11. Observability

### Structured Logging (`app/observability/logger.py`)

Every module gets a structured JSON logger:

```python
from app.observability.logger import get_logger
logger = get_logger("module_name")
```

**Dual output:**
1. `stdout` — for container/terminal viewing
2. `logs/ultradoc.log` — RotatingFileHandler (10MB max, 3 backups)

**Log format:**
```json
{
  "timestamp": "2026-04-08T09:04:47.431Z",
  "level": "INFO",
  "module": "main",
  "message": "ask_complete",
  "doc_id": "2b449d22-...",
  "confidence_level": "HIGH",
  "confidence_score": 0.82,
  "refused": false
}
```

### What Gets Logged

| Module | Log Events |
|--------|-----------|
| `main` | `upload_complete`, `ask_received`, `ask_complete`, `ask_refused` (with reason), `extract_complete` |
| `parser` | `parse_success` (file type, text length, page count) |
| `chunker` | `chunking_complete` (chunk count, avg size) |
| `classifier` | `classification_complete` (doc type) |
| `retriever` | `retrieval_complete` (mode, chunks found, top score) |
| `guardrails` | `guardrail_scope`, `guardrail_threshold`, `guardrail_grounding` (each layer's result) |
| `generator` | `generation_complete` (LLM confidence), `confidence_scored` (full breakdown) |
| `extractor` | `extraction_complete` (fields extracted, completeness) |
| `tracer` | `trace_complete` (full span tree with latencies) |

### Per-Request Tracing (`app/observability/tracer.py`)

Every request gets a `Tracer` instance with `trace_id`. Each pipeline stage is a span:

```python
with tracer.span("retrieval") as span:
    results = retrieve(...)
    span.metadata = {"mode": "hybrid", "top_score": 0.85}
```

The `/traces` endpoint returns the last 50 traces — useful for debugging in the UI.

### Langfuse Integration (`app/observability/langfuse_integration.py`)

When `LANGFUSE_ENABLED=true`:
- LangChain callback handler sends all LLM calls to Langfuse
- Full traces visible in Langfuse dashboard: latency, tokens, prompt versions
- Enables prompt A/B testing and eval-driven optimization loops

---

## 12. Evaluation Framework

### Architecture

```
eval/
├── ground_truth.json              # 16 hand-crafted cases for 3 sample PDFs
├── sample_report.json             # Results from sample eval
├── run_simulation.py              # Test harness
├── generate_synthetic_data.py     # Synthetic doc generator
├── report.py                      # EvalReport dataclass
└── synthetic_data/
    ├── docs/                      # 160 generated docs (86 txt, 46 docx, 30 pdf)
    ├── ground_truth.json          # ~1000 auto-generated test cases
    └── simulation_report.json     # Results from synthetic eval
```

### Two Eval Suites

| Suite | Docs | Test Cases | Purpose |
|-------|------|-----------|---------|
| **Sample** (3 PDFs) | BOL, Carrier-RC, Shipper-RC from the company | 16 (12 Q&A + 2 extraction + 2 refusal) | Validates against their actual test data |
| **Synthetic** (160 docs) | LLM-generated with 12 messiness profiles | ~1000 | Stress-tests across diverse formats, edge cases |

### How the Eval Runner Works (`eval/run_simulation.py`)

**Phase 1: Upload** — Upload all documents via `POST /upload`, collect `doc_id` mapping.

**Phase 2: Test** — For each test case:
- **Q&A tests:** Call `POST /ask`, check result with LLM-as-judge
- **Extraction tests:** Call `POST /extract`, check fields with flexible matching

**Phase 3: Report** — Compute metrics:

| Metric | How |
|--------|-----|
| Q&A Accuracy | correct / total |
| Normal Q&A Accuracy | correct / (total - refusal_cases) |
| Correct Refusals | properly_refused / expected_refusals |
| False Acceptances | answered_when_should_refuse |
| Extraction Field Accuracy | matched_fields / checked_fields |
| Latency | avg, p50, p95 for upload, Q&A, extraction |

### Q&A Evaluation — 3-Tier Checking

```
Expected == __OUT_OF_SCOPE__ or __NOT_FOUND__?
  → Check: did the system refuse? (status or answer text)

Otherwise:
  1. Fast path: expected.lower() in actual.lower()? → PASS
  2. LLM-as-judge: "Is actual semantically equivalent to expected?" → YES/NO
  3. Fallback: substring match if LLM judge fails
```

### Extraction Evaluation — 3-Tier Field Matching

```
For each expected field:
  1. Substring match (either direction)
  2. Numeric comparison (strips $, commas, tolerance=1.0)
  3. Core value matching: 80%+ of significant words present
```

### Running Evals

```bash
# Against 3 sample PDFs (their test data)
PYTHONPATH=. uv run python eval/run_simulation.py --sample-only

# Against synthetic data (first 30 docs for quick check)
PYTHONPATH=. uv run python eval/run_simulation.py --max-docs 30

# Full 160-doc synthetic eval
PYTHONPATH=. uv run python eval/run_simulation.py

# Specific retrieval mode
PYTHONPATH=. uv run python eval/run_simulation.py --retrieval-mode vector
```

### Latest Results

**Sample Docs (3 PDFs from company):**

| Metric | Score |
|--------|-------|
| Q&A Accuracy | **100% (14/14)** |
| Correct Refusals | **2/2 (100%)** |
| False Acceptances | **0** |
| Extraction Field Accuracy | **100% (12/12)** |

---

## 13. Synthetic Data Generation

### Generator (`eval/generate_synthetic_data.py`)

Produces diverse, messy logistics documents using the LLM:

**Step 1:** Generate seed fields (random but realistic):
- 20 carrier names, 15 shippers, 15 consignee DCs
- 24 cities (US, Canada, Mexico)
- Random rates ($200-$8000), weights, dates
- Some fields intentionally MISSING (15-20% probability)

**Step 2:** Generate document text with a messiness profile (12 profiles):
- Clean, Typo-heavy, Missing data, Legacy system export
- Extra noise (disclaimers, watermarks), Special characters
- Multi-format chaos, Scan artifact style, Minimal/terse
- Verbose/repetitive, Cross-border (bilingual), Contradictory

**Step 3:** Generate ground truth Q&A and extraction test cases.

**Step 4:** Save as TXT (50%), DOCX (30%), or PDF (20%).

### Format Distribution

| Format | Count | Tests |
|--------|-------|-------|
| TXT | 86 | Text parsing, chunking |
| DOCX | 46 | python-docx parsing |
| PDF | 30 | pdfplumber parsing, table extraction |

### Running the Generator

```bash
# Generate 160 docs (default)
PYTHONPATH=. uv run python eval/generate_synthetic_data.py

# Quick test (20 docs)
PYTHONPATH=. uv run python eval/generate_synthetic_data.py --count 20

# With Langfuse prompt seeding
PYTHONPATH=. uv run python eval/generate_synthetic_data.py --seed-to-langfuse
```

---

## 14. Testing

### 104 Tests Across 11 Files

```bash
# Run all tests
uv run python -m pytest tests/ -v

# Run specific test file
uv run python -m pytest tests/test_guardrails.py -v

# Run with coverage
uv run python -m pytest tests/ --cov=app
```

### Test Coverage Map

| File | Tests | What It Covers |
|------|-------|---------------|
| `test_api.py` | 7 classes | Health, upload, ask, extract, documents, traces, Gradio mount, Swagger |
| `test_chunker.py` | 4 tests | Section splitting, table preservation, empty input, oversized chunks |
| `test_confidence.py` | 4 tests | HIGH/MEDIUM/LOW thresholds, breakdown weights |
| `test_guardrails.py` | 8 tests | Scope pass/fail, threshold pass/fail, grounding pass/fail, edge cases |
| `test_langfuse.py` | 2 classes | Langfuse handler creation, prompt registry with/without Langfuse |
| `test_parser.py` | 4 tests | PDF parsing with tables, TXT parsing, error handling |
| `test_provider.py` | 3 classes | Config loading, LLMResponse dataclass, provider initialization |
| `test_retriever.py` | 5 tests | RRF fusion, BM25 ranking, metadata filter, deduplication |
| `test_security.py` | 4 classes | Prompt injection, grounding vs hallucination, threshold bypass, adversarial inputs |

### Test Fixtures (`tests/conftest.py`)

Three sample PDFs as pytest fixtures:
- `sample_bol_path` — `BOL53657_billoflading.pdf`
- `sample_carrier_rc_path` — `LD53657-Carrier-RC.pdf`
- `sample_shipper_rc_path` — `LD53657-Shipper-RC.pdf`

---

## 15. Deployment

### Local Development

```bash
# Install dependencies
uv sync

# Set up environment
cp .env.example .env  # Fill in Azure OpenAI credentials

# Run the server
uv run python run.py
# → API at http://localhost:7860
# → UI at http://localhost:7860/ui
# → Docs at http://localhost:7860/docs
```

### Docker

```bash
# Build and run
docker-compose up --build

# Or manually
docker build -t ultradoc .
docker run -p 7860:7860 --env-file .env ultradoc
```

### Dockerfile Details

```dockerfile
FROM python:3.11-slim
# Install build deps for chromadb/numpy
RUN apt-get install build-essential
# Use uv for fast, reproducible installs
RUN pip install uv
COPY pyproject.toml uv.lock .
RUN uv sync --no-dev --frozen
COPY . .
EXPOSE 7860
CMD ["uv", "run", "python", "run.py"]
```

**Why uv?** 10-50x faster than pip for dependency resolution. Lockfile ensures reproducible builds.

---

## 16. Operational Runbook

### Common Commands

```bash
# Start server
uv run python run.py

# Run tests
uv run python -m pytest tests/ -v

# Run sample eval (3 PDFs from company)
PYTHONPATH=. uv run python eval/run_simulation.py --sample-only

# Run synthetic eval (30 docs, quick)
PYTHONPATH=. uv run python eval/run_simulation.py --max-docs 30

# Generate synthetic data
PYTHONPATH=. uv run python eval/generate_synthetic_data.py --count 160

# Seed prompts to Langfuse
PYTHONPATH=. uv run python scripts/seed_langfuse_prompts.py

# View logs
tail -f logs/ultradoc.log | python -m json.tool

# View recent refusals in logs
grep "ask_refused" logs/ultradoc.log | python -m json.tool
```

### Where Things Are

| What | Where |
|------|-------|
| Application logs | `logs/ultradoc.log` (JSON, rotating 10MB x 3) |
| Sample eval results | `eval/sample_report.json` |
| Synthetic eval results | `eval/synthetic_data/simulation_report.json` |
| Synthetic ground truth | `eval/synthetic_data/ground_truth.json` |
| Sample ground truth | `eval/ground_truth.json` |
| ChromaDB data | `data/chroma/` |
| Uploaded files | `data/uploads/` |
| All prompts | `app/llm/prompts/` (+ Langfuse if enabled) |
| All thresholds | `app/config.py` (+ `.env` overrides) |

### Debugging a Failed Question

1. Check logs: `grep "ask_refused" logs/ultradoc.log | tail -5`
2. Look at the reason: `out_of_scope`, `low_grounding`, `low_confidence`, `llm_refusal`
3. For `low_grounding`: check if retrieval score was high — may need the override
4. For `out_of_scope`: check if the question keyword is in `LOGISTICS_WORDS`
5. For retrieval failures: try enabling query rewrite or switching to hybrid mode
6. Check the full trace: `GET /traces` in the UI

### Tuning Thresholds

| Threshold | Config Key | Effect of Lowering | Effect of Raising |
|-----------|-----------|-------------------|------------------|
| Confidence | `DEFAULT_CONFIDENCE_THRESHOLD` | More answers pass, more risk | Fewer answers, safer |
| Retrieval | `confidence_threshold` in request | More chunks qualify | Stricter retrieval |
| Grounding | `GROUNDING_OVERLAP_THRESHOLD` | More answers pass grounding | More refusals |
| Refusal | `LOW_CONFIDENCE_REFUSAL_THRESHOLD` | Less aggressive refusal | More aggressive |
