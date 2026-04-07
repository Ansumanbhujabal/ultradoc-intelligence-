# Ultra Doc-Intelligence — Design Spec

**Date:** 2026-04-07
**Author:** Ansuman SS Bhujabala
**Status:** Approved
**Approach:** B++ (LLM-First with Vector Verification + Observability + Eval)

---

## 1. Overview

A POC AI system that allows users to upload logistics documents (PDF, DOCX, TXT) and interact with them using natural language questions. The system retrieves relevant content, answers grounded questions, applies guardrails, returns confidence scores, and extracts structured shipment data. Simulates an AI assistant inside a Transportation Management System (TMS).

### Design Principles

- **Provider-agnostic** — Azure OpenAI primary, swap to OpenAI/Anthropic/Ollama via config
- **Deterministic pipeline** — parse, classify, retrieve, generate, validate — each stage testable
- **Observable** — structured traces at every stage, latency + token tracking
- **Debuggable** — composite confidence with breakdown, guardrails are heuristic not LLM-based
- **Experiment-friendly** — A/B flags for query rewrite, retrieval mode, thresholds

---

## 2. Project Structure

```
ultradoc-intelligence/
├── app/
│   ├── main.py                  # FastAPI app + endpoints
│   ├── config.py                # Settings via pydantic-settings (.env)
│   ├── models/
│   │   ├── schemas.py           # Request/Response Pydantic models
│   │   └── extraction.py        # Shipment extraction schema
│   ├── pipeline/
│   │   ├── parser.py            # PDF/DOCX/TXT parsing (pdfplumber + fallback)
│   │   ├── classifier.py        # Doc type detection (BOL/RC/Invoice)
│   │   ├── chunker.py           # Section-based chunking (tables intact)
│   │   ├── embedder.py          # Embedding + ChromaDB storage
│   │   ├── retriever.py         # Hybrid retrieval (vector + BM25 + metadata + RRF)
│   │   ├── generator.py         # LLM Q&A generation with hidden CoT
│   │   ├── extractor.py         # Structured extraction (Pydantic + function calling)
│   │   └── guardrails.py        # 3-layer guardrail system
│   ├── llm/
│   │   ├── provider.py          # Provider-agnostic LLM abstraction
│   │   └── prompts/
│   │       ├── registry.py      # Prompt registry (load by name + version)
│   │       ├── system.py        # Base system prompts
│   │       ├── qa.py            # Q&A prompt templates
│   │       ├── extraction/
│   │       │   ├── bol.py       # BOL-specific extraction prompt
│   │       │   ├── rate_confirm.py  # Rate Confirmation extraction prompt
│   │       │   └── generic.py   # Fallback extraction prompt
│   │       └── guardrails.py    # Out-of-scope detection prompts
│   ├── observability/
│   │   ├── tracer.py            # Request tracing (per-stage spans)
│   │   ├── metrics.py           # Latency + token + cost tracking
│   │   └── logger.py            # Structured JSON logger
│   └── storage/
│       ├── vector_store.py      # ChromaDB wrapper
│       └── cache.py             # Document + embedding cache
├── eval/
│   ├── ground_truth.json        # 15-20 Q&A pairs from sample docs
│   ├── run_eval.py              # Automated eval script
│   └── report.py                # Metrics + output
├── ui/
│   └── gradio_app.py            # Gradio UI (4 tabs: Upload, Chat, Extract, Traces)
├── tests/
│   └── ...
├── ultradoc_sample_test_data/   # Sample logistics docs
├── run.py                       # Single entrypoint (FastAPI + Gradio)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env
└── README.md
```

---

## 3. API Endpoints

### POST /upload

**Request:** multipart file (PDF/DOCX/TXT)

**Pipeline:**
1. Parse — pdfplumber (PDF) / python-docx (DOCX) / raw read (TXT)
   - Failure → LLM vision fallback (send page as image to Azure OpenAI)
2. Classify — LLM call: detect doc type → BOL | Rate Confirmation | Invoice | Unknown
3. Chunk — Section-based splitting, keep tables intact within chunks
4. Embed — Azure OpenAI embeddings → store in ChromaDB with `doc_id` metadata
5. Cache — Store full text + doc_type + chunks for fast re-access
6. Trace — Log: parse_time, classify_time, chunk_count, embed_time, doc_type

**Response:**
```json
{
  "doc_id": "uuid",
  "doc_type": "rate_confirmation",
  "page_count": 2,
  "chunk_count": 8,
  "status": "success"
}
```

### POST /ask

**Request:**
```json
{
  "doc_id": "uuid",
  "question": "What is the carrier rate?",
  "enable_query_rewrite": false,
  "retrieval_mode": "hybrid",
  "confidence_threshold": 0.3
}
```

**Pipeline:**
1. Guardrail #1 (Pre-Retrieval) — Out-of-scope check (keyword classifier, no LLM)
2. Query Rewrite (optional) — gpt-4.1-mini reformulates question for retrieval
3. Hybrid Retrieval:
   a. Vector search → ChromaDB top-5
   b. BM25 keyword search → top-5
   c. Metadata filter → match section headers if field detected in question
   d. RRF fusion → merge & re-rank → final top-3
4. Guardrail #2 (Post-Retrieval) — If best chunk similarity < threshold → "Not found in document"
5. Generate — Stuff full doc text + question into LLM with hidden CoT
   - System prompt: "Answer ONLY from document context"
   - CoT: LLM identifies relevant section first, then answers
6. Guardrail #3 (Post-Generation) — Grounding check: answer tokens vs source chunk overlap
   - Low overlap → flag `guardrail_status: "low_grounding"`
7. Confidence — Composite score:
   - 40% retrieval similarity (from hybrid RRF)
   - 35% answer-source token overlap
   - 25% LLM self-assessment (from hidden CoT: HIGH/MEDIUM/LOW)
8. Trace — Log all stages with latency, tokens, metadata

**Response:**
```json
{
  "answer": "The carrier rate is $400.00 USD",
  "source_text": "Carrier Pay: Flatbed:$ 400.00 USD ... Total: 400.00 USD",
  "confidence": {
    "score": 0.82,
    "level": "HIGH",
    "breakdown": {
      "retrieval": 0.91,
      "grounding": 0.78,
      "llm_assessment": 0.80
    }
  },
  "guardrail_status": "passed",
  "query_rewritten": false,
  "retrieval_mode": "hybrid"
}
```

### POST /extract

**Request:**
```json
{
  "doc_id": "uuid"
}
```

**Pipeline:**
1. Load cached full text + doc_type
2. Select prompt template based on doc_type (BOL/RC/Generic)
3. LLM + function calling with Pydantic schema
4. Validate — Pydantic parses response, nulls for missing fields
5. Completeness — Score = % of non-null fields
6. Trace — Log extract_time, tokens, completeness, null_fields

**Response:**
```json
{
  "extracted_data": {
    "shipment_id": "LD53657",
    "shipper": "AAA, Los Angeles International Airport (LAX), World Way, Los Angeles, CA, USA",
    "consignee": "xyz, 7470 Cherry Avenue, Fontana, CA 92336, USA",
    "pickup_datetime": "2026-02-08T09:00:00",
    "delivery_datetime": "2026-02-08T09:00:00",
    "equipment_type": "Flatbed",
    "mode": "FTL",
    "rate": 400.00,
    "currency": "USD",
    "weight": "56000.00 lbs",
    "carrier_name": "SWIFT SHIFT LOGISTICS LLC"
  },
  "completeness_score": 1.0,
  "doc_type": "rate_confirmation"
}
```

---

## 4. LLM Abstraction Layer

### Provider Interface

```python
class LLMProvider(ABC):
    async def generate(self, messages, **kwargs) -> LLMResponse
    async def generate_structured(self, messages, schema, **kwargs) -> dict
    async def embed(self, texts) -> list[list[float]]
```

### Implementations

- `AzureOpenAIProvider` — primary, uses Azure deployments from .env
- `OpenAIProvider` — swap-in compatible
- `AnthropicProvider` — swap-in compatible

### LLMResponse

Every call returns:
- `content` — generated text or structured output
- `tokens_in` / `tokens_out` — for cost tracking
- `latency_ms` — for performance tracking
- `model_name` — which model was used

### Model Routing

- **Classification + Query Rewrite** → gpt-4.1-mini (cheap, fast)
- **Q&A Generation** → gpt-4o (accuracy matters)
- **Structured Extraction** → gpt-4o with function calling
- **Embeddings** → text-embedding-ada-002 or text-embedding-3-small

Config-driven — change `.env` to switch any of these.

---

## 5. Prompt Architecture

### Prompt Registry

- Templates stored as Python modules with versioning
- Load by name: `registry.get("qa_with_cot", version="v1")`
- Variable injection: `{document_text}`, `{question}`, `{doc_type}`
- Swappable without touching pipeline logic

### Prompt Templates

**System Prompt (all calls):**
- Role: "You are a logistics document analyst for a TMS system"
- Rules: "Answer ONLY from the provided document. If information is not found, say 'Not found in document.' Never infer or assume."

**Q&A Prompt (with hidden CoT):**
- Instruct LLM to first identify the relevant section
- Then formulate the answer
- Include confidence self-assessment: HIGH/MEDIUM/LOW
- Parse out only the answer for user; CoT stays internal for logging

**Extraction Prompts (per doc type):**
- BOL-specific: knows field locations typical in Bills of Lading
- Rate Confirmation-specific: knows carrier details, stops, rate breakdown
- Generic fallback: broad extraction
- Each includes 1-2 few-shot examples for accuracy

**Guardrail Prompts:**
- Out-of-scope keyword list (no LLM call needed)

---

## 6. Hybrid Retrieval

### Three Signals + Fusion

1. **Vector Search (semantic)** — ChromaDB cosine similarity, top-5 chunks
   - Catches semantic matches: "Who is the receiver?" → consignee section

2. **BM25 Keyword Search (lexical)** — `rank_bm25` library, in-memory, top-5 chunks
   - Catches exact terms: "LD53657", "PO 112233ABC", carrier names

3. **Metadata Filter (structured)** — Section header matching
   - Tag each chunk with section name during chunking (e.g., "Rate Breakdown", "Carrier Details")
   - If question mentions a field → jump directly to that section

4. **RRF Fusion** — Reciprocal Rank Fusion to merge all 3 result lists
   - `score(chunk) = sum(1 / (k + rank_i + 1))` across all lists
   - Final top-3 chunks with normalized hybrid score

### Query Rewrite (Optional)

- Flag: `enable_query_rewrite: true`
- Uses gpt-4.1-mini to reformulate casual questions
- "When do they pick it up?" → "What is the pickup date and time for the shipment?"
- A/B testable via eval harness

### Retrieval Modes (Configurable)

- `"hybrid"` — all 3 signals + RRF (default)
- `"vector"` — ChromaDB only
- `"bm25"` — keyword only
- Switchable per request for experimentation

---

## 7. Guardrails

### Layer 1: Pre-Retrieval (cheap, fast, deterministic)

- **Out-of-scope detection** — keyword classifier checks if question has any logistics/document relevance
- Off-topic questions ("What's the weather?") → rejected immediately
- No LLM call needed

### Layer 2: Post-Retrieval (deterministic)

- **Retrieval threshold** — if best hybrid chunk similarity < `confidence_threshold` (default 0.3) → return "Not found in document"
- Zero chunks returned → same response
- Threshold is configurable per request

### Layer 3: Post-Generation (deterministic)

- **Grounding check** — token overlap ratio between answer and source chunks
- If overlap < 40% → flag `guardrail_status: "low_grounding"` (threshold tunable via eval)
- Still returns the answer, but with flag + lower confidence
- UI shows a warning for low-grounding answers

### Design Choice

All guardrails are heuristic/deterministic, not LLM-based. Fast, cheap, predictable, debuggable. No extra LLM calls.

---

## 8. Confidence Scoring

### Composite Score (3 signals)

| Signal | Source | Weight | Type |
|--------|--------|--------|------|
| Retrieval quality | Hybrid RRF normalized score | 40% | Deterministic |
| Grounding strength | Answer-source token overlap ratio | 35% | Deterministic |
| LLM self-assessment | Extracted from hidden CoT (HIGH/MED/LOW) | 25% | LLM-derived |

### Output

```json
{
  "score": 0.82,
  "level": "HIGH",
  "breakdown": {
    "retrieval": 0.91,
    "grounding": 0.78,
    "llm_assessment": 0.80
  }
}
```

### Levels

- **HIGH** (> 0.7) — confident, well-grounded answer
- **MEDIUM** (0.4 - 0.7) — answer found but weaker grounding
- **LOW** (< 0.4) — low confidence, may trigger guardrail

### Extraction Confidence

For `POST /extract`: completeness score = % of non-null fields in the Pydantic model.

---

## 9. Observability

### Structured Tracing

Every request produces a trace with spans per pipeline stage:

```json
{
  "trace_id": "tr_abc123",
  "doc_id": "doc_xyz",
  "endpoint": "/ask",
  "timestamp": "2026-04-07T14:30:00Z",
  "total_latency_ms": 2340,
  "total_tokens": 1850,
  "spans": [
    {"stage": "query_rewrite", "status": "skipped", "latency_ms": 0},
    {"stage": "retrieval", "status": "success", "latency_ms": 45, "metadata": {...}},
    {"stage": "guardrail_threshold", "status": "passed", "latency_ms": 1},
    {"stage": "generation", "status": "success", "latency_ms": 2150, "metadata": {...}},
    {"stage": "guardrail_grounding", "status": "passed", "latency_ms": 3},
    {"stage": "confidence_scoring", "status": "completed", "latency_ms": 1}
  ]
}
```

### Implementation

Context-manager based tracer — `with tracer.span("stage")` wraps each pipeline step. Zero-dependency, pluggable. Can swap to Langfuse/Langsmith later if needed.

### Metrics Tracked

- Per-stage latency (ms)
- Token usage (in/out) per LLM call
- Estimated cost per request (USD)
- Confidence breakdown per request

### UI Integration

Traces tab in Gradio — expandable accordion showing recent traces with all spans.

---

## 10. Eval Framework

### Ground Truth

`eval/ground_truth.json` — 15-20 test cases from sample docs:
- Q&A pairs with expected answers and expected source substrings
- Extraction cases with expected field values
- Edge cases: off-topic questions, questions about missing data

### Metrics

**Q&A:**
- Answer accuracy (fuzzy match against expected)
- Source hit rate (does source_text contain expected string?)
- Average confidence across correct answers
- Confidence calibration (high-confidence = higher accuracy?)

**Extraction:**
- Field accuracy (exact match per field)
- Completeness (% non-null when expected non-null)
- Null correctness (correctly null when field missing)

**Guardrails:**
- False refusal rate (refused a valid question)
- False acceptance rate (answered when should have refused)

**Performance:**
- Average latency per endpoint
- Average tokens per request
- Estimated cost per request

### A/B Testing

Run eval with different flags to compare:
```bash
python -m eval.run_eval --flags '{"enable_query_rewrite": false}'
python -m eval.run_eval --flags '{"enable_query_rewrite": true}'
python -m eval.run_eval --flags '{"retrieval_mode": "vector"}'
python -m eval.run_eval --flags '{"retrieval_mode": "hybrid"}'
```

---

## 11. UI — Gradio

### 4 Tabs

1. **Upload** — File upload, shows doc_id, doc_type, page/chunk count
2. **Ask** — Document selector, question input, answer + source + confidence display, toggle switches for query rewrite and retrieval mode
3. **Extract** — Document selector, "Run Extraction" button, JSON output + completeness score
4. **Traces** — Expandable accordion of recent request traces

### Design Notes

- Gradio calls FastAPI endpoints (or imports pipeline directly)
- Not a stateful chat — each question is independent
- Confidence shown as color-coded badges: green (HIGH), yellow (MEDIUM), red (LOW)
- Source text in collapsible sections

---

## 12. Deployment

### Local — Docker

```yaml
services:
  app:
    build: .
    ports:
      - "8000:8000"   # FastAPI
      - "7860:7860"   # Gradio
    env_file: .env
    volumes:
      - ./data:/app/data   # persist ChromaDB + uploads
```

### Hosted — HuggingFace Spaces

- Same Dockerfile → push to HF Space
- Azure OpenAI key as HF Space secret
- Public URL for submission

### Single Entrypoint

`run.py` starts both FastAPI (port 8000) and Gradio (port 7860). For HF Spaces, Gradio mounts inside FastAPI on a single port.

---

## 13. Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI, Python 3.11 |
| LLM | Azure OpenAI (gpt-4o, gpt-4.1-mini) |
| Embeddings | Azure OpenAI text-embedding-3-small |
| Vector Store | ChromaDB (local, persistent) |
| PDF Parsing | pdfplumber + LLM vision fallback |
| DOCX Parsing | python-docx |
| BM25 Search | rank_bm25 |
| Schema Validation | Pydantic v2 |
| UI | Gradio |
| Deployment | Docker + HuggingFace Spaces |
| Eval | Custom harness (replaceable with Langfuse/Langsmith) |

---

## 14. Key Dependencies

```
fastapi
uvicorn
pdfplumber
python-docx
chromadb
openai
rank-bm25
pydantic>=2.0
pydantic-settings
gradio
httpx
python-multipart
```

---

## 15. Failure Cases & Known Limitations

- **Scanned PDFs** — pdfplumber fails, LLM vision fallback handles but at higher cost/latency
- **Very large documents** — full-context stuffing won't scale past ~50 pages; would need true chunked RAG
- **Multi-document queries** — not supported; each question targets one uploaded doc
- **Multi-turn conversation** — not supported; each question is independent
- **Rate limits** — Azure OpenAI rate limits could throttle under heavy eval runs

---

## 16. Improvement Ideas (for README)

- Langfuse/Langsmith for production prompt management and eval
- Reranker model (Cohere/cross-encoder) after hybrid retrieval
- Multi-document queries (cross-doc comparison)
- Multi-turn conversation with memory
- Fine-tuned extraction model for logistics-specific fields
- OCR pipeline for scanned documents (Tesseract/PaddleOCR)
- Webhook/async processing for large documents
- GraphRAG for relationship extraction across shipment entities
