# Ultra Doc-Intelligence

AI-powered logistics document Q&A system that allows users to upload logistics documents and interact with them using natural language. Built as a TMS (Transportation Management System) AI assistant.

## Features

- **Document Upload & Processing** — PDF, DOCX, TXT with table-aware parsing
- **Natural Language Q&A** — Ask questions, get grounded answers with source text and confidence scores
- **Structured Extraction** — Extract shipment data (shipper, consignee, rate, dates, etc.) as JSON
- **4-Layer Guardrails** — Deterministic hallucination prevention (document-level + question + retrieval + grounding)
- **Hybrid Retrieval** — Vector search + BM25 + metadata filtering with RRF fusion
- **Observability** — Structured traces with per-stage latency and token tracking
- **Evaluation Framework** — Ground truth testing with accuracy metrics

## Architecture

```
User -> Gradio UI -> FastAPI API -> Pipeline Orchestrator
                                     |-- Parser (pdfplumber + fallback)
                                     |-- Classifier (LLM-based doc type detection)
                                     |-- Chunker (section-based, tables intact)
                                     |-- Embedder (configurable: local or Azure OpenAI -> ChromaDB)
                                     |-- Retriever (hybrid: vector + BM25 + metadata + RRF)
                                     |-- Generator (LangChain LLM + hidden CoT)
                                     |-- Extractor (Pydantic + function calling)
                                     |-- Guardrails (3 deterministic layers)
                                     +-- Langfuse (tracing + prompt management)
```

LLM calls go through **LangChain** as the abstraction layer — vendor-agnostic, swap providers (Azure OpenAI, OpenAI, etc.) via env var. **Langfuse** provides tracing for every pipeline stage and optional prompt management for production prompt iteration without redeploying.

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

Four deterministic layers — no extra LLM calls, zero additional cost:

| Layer | When | What | Example |
|-------|------|------|---------|
| **Layer 0: Document type** | After upload | Rejects non-logistics documents (resumes, contracts, etc.) | Resume uploaded → "Not a logistics document" |
| **Layer 1: Question scope** | Before retrieval | Keyword + word-boundary matching for UNKNOWN doc types. Skipped for classified docs (BOL/RC/Invoice) since the doc is already trusted | "What's the weather?" on unknown doc → blocked |
| **Layer 2: Retrieval threshold** | After retrieval | Rejects queries where best chunk similarity < threshold | Unrelated question → low similarity → "Not found in document" |
| **Layer 3: Grounding check** | After generation | Token overlap between answer and source text | Hallucinated answer → low overlap → flagged as LOW_GROUNDING |

Design decision: Layer 1 (question scope) is skipped for known logistics doc types. Rationale: if the classifier already confirmed it's a BOL, any question about it is valid — Layers 2 and 3 catch irrelevant questions via retrieval failure and grounding checks. This prevents false rejections like "what is this about?" on a real Rate Confirmation.

### Confidence Scoring

Composite score from 3 signals:
| Signal | Weight | Type |
|--------|--------|------|
| Retrieval similarity | 40% | Deterministic |
| Answer-source token overlap | 35% | Deterministic |
| LLM self-assessment | 25% | LLM-derived |

Each response includes the full breakdown for debuggability.

### Extraction Post-Processing

LLM-generated JSON goes through robust validation before returning:

- **Currency stripping**: `"$400.00 USD"` → `400.0` (handles $, €, £, ¥, commas)
- **Null normalization**: `"-"`, `"N/A"`, `"TBD"`, `""` → `null`
- **Date normalization**: `"02/08/2026"`, `"08-Feb-2026"` → `"2026-02-08T00:00:00"` (ISO format)
- **Field-level fallback**: If one field fails validation, only that field becomes null — the other 10 are preserved (not nuked)
- **COD/rate distinction**: Prompts explicitly instruct the LLM to not confuse COD value, insurance, or declared value with the freight rate

### Query Rewrite

Enabled by default. Before retrieval, the LLM rewrites the user's question:
1. **Typo correction**: `"caddress"` → `"address"`, `"shipmnt"` → `"shipment"`
2. **Terminology sharpening**: Uses precise logistics terms for better retrieval
3. Uses the fast model (gpt-4.1-mini) for low latency (~200ms)

### Synthetic Data & Evaluation

LLM-powered evaluation pipeline for testing at scale:

```bash
# Generate 160+ messy synthetic documents using Azure OpenAI
PYTHONPATH=. uv run python eval/generate_synthetic_data.py --count 160

# Run simulation against all docs + 3 sample PDFs
PYTHONPATH=. uv run python eval/run_simulation.py --sample-docs
```

- **12 messiness profiles**: typos, missing fields, special characters, legacy format exports, multi-language, contradictory info, scan artifacts, etc.
- **3 file formats**: TXT (50%), DOCX (30%), PDF (20%)
- **6 document types**: BOL, Rate Confirmation, Invoice, Shipment Instructions, Delivery Receipt, Freight Quote
- All generation prompts managed in Langfuse alongside app prompts
- Simulation report includes: accuracy, field-level extraction accuracy, latency (avg/p50/p95), confidence distribution, correct refusals

### Failure Cases

- Scanned/image PDFs with no text layer (no OCR — returns empty text warning)
- Documents > 50 pages (context window limit)
- Multi-document queries not supported (single-doc Q&A only)
- Questions requiring cross-document reasoning (e.g., comparing shipper RC rate vs carrier RC rate)
- BOLs typically don't contain numeric freight rates (charges are "Collect" or "Prepaid") — extraction correctly returns null

### Edge Cases Found & Fixed

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Resume uploaded, system answered questions about it | No document-level guardrail | Added Layer 0: NOT_LOGISTICS classification |
| `"$400.00 USD"` as rate crashed Pydantic, nuked all 11 fields | No post-processing on LLM output | Added field validators (currency strip, null normalize, date parse) + field-level fallback |
| COD value ($64,000) extracted as freight rate | Prompt didn't distinguish COD from rate | Added explicit COD exclusion to all extraction prompts |
| `carrier_name: "-"` stored as dash instead of null | No null-value normalization | Added NULL_VALUES set (`-`, `N/A`, `TBD`, etc.) |
| "what is this logistic about?" rejected on valid RC | Scope check ran on all questions | Scope check skipped for classified doc types (Layer 2+3 catch irrelevant queries) |
| "what is this doc about?" passed scope check | `"doc"` was a keyword (substring match) | Switched to word-boundary regex matching |
| Typo "caddress" → interpreted as "customs address" | Query rewrite didn't fix typos first | Updated rewrite prompt: fix typos before applying terminology |
| Default MEDIUM confidence on CoT parse failure | Parser defaulted to MEDIUM | Changed default to LOW |

### Improvement Ideas

- Hybrid extraction: regex/rule-based first pass for obvious fields (dates, amounts, IDs), LLM for ambiguous fields
- Reranker model (Cohere/cross-encoder) after hybrid retrieval
- Multi-document queries and cross-doc comparison
- Separate customer RC vs carrier RC extraction prompts
- OCR pipeline for scanned documents (Tesseract/Azure Document Intelligence)
- LLM-based question scope check as a fallback for keyword matching
- Cache persistence (SQLite) to survive container restarts
- GraphRAG for entity relationship extraction

## Quick Start

### Local (Docker)

```bash
git clone <repo-url>
cd ultradoc-intelligence
cp .env.example .env  # Fill in your API keys (Azure OpenAI, Langfuse, etc.)

docker-compose up --build

# App (API + UI): http://localhost:7860
# Swagger docs:   http://localhost:7860/docs
# Gradio UI:      http://localhost:7860/ui
```

### Local (Python with uv)

```bash
cp .env.example .env  # Fill in your API keys
uv sync
uv run python run.py
# App runs on http://localhost:7860
```

### API Endpoints

Full Swagger docs available at `/docs`. Gradio UI at `/ui`.

```bash
# Upload a document
curl -X POST http://localhost:7860/upload -F "file=@document.pdf"

# Ask a question
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "...", "question": "What is the carrier rate?"}'

# Extract structured data
curl -X POST http://localhost:7860/extract \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "..."}'
```

### Run Evaluation

```bash
# Start the server first, then:
uv run python -m eval.run_eval

# Compare with query rewriting:
uv run python -m eval.run_eval --flags '{"enable_query_rewrite": true}'
```

## Configuration

Copy `.env.example` to `.env` and set the relevant keys. Key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM backend to use | `azure_openai` |
| `EMBEDDING_PROVIDER` | `local` (all-MiniLM-L6-v2) or `azure` | `local` |
| `LANGFUSE_ENABLED` | Enable Langfuse tracing | `false` |
| `LANGFUSE_PROMPT_MANAGEMENT` | Fetch prompts from Langfuse instead of local templates | `false` |

See `.env.example` for the full list (Azure OpenAI credentials, Langfuse keys, etc.).

## Development Approach

### Phase 1: Core Pipeline
Built the end-to-end RAG pipeline: PDF/DOCX/TXT parser → section-based chunker → ChromaDB embeddings → hybrid retrieval (vector + BM25 + RRF) → LLM generation with hidden CoT → 3-layer deterministic guardrails → composite confidence scoring. FastAPI API + Gradio UI.

### Phase 2: Production Hardening
Replaced bare OpenAI SDK with **LangChain** for vendor-agnostic LLM access. Added **Langfuse** for tracing and prompt management. Consolidated to single-port deployment (FastAPI + Gradio mounted together) for HuggingFace Spaces compatibility.

### Phase 3: Evaluation & Edge Case Discovery
- Built LLM-powered synthetic data generator (160+ messy documents across 12 messiness profiles)
- All prompts (app + synthetic generation) managed through Langfuse
- Ran systematic edge case audits discovering 15+ issues across extraction, guardrails, and retrieval
- Fixed critical bugs: extraction post-processing, field-level validation fallback, scope guardrail precision

### Phase 4: Testing
104 automated tests covering:
- Unit tests: parser, chunker, retriever, confidence scoring
- Integration tests: all API endpoints, Gradio mount, Swagger docs
- Security tests: prompt injection (6 attack types), hallucination grounding, adversarial inputs
- Provider tests: LangChain provider, config defaults, model selection
- Langfuse tests: handler/client factories, prompt registry fallback

### Key Design Decisions
1. **Deterministic guardrails over LLM-based** — Zero extra cost, predictable, debuggable. LLM-based scope check considered but rejected: Layers 2+3 already catch edge cases at retrieval/grounding level.
2. **LangChain over bare SDK** — Vendor agnostic. Swap Azure OpenAI → OpenAI → Anthropic → Ollama with one env var change.
3. **Langfuse over custom tracing** — Production-grade observability + prompt versioning. Custom tracer kept for the `/traces` endpoint.
4. **Local embeddings by default** — No API key needed for ChromaDB embeddings. Azure OpenAI embeddings available via `EMBEDDING_PROVIDER=azure`.
5. **Query rewrite on by default** — Handles typos and vague questions. Users can disable for raw queries.
6. **Field-level extraction fallback** — One bad field (e.g., rate with currency symbol) doesn't nuke the other 10 correct fields.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI, Python 3.11 |
| LLM | LangChain (langchain-openai) — Azure OpenAI (gpt-4o, gpt-4.1-mini) |
| Embeddings | Configurable: local (all-MiniLM-L6-v2) or Azure OpenAI (text-embedding-3-small) |
| Vector Store | ChromaDB |
| Observability | Langfuse (tracing, prompt management) |
| PDF Parsing | pdfplumber |
| BM25 Search | rank_bm25 |
| Validation | Pydantic v2 |
| UI | Gradio |
| Deployment | Docker, HuggingFace Spaces |
