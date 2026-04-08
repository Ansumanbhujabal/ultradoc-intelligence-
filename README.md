---
title: Ultra Doc-Intelligence
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Ultra Doc-Intelligence

An AI-powered logistics document Q&A system that lets users upload shipping documents (BOL, Rate Confirmations, Invoices) and interact with them using natural language. Built as a proof-of-concept AI assistant for a Transportation Management System (TMS). The system retrieves relevant content, answers grounded questions with source citations, applies multi-layer guardrails against hallucination, returns calibrated confidence scores, and extracts structured shipment data -- all through a clean API and lightweight UI.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Docker](#docker)
- [Environment Variables](#environment-variables)
- [API Endpoints](#api-endpoints)
- [Architecture Overview](#architecture-overview)
- [Chunking Strategy](#chunking-strategy)
- [Retrieval Method](#retrieval-method)
- [Guardrails Approach](#guardrails-approach)
- [Confidence Scoring](#confidence-scoring)
- [Evaluation Results](#evaluation-results)
- [Commands Reference](#commands-reference)
- [Failure Cases and Edge Cases](#failure-cases-and-edge-cases)
- [Improvement Ideas](#improvement-ideas)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)

---

## Quick Start

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/), Azure OpenAI API access

```bash
git clone <repo-url>
cd ultradoc-intelligence

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your Azure OpenAI credentials (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT)

# Start the server
uv run python run.py
```

The app runs on **http://localhost:7860**:
- Gradio UI: http://localhost:7860/ui
- Swagger docs: http://localhost:7860/docs

---

## Docker

Single command to build and run:

```bash
cp .env.example .env   # Fill in your API keys first
docker compose up --build
```

The container exposes port 7860. Data is persisted via a volume mount to `./data`.

---

## Environment Variables

Copy `.env.example` to `.env` and configure. All variables are loaded via Pydantic Settings.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AZURE_OPENAI_API_KEY` | Yes | -- | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Yes | -- | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_VERSION` | No | `2024-02-15-preview` | API version |
| `AZURE_OPENAI_MODEL` | No | `gpt-4o` | Primary LLM model |
| `AZURE_OPENAI_FAST_MODEL` | No | `gpt-4.1-mini` | Fast model for query rewrite, classification |
| `AZURE_OPENAI_EMBEDDING_MODEL` | No | `text-embedding-3-small` | Azure embedding model (when `EMBEDDING_PROVIDER=azure`) |
| `LLM_PROVIDER` | No | `azure_openai` | LLM backend (`azure_openai`, `openai`, etc.) |
| `EMBEDDING_PROVIDER` | No | `local` | `local` (all-MiniLM-L6-v2, no API key) or `azure` |
| `CHROMA_PERSIST_DIR` | No | `./data/chroma` | ChromaDB storage path |
| `UPLOAD_DIR` | No | `./data/uploads` | Uploaded file storage path |
| `DEFAULT_CONFIDENCE_THRESHOLD` | No | `0.35` | Minimum similarity for retrieval |
| `DEFAULT_RETRIEVAL_MODE` | No | `hybrid` | Retrieval strategy |
| `GROUNDING_OVERLAP_THRESHOLD` | No | `0.4` | Token overlap threshold for grounding check |
| `LOW_CONFIDENCE_REFUSAL_THRESHOLD` | No | `0.3` | Below this, system refuses to answer |
| `RETRIEVAL_TOP_K` | No | `5` | Chunks retrieved per strategy |
| `FINAL_TOP_K` | No | `3` | Chunks passed to LLM after fusion |
| `LANGFUSE_ENABLED` | No | `false` | Enable Langfuse tracing |
| `LANGFUSE_PUBLIC_KEY` | No | -- | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | No | -- | Langfuse secret key |
| `LANGFUSE_HOST` | No | `https://us.cloud.langfuse.com` | Langfuse host URL |
| `LANGFUSE_PROMPT_MANAGEMENT` | No | `false` | Fetch prompts from Langfuse instead of local templates |
| `API_PORT` | No | `7860` | Server port |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `LOG_FILE` | No | `./logs/ultradoc.log` | Log file path |

---

## API Endpoints

Full interactive docs at `/docs` (Swagger UI). Gradio UI at `/ui`.

### POST /upload

Upload a logistics document (PDF, DOCX, or TXT).

```bash
curl -X POST http://localhost:7860/upload \
  -F "file=@rate_confirmation.pdf"
```

Returns a `doc_id` used for subsequent queries.

### POST /ask

Ask a natural language question about an uploaded document.

```bash
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "abc123",
    "question": "What is the carrier rate?"
  }'
```

Returns: answer, supporting source text, confidence score with breakdown, guardrail status.

### POST /extract

Extract structured shipment data from an uploaded document.

```bash
curl -X POST http://localhost:7860/extract \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "abc123"}'
```

Returns JSON with fields: `shipment_id`, `shipper`, `consignee`, `pickup_datetime`, `delivery_datetime`, `equipment_type`, `mode`, `rate`, `currency`, `weight`, `carrier_name`. Missing fields are `null`.

---

## Architecture Overview

```
User --> Gradio UI --> FastAPI API --> Pipeline Orchestrator
                                       |-- Parser (pdfplumber + python-docx + TXT)
                                       |-- Classifier (LLM-based doc type detection)
                                       |-- Chunker (section-based, tables intact)
                                       |-- Embedder (local or Azure OpenAI --> ChromaDB)
                                       |-- Retriever (hybrid: vector + BM25 + metadata + RRF)
                                       |-- Generator (LangChain LLM + hidden CoT)
                                       |-- Extractor (Pydantic + function calling)
                                       |-- Guardrails (4 deterministic layers)
                                       +-- Langfuse (tracing + prompt management)
```

**Request flows:**
- **Upload:** File --> Parse --> Classify doc type --> Chunk by section --> Embed --> Store in ChromaDB --> return `doc_id`
- **Ask:** Question --> Scope guard --> Hybrid retrieve --> Threshold guard --> LLM generate (hidden CoT) --> Grounding guard --> Confidence score --> Answer
- **Extract:** doc_id --> Type-specific prompt --> JSON mode --> Pydantic validate + post-process --> ShipmentData

LLM calls go through **LangChain** for vendor abstraction -- swap providers (Azure OpenAI, OpenAI, Anthropic, Ollama) via a single env var. **Langfuse** provides optional tracing and prompt management for production iteration without redeployment.

For full architectural details and design decisions, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Chunking Strategy

The system uses **section-based chunking** that detects logistics document headers (Carrier Details, Stops, Rate Breakdown, etc.) and splits on section boundaries rather than arbitrary character counts. Tables are preserved intact within their parent chunk. This matters because naive character-splitting destroys tabular structure -- a rate table split mid-row produces two meaningless fragments that neither vector search nor LLM generation can use correctly.

---

## Retrieval Method

The system uses **hybrid retrieval** combining three signals fused with Reciprocal Rank Fusion (RRF):

1. **Vector search** (ChromaDB, cosine similarity) -- semantic matching for paraphrased questions ("Who is receiving the shipment?" matches "consignee" chunks)
2. **BM25 keyword search** -- exact term matching critical for IDs, names, and numbers (e.g., "LD53657")
3. **Metadata filtering** -- section header matching for field-specific questions (e.g., "What equipment type?" prioritizes "Carrier Details" sections)

No single retrieval strategy handles the full range of logistics queries. Vector search misses exact identifiers; BM25 misses semantic paraphrases; metadata filtering narrows results when the question maps to a known document section. RRF fusion combines their rankings without needing score calibration across strategies.

Optional **query rewriting** (enabled by default) reformulates questions before retrieval -- fixing typos and sharpening terminology using the fast model.

---

## Guardrails Approach

Four deterministic layers that prevent hallucination without additional LLM calls (zero extra cost, fully predictable):

| Layer | Stage | What It Does |
|-------|-------|-------------|
| **Layer 0: Document type** | After upload | Classifies the document. Rejects non-logistics documents (resumes, contracts, recipes). A resume upload returns "Not a logistics document" instead of hallucinated answers. |
| **Layer 1: Question scope** | Before retrieval | Keyword + word-boundary matching blocks off-topic questions on unclassified documents. Skipped for classified logistics docs (BOL, RC, Invoice) because Layers 2-3 handle scope naturally via retrieval failure. |
| **Layer 2: Retrieval threshold** | After retrieval | Rejects queries where the best chunk similarity falls below the configured threshold. An unrelated question produces low similarity and triggers "Not found in document." |
| **Layer 3: Grounding check** | After generation | Measures token overlap between the generated answer and source chunks. A hallucinated answer with low overlap is flagged as LOW_GROUNDING and the confidence score is penalized. |

Additionally, a **post-generation refusal gate** detects when the LLM itself says "not found in document" and converts it to a proper NOT_FOUND response. A **low-confidence refusal** triggers when composite confidence is below 0.3 and the LLM self-assessed LOW -- strong signal the answer is unreliable.

Design decision: For classified logistics docs (BOL, RC, Invoice), Layer 1 only blocks obviously off-topic questions (weather, sports, etc.). Generic questions like "Who is the customer?" pass through to retrieval -- Layers 2 and 3 catch genuinely unanswerable questions through retrieval failure and grounding checks. This prevents false rejections on valid documents.

---

## Confidence Scoring

Each answer includes a **weighted composite confidence score** built from three independent signals:

| Signal | Weight | Type | What It Measures |
|--------|--------|------|-----------------|
| Retrieval similarity | 40% | Deterministic | How well the retrieved chunks match the question (cosine similarity) |
| Answer-source token overlap | 35% | Deterministic | How grounded the answer is in the source text (token-level overlap ratio) |
| LLM self-assessment | 25% | LLM-derived | Model's own certainty extracted from hidden chain-of-thought |

The full breakdown is returned with every response for transparency and debuggability. When the composite score falls below the refusal threshold (default 0.3), the system refuses to answer rather than returning a low-confidence guess.

---

## Evaluation Results

### Sample Documents (Company Test Data -- 3 PDFs, 16 test cases)

| Metric | Score |
|--------|-------|
| Q&A Accuracy | **100.0% (14/14)** |
| Extraction Field Accuracy | **100.0% (12/12)** |
| Correct Refusals | **2/2 (100%)** |
| False Acceptances | **0** |
| Avg Q&A Latency | 5.2s |
| Avg Confidence | 0.69 |

### Synthetic Stress Test (160 docs, 960 test cases, 12 messiness profiles)

| Metric | Score |
|--------|-------|
| Q&A Accuracy | **91.9% (735/800)** |
| Correct Refusals | **315/320 (98.4%)** |
| False Acceptances | **5** |
| Extraction Field Accuracy | 66.5% (1376/2069) |
| Extraction Completeness | 95.3% |
| Avg Q&A Latency | 2.2s |

Synthetic extraction accuracy is lower because LLM-generated messy documents (OCR artifacts, contradictory data, missing fields) produce ambiguous ground truth for date/rate fields. On real-world sample documents, extraction is 100%.

For the full evaluation report with confidence distributions, latency percentiles, and per-case breakdown, see [eval/EVAL_RESULTS.md](eval/EVAL_RESULTS.md).

The evaluation framework includes:
- **Ground truth tests** against 3 sample logistics documents (BOL, Carrier RC, Shipper RC) with 16 test cases covering Q&A accuracy, extraction accuracy, out-of-scope rejection, and not-found handling
- **Synthetic simulation** across 160+ LLM-generated messy documents (12 messiness profiles, 6 doc types, 3 file formats)
- **Metrics tracked:** answer accuracy, field-level extraction accuracy, latency (avg/p50/p95), confidence distribution, correct refusals

---

## Commands Reference

| Command | Description |
|---------|-------------|
| `uv run python run.py` | Start the server (API + UI on port 7860) |
| `uv run pytest -v` | Run all 104 tests (unit + integration + security) |
| `PYTHONPATH=. uv run python eval/run_simulation.py --sample-only` | Run eval on 3 sample PDFs only |
| `PYTHONPATH=. uv run python eval/run_simulation.py` | Run full eval on all 160 synthetic docs |
| `PYTHONPATH=. uv run python eval/run_simulation.py --max-docs 30` | Run eval on first 30 synthetic docs (quick) |
| `PYTHONPATH=. uv run python eval/compile_results.py` | Compile eval results into formatted report |
| `PYTHONPATH=. uv run python eval/generate_synthetic_data.py` | Generate 160 synthetic messy documents |
| `PYTHONPATH=. uv run python scripts/seed_langfuse_prompts.py` | Seed all 10 prompts to Langfuse |
| `tail -f logs/ultradoc.log \| python -m json.tool` | View structured logs (pretty-printed) |
| `grep "ask_refused" logs/ultradoc.log` | Debug guardrail refusals |
| `docker compose up --build` | Build and run via Docker |
| `docker compose down` | Stop Docker containers |

---

## Failure Cases and Edge Cases

**Known limitations:**

- **Scanned/image PDFs** with no text layer return empty text (no OCR pipeline -- would need Tesseract or Azure Document Intelligence)
- **Documents over 50 pages** may exceed context window limits
- **Multi-document queries** are not supported -- the system operates on a single uploaded document at a time
- **Cross-document reasoning** (e.g., comparing shipper RC rate vs carrier RC rate) requires uploading and querying documents separately
- **BOLs typically lack numeric freight rates** -- charges are listed as "Collect" or "Prepaid", so extraction correctly returns `null` for the rate field

**Edge cases discovered and fixed during development:**

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Resume uploaded, system answered questions about it | No document-level guardrail | Added Layer 0: NOT_LOGISTICS classification |
| `"$400.00 USD"` as rate crashed Pydantic, nuked all 11 fields | No post-processing on LLM output | Added currency stripping, null normalization, date parsing + field-level fallback |
| COD value ($64,000) extracted as freight rate | Prompt did not distinguish COD from rate | Added explicit COD exclusion to extraction prompts |
| `carrier_name: "-"` stored as dash instead of null | No null-value normalization | Added NULL_VALUES set (`-`, `N/A`, `TBD`, etc.) |
| "what is this logistic about?" rejected on valid RC | Scope check ran on all questions | Scope check skipped for classified doc types |
| "what is this doc about?" passed scope check | `"doc"` was a keyword (substring match) | Switched to word-boundary regex matching |
| Typo "caddress" interpreted as "customs address" | Query rewrite did not fix typos first | Updated rewrite prompt: fix typos before applying terminology |
| Default MEDIUM confidence on CoT parse failure | Parser defaulted to MEDIUM | Changed default to LOW |

---

## Improvement Ideas

What would move this from POC to production:

- **Hybrid extraction** -- regex/rule-based first pass for obvious fields (dates, amounts, IDs), LLM for ambiguous fields only
- **Reranker model** (Cohere or cross-encoder) after hybrid retrieval for better precision
- **Multi-document support** -- cross-document queries and comparison (e.g., shipper vs carrier rate discrepancies)
- **OCR pipeline** for scanned documents (Tesseract or Azure Document Intelligence)
- **LLM-based scope check** as a fallback layer for the keyword-based guardrail
- **Persistent cache** (SQLite or Redis) to survive container restarts
- **GraphRAG** for entity relationship extraction across shipment parties
- **Separate extraction prompts** per document subtype (customer RC vs carrier RC have different rate semantics)

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| API | FastAPI | Async, auto-generated OpenAPI docs, Pydantic-native validation |
| LLM Abstraction | LangChain (langchain-openai) | Vendor-agnostic -- swap Azure/OpenAI/Anthropic/Ollama via env var |
| LLM Models | Azure OpenAI (GPT-4o, GPT-4.1-mini) | GPT-4o for generation/extraction, 4.1-mini for fast tasks (classification, rewrite) |
| Embeddings | all-MiniLM-L6-v2 (local default) | No API key needed, fast, good quality. Azure option available. |
| Vector Store | ChromaDB | Zero-config, embedded, persistent, no infrastructure to manage |
| BM25 Search | rank-bm25 | Lexical retrieval for exact term matching (IDs, numbers, names) |
| PDF Parsing | pdfplumber | Table-aware extraction, not just raw text |
| DOCX Parsing | python-docx | Native Word document support |
| Validation | Pydantic v2 | Strict schema enforcement for extraction output |
| Observability | Langfuse | Tracing per pipeline stage, prompt versioning, cost tracking |
| UI | Gradio | Functional interface in minutes, mounts directly on FastAPI |
| Deployment | Docker, HuggingFace Spaces | Single-port container, cloud-hostable |
| Testing | pytest (104 tests) | Unit, integration, security (prompt injection), provider tests |

---

## Project Structure

```
ultradoc-intelligence/
|-- app/
|   |-- main.py              # FastAPI app, endpoint definitions
|   |-- config.py             # Pydantic Settings (all env vars)
|   |-- models/               # Pydantic data models (ShipmentData, etc.)
|   |-- pipeline/             # Core logic: parser, chunker, retriever, generator, extractor, guardrails
|   |-- llm/                  # LangChain provider, prompt templates
|   |-- storage/              # ChromaDB vector store
|   |-- observability/        # Langfuse integration, tracing
|-- ui/                       # Gradio UI
|-- eval/
|   |-- ground_truth.json     # Test cases for 3 sample documents
|   |-- run_simulation.py     # Eval runner (sample + synthetic)
|   |-- compile_results.py    # Results formatter (terminal + markdown)
|   |-- generate_synthetic_data.py  # LLM-powered messy doc generator
|   |-- EVAL_RESULTS.md       # Compiled evaluation report
|   |-- synthetic_data/       # Generated synthetic documents + ground truth
|-- tests/                    # 104 tests (unit, integration, security)
|-- scripts/                  # Utility scripts (Langfuse prompt seeding, etc.)
|-- docs/
|   |-- ARCHITECTURE.md       # Full architectural reference
|-- data/                     # ChromaDB storage + uploaded files
|-- logs/                     # Application logs
|-- Dockerfile
|-- docker-compose.yml
|-- pyproject.toml
|-- run.py                    # Server entrypoint
```
