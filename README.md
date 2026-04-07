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

- Langfuse dashboard for production monitoring (integrated)
- Reranker model (Cohere/cross-encoder) after hybrid retrieval
- Multi-document queries and cross-doc comparison
- Fine-tuned extraction model for logistics fields
- OCR pipeline for scanned documents
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
