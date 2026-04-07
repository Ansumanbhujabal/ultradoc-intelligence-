# LangChain + Langfuse Refactor — Design Spec

**Date:** 2026-04-07
**Goal:** Replace bare OpenAI SDK calls with LangChain for vendor-agnostic LLM access, add Langfuse for tracing and prompt management. Deploy on HuggingFace Spaces free tier.

**Motivation:** Bare `openai.AzureOpenAI` calls create vendor lock-in. LangChain is listed in the JD qualifications. Langfuse adds production-grade observability and prompt lifecycle management — both strong signals for the reviewer.

**Constraint:** "Clarity and correctness > framework complexity." This refactor replaces the LLM call layer only. Pipeline logic, guardrails, chunking, retrieval, and confidence scoring stay untouched.

---

## 1. LLM Provider Layer

### Current State
- `app/llm/provider.py` has `LLMProvider` ABC + `AzureOpenAIProvider`
- Direct `openai.AzureOpenAI` SDK calls for `generate()`, `generate_structured()`, `embed()`
- Singleton via `get_provider()`
- 4 call sites: `classifier.py`, `generator.py` (x2), `extractor.py`

### New Design
Replace `AzureOpenAIProvider` with `LangChainProvider`. Keep `LLMProvider` ABC and `LLMResponse` dataclass for backward compat.

```python
class LangChainProvider(LLMProvider):
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_deployment=settings.azure_openai_model,
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
        self.fast_llm = AzureChatOpenAI(
            azure_deployment=settings.azure_openai_fast_model,
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
        if settings.embedding_provider == "azure":
            self.embeddings = AzureOpenAIEmbeddings(
                model=settings.azure_openai_embedding_model,
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                openai_api_version=settings.azure_openai_api_version,
            )
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

**Methods:**
- `generate(messages, model, **kwargs)` → `self.llm.invoke(messages, config={"callbacks": [langfuse_handler]})` → map to `LLMResponse`
- `generate_structured(messages, schema, model, **kwargs)` → `self.llm.bind(response_format={"type": "json_object"}).invoke(...)` → map to `LLMResponse`
- `embed(texts)` → `self.embeddings.embed_documents(texts)`

**Model selection:** `model` param selects `self.llm` (default) or `self.fast_llm` (fast_model). Callers pass `model=provider.fast_model` as before.

**Vendor swap:** Change `LLM_PROVIDER` env var + add provider-specific keys. No pipeline code changes. Future providers (Anthropic, Ollama) implement same ABC.

---

## 2. Langfuse Integration

### 2a. Tracing

New module: `app/observability/langfuse_integration.py`

```python
from langfuse.langchain import CallbackHandler

def get_langfuse_handler(trace_name: str, metadata: dict = None):
    if not settings.langfuse_enabled:
        return None
    return CallbackHandler(
        trace_name=trace_name,
        metadata=metadata,
    )
```

- Every LLM call passes handler via `config={"callbacks": [handler]}`
- When `LANGFUSE_ENABLED=false`, returns `None` — callbacks list is empty, zero overhead
- Captures: model, tokens, latency, prompt/completion, cost automatically

Existing `app/observability/tracer.py` stays. It powers the `/traces` API endpoint. Langfuse is additive — captures LLM-level detail the custom tracer doesn't.

### 2b. Prompt Management

Modify `app/llm/prompts/registry.py`:

```python
class PromptRegistry:
    def get(self, name: str, version: str = "v1", **kwargs) -> str:
        if settings.langfuse_prompt_management:
            try:
                langfuse = get_client()
                prompt = langfuse.get_prompt(name, version=version)
                return prompt.compile(**kwargs)
            except Exception:
                pass  # fall through to local
        return self._local_templates[name].render(**kwargs)
```

- `LANGFUSE_PROMPT_MANAGEMENT=true` → fetch from Langfuse cloud, compile with template variables
- `LANGFUSE_PROMPT_MANAGEMENT=false` or Langfuse unreachable → use local templates (current behavior)
- All 7 existing prompts (system, classification, query_rewrite, qa_with_cot, extraction_bol, extraction_rc, extraction_generic) remain as local fallbacks
- One-time seed script to upload local prompts to Langfuse

### Env vars

```
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://us.cloud.langfuse.com
LANGFUSE_PROMPT_MANAGEMENT=false
```

---

## 3. Single-Port Deployment

### Current State
- `run.py` spawns FastAPI (port 8000) and Gradio (port 7860) in separate threads
- `ui/gradio_app.py` uses `httpx` to call `http://localhost:8000`

### New Design
Mount Gradio as a sub-app on FastAPI. Single port (7860).

```python
# run.py
import gradio as gr
from app.main import app
from ui.gradio_app import build_ui

demo = build_ui()
app = gr.mount_gradio_app(app, demo, path="/ui")
uvicorn.run(app, host="0.0.0.0", port=settings.api_port)
```

**Route map (port 7860):**
```
/docs          → Swagger UI
/redoc         → ReDoc
/upload        → POST API
/ask           → POST API
/extract       → POST API
/health        → GET
/documents     → GET
/traces        → GET
/ui            → Gradio app
```

**UI change:** `ui/gradio_app.py` calls pipeline functions directly (same process) instead of httpx to localhost. This eliminates the internal HTTP hop.

**Locally:** Same single-port behavior. Set `API_PORT=7860` in `.env`.

---

## 4. Embedding Configuration

```python
# app/config.py
embedding_provider: str = "local"  # "local" | "azure"
```

- `EMBEDDING_PROVIDER=local` → `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` — free, no API key, works on HF Spaces
- `EMBEDDING_PROVIDER=azure` → `AzureOpenAIEmbeddings(model=settings.azure_openai_embedding_model)` — better quality, costs money

Default is `local`. One env var change to switch.

ChromaDB continues to use its own internal embeddings for vector storage. The provider embeddings are available if callers need them directly.

---

## 5. Dependency Changes

**Adding (via `uv add`):**
- `langchain-openai >= 0.3.0` — AzureChatOpenAI, AzureOpenAIEmbeddings
- `langchain-community >= 0.3.0` — HuggingFaceEmbeddings
- `langfuse >= 3.8.0` — Tracing, prompt management, LangChain callback

**Removing:**
- `openai` as direct dependency (transitive via langchain-openai)

**Keeping:** All existing deps unchanged.

---

## 6. Config Changes

New fields in `app/config.py` Settings:

```python
# Embedding
embedding_provider: str = "local"

# Langfuse
langfuse_enabled: bool = False
langfuse_public_key: str = ""
langfuse_secret_key: str = ""
langfuse_host: str = "https://us.cloud.langfuse.com"
langfuse_prompt_management: bool = False
```

New file: `.env.example` (committed, no secrets):
```env
# LLM
LLM_PROVIDER=azure_openai
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_MODEL=gpt-4o
AZURE_OPENAI_FAST_MODEL=gpt-4.1-mini
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Embeddings
EMBEDDING_PROVIDER=local

# Langfuse
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://us.cloud.langfuse.com
LANGFUSE_PROMPT_MANAGEMENT=false

# Pipeline
DEFAULT_CONFIDENCE_THRESHOLD=0.1
DEFAULT_RETRIEVAL_MODE=hybrid
GROUNDING_OVERLAP_THRESHOLD=0.4
RETRIEVAL_TOP_K=5
FINAL_TOP_K=3

# Storage
CHROMA_PERSIST_DIR=./data/chroma
UPLOAD_DIR=./data/uploads

# Server
API_PORT=7860
LOG_LEVEL=INFO
```

---

## 7. File Change Map

| File | Action | Description |
|------|--------|-------------|
| `app/llm/provider.py` | Rewrite | LangChain-based provider, same ABC |
| `app/llm/prompts/registry.py` | Modify | Add Langfuse prompt fetch + local fallback |
| `app/config.py` | Modify | Add embedding, Langfuse settings |
| `app/pipeline/classifier.py` | Modify | Update provider call pattern |
| `app/pipeline/generator.py` | Modify | Update provider call pattern |
| `app/pipeline/extractor.py` | Modify | Update provider call pattern |
| `app/observability/langfuse_integration.py` | New | Langfuse callback handler factory |
| `run.py` | Rewrite | Single-port FastAPI + Gradio mount |
| `ui/gradio_app.py` | Modify | Direct pipeline calls, no httpx |
| `pyproject.toml` | Modify | Add 3 deps via uv |
| `.env.example` | New | Template env file, no secrets |
| `README.md` | Modify | Update architecture, add Langfuse section |

**Untouched:**
- `app/pipeline/parser.py` — Document parsing
- `app/pipeline/chunker.py` — Section-based chunking
- `app/pipeline/guardrails.py` — 3-layer guardrail system
- `app/pipeline/retriever.py` — Hybrid retrieval (vector + BM25 + RRF)
- `app/pipeline/embedder.py` — ChromaDB storage
- `app/storage/` — Cache, vector store
- `app/models/` — Pydantic schemas
- `app/llm/prompts/*.py` — Prompt templates (content unchanged)
- `eval/` — Evaluation framework
- `tests/` — Existing unit tests (may need import updates)

---

## 8. HuggingFace Spaces Deployment

**Dockerfile:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install uv && uv sync --no-dev
EXPOSE 7860
CMD ["uv", "run", "python", "run.py"]
```

**Secrets** (set in HF Space Settings → Variables and Secrets):
- `AZURE_OPENAI_API_KEY` — secret
- `AZURE_OPENAI_ENDPOINT` — variable
- `LANGFUSE_SECRET_KEY` — secret
- `LANGFUSE_PUBLIC_KEY` — variable

**Constraints:**
- 16 GB RAM, 2 CPU cores
- Only port 7860 exposed
- Write only to `/tmp`
- Outbound: ports 80, 443, 8080 only

**Adjustments:**
- `CHROMA_PERSIST_DIR=/tmp/chroma` and `UPLOAD_DIR=/tmp/uploads` for HF Spaces
- Single port 7860 (handled by Section 3)
