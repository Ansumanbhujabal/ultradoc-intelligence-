"""Document type classification using LLM."""

from app.llm.provider import get_provider
from app.llm.prompts.system import *  # registers prompts
from app.llm.prompts.registry import registry
from app.models.schemas import DocType
from app.observability.tracer import Tracer


def classify_document(text: str, tracer: Tracer) -> DocType:
    """Classify a logistics document by type using LLM."""
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
        "not_logistics": DocType.NOT_LOGISTICS,
        "unknown": DocType.UNKNOWN,
    }
    return type_map.get(result, DocType.UNKNOWN)
