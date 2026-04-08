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

DOC_TYPE_PROMPT_MAP = {
    DocType.BOL: "extraction_bol",
    DocType.RATE_CONFIRMATION: "extraction_rc",
    DocType.INVOICE: "extraction_generic",
    DocType.UNKNOWN: "extraction_generic",
}


def extract_shipment_data(full_text: str, doc_type: DocType, tracer: Tracer) -> dict:
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

    try:
        raw = json.loads(response.content)
        shipment = ShipmentData.model_validate(raw)
    except json.JSONDecodeError as e:
        logger.error(f"Extraction JSON parse failed: {e}")
        shipment = ShipmentData()
    except Exception as e:
        # Field-level fallback: try to salvage individual fields
        logger.warning(f"Extraction validation failed, attempting field-level fallback: {e}")
        try:
            salvaged = {}
            for field_name in ShipmentData.model_fields:
                if field_name in raw:
                    try:
                        partial = ShipmentData.model_validate({field_name: raw[field_name]})
                        salvaged[field_name] = getattr(partial, field_name)
                    except Exception:
                        salvaged[field_name] = None
            shipment = ShipmentData(**salvaged)
        except Exception:
            shipment = ShipmentData()

    completeness = shipment.completeness_score()

    with tracer.span("extraction_validation") as span:
        null_fields = [f for f in shipment.model_fields if getattr(shipment, f) is None]
        span.metadata = {
            "completeness_score": completeness,
            "null_fields": null_fields,
            "non_null_count": len(shipment.model_fields) - len(null_fields),
        }

    non_null = len(shipment.model_fields) - len([f for f in shipment.model_fields if getattr(shipment, f) is None])
    logger.info("extraction_complete", extra={"extra_data": {
        "fields_extracted": non_null, "completeness_score": completeness,
    }})
    return {
        "shipment_data": shipment,
        "completeness_score": completeness,
        "tokens_in": response.tokens_in,
        "tokens_out": response.tokens_out,
    }
