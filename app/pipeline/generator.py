"""Q&A generation with hidden Chain-of-Thought and confidence scoring."""

import re
from app.config import settings
from app.llm.provider import get_provider
from app.llm.prompts.system import *  # register prompts
from app.llm.prompts.qa import *  # register prompts
from app.llm.prompts.registry import registry
from app.models.schemas import ConfidenceResult, ConfidenceBreakdown, ConfidenceLevel
from app.observability.tracer import Tracer


def generate_answer(question: str, full_text: str, source_chunks: list[dict], tracer: Tracer) -> dict:
    prompt = registry.get("qa_with_cot")
    source_text = "\n\n".join(c["text"] for c in source_chunks[:3])
    rendered = prompt.render(document_text=full_text, source_chunks=source_text, question=question)
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


def compute_confidence(retrieval_score: float, grounding_ratio: float, llm_assessment: str) -> ConfidenceResult:
    assessment_map = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.2}
    llm_score = assessment_map.get(llm_assessment.upper(), 0.4)

    composite = (
        0.40 * min(retrieval_score, 1.0)
        + 0.35 * min(grounding_ratio, 1.0)
        + 0.25 * llm_score
    )
    composite = round(composite, 2)

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
    section = ""
    answer = response_text
    confidence = "MEDIUM"

    section_match = re.search(r"SECTION:\s*(.+?)(?:\n|$)", response_text)
    answer_match = re.search(r"ANSWER:\s*(.+?)(?:\nCONFIDENCE:|$)", response_text, re.DOTALL)
    confidence_match = re.search(r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)", response_text, re.IGNORECASE)

    if section_match:
        section = section_match.group(1).strip()
    if answer_match:
        answer = answer_match.group(1).strip()
    if confidence_match:
        confidence = confidence_match.group(1).strip().upper()

    return {"section": section, "answer": answer, "confidence": confidence}
