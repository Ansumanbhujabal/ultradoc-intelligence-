"""3-layer deterministic guardrail system."""
import re
from app.models.schemas import GuardrailStatus
from app.llm.prompts.guardrails import is_in_scope
from app.observability.logger import get_logger

logger = get_logger("guardrails")


def check_scope(question: str) -> GuardrailStatus:
    """Layer 1: Pre-retrieval out-of-scope check (no LLM call)."""
    if not question.strip():
        return GuardrailStatus.OUT_OF_SCOPE
    if is_in_scope(question):
        logger.info("guardrail_scope", extra={"extra_data": {"result": "passed"}})
        return GuardrailStatus.PASSED
    logger.info("guardrail_scope", extra={"extra_data": {"result": "out_of_scope"}})
    return GuardrailStatus.OUT_OF_SCOPE


def check_retrieval_threshold(chunks: list[dict], threshold: float) -> GuardrailStatus:
    """Layer 2: Post-retrieval — check if best chunk meets similarity threshold."""
    if not chunks:
        return GuardrailStatus.NOT_FOUND
    best_similarity = max(c.get("similarity", 0) for c in chunks)
    if best_similarity < threshold:
        logger.info("guardrail_threshold", extra={"extra_data": {
            "result": "not_found", "best_similarity": best_similarity, "threshold": threshold,
        }})
        return GuardrailStatus.NOT_FOUND
    logger.info("guardrail_threshold", extra={"extra_data": {
        "result": "passed", "best_similarity": best_similarity, "threshold": threshold,
    }})
    return GuardrailStatus.PASSED


def check_grounding(answer: str, source_text: str, threshold: float = 0.4) -> tuple[GuardrailStatus, float]:
    """Layer 3: Post-generation — check answer-source token overlap."""
    if not answer or not source_text:
        return GuardrailStatus.LOW_GROUNDING, 0.0

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
        logger.info("guardrail_grounding", extra={"extra_data": {
            "result": "low_grounding", "overlap_ratio": round(ratio, 2), "threshold": threshold,
        }})
        return GuardrailStatus.LOW_GROUNDING, round(ratio, 2)
    logger.info("guardrail_grounding", extra={"extra_data": {
        "result": "passed", "overlap_ratio": round(ratio, 2), "threshold": threshold,
    }})
    return GuardrailStatus.PASSED, round(ratio, 2)
