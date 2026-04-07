"""3-layer deterministic guardrail system."""
import re
from app.models.schemas import GuardrailStatus
from app.llm.prompts.guardrails import is_in_scope


def check_scope(question: str) -> GuardrailStatus:
    """Layer 1: Pre-retrieval out-of-scope check (no LLM call)."""
    if not question.strip():
        return GuardrailStatus.OUT_OF_SCOPE
    if is_in_scope(question):
        return GuardrailStatus.PASSED
    return GuardrailStatus.OUT_OF_SCOPE


def check_retrieval_threshold(chunks: list[dict], threshold: float) -> GuardrailStatus:
    """Layer 2: Post-retrieval — check if best chunk meets similarity threshold."""
    if not chunks:
        return GuardrailStatus.NOT_FOUND
    best_similarity = max(c.get("similarity", 0) for c in chunks)
    if best_similarity < threshold:
        return GuardrailStatus.NOT_FOUND
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
        return GuardrailStatus.LOW_GROUNDING, round(ratio, 2)
    return GuardrailStatus.PASSED, round(ratio, 2)
