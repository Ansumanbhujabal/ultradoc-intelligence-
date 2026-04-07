"""Tests for confidence scoring."""
from app.pipeline.generator import compute_confidence
from app.models.schemas import ConfidenceLevel

def test_confidence_high():
    result = compute_confidence(retrieval_score=0.9, grounding_ratio=0.8, llm_assessment="HIGH")
    assert result.score > 0.7
    assert result.level == ConfidenceLevel.HIGH

def test_confidence_low():
    result = compute_confidence(retrieval_score=0.2, grounding_ratio=0.1, llm_assessment="LOW")
    assert result.score < 0.4
    assert result.level == ConfidenceLevel.LOW

def test_confidence_medium():
    result = compute_confidence(retrieval_score=0.5, grounding_ratio=0.5, llm_assessment="MEDIUM")
    assert 0.4 <= result.score <= 0.7
    assert result.level == ConfidenceLevel.MEDIUM

def test_confidence_breakdown_present():
    result = compute_confidence(0.9, 0.8, "HIGH")
    assert result.breakdown.retrieval > 0
    assert result.breakdown.grounding > 0
    assert result.breakdown.llm_assessment > 0
