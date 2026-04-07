"""Tests for the 3-layer guardrail system."""
from app.pipeline.guardrails import check_scope, check_retrieval_threshold, check_grounding
from app.models.schemas import GuardrailStatus

def test_scope_in_scope():
    assert check_scope("What is the carrier rate?") == GuardrailStatus.PASSED

def test_scope_out_of_scope():
    assert check_scope("What is the weather today?") == GuardrailStatus.OUT_OF_SCOPE

def test_scope_edge_case_empty():
    assert check_scope("") == GuardrailStatus.OUT_OF_SCOPE

def test_retrieval_threshold_passes():
    chunks = [{"text": "rate is 400", "similarity": 0.8}]
    assert check_retrieval_threshold(chunks, 0.3) == GuardrailStatus.PASSED

def test_retrieval_threshold_fails():
    chunks = [{"text": "rate is 400", "similarity": 0.1}]
    assert check_retrieval_threshold(chunks, 0.3) == GuardrailStatus.NOT_FOUND

def test_retrieval_threshold_empty():
    assert check_retrieval_threshold([], 0.3) == GuardrailStatus.NOT_FOUND

def test_grounding_high_overlap():
    answer = "The carrier rate is 400 USD for flatbed"
    source = "Carrier Pay Flatbed: $400.00 USD Total: 400.00 USD"
    status, ratio = check_grounding(answer, source, 0.4)
    assert status == GuardrailStatus.PASSED
    assert ratio > 0.4

def test_grounding_low_overlap():
    answer = "The shipment will arrive on Mars next Tuesday via teleportation"
    source = "Pickup from Los Angeles Airport"
    status, ratio = check_grounding(answer, source, 0.4)
    assert status == GuardrailStatus.LOW_GROUNDING
    assert ratio < 0.4
