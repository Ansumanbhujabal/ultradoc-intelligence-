"""Request and response schemas for API endpoints."""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class DocType(str, Enum):
    BOL = "bill_of_lading"
    RATE_CONFIRMATION = "rate_confirmation"
    INVOICE = "invoice"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class GuardrailStatus(str, Enum):
    PASSED = "passed"
    LOW_GROUNDING = "low_grounding"
    NOT_FOUND = "not_found"
    OUT_OF_SCOPE = "out_of_scope"


class ConfidenceBreakdown(BaseModel):
    retrieval: float = Field(description="Hybrid retrieval similarity score")
    grounding: float = Field(description="Answer-source token overlap ratio")
    llm_assessment: float = Field(description="LLM self-assessed confidence")


class ConfidenceResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    level: ConfidenceLevel
    breakdown: ConfidenceBreakdown


class UploadResponse(BaseModel):
    doc_id: str
    doc_type: DocType
    page_count: int
    chunk_count: int
    status: str = "success"


class AskRequest(BaseModel):
    doc_id: str
    question: str
    enable_query_rewrite: bool = False
    retrieval_mode: str = "hybrid"
    confidence_threshold: float = 0.3


class AskResponse(BaseModel):
    answer: str
    source_text: str
    confidence: ConfidenceResult
    guardrail_status: GuardrailStatus
    query_rewritten: bool = False
    retrieval_mode: str = "hybrid"


class ExtractRequest(BaseModel):
    doc_id: str


class ExtractResponse(BaseModel):
    extracted_data: dict
    completeness_score: float = Field(ge=0.0, le=1.0)
    doc_type: DocType
