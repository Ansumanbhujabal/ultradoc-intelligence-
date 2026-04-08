"""FastAPI application with /upload, /ask, and /extract endpoints."""

import hashlib
import os
import uuid
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.config import settings
from app.models.schemas import (
    AskRequest, AskResponse, ExtractRequest, ExtractResponse,
    UploadResponse, GuardrailStatus, DocType,
)
from app.pipeline.parser import parse_document
from app.pipeline.classifier import classify_document
from app.pipeline.chunker import chunk_document
from app.pipeline.embedder import embed_and_store
from app.pipeline.retriever import retrieve
from app.pipeline.generator import generate_answer, rewrite_query, compute_confidence
from app.pipeline.extractor import extract_shipment_data
from app.pipeline.guardrails import check_scope, check_retrieval_threshold, check_grounding
from app.llm.prompts.guardrails import is_obviously_off_topic
from app.storage.cache import DocumentCache, DocumentRecord
from app.observability.tracer import Tracer
from app.observability.logger import get_logger

logger = get_logger("main")

app = FastAPI(
    title="Ultra Doc-Intelligence",
    description="AI-powered logistics document Q&A system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(400, f"Unsupported file type: {ext}. Use PDF, DOCX, or TXT.")

    # Read file content and compute hash for dedup
    file_content = await file.read()
    content_hash = hashlib.sha256(file_content).hexdigest()

    # Check if same content already uploaded
    cache = DocumentCache.get_instance()
    for doc in cache.list_documents():
        existing = cache.get(doc["doc_id"])
        if existing and getattr(existing, "content_hash", None) == content_hash:
            return UploadResponse(
                doc_id=existing.doc_id,
                doc_type=existing.doc_type,
                page_count=existing.page_count,
                chunk_count=len(existing.chunks),
                status="already_uploaded",
            )

    doc_id = str(uuid.uuid4())
    tracer = Tracer("/upload", doc_id)

    os.makedirs(settings.upload_dir, exist_ok=True)
    file_path = os.path.join(settings.upload_dir, f"{doc_id}{ext}")
    with open(file_path, "wb") as f:
        f.write(file_content)

    with tracer.span("parsing") as span:
        parse_result = parse_document(file_path)
        if parse_result["status"] != "success":
            raise HTTPException(422, f"Failed to parse document: {parse_result.get('error', 'unknown')}")
        span.metadata = {"page_count": parse_result["page_count"]}

    doc_type = classify_document(parse_result["text"], tracer)

    with tracer.span("chunking") as span:
        chunks = chunk_document(parse_result["text"])
        span.metadata = {"chunk_count": len(chunks)}

    embed_and_store(doc_id, chunks, tracer)

    record = DocumentRecord(
        doc_id=doc_id,
        file_name=file.filename or "unknown",
        full_text=parse_result["text"],
        doc_type=doc_type,
        chunks=chunks,
        page_count=parse_result["page_count"],
    )
    record.content_hash = content_hash
    cache.store(record)

    logger.info("upload_complete", extra={"extra_data": {
        "filename": file.filename, "doc_type": doc_type.value,
        "chunk_count": len(chunks), "doc_id": doc_id,
    }})

    tracer.finish()

    return UploadResponse(
        doc_id=doc_id,
        doc_type=doc_type,
        page_count=parse_result["page_count"],
        chunk_count=len(chunks),
    )


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    tracer = Tracer("/ask", request.doc_id)
    logger.info("ask_received", extra={"extra_data": {
        "question": request.question, "doc_id": request.doc_id,
    }})

    cache = DocumentCache.get_instance()
    record = cache.get(request.doc_id)
    if not record:
        raise HTTPException(404, f"Document {request.doc_id} not found. Upload it first.")

    # Layer 0: Document-level guardrail — refuse non-logistics documents
    if record.doc_type == DocType.NOT_LOGISTICS:
        tracer.skip("guardrail_doc_type", "not_logistics")
        tracer.finish()
        return AskResponse(
            answer="This document is not a logistics document. The system only supports logistics-related documents such as Bills of Lading, Rate Confirmations, and Invoices.",
            source_text="",
            confidence=compute_confidence(0.0, 0.0, "LOW"),
            guardrail_status=GuardrailStatus.OUT_OF_SCOPE,
            query_rewritten=False,
            retrieval_mode=request.retrieval_mode,
        )

    # Layer 1: Question scope check
    # For classified logistics docs: only hard-block obviously off-topic questions
    # (weather, sports, etc.) — let ambiguous questions through to retrieval.
    # For unclassified docs: use the full keyword-based scope check.
    with tracer.span("guardrail_scope") as span:
        is_classified = record.doc_type not in (DocType.UNKNOWN, DocType.NOT_LOGISTICS)
        if is_classified:
            scope_blocked = is_obviously_off_topic(request.question)
        else:
            scope_status = check_scope(request.question)
            scope_blocked = scope_status == GuardrailStatus.OUT_OF_SCOPE
        span.metadata = {"status": "out_of_scope" if scope_blocked else "passed"}

    if scope_blocked:
        tracer.finish()
        logger.info("ask_refused", extra={"extra_data": {"reason": "out_of_scope", "question": request.question}})
        return AskResponse(
            answer="This question does not appear to be related to logistics or the document content.",
            source_text="",
            confidence=compute_confidence(0.0, 0.0, "LOW"),
            guardrail_status=GuardrailStatus.OUT_OF_SCOPE,
            query_rewritten=False,
            retrieval_mode=request.retrieval_mode,
        )

    question = request.question
    query_rewritten = False
    if request.enable_query_rewrite:
        question = rewrite_query(request.question, tracer)
        query_rewritten = True
    else:
        tracer.skip("query_rewrite", "enable_query_rewrite=false")

    chunks = retrieve(
        question=question,
        doc_id=request.doc_id,
        tracer=tracer,
        mode=request.retrieval_mode,
    )

    with tracer.span("guardrail_threshold") as span:
        threshold_status = check_retrieval_threshold(chunks, request.confidence_threshold)
        span.metadata = {
            "status": threshold_status.value,
            "threshold": request.confidence_threshold,
            "best_similarity": chunks[0]["similarity"] if chunks else 0,
        }

    if threshold_status == GuardrailStatus.NOT_FOUND:
        tracer.finish()
        return AskResponse(
            answer="Not found in document. The information you're looking for does not appear to be in this document.",
            source_text="",
            confidence=compute_confidence(
                chunks[0]["similarity"] if chunks else 0.0, 0.0, "LOW"
            ),
            guardrail_status=GuardrailStatus.NOT_FOUND,
            query_rewritten=query_rewritten,
            retrieval_mode=request.retrieval_mode,
        )

    gen_result = generate_answer(
        question=question,
        full_text=record.full_text,
        source_chunks=chunks,
        tracer=tracer,
    )

    with tracer.span("guardrail_grounding") as span:
        grounding_status, grounding_ratio = check_grounding(
            gen_result["answer"],
            gen_result["source_text"],
            settings.grounding_overlap_threshold,
        )
        span.metadata = {
            "status": grounding_status.value,
            "overlap_ratio": grounding_ratio,
        }

    retrieval_score = chunks[0]["similarity"] if chunks else 0.0
    with tracer.span("confidence_scoring") as span:
        confidence = compute_confidence(
            retrieval_score=retrieval_score,
            grounding_ratio=grounding_ratio,
            llm_assessment=gen_result["llm_confidence"],
        )
        span.metadata = {"score": confidence.score, "level": confidence.level.value}

    final_status = grounding_status if grounding_status != GuardrailStatus.PASSED else GuardrailStatus.PASSED

    # Layer 4: Post-generation refusal gate
    # If the LLM itself says "not found" in its answer, honour that as a NOT_FOUND refusal
    answer_lower = gen_result["answer"].lower()
    llm_refused = (
        "not found in document" in answer_lower
        or "not found in the document" in answer_lower
        or "not present in" in answer_lower
        or "does not contain" in answer_lower
        or "no information" in answer_lower
        or "not mentioned" in answer_lower
        or "not included in" in answer_lower
        or "not provided in" in answer_lower
        or "not available in" in answer_lower
        or "not specified in" in answer_lower
    )

    if llm_refused:
        logger.info("ask_refused", extra={"extra_data": {
            "doc_id": request.doc_id, "reason": "llm_refusal",
        }})
        tracer.finish()
        return AskResponse(
            answer="Not found in document. The information you're looking for does not appear to be in this document.",
            source_text="",
            confidence=compute_confidence(retrieval_score, 0.0, "LOW"),
            guardrail_status=GuardrailStatus.NOT_FOUND,
            query_rewritten=query_rewritten,
            retrieval_mode=request.retrieval_mode,
        )

    # Also refuse when composite confidence is very low (below refusal threshold)
    # AND the LLM self-assessed as LOW — strong signal the answer is unreliable
    if (confidence.score < settings.low_confidence_refusal_threshold
            and gen_result["llm_confidence"].upper() == "LOW"):
        logger.info("ask_refused", extra={"extra_data": {
            "doc_id": request.doc_id, "reason": "low_confidence",
            "score": confidence.score,
        }})
        tracer.finish()
        return AskResponse(
            answer="Not found in document. The information you're looking for does not appear to be in this document.",
            source_text="",
            confidence=confidence,
            guardrail_status=GuardrailStatus.NOT_FOUND,
            query_rewritten=query_rewritten,
            retrieval_mode=request.retrieval_mode,
        )

    # Refuse on low grounding — answer is not supported by source text
    # Exception: if retrieval was strong AND LLM is highly confident, trust the answer.
    # Short entity answers (e.g., "SWIFT SHIFT LOGISTICS LLC") naturally have low token
    # overlap with large source chunks — grounding check is unreliable for these.
    llm_confident = gen_result["llm_confidence"].upper() == "HIGH"
    if final_status == GuardrailStatus.LOW_GROUNDING and (retrieval_score >= 0.5 and llm_confident):
        # Override: strong retrieval + high LLM confidence → trust the answer
        final_status = GuardrailStatus.PASSED
    if final_status == GuardrailStatus.LOW_GROUNDING:
        logger.info("ask_refused", extra={"extra_data": {
            "doc_id": request.doc_id, "reason": "low_grounding",
        }})
        tracer.finish()
        return AskResponse(
            answer="Not found in document. The information you're looking for does not appear to be in this document.",
            source_text="",
            confidence=confidence,
            guardrail_status=GuardrailStatus.NOT_FOUND,
            query_rewritten=query_rewritten,
            retrieval_mode=request.retrieval_mode,
        )

    logger.info("ask_complete", extra={"extra_data": {
        "doc_id": request.doc_id, "guardrail": final_status.value,
        "confidence_level": confidence.level.value,
        "confidence_score": confidence.score, "refused": False,
    }})

    tracer.finish()

    return AskResponse(
        answer=gen_result["answer"],
        source_text=gen_result["source_text"],
        confidence=confidence,
        guardrail_status=final_status,
        query_rewritten=query_rewritten,
        retrieval_mode=request.retrieval_mode,
    )


@app.post("/extract", response_model=ExtractResponse)
async def extract_data(request: ExtractRequest):
    cache = DocumentCache.get_instance()
    record = cache.get(request.doc_id)
    if not record:
        raise HTTPException(404, f"Document {request.doc_id} not found. Upload it first.")

    if record.doc_type == DocType.NOT_LOGISTICS:
        raise HTTPException(422, "This document is not a logistics document. Extraction is only supported for logistics documents.")

    # Return cached result if already extracted
    if record.extraction_result is not None:
        tracer = Tracer("/extract", request.doc_id)
        tracer.skip("extraction", "cached — same document content")
        tracer.finish()
        return ExtractResponse(**record.extraction_result)

    tracer = Tracer("/extract", request.doc_id)

    result = extract_shipment_data(
        full_text=record.full_text,
        doc_type=record.doc_type,
        tracer=tracer,
    )

    tracer.finish()

    non_null = sum(1 for f in result["shipment_data"].model_fields if getattr(result["shipment_data"], f) is not None)
    logger.info("extract_complete", extra={"extra_data": {
        "doc_id": request.doc_id, "fields_extracted": non_null,
        "completeness_score": result["completeness_score"],
    }})

    response_data = {
        "extracted_data": result["shipment_data"].model_dump(),
        "completeness_score": result["completeness_score"],
        "doc_type": record.doc_type,
    }

    # Cache the result
    record.extraction_result = response_data

    return ExtractResponse(**response_data)


@app.get("/documents")
def list_documents():
    cache = DocumentCache.get_instance()
    return {"documents": cache.list_documents()}


@app.get("/traces")
def get_traces():
    return {"traces": Tracer.get_recent_traces()}
