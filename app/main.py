"""FastAPI application with /upload, /ask, and /extract endpoints."""

import hashlib
import os
import uuid
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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

    cache = DocumentCache.get_instance()
    record = cache.get(request.doc_id)
    if not record:
        raise HTTPException(404, f"Document {request.doc_id} not found. Upload it first.")

    with tracer.span("guardrail_scope") as span:
        scope_status = check_scope(request.question)
        span.metadata = {"status": scope_status.value}

    if scope_status == GuardrailStatus.OUT_OF_SCOPE:
        tracer.finish()
        return AskResponse(
            answer="This question does not appear to be related to the uploaded document.",
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
