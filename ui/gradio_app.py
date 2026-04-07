"""Gradio UI with 4 tabs: Upload, Ask, Extract, Traces."""

import json
import os
import hashlib
import uuid
import shutil
import gradio as gr

from app.config import settings
from app.models.schemas import DocType, GuardrailStatus
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


def get_doc_choices():
    try:
        cache = DocumentCache.get_instance()
        docs = cache.list_documents()
        return {f"{d['file_name']} ({d['doc_id'][:8]}...)": d["doc_id"] for d in docs}
    except Exception:
        return {}


def upload_file(file):
    if file is None:
        return "No file selected.", gr.update(choices=[]), gr.update(choices=[])
    try:
        ext = os.path.splitext(file.name)[1].lower()
        if ext not in {".pdf", ".docx", ".txt"}:
            return f"Unsupported file type: {ext}", gr.update(), gr.update()

        with open(file.name, "rb") as f:
            file_content = f.read()
        content_hash = hashlib.sha256(file_content).hexdigest()

        cache = DocumentCache.get_instance()
        for doc in cache.list_documents():
            existing = cache.get(doc["doc_id"])
            if existing and getattr(existing, "content_hash", None) == content_hash:
                result = (
                    f"**Already Uploaded**\n\n"
                    f"- **Doc ID:** `{existing.doc_id}`\n"
                    f"- **Type:** {existing.doc_type}\n"
                    f"- **Pages:** {existing.page_count}\n"
                    f"- **Chunks:** {len(existing.chunks)}"
                )
                choices = get_doc_choices()
                choice_list = list(choices.keys())
                return result, gr.update(choices=choice_list), gr.update(choices=choice_list)

        doc_id = str(uuid.uuid4())
        tracer = Tracer("/upload", doc_id)

        os.makedirs(settings.upload_dir, exist_ok=True)
        file_path = os.path.join(settings.upload_dir, f"{doc_id}{ext}")
        with open(file_path, "wb") as f:
            f.write(file_content)

        with tracer.span("parsing") as span:
            parse_result = parse_document(file_path)
            if parse_result["status"] != "success":
                return f"Parse failed: {parse_result.get('error')}", gr.update(), gr.update()
            span.metadata = {"page_count": parse_result["page_count"]}

        doc_type = classify_document(parse_result["text"], tracer)

        with tracer.span("chunking") as span:
            chunks = chunk_document(parse_result["text"])
            span.metadata = {"chunk_count": len(chunks)}

        embed_and_store(doc_id, chunks, tracer)

        record = DocumentRecord(
            doc_id=doc_id,
            file_name=os.path.basename(file.name),
            full_text=parse_result["text"],
            doc_type=doc_type,
            chunks=chunks,
            page_count=parse_result["page_count"],
        )
        record.content_hash = content_hash
        cache.store(record)
        tracer.finish()

        result = (
            f"**Upload Successful**\n\n"
            f"- **Doc ID:** `{doc_id}`\n"
            f"- **Type:** {doc_type}\n"
            f"- **Pages:** {parse_result['page_count']}\n"
            f"- **Chunks:** {len(chunks)}"
        )
        choices = get_doc_choices()
        choice_list = list(choices.keys())
        dropdown_update = gr.update(choices=choice_list, value=choice_list[-1] if choice_list else None)
        return result, dropdown_update, dropdown_update
    except Exception as e:
        return f"Error: {str(e)}", gr.update(), gr.update()


def ask_question(doc_label, question, enable_rewrite, retrieval_mode, threshold):
    if not doc_label or not question:
        return "Please select a document and enter a question."
    choices = get_doc_choices()
    doc_id = choices.get(doc_label)
    if not doc_id:
        return "Document not found. Please re-upload."
    try:
        tracer = Tracer("/ask", doc_id)
        cache = DocumentCache.get_instance()
        record = cache.get(doc_id)
        if not record:
            return "Document not found in cache."

        # Layer 0: Document-level guardrail
        from app.models.schemas import DocType
        if record.doc_type == DocType.NOT_LOGISTICS:
            tracer.skip("guardrail_doc_type", "not_logistics")
            tracer.finish()
            conf = compute_confidence(0.0, 0.0, "LOW")
            return _format_answer("This document is not a logistics document. The system only supports logistics-related documents such as Bills of Lading, Rate Confirmations, and Invoices.", "", conf, GuardrailStatus.OUT_OF_SCOPE, False)

        with tracer.span("guardrail_scope") as span:
            scope_status = check_scope(question)
            span.metadata = {"status": scope_status.value}

        if scope_status == GuardrailStatus.OUT_OF_SCOPE:
            tracer.finish()
            conf = compute_confidence(0.0, 0.0, "LOW")
            return _format_answer("This question does not appear to be related to logistics.", "", conf, GuardrailStatus.OUT_OF_SCOPE, False)

        q = question
        query_rewritten = False
        if enable_rewrite:
            q = rewrite_query(question, tracer)
            query_rewritten = True
        else:
            tracer.skip("query_rewrite", "disabled")

        chunks = retrieve(question=q, doc_id=doc_id, tracer=tracer, mode=retrieval_mode)

        with tracer.span("guardrail_threshold") as span:
            threshold_status = check_retrieval_threshold(chunks, threshold)
            span.metadata = {"status": threshold_status.value, "threshold": threshold, "best_similarity": chunks[0]["similarity"] if chunks else 0}

        if threshold_status == GuardrailStatus.NOT_FOUND:
            tracer.finish()
            conf = compute_confidence(chunks[0]["similarity"] if chunks else 0.0, 0.0, "LOW")
            return _format_answer("Not found in document.", "", conf, GuardrailStatus.NOT_FOUND, query_rewritten)

        gen_result = generate_answer(question=q, full_text=record.full_text, source_chunks=chunks, tracer=tracer)

        with tracer.span("guardrail_grounding") as span:
            grounding_status, grounding_ratio = check_grounding(gen_result["answer"], gen_result["source_text"], settings.grounding_overlap_threshold)
            span.metadata = {"status": grounding_status.value, "overlap_ratio": grounding_ratio}

        retrieval_score = chunks[0]["similarity"] if chunks else 0.0
        with tracer.span("confidence_scoring") as span:
            confidence = compute_confidence(retrieval_score, grounding_ratio, gen_result["llm_confidence"])
            span.metadata = {"score": confidence.score, "level": confidence.level.value}

        final_status = grounding_status if grounding_status != GuardrailStatus.PASSED else GuardrailStatus.PASSED
        tracer.finish()
        return _format_answer(gen_result["answer"], gen_result["source_text"], confidence, final_status, query_rewritten)
    except Exception as e:
        return f"Error: {str(e)}"


def _format_answer(answer, source_text, confidence, guardrail_status, query_rewritten):
    """Format the answer with confidence and guardrail info."""
    conf = confidence
    level = conf.level.value
    color = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}.get(level, "gray")
    badge = f'<span style="color:{color};font-weight:bold">{level} ({conf.score})</span>'
    result = f"### Answer\n\n{answer}\n\n"
    result += f"### Confidence: {badge}\n\n"
    result += f"| Signal | Score |\n|--------|-------|\n"
    result += f"| Retrieval | {conf.breakdown.retrieval} |\n"
    result += f"| Grounding | {conf.breakdown.grounding} |\n"
    result += f"| LLM Assessment | {conf.breakdown.llm_assessment} |\n\n"
    result += f"### Guardrail: `{guardrail_status.value if hasattr(guardrail_status, 'value') else guardrail_status}`\n\n"
    if query_rewritten:
        result += f"*Query was rewritten for better retrieval*\n\n"
    if source_text:
        result += f"<details><summary>Source Text</summary>\n\n```\n{source_text}\n```\n</details>"
    return result


def extract_data(doc_label):
    if not doc_label:
        return "Please select a document.", ""
    choices = get_doc_choices()
    doc_id = choices.get(doc_label)
    if not doc_id:
        return "Document not found.", ""
    try:
        cache = DocumentCache.get_instance()
        record = cache.get(doc_id)
        if not record:
            return "Document not found in cache.", ""

        if record.extraction_result is not None:
            data = record.extraction_result
            summary = (
                f"**Document Type:** {data['doc_type']}\n\n"
                f"**Completeness:** {data['completeness_score'] * 100:.0f}% fields extracted"
            )
            return summary, json.dumps(data["extracted_data"], indent=2)

        tracer = Tracer("/extract", doc_id)
        result = extract_shipment_data(full_text=record.full_text, doc_type=record.doc_type, tracer=tracer)
        tracer.finish()

        response_data = {
            "extracted_data": result["shipment_data"].model_dump(),
            "completeness_score": result["completeness_score"],
            "doc_type": record.doc_type,
        }
        record.extraction_result = response_data

        doc_type_val = record.doc_type.value if hasattr(record.doc_type, 'value') else record.doc_type
        summary = (
            f"**Document Type:** {doc_type_val}\n\n"
            f"**Completeness:** {result['completeness_score'] * 100:.0f}% fields extracted"
        )
        return summary, json.dumps(response_data["extracted_data"], indent=2)
    except Exception as e:
        return f"Error: {str(e)}", ""


def get_traces():
    try:
        traces = Tracer.get_recent_traces()
        if not traces:
            return "No traces yet. Upload a document or ask a question first."
        output = ""
        for trace in traces[:10]:
            output += f"### {trace['trace_id']} | `{trace['endpoint']}` | {trace['total_latency_ms']}ms | {trace['total_tokens']} tokens\n\n"
            for span in trace.get("spans", []):
                status_icon = {"success": "OK", "error": "ERR", "skipped": "SKIP"}.get(span["status"], "?")
                output += f"- **{span['stage']}**: {status_icon} ({span['latency_ms']}ms)"
                if span.get("metadata"):
                    meta_str = ", ".join(f"{k}={v}" for k, v in span["metadata"].items())
                    output += f" — {meta_str}"
                output += "\n"
            output += "\n---\n\n"
        return output
    except Exception as e:
        return f"Error fetching traces: {str(e)}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Ultra Doc-Intelligence", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Ultra Doc-Intelligence\n"
            "*AI-powered logistics document Q&A system*\n\n"
            "API docs: [Swagger](/docs) | "
            "Observability: [Langfuse Dashboard](https://us.cloud.langfuse.com)"
        )
        with gr.Tabs():
            with gr.Tab("Upload"):
                file_input = gr.File(label="Upload Document (PDF, DOCX, TXT)")
                upload_btn = gr.Button("Upload & Process", variant="primary")
                upload_output = gr.Markdown()

            with gr.Tab("Ask"):
                with gr.Row():
                    doc_dropdown = gr.Dropdown(label="Select Document", choices=[], interactive=True)
                    refresh_btn = gr.Button("Refresh", size="sm")
                question_input = gr.Textbox(label="Your Question", placeholder="e.g., What is the carrier rate?")
                with gr.Row():
                    rewrite_toggle = gr.Checkbox(label="Enable Query Rewrite", value=False)
                    retrieval_dropdown = gr.Dropdown(label="Retrieval Mode", choices=["hybrid", "vector", "bm25"], value="hybrid")
                    threshold_slider = gr.Slider(label="Confidence Threshold", minimum=0.0, maximum=1.0, value=0.1, step=0.05)
                ask_btn = gr.Button("Ask", variant="primary")
                ask_output = gr.Markdown()

            with gr.Tab("Extract"):
                with gr.Row():
                    extract_doc_dropdown = gr.Dropdown(label="Select Document", choices=[], interactive=True)
                    extract_refresh_btn = gr.Button("Refresh", size="sm")
                extract_btn = gr.Button("Run Extraction", variant="primary")
                extract_summary = gr.Markdown()
                extract_json = gr.Code(label="Extracted Data (JSON)", language="json")

            with gr.Tab("Traces"):
                traces_btn = gr.Button("Refresh Traces", variant="secondary")
                traces_output = gr.Markdown()

        upload_btn.click(fn=upload_file, inputs=[file_input], outputs=[upload_output, doc_dropdown, extract_doc_dropdown])

        def refresh_docs():
            choices = list(get_doc_choices().keys())
            return gr.update(choices=choices)

        refresh_btn.click(fn=refresh_docs, outputs=[doc_dropdown])
        extract_refresh_btn.click(fn=refresh_docs, outputs=[extract_doc_dropdown])
        ask_btn.click(fn=ask_question, inputs=[doc_dropdown, question_input, rewrite_toggle, retrieval_dropdown, threshold_slider], outputs=[ask_output])
        extract_btn.click(fn=extract_data, inputs=[extract_doc_dropdown], outputs=[extract_summary, extract_json])
        traces_btn.click(fn=get_traces, outputs=[traces_output])

    return demo
