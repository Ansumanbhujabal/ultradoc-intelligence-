"""Gradio UI with 4 tabs: Upload, Ask, Extract, Traces."""

import json
import gradio as gr
import httpx

API_BASE = "http://localhost:8000"


def get_doc_choices():
    try:
        resp = httpx.get(f"{API_BASE}/documents", timeout=5)
        docs = resp.json().get("documents", [])
        return {f"{d['file_name']} ({d['doc_id'][:8]}...)": d["doc_id"] for d in docs}
    except Exception:
        return {}


def upload_file(file):
    if file is None:
        return "No file selected.", gr.update(choices=[]), gr.update(choices=[])
    try:
        with open(file.name, "rb") as f:
            resp = httpx.post(
                f"{API_BASE}/upload",
                files={"file": (file.name.split("/")[-1], f)},
                timeout=60,
            )
        if resp.status_code != 200:
            return f"Upload failed: {resp.text}", gr.update(), gr.update()
        data = resp.json()
        result = (
            f"**Upload Successful**\n\n"
            f"- **Doc ID:** `{data['doc_id']}`\n"
            f"- **Type:** {data['doc_type']}\n"
            f"- **Pages:** {data['page_count']}\n"
            f"- **Chunks:** {data['chunk_count']}"
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
        resp = httpx.post(
            f"{API_BASE}/ask",
            json={
                "doc_id": doc_id,
                "question": question,
                "enable_query_rewrite": enable_rewrite,
                "retrieval_mode": retrieval_mode,
                "confidence_threshold": threshold,
            },
            timeout=60,
        )
        if resp.status_code != 200:
            return f"Error: {resp.text}"
        data = resp.json()
        conf = data["confidence"]
        level = conf["level"]
        color = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}.get(level, "gray")
        badge = f'<span style="color:{color};font-weight:bold">{level} ({conf["score"]})</span>'
        result = f"### Answer\n\n{data['answer']}\n\n"
        result += f"### Confidence: {badge}\n\n"
        result += f"| Signal | Score |\n|--------|-------|\n"
        result += f"| Retrieval | {conf['breakdown']['retrieval']} |\n"
        result += f"| Grounding | {conf['breakdown']['grounding']} |\n"
        result += f"| LLM Assessment | {conf['breakdown']['llm_assessment']} |\n\n"
        result += f"### Guardrail: `{data['guardrail_status']}`\n\n"
        if data.get("query_rewritten"):
            result += f"*Query was rewritten for better retrieval*\n\n"
        if data.get("source_text"):
            result += f"<details><summary>Source Text</summary>\n\n```\n{data['source_text']}\n```\n</details>"
        return result
    except Exception as e:
        return f"Error: {str(e)}"


def extract_data(doc_label):
    if not doc_label:
        return "Please select a document.", ""
    choices = get_doc_choices()
    doc_id = choices.get(doc_label)
    if not doc_id:
        return "Document not found.", ""
    try:
        resp = httpx.post(
            f"{API_BASE}/extract",
            json={"doc_id": doc_id},
            timeout=60,
        )
        if resp.status_code != 200:
            return f"Error: {resp.text}", ""
        data = resp.json()
        summary = (
            f"**Document Type:** {data['doc_type']}\n\n"
            f"**Completeness:** {data['completeness_score'] * 100:.0f}% fields extracted"
        )
        return summary, json.dumps(data["extracted_data"], indent=2)
    except Exception as e:
        return f"Error: {str(e)}", ""


def get_traces():
    try:
        resp = httpx.get(f"{API_BASE}/traces", timeout=5)
        traces = resp.json().get("traces", [])
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
        gr.Markdown("# Ultra Doc-Intelligence\n*AI-powered logistics document Q&A system*")
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
