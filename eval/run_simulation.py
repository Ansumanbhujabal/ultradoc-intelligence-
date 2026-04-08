"""Simulation runner — uploads synthetic docs, runs tests, generates report.

Usage:
    PYTHONPATH=. uv run python eval/run_simulation.py
    PYTHONPATH=. uv run python eval/run_simulation.py --max-docs 10
    PYTHONPATH=. uv run python eval/run_simulation.py --sample-docs
    PYTHONPATH=. uv run python eval/run_simulation.py --sample-only
    PYTHONPATH=. uv run python eval/run_simulation.py --retrieval-mode vector
"""

import json
import os
import time
import argparse
from datetime import datetime
from collections import defaultdict

import httpx

API_BASE = os.environ.get("API_BASE", "http://localhost:7860")
SYNTHETIC_DIR = os.path.join(os.path.dirname(__file__), "synthetic_data")
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ultradoc_sample_test_data")
SAMPLE_GT = os.path.join(os.path.dirname(__file__), "ground_truth.json")
TIMEOUT = 120


def percentile(values: list, pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * pct / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def upload_document(filepath: str, client: httpx.Client) -> dict:
    """Upload a document and return response + latency."""
    filename = os.path.basename(filepath)
    start = time.perf_counter()
    with open(filepath, "rb") as f:
        resp = client.post(
            f"{API_BASE}/upload",
            files={"file": (filename, f)},
            timeout=TIMEOUT,
        )
    latency = round((time.perf_counter() - start) * 1000)

    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}", "latency_ms": latency}

    data = resp.json()
    data["latency_ms"] = latency
    return data


def run_qa_test(doc_id: str, question: str, client: httpx.Client,
                retrieval_mode: str = "hybrid") -> dict:
    """Run a Q&A test and return response + latency."""
    start = time.perf_counter()
    resp = client.post(
        f"{API_BASE}/ask",
        json={
            "doc_id": doc_id,
            "question": question,
            "enable_query_rewrite": True,
            "retrieval_mode": retrieval_mode,
            "confidence_threshold": 0.35,
        },
        timeout=TIMEOUT,
    )
    latency = round((time.perf_counter() - start) * 1000)

    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}", "latency_ms": latency}

    data = resp.json()
    data["latency_ms"] = latency
    return data


def run_extract_test(doc_id: str, client: httpx.Client) -> dict:
    """Run an extraction test and return response + latency."""
    start = time.perf_counter()
    resp = client.post(
        f"{API_BASE}/extract",
        json={"doc_id": doc_id},
        timeout=TIMEOUT,
    )
    latency = round((time.perf_counter() - start) * 1000)

    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}", "latency_ms": latency}

    data = resp.json()
    data["latency_ms"] = latency
    return data


def _is_refusal(response: dict) -> bool:
    """Check if a response is a refusal (out-of-scope, not-found, etc.)."""
    status = response.get("guardrail_status", "")
    answer = response.get("answer", "").lower()
    refusal_statuses = {"OUT_OF_SCOPE", "NOT_FOUND"}
    refusal_phrases = [
        "not found", "not a logistics", "not related to", "does not appear",
        "not present", "not mentioned", "not provided", "not included",
        "not specified", "does not contain", "no information", "not available",
    ]
    return (status in refusal_statuses
            or any(p in answer for p in refusal_phrases))


# LLM-as-judge for semantic equivalence
_judge_provider = None

def _get_judge():
    global _judge_provider
    if _judge_provider is None:
        from app.llm.provider import get_provider
        _judge_provider = get_provider()
    return _judge_provider


def _llm_judge_equivalent(question: str, expected: str, actual: str) -> bool:
    """Use LLM to judge if actual answer is semantically equivalent to expected."""
    try:
        import app.llm.prompts.eval  # noqa: registers eval prompts
        from app.llm.prompts.registry import registry
        judge = _get_judge()
        template = registry.get("eval_qa_judge")
        prompt = template.render(question=question, expected=expected, actual=actual)
        resp = judge.generate(
            messages=[{"role": "user", "content": prompt}],
            model=judge.fast_model,
            max_tokens=5,
            temperature=0.0,
        )
        return resp.content.strip().upper().startswith("YES")
    except Exception:
        # Fallback to substring match if LLM judge fails
        return expected.lower() in actual.lower()


def check_qa_result(expected_answer: str, response: dict, question: str = "") -> bool:
    """Check if a Q&A response matches expectations using LLM-as-judge."""
    if expected_answer in ("__OUT_OF_SCOPE__", "__NOT_FOUND__"):
        return _is_refusal(response)

    # If the system refused but shouldn't have, that's a fail
    if _is_refusal(response):
        return False

    actual = response.get("answer", "")

    # Quick check: exact substring match (skip LLM call if obvious)
    if expected_answer.lower() in actual.lower():
        return True

    # LLM-as-judge for semantic equivalence
    return _llm_judge_equivalent(question, expected_answer, actual)


def check_extraction_result(expected_fields: dict, response: dict) -> tuple:
    """Check extraction fields with flexible matching. Returns (fields_checked, fields_matched)."""
    extracted = response.get("extracted_data", {})
    checked = 0
    matched = 0

    for field, expected_val in expected_fields.items():
        if expected_val is None:
            continue
        checked += 1
        actual_val = extracted.get(field)
        if actual_val is None:
            continue

        expected_str = str(expected_val).lower().strip()
        actual_str = str(actual_val).lower().strip()

        # Substring match (either direction)
        if expected_str in actual_str or actual_str in expected_str:
            matched += 1
        # Numeric comparison (rates, weights, etc.)
        elif _is_numeric_match(expected_val, actual_val):
            matched += 1
        # Fuzzy: expected is a name/entity — check if core value is present
        # e.g., expected="Atlas Construction Materials", actual="Atlas Construction Materials, 1234 Blvd"
        elif _core_value_present(expected_str, actual_str):
            matched += 1

    return checked, matched


def _is_numeric_match(expected, actual, tolerance=1.0) -> bool:
    """Check if two values are numerically close."""
    try:
        # Strip currency symbols and commas
        e = float(str(expected).replace(",", "").replace("$", "").strip())
        a = float(str(actual).replace(",", "").replace("$", "").strip())
        return abs(e - a) < tolerance
    except (ValueError, TypeError):
        return False


def _core_value_present(expected: str, actual: str) -> bool:
    """Check if all significant words from expected appear in actual."""
    stop_words = {"the", "a", "an", "of", "in", "at", "to", "for", "and", "or", "is"}
    expected_words = [w for w in expected.split() if w not in stop_words and len(w) > 1]
    if not expected_words:
        return False
    matches = sum(1 for w in expected_words if w in actual)
    return matches / len(expected_words) >= 0.8


def run_simulation(max_docs=None, sample_docs=False, sample_only=False,
                   retrieval_mode="hybrid"):
    """Run the full simulation."""
    client = httpx.Client()

    # Verify API is running
    try:
        resp = client.get(f"{API_BASE}/health", timeout=5)
        if resp.status_code != 200:
            print(f"ERROR: API not healthy at {API_BASE}")
            return
    except Exception as e:
        print(f"ERROR: Cannot reach API at {API_BASE}: {e}")
        return

    print(f"API healthy at {API_BASE}")
    print(f"Retrieval mode: {retrieval_mode}")
    print()

    # Load ground truth — sample-only skips synthetic entirely
    if sample_only:
        if not os.path.exists(SAMPLE_GT):
            print(f"ERROR: Sample ground truth not found at {SAMPLE_GT}")
            return
        with open(SAMPLE_GT) as f:
            test_cases = json.load(f)
        print(f"Sample-only mode: {len(test_cases)} test cases from {SAMPLE_GT}")
    else:
        gt_path = os.path.join(SYNTHETIC_DIR, "ground_truth.json")
        if not os.path.exists(gt_path):
            print(f"ERROR: Ground truth not found at {gt_path}")
            print("Run generate_synthetic_data.py first.")
            return

        with open(gt_path) as f:
            test_cases = json.load(f)

        # Optionally add sample doc test cases
        if sample_docs and os.path.exists(SAMPLE_GT):
            with open(SAMPLE_GT) as f:
                sample_cases = json.load(f)
                test_cases.extend(sample_cases)
            print(f"Added {len(sample_cases)} sample doc test cases")

    # Get unique documents
    doc_files = sorted(set(tc["doc_file"] for tc in test_cases))
    if max_docs:
        doc_files = doc_files[:max_docs]
        test_cases = [tc for tc in test_cases if tc["doc_file"] in set(doc_files)]

    print(f"Documents to upload: {len(doc_files)}")
    print(f"Test cases to run: {len(test_cases)}")
    print()

    # Phase 1: Upload documents
    print("--- Phase 1: Uploading documents ---")
    doc_id_map = {}
    upload_latencies = []
    upload_errors = []

    for i, filename in enumerate(doc_files, 1):
        # Determine file path — check sample dir first for PDFs, then synthetic
        filepath = None
        if os.path.exists(os.path.join(SAMPLE_DIR, filename)):
            filepath = os.path.join(SAMPLE_DIR, filename)
        elif os.path.exists(os.path.join(SYNTHETIC_DIR, "docs", filename)):
            filepath = os.path.join(SYNTHETIC_DIR, "docs", filename)
        else:
            # Defensive fallback: check if file exists with different extension
            # (handles cases where PDF generation fell back to .txt)
            base = os.path.splitext(filename)[0]
            for ext in (".txt", ".docx", ".pdf"):
                candidate = os.path.join(SYNTHETIC_DIR, "docs", f"{base}{ext}")
                if os.path.exists(candidate):
                    filepath = candidate
                    print(f"  [{i:3d}/{len(doc_files)}] NOTE {filename} -> {os.path.basename(candidate)}")
                    break

        if filepath is None:
            upload_errors.append({"doc_file": filename, "error": "File not found"})
            print(f"  [{i:3d}/{len(doc_files)}] SKIP {filename} — not found")
            continue

        result = upload_document(filepath, client)

        if "error" in result:
            upload_errors.append({"doc_file": filename, "error": result["error"]})
            print(f"  [{i:3d}/{len(doc_files)}] ERROR {filename} — {result['error'][:60]}")
        else:
            doc_id_map[filename] = result["doc_id"]
            upload_latencies.append(result["latency_ms"])
            status = result.get("status", "uploaded")
            if i % 10 == 0 or i == len(doc_files):
                print(f"  [{i:3d}/{len(doc_files)}] {filename} — {status} ({result['latency_ms']}ms)")

    print(f"\nUploaded: {len(doc_id_map)}/{len(doc_files)}, Errors: {len(upload_errors)}")
    print()

    # Phase 2: Run test cases
    print("--- Phase 2: Running test cases ---")
    qa_results = []
    ext_results = []
    qa_latencies = []
    ext_latencies = []
    errors = list(upload_errors)
    failed_cases = []

    for i, tc in enumerate(test_cases, 1):
        doc_file = tc["doc_file"]
        doc_id = doc_id_map.get(doc_file)

        if not doc_id:
            errors.append({"doc_file": doc_file, "error": "No doc_id — upload failed"})
            continue

        if tc["type"] == "qa":
            response = run_qa_test(doc_id, tc["question"], client, retrieval_mode)

            if "error" in response:
                errors.append({"doc_file": doc_file, "question": tc["question"], "error": response["error"]})
            else:
                passed = check_qa_result(tc["expected_answer"], response, tc["question"])
                qa_latencies.append(response["latency_ms"])

                conf = response.get("confidence", {})
                qa_results.append({
                    "passed": passed,
                    "confidence_score": conf.get("score", 0),
                    "confidence_level": conf.get("level", "UNKNOWN"),
                    "guardrail_status": response.get("guardrail_status", ""),
                    "expected_type": "out_of_scope" if tc["expected_answer"] == "__OUT_OF_SCOPE__"
                                    else "not_found" if tc["expected_answer"] == "__NOT_FOUND__"
                                    else "normal",
                })

                if not passed:
                    failed_cases.append({
                        "doc_file": doc_file,
                        "question": tc["question"],
                        "expected": tc["expected_answer"],
                        "actual": response.get("answer", "")[:200],
                        "guardrail": response.get("guardrail_status", ""),
                        "confidence": conf.get("score", 0),
                    })

        elif tc["type"] == "extraction":
            response = run_extract_test(doc_id, client)

            if "error" in response:
                errors.append({"doc_file": doc_file, "type": "extraction", "error": response["error"]})
            else:
                checked, matched = check_extraction_result(tc["expected_fields"], response)
                ext_latencies.append(response["latency_ms"])
                ext_results.append({
                    "fields_checked": checked,
                    "fields_matched": matched,
                    "completeness": response.get("completeness_score", 0),
                })

                if checked > 0 and matched < checked:
                    failed_cases.append({
                        "doc_file": doc_file,
                        "question": "extraction",
                        "expected": json.dumps({k: v for k, v in tc["expected_fields"].items() if v is not None}),
                        "actual": json.dumps(response.get("extracted_data", {}))[:200],
                    })

        if i % 20 == 0 or i == len(test_cases):
            qa_pass = sum(1 for r in qa_results if r["passed"])
            print(f"  [{i:3d}/{len(test_cases)}] QA: {qa_pass}/{len(qa_results)} passed | Ext: {len(ext_results)} done")

    # Phase 3: Compute report
    print()
    print("--- Phase 3: Computing report ---")

    # QA metrics
    qa_total = len(qa_results)
    qa_passed = sum(1 for r in qa_results if r["passed"])
    qa_normal = [r for r in qa_results if r["expected_type"] == "normal"]
    qa_refusal = [r for r in qa_results if r["expected_type"] in ("out_of_scope", "not_found")]
    correct_refusals = sum(1 for r in qa_refusal if r["passed"])
    false_acceptances = sum(1 for r in qa_refusal if not r["passed"])
    confidence_levels = defaultdict(int)
    for r in qa_results:
        confidence_levels[r["confidence_level"]] += 1

    # Extraction metrics
    ext_total = len(ext_results)
    total_fields_checked = sum(r["fields_checked"] for r in ext_results)
    total_fields_matched = sum(r["fields_matched"] for r in ext_results)

    report = {
        "timestamp": datetime.now().isoformat(),
        "api_base": API_BASE,
        "retrieval_mode": retrieval_mode,
        "total_documents": len(doc_id_map),
        "total_test_cases": len(test_cases),
        "results": {
            "qa": {
                "total": qa_total,
                "passed": qa_passed,
                "failed": qa_total - qa_passed,
                "accuracy": round(qa_passed / qa_total, 4) if qa_total else 0,
                "normal_accuracy": round(sum(1 for r in qa_normal if r["passed"]) / len(qa_normal), 4) if qa_normal else 0,
                "avg_latency_ms": round(sum(qa_latencies) / len(qa_latencies)) if qa_latencies else 0,
                "avg_confidence": round(sum(r["confidence_score"] for r in qa_results) / qa_total, 4) if qa_total else 0,
                "correct_refusals": correct_refusals,
                "total_refusal_cases": len(qa_refusal),
                "false_acceptances": false_acceptances,
                "by_confidence_level": dict(confidence_levels),
            },
            "extraction": {
                "total": ext_total,
                "fields_checked": total_fields_checked,
                "fields_matched": total_fields_matched,
                "field_accuracy": round(total_fields_matched / total_fields_checked, 4) if total_fields_checked else 0,
                "avg_completeness": round(sum(r["completeness"] for r in ext_results) / ext_total, 4) if ext_total else 0,
                "avg_latency_ms": round(sum(ext_latencies) / len(ext_latencies)) if ext_latencies else 0,
            },
        },
        "latency": {
            "upload_avg_ms": round(sum(upload_latencies) / len(upload_latencies)) if upload_latencies else 0,
            "upload_p50_ms": round(percentile(upload_latencies, 50)),
            "upload_p95_ms": round(percentile(upload_latencies, 95)),
            "qa_avg_ms": round(sum(qa_latencies) / len(qa_latencies)) if qa_latencies else 0,
            "qa_p50_ms": round(percentile(qa_latencies, 50)),
            "qa_p95_ms": round(percentile(qa_latencies, 95)),
            "extract_avg_ms": round(sum(ext_latencies) / len(ext_latencies)) if ext_latencies else 0,
            "extract_p50_ms": round(percentile(ext_latencies, 50)),
            "extract_p95_ms": round(percentile(ext_latencies, 95)),
        },
        "errors": errors[:50],
        "failed_cases": failed_cases[:100],
    }

    # Save report — separate files for sample vs synthetic evals
    if sample_only:
        report_path = os.path.join(os.path.dirname(__file__), "sample_report.json")
    else:
        report_path = os.path.join(SYNTHETIC_DIR, "simulation_report.json")
        os.makedirs(SYNTHETIC_DIR, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print_report(report)
    print(f"\nFull report saved to: {report_path}")


def print_report(report):
    """Print formatted simulation report."""
    print()
    print("=" * 60)
    print("SIMULATION REPORT")
    print("=" * 60)
    print(f"  Timestamp:      {report['timestamp']}")
    print(f"  API:            {report['api_base']}")
    print(f"  Retrieval Mode: {report['retrieval_mode']}")
    print(f"  Documents:      {report['total_documents']}")
    print(f"  Test Cases:     {report['total_test_cases']}")
    print()

    qa = report["results"]["qa"]
    print(f"--- Q&A Results ---")
    print(f"  Overall Accuracy:   {qa['accuracy']:.1%} ({qa['passed']}/{qa['total']})")
    print(f"  Normal Q&A Accuracy:{qa['normal_accuracy']:.1%}")
    print(f"  Avg Latency:        {qa['avg_latency_ms']}ms")
    print(f"  Avg Confidence:     {qa['avg_confidence']:.3f}")
    print(f"  Correct Refusals:   {qa['correct_refusals']}/{qa['total_refusal_cases']}")
    print(f"  False Acceptances:  {qa['false_acceptances']}")
    print(f"  By Confidence:      {qa['by_confidence_level']}")
    print()

    ext = report["results"]["extraction"]
    print(f"--- Extraction Results ---")
    print(f"  Field Accuracy:     {ext['field_accuracy']:.1%} ({ext['fields_matched']}/{ext['fields_checked']})")
    print(f"  Avg Completeness:   {ext['avg_completeness']:.1%}")
    print(f"  Avg Latency:        {ext['avg_latency_ms']}ms")
    print()

    lat = report["latency"]
    print(f"--- Latency Distribution ---")
    print(f"  Upload:  avg={lat['upload_avg_ms']}ms  p50={lat['upload_p50_ms']}ms  p95={lat['upload_p95_ms']}ms")
    print(f"  Q&A:     avg={lat['qa_avg_ms']}ms  p50={lat['qa_p50_ms']}ms  p95={lat['qa_p95_ms']}ms")
    print(f"  Extract: avg={lat['extract_avg_ms']}ms  p50={lat['extract_p50_ms']}ms  p95={lat['extract_p95_ms']}ms")

    if report["errors"]:
        print(f"\n--- Errors ({len(report['errors'])}) ---")
        for err in report["errors"][:5]:
            print(f"  {err.get('doc_file', '?')}: {err.get('error', '?')[:80]}")
        if len(report["errors"]) > 5:
            print(f"  ... and {len(report['errors']) - 5} more")

    if report["failed_cases"]:
        print(f"\n--- Failed Cases (first 10 of {len(report['failed_cases'])}) ---")
        for fc in report["failed_cases"][:10]:
            print(f"  {fc['doc_file']} | Q: {fc.get('question', 'extraction')[:60]}")
            print(f"    Expected: {str(fc['expected'])[:80]}")
            print(f"    Actual:   {str(fc['actual'])[:80]}")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation against synthetic + sample data")
    parser.add_argument("--max-docs", type=int, default=None, help="Limit number of docs to upload")
    parser.add_argument("--sample-docs", action="store_true", help="Include 3 sample PDFs from test data")
    parser.add_argument("--sample-only", action="store_true",
                        help="Run ONLY against 3 sample PDFs (uses eval/ground_truth.json)")
    parser.add_argument("--retrieval-mode", default="hybrid", choices=["hybrid", "vector", "bm25"])
    args = parser.parse_args()

    run_simulation(
        max_docs=args.max_docs,
        sample_docs=args.sample_docs,
        sample_only=args.sample_only,
        retrieval_mode=args.retrieval_mode,
    )
