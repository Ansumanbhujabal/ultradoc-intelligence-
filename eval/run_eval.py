"""Automated evaluation runner."""

import json
import os
import time
import argparse
import httpx
from eval.report import EvalReport

API_BASE = "http://localhost:8000"
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ultradoc_sample_test_data")


def run_evaluation(ground_truth_path: str, flags: dict = None):
    flags = flags or {}
    report = EvalReport()

    with open(ground_truth_path) as f:
        test_cases = json.load(f)

    doc_map = {}
    unique_files = set(tc["doc_file"] for tc in test_cases)

    print(f"Uploading {len(unique_files)} documents...")
    for file_name in unique_files:
        file_path = os.path.join(SAMPLE_DIR, file_name)
        if not os.path.exists(file_path):
            print(f"  SKIP: {file_name} not found")
            continue
        with open(file_path, "rb") as f:
            resp = httpx.post(f"{API_BASE}/upload", files={"file": (file_name, f)}, timeout=60)
        if resp.status_code == 200:
            doc_map[file_name] = resp.json()["doc_id"]
            print(f"  OK: {file_name} -> {doc_map[file_name][:8]}...")
        else:
            print(f"  FAIL: {file_name}: {resp.text}")

    print(f"\nRunning {len(test_cases)} test cases...")
    for i, tc in enumerate(test_cases):
        doc_id = doc_map.get(tc["doc_file"])
        if not doc_id:
            print(f"  [{i+1}] SKIP: doc not uploaded")
            continue
        if tc["type"] == "qa":
            _run_qa_case(tc, doc_id, report, flags, i + 1)
        elif tc["type"] == "extraction":
            _run_extraction_case(tc, doc_id, report, i + 1)

    report.print_report()
    return report


def _run_qa_case(tc, doc_id, report, flags, idx):
    report.total_qa += 1
    start = time.perf_counter()
    try:
        resp = httpx.post(f"{API_BASE}/ask", json={"doc_id": doc_id, "question": tc["question"], "enable_query_rewrite": flags.get("enable_query_rewrite", False), "retrieval_mode": flags.get("retrieval_mode", "hybrid")}, timeout=60)
        latency = (time.perf_counter() - start) * 1000
        report.latencies.append(latency)
        if resp.status_code != 200:
            print(f"  [{idx}] ERROR: {resp.status_code}")
            return
        data = resp.json()
        answer = data["answer"]
        expected = tc["expected_answer"]
        guardrail = data["guardrail_status"]

        if expected == "__OUT_OF_SCOPE__":
            if guardrail == "out_of_scope":
                report.correct_refusals += 1
                report.correct_answers += 1
                print(f"  [{idx}] PASS (correct refusal): {tc['question'][:40]}...")
            else:
                report.false_acceptances += 1
                print(f"  [{idx}] FAIL (should have refused): {tc['question'][:40]}...")
            return

        if expected == "__NOT_FOUND__":
            if guardrail == "not_found" or "not found" in answer.lower():
                report.correct_refusals += 1
                report.correct_answers += 1
                print(f"  [{idx}] PASS (correct not-found): {tc['question'][:40]}...")
            else:
                report.false_acceptances += 1
                print(f"  [{idx}] FAIL (should have said not found): {tc['question'][:40]}...")
            return

        if expected.lower() in answer.lower():
            report.correct_answers += 1
            print(f"  [{idx}] PASS: {tc['question'][:40]}...")
        else:
            print(f"  [{idx}] FAIL: expected '{expected}' in answer '{answer[:60]}...'")

        source_keyword = tc.get("expected_source_contains", "")
        if source_keyword and source_keyword.lower() in data.get("source_text", "").lower():
            report.source_hits += 1
        report.confidence_sum += data["confidence"]["score"]
    except Exception as e:
        print(f"  [{idx}] ERROR: {e}")


def _run_extraction_case(tc, doc_id, report, idx):
    report.total_extraction += 1
    try:
        resp = httpx.post(f"{API_BASE}/extract", json={"doc_id": doc_id}, timeout=60)
        report.latencies.append(0)
        if resp.status_code != 200:
            print(f"  [{idx}] EXTRACT ERROR: {resp.status_code}")
            return
        extracted = resp.json()["extracted_data"]
        expected = tc["expected_fields"]
        for field_name, expected_val in expected.items():
            report.total_fields_checked += 1
            actual = extracted.get(field_name)
            if actual is None:
                print(f"  [{idx}] EXTRACT MISS: {field_name} (got null)")
                continue
            actual_str = str(actual).lower()
            expected_str = str(expected_val).lower()
            if expected_str in actual_str:
                report.field_matches += 1
                print(f"  [{idx}] EXTRACT PASS: {field_name}")
            else:
                print(f"  [{idx}] EXTRACT FAIL: {field_name} expected='{expected_val}' got='{actual}'")
    except Exception as e:
        print(f"  [{idx}] EXTRACT ERROR: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ultra Doc-Intelligence evaluation")
    parser.add_argument("--ground-truth", default="eval/ground_truth.json")
    parser.add_argument("--flags", default="{}")
    args = parser.parse_args()
    flags = json.loads(args.flags)
    run_evaluation(args.ground_truth, flags)
