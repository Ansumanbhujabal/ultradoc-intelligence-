"""
Compile eval results into a formatted terminal report and EVAL_RESULTS.md.

Usage:
    PYTHONPATH=. uv run python eval/compile_results.py
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).parent
SAMPLE_REPORT = EVAL_DIR / "sample_report.json"
SYNTHETIC_REPORT = EVAL_DIR / "synthetic_data" / "simulation_report.json"
OUTPUT_MD = EVAL_DIR / "EVAL_RESULTS.md"


# ── helpers ──────────────────────────────────────────────────────────────────


def load_report(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def ms(val: int | float) -> str:
    if val >= 1000:
        return f"{val / 1000:.1f}s"
    return f"{int(val)}ms"


def bar(val: float, width: int = 20) -> str:
    filled = int(val * width)
    return "█" * filled + "░" * (width - filled)


# ── terminal output (box drawing) ───────────────────────────────────────────


BOX_TL = "╔"
BOX_TR = "╗"
BOX_BL = "╚"
BOX_BR = "╝"
BOX_H = "═"
BOX_V = "║"
BOX_ML = "╠"
BOX_MR = "╣"
BOX_T = "╦"
BOX_B = "╩"
BOX_X = "╬"
W = 72  # total box width (inner)


def box_top():
    return f"{BOX_TL}{BOX_H * W}{BOX_TR}"


def box_bot():
    return f"{BOX_BL}{BOX_H * W}{BOX_BR}"


def box_mid():
    return f"{BOX_ML}{BOX_H * W}{BOX_MR}"


def box_row(text: str):
    stripped = text
    pad = W - len(stripped)
    if pad < 0:
        stripped = stripped[: W - 1] + "~"
        pad = 0
    return f"{BOX_V}{stripped}{' ' * pad}{BOX_V}"


def box_center(text: str):
    pad = W - len(text)
    left = pad // 2
    right = pad - left
    return f"{BOX_V}{' ' * left}{text}{' ' * right}{BOX_V}"


def box_blank():
    return box_row("")


def section_header(title: str):
    lines = [box_mid()]
    lines.append(box_center(f"[ {title} ]"))
    lines.append(box_mid())
    return lines


def kv(key: str, val: str, indent: int = 2) -> str:
    return box_row(f"{' ' * indent}{key:<28}{val}")


def print_report_section(report: dict, label: str) -> list[str]:
    """Generate terminal lines for one report section."""
    lines = []
    lines.extend(section_header(label))

    qa = report["results"]["qa"]
    ext = report["results"]["extraction"]
    lat = report["latency"]

    lines.append(box_row(f"  Documents: {report['total_documents']}    "
                         f"Test Cases: {report['total_test_cases']}    "
                         f"Mode: {report.get('retrieval_mode', 'N/A')}"))
    lines.append(box_blank())

    # QA
    lines.append(box_row("  Q&A PERFORMANCE"))
    lines.append(kv("Accuracy:", f"{pct(qa['accuracy'])}  {bar(qa['accuracy'])}"))
    lines.append(kv("Normal Q&A Accuracy:", pct(qa['normal_accuracy'])))
    lines.append(kv("Passed / Total:", f"{qa['passed']} / {qa['total']}"))
    lines.append(kv("Avg Confidence:", f"{qa['avg_confidence']:.2f}"))
    lines.append(kv("Avg Latency:", ms(qa['avg_latency_ms'])))
    lines.append(box_blank())
    lines.append(box_row("  GUARDRAILS"))
    lines.append(kv("Correct Refusals:", f"{qa['correct_refusals']} / {qa['total_refusal_cases']}"))
    lines.append(kv("False Acceptances:", str(qa['false_acceptances'])))
    lines.append(box_blank())

    # Confidence breakdown
    lines.append(box_row("  CONFIDENCE BREAKDOWN"))
    by_conf = qa.get("by_confidence_level", {})
    total_q = qa["total"] or 1
    for level in ["HIGH", "MEDIUM", "LOW"]:
        count = by_conf.get(level, 0)
        frac = count / total_q
        lines.append(kv(f"  {level}:", f"{count:>4}  ({pct(frac)})  {bar(frac, 15)}"))
    lines.append(box_blank())

    # Extraction
    lines.append(box_row("  EXTRACTION PERFORMANCE"))
    lines.append(kv("Field Accuracy:", f"{pct(ext['field_accuracy'])}  {bar(ext['field_accuracy'])}"))
    lines.append(kv("Fields Matched:", f"{ext['fields_matched']} / {ext['fields_checked']}"))
    lines.append(kv("Avg Completeness:", pct(ext['avg_completeness'])))
    lines.append(kv("Avg Latency:", ms(ext['avg_latency_ms'])))
    lines.append(box_blank())

    # Latency
    lines.append(box_row("  LATENCY (p50 / p95)"))
    lines.append(kv("Upload:", f"{ms(lat['upload_p50_ms'])} / {ms(lat['upload_p95_ms'])}"))
    lines.append(kv("Q&A:", f"{ms(lat['qa_p50_ms'])} / {ms(lat['qa_p95_ms'])}"))
    lines.append(kv("Extraction:", f"{ms(lat['extract_p50_ms'])} / {ms(lat['extract_p95_ms'])}"))

    # Errors
    errors = report.get("errors", [])
    failed = report.get("failed_cases", [])
    if errors or failed:
        lines.append(box_blank())
        lines.append(box_row(f"  ISSUES: {len(errors)} errors, {len(failed)} failed cases"))
        for fc in failed[:5]:
            doc = fc.get("doc_file", "?")
            q = fc.get("question", "?")[:40]
            lines.append(box_row(f"    - {doc}: {q}"))
        if len(failed) > 5:
            lines.append(box_row(f"    ... and {len(failed) - 5} more"))

    return lines


def print_comparison_table(sample: dict | None, synthetic: dict | None) -> list[str]:
    """Side-by-side comparison table."""
    lines = section_header("COMPARISON")

    header = f"  {'Metric':<30}{'Sample':>18}{'Synthetic':>18}"
    sep = f"  {'─' * 30}{'─' * 18}{'─' * 18}"
    lines.append(box_row(header))
    lines.append(box_row(sep))

    def row(label, s_val, y_val):
        sv = s_val if sample else "—"
        yv = y_val if synthetic else "—"
        lines.append(box_row(f"  {label:<30}{str(sv):>18}{str(yv):>18}"))

    s_qa = sample["results"]["qa"] if sample else {}
    y_qa = synthetic["results"]["qa"] if synthetic else {}
    s_ext = sample["results"]["extraction"] if sample else {}
    y_ext = synthetic["results"]["extraction"] if synthetic else {}

    row("Q&A Accuracy",
        pct(s_qa["accuracy"]) if s_qa else "—",
        pct(y_qa["accuracy"]) if y_qa else "—")
    row("Normal Q&A Accuracy",
        pct(s_qa["normal_accuracy"]) if s_qa else "—",
        pct(y_qa["normal_accuracy"]) if y_qa else "—")
    row("Extraction Field Accuracy",
        pct(s_ext["field_accuracy"]) if s_ext else "—",
        pct(y_ext["field_accuracy"]) if y_ext else "—")
    row("Avg Completeness",
        pct(s_ext["avg_completeness"]) if s_ext else "—",
        pct(y_ext["avg_completeness"]) if y_ext else "—")
    row("False Acceptances",
        str(s_qa.get("false_acceptances", "—")) if s_qa else "—",
        str(y_qa.get("false_acceptances", "—")) if y_qa else "—")
    row("Correct Refusals",
        f"{s_qa['correct_refusals']}/{s_qa['total_refusal_cases']}" if s_qa else "—",
        f"{y_qa['correct_refusals']}/{y_qa['total_refusal_cases']}" if y_qa else "—")
    row("Avg Confidence",
        f"{s_qa['avg_confidence']:.2f}" if s_qa else "—",
        f"{y_qa['avg_confidence']:.2f}" if y_qa else "—")
    row("Q&A Avg Latency",
        ms(s_qa["avg_latency_ms"]) if s_qa else "—",
        ms(y_qa["avg_latency_ms"]) if y_qa else "—")
    row("Extraction Avg Latency",
        ms(s_ext["avg_latency_ms"]) if s_ext else "—",
        ms(y_ext["avg_latency_ms"]) if y_ext else "—")

    return lines


def find_highlights(sample: dict | None, synthetic: dict | None) -> list[str]:
    """Pick out the best metrics to highlight."""
    highlights = []
    for label, report in [("Sample", sample), ("Synthetic", synthetic)]:
        if not report:
            continue
        qa = report["results"]["qa"]
        ext = report["results"]["extraction"]
        if qa["accuracy"] == 1.0:
            highlights.append(f"{label}: Perfect Q&A accuracy (100%)")
        elif qa["accuracy"] >= 0.95:
            highlights.append(f"{label}: Excellent Q&A accuracy ({pct(qa['accuracy'])})")
        if ext["field_accuracy"] == 1.0:
            highlights.append(f"{label}: Perfect extraction field accuracy (100%)")
        if qa["false_acceptances"] == 0:
            highlights.append(f"{label}: Zero false acceptances (guardrails hold)")
        if qa["total_refusal_cases"] > 0:
            refusal_rate = qa["correct_refusals"] / qa["total_refusal_cases"]
            if refusal_rate >= 0.95:
                highlights.append(f"{label}: Refusal accuracy {pct(refusal_rate)} "
                                  f"({qa['correct_refusals']}/{qa['total_refusal_cases']})")
    return highlights


def render_terminal(sample: dict | None, synthetic: dict | None) -> str:
    """Build full terminal report."""
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("")
    lines.append(box_top())
    lines.append(box_center("ULTRADOC INTELLIGENCE  --  EVALUATION RESULTS"))
    lines.append(box_center(f"Compiled: {now}"))

    if sample:
        lines.extend(print_report_section(sample, "SAMPLE DOCS (Company Test Data)"))
    else:
        lines.extend(section_header("SAMPLE DOCS (Company Test Data)"))
        lines.append(box_center("[ not yet run -- run: uv run python eval/run_simulation.py --sample ]"))

    if synthetic:
        lines.extend(print_report_section(synthetic, "SYNTHETIC DOCS (Stress Test)"))
    else:
        lines.extend(section_header("SYNTHETIC DOCS (Stress Test)"))
        lines.append(box_center("[ not yet run -- run: uv run python eval/run_simulation.py ]"))

    if sample or synthetic:
        lines.extend(print_comparison_table(sample, synthetic))

    # Highlights
    highlights = find_highlights(sample, synthetic)
    if highlights:
        lines.extend(section_header("HIGHLIGHTS"))
        for h in highlights:
            lines.append(box_row(f"  * {h}"))

    lines.append(box_bot())
    lines.append("")
    return "\n".join(lines)


# ── markdown output ──────────────────────────────────────────────────────────


def md_section(report: dict, label: str) -> str:
    qa = report["results"]["qa"]
    ext = report["results"]["extraction"]
    lat = report["latency"]
    errors = report.get("errors", [])
    failed = report.get("failed_cases", [])

    by_conf = qa.get("by_confidence_level", {})
    total_q = qa["total"] or 1

    s = f"### {label}\n\n"
    s += f"**Run:** {report['timestamp']}  |  "
    s += f"**Docs:** {report['total_documents']}  |  "
    s += f"**Test Cases:** {report['total_test_cases']}  |  "
    s += f"**Mode:** {report.get('retrieval_mode', 'N/A')}\n\n"

    # QA table
    s += "#### Q&A Performance\n\n"
    s += "| Metric | Value |\n|---|---|\n"
    s += f"| **Accuracy** | **{pct(qa['accuracy'])}** |\n"
    s += f"| Normal Q&A Accuracy | {pct(qa['normal_accuracy'])} |\n"
    s += f"| Passed / Total | {qa['passed']} / {qa['total']} |\n"
    s += f"| Avg Confidence | {qa['avg_confidence']:.2f} |\n"
    s += f"| Avg Latency | {ms(qa['avg_latency_ms'])} |\n"
    s += f"| Correct Refusals | {qa['correct_refusals']} / {qa['total_refusal_cases']} |\n"
    s += f"| False Acceptances | {qa['false_acceptances']} |\n\n"

    # Confidence breakdown
    s += "#### Confidence Breakdown\n\n"
    s += "| Level | Count | % |\n|---|---|---|\n"
    for level in ["HIGH", "MEDIUM", "LOW"]:
        count = by_conf.get(level, 0)
        s += f"| {level} | {count} | {pct(count / total_q)} |\n"
    s += "\n"

    # Extraction table
    s += "#### Extraction Performance\n\n"
    s += "| Metric | Value |\n|---|---|\n"
    s += f"| **Field Accuracy** | **{pct(ext['field_accuracy'])}** |\n"
    s += f"| Fields Matched | {ext['fields_matched']} / {ext['fields_checked']} |\n"
    s += f"| Avg Completeness | {pct(ext['avg_completeness'])} |\n"
    s += f"| Avg Latency | {ms(ext['avg_latency_ms'])} |\n\n"

    # Latency
    s += "#### Latency (p50 / p95)\n\n"
    s += "| Operation | p50 | p95 |\n|---|---|---|\n"
    s += f"| Upload | {ms(lat['upload_p50_ms'])} | {ms(lat['upload_p95_ms'])} |\n"
    s += f"| Q&A | {ms(lat['qa_p50_ms'])} | {ms(lat['qa_p95_ms'])} |\n"
    s += f"| Extraction | {ms(lat['extract_p50_ms'])} | {ms(lat['extract_p95_ms'])} |\n\n"

    # Errors
    if errors or failed:
        s += "#### Issues\n\n"
        s += f"- **{len(errors)}** errors, **{len(failed)}** failed cases\n"
        for fc in failed[:10]:
            doc = fc.get("doc_file", "?")
            q = fc.get("question", "?")[:60]
            s += f"  - `{doc}`: {q}\n"
        if len(failed) > 10:
            s += f"  - ... and {len(failed) - 10} more\n"
        s += "\n"

    return s


def md_comparison(sample: dict | None, synthetic: dict | None) -> str:
    s = "### Side-by-Side Comparison\n\n"
    s += "| Metric | Sample | Synthetic |\n|---|---|---|\n"

    def val(report, path):
        if not report:
            return "---"
        obj = report
        for key in path:
            obj = obj[key]
        return obj

    def row(label, path, fmt=pct):
        sv = fmt(val(sample, path)) if sample else "---"
        yv = fmt(val(synthetic, path)) if synthetic else "---"
        s_line = f"| {label} | {sv} | {yv} |\n"
        return s_line

    s += row("**Q&A Accuracy**", ["results", "qa", "accuracy"])
    s += row("Normal Q&A Accuracy", ["results", "qa", "normal_accuracy"])
    s += row("**Extraction Accuracy**", ["results", "extraction", "field_accuracy"])
    s += row("Avg Completeness", ["results", "extraction", "avg_completeness"])
    s += row("False Acceptances", ["results", "qa", "false_acceptances"], fmt=str)
    s += row("Avg Confidence", ["results", "qa", "avg_confidence"], fmt=lambda x: f"{x:.2f}")
    s += row("Q&A Avg Latency", ["results", "qa", "avg_latency_ms"], fmt=ms)
    s += row("Extraction Avg Latency", ["results", "extraction", "avg_latency_ms"], fmt=ms)

    # Refusal row (special format)
    if sample:
        sq = sample["results"]["qa"]
        sv = f"{sq['correct_refusals']}/{sq['total_refusal_cases']}"
    else:
        sv = "---"
    if synthetic:
        yq = synthetic["results"]["qa"]
        yv = f"{yq['correct_refusals']}/{yq['total_refusal_cases']}"
    else:
        yv = "---"
    s += f"| Correct Refusals | {sv} | {yv} |\n"

    s += "\n"
    return s


def render_markdown(sample: dict | None, synthetic: dict | None) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = "# Ultradoc Intelligence -- Evaluation Results\n\n"
    md += f"> Compiled: {now}\n\n"

    if sample:
        md += md_section(sample, "Sample Docs (Company Test Data)")
    else:
        md += "### Sample Docs (Company Test Data)\n\n"
        md += "_Not yet run._ Execute: `uv run python eval/run_simulation.py --sample`\n\n"

    md += "---\n\n"

    if synthetic:
        md += md_section(synthetic, "Synthetic Docs (Stress Test)")
    else:
        md += "### Synthetic Docs (Stress Test)\n\n"
        md += "_Not yet run._ Execute: `uv run python eval/run_simulation.py`\n\n"

    md += "---\n\n"

    if sample or synthetic:
        md += md_comparison(sample, synthetic)

    highlights = find_highlights(sample, synthetic)
    if highlights:
        md += "### Highlights\n\n"
        for h in highlights:
            md += f"- {h}\n"
        md += "\n"

    return md


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    sample = load_report(SAMPLE_REPORT)
    synthetic = load_report(SYNTHETIC_REPORT)

    if not sample and not synthetic:
        print("No eval reports found. Run evaluations first:")
        print(f"  Sample:    PYTHONPATH=. uv run python eval/run_simulation.py --sample")
        print(f"  Synthetic: PYTHONPATH=. uv run python eval/run_simulation.py")
        sys.exit(1)

    # Terminal output
    terminal = render_terminal(sample, synthetic)
    print(terminal)

    # Markdown output
    md = render_markdown(sample, synthetic)
    OUTPUT_MD.write_text(md, encoding="utf-8")
    print(f"Markdown report saved to: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
