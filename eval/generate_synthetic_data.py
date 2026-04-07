"""LLM-powered synthetic logistics data generator.

Generates 160+ messy, diverse logistics documents using Azure OpenAI,
along with ground truth Q&A and extraction test cases.

Usage:
    PYTHONPATH=. uv run python eval/generate_synthetic_data.py
    PYTHONPATH=. uv run python eval/generate_synthetic_data.py --count 20  # quick test
    PYTHONPATH=. uv run python eval/generate_synthetic_data.py --count 160 --seed-to-langfuse
"""

import json
import os
import random
import argparse
from datetime import datetime, timedelta

from app.llm.provider import get_provider
from app.llm.prompts.synthetic import *  # registers prompts
from app.llm.prompts.registry import registry
from app.observability.logger import get_logger

logger = get_logger("synthetic_generator")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "synthetic_data", "docs")
GROUND_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "synthetic_data", "ground_truth.json")

# --- Seed Data Pools ---

CARRIER_NAMES = [
    "SWIFT SHIFT LOGISTICS LLC", "EAGLE TRANSPORT INC", "PACIFIC FREIGHT SOLUTIONS",
    "MIDWEST HAULERS CO", "ATLANTIC CARGO SERVICES", "DESERT WIND TRUCKING LLC",
    "NORTHERN STAR LOGISTICS", "BLUE RIDGE CARRIERS", "SUMMIT FREIGHT GROUP",
    "COASTAL DRAYAGE INC", "HEARTLAND EXPRESS LLC", "GOLDEN STATE TRANSPORT",
    "CROSSROADS FREIGHT CO", "IRON HORSE LOGISTICS", "SILVER LINE CARRIERS",
    "Jose Garcia Transportes SA", "MAPLE LEAF SHIPPING LTD", "Rio Grande Logistics",
    "Trans-Pacific Freight Corp", "GREAT PLAINS HAULING LLC",
]

SHIPPER_NAMES = [
    "ABC Manufacturing Corp", "XYZ Industrial Supplies", "Global Tech Industries",
    "Fresh Harvest Foods Inc", "Steel Dynamics LLC", "Acme Chemical Solutions",
    "Pacific Rim Electronics", "Midwest Grain Cooperative", "Santos Agricola SA de CV",
    "Northern Timber Co Ltd", "Sunrise Pharmaceuticals", "Heavy Metal Fabricators Inc",
    "QuickParts Automotive", "Green Valley Organics", "Atlas Construction Materials",
]

CONSIGNEE_NAMES = [
    "Walmart Distribution Center #4521", "Amazon Fulfillment Center FTW1",
    "Target Regional DC - Southeast", "Costco Wholesale Depot",
    "Home Depot Supply Chain Hub", "FedEx Ground Hub - Memphis",
    "Dollar General DC #12", "Kroger Fresh Distribution", "UPS Freight Terminal",
    "Sysco Foods Warehouse", "McLane Company Inc", "Cardinal Health DC",
    "Grainger Industrial Supply", "Fastenal Distribution Center", "ALDI Regional Warehouse",
]

CITIES = [
    ("Los Angeles", "CA", "90001"), ("Chicago", "IL", "60601"), ("Houston", "TX", "77001"),
    ("Phoenix", "AZ", "85001"), ("Dallas", "TX", "75201"), ("Atlanta", "GA", "30301"),
    ("Denver", "CO", "80201"), ("Seattle", "WA", "98101"), ("Miami", "FL", "33101"),
    ("Memphis", "TN", "38101"), ("Portland", "OR", "97201"), ("Detroit", "MI", "48201"),
    ("Minneapolis", "MN", "55401"), ("Kansas City", "MO", "64101"), ("Nashville", "TN", "37201"),
    ("Toronto", "ON", "M5V 3A8"), ("Vancouver", "BC", "V6B 1A1"), ("Montreal", "QC", "H3B 1A2"),
    ("Monterrey", "NL", "64000"), ("Laredo", "TX", "78040"), ("El Paso", "TX", "79901"),
    ("San Diego", "CA", "92101"), ("Nogales", "AZ", "85621"), ("Buffalo", "NY", "14201"),
]

EQUIPMENT_TYPES = ["Dry Van", "Flatbed", "Reefer", "Step Deck", "Tanker", "Intermodal", "Lowboy", "Conestoga", "Power Only"]
MODES = ["FTL", "LTL", "Partial", "Drayage", "Intermodal", "Expedited"]
CURRENCIES = ["USD", "CAD", "EUR", "MXN"]
WEIGHT_UNITS = ["lbs", "kg", "tons"]

DOC_TYPES = [
    ("bill_of_lading", 40),
    ("rate_confirmation", 40),
    ("invoice", 25),
    ("shipment_instructions", 25),
    ("delivery_receipt", 15),
    ("freight_quote", 15),
]

MESSINESS_PROFILES = [
    # Clean-ish
    "Mostly clean formatting but with inconsistent date formats (mix MM/DD/YYYY and DD-Mon-YYYY in the same document). Some fields use ALL CAPS labels, others use Title Case.",

    # Typo-heavy
    "Multiple typos in section headers: 'Consginee', 'Rat Confirmation', 'Equipmnt Type', 'Delvery Date'. Some field values have trailing spaces. Phone numbers are in different formats (xxx-xxx-xxxx, (xxx) xxx-xxxx, xxx.xxx.xxxx).",

    # Missing data
    "Several key fields are intentionally MISSING — leave blanks, dashes (---), 'N/A', or 'TBD' instead of actual values. Some sections are completely absent. The document feels incomplete, like a draft.",

    # Legacy system export
    "Format as if exported from an old legacy TMS: pipe-delimited tables, fixed-width columns that don't align perfectly, truncated field names (SHIP_NM, CONS_ADDR, EQ_TYP), system-generated headers with timestamps and user IDs.",

    # Extra noise
    "Include lots of noise: legal disclaimers ('THIS DOCUMENT IS CONFIDENTIAL AND PROPRIETARY'), page headers/footers with page numbers, a 'DO NOT DUPLICATE' watermark text, terms and conditions section, and repeated company logos described as [LOGO] or [COMPANY LETTERHEAD].",

    # Special characters
    "Use special characters throughout: accented names (Jose Garcia, Francois Leblanc, Munoz), currency symbols mixed ($ and USD and dollars), em-dashes instead of hyphens, bullet points (•, *, -) mixed, trademark symbols (TM), copyright notices.",

    # Multi-format chaos
    "Mix multiple formatting styles in one document: some sections as key-value pairs, others as tables (pipe-delimited), others as flowing paragraphs. Dates appear in at least 3 different formats. Some values are repeated in different sections with slight variations.",

    # Scan artifact style
    "Format as if poorly OCR'd from a scanned document: some characters replaced (0/O confusion, 1/l/I confusion, $ as S), random line breaks in the middle of words, some text running together without spaces, garbled header text.",

    # Minimal/terse
    "Very terse and minimal — abbreviate everything. 'PU' for pickup, 'DEL' for delivery, 'EQ' for equipment, 'WT' for weight. No full sentences, just abbreviated field labels and values. Total document under 200 words.",

    # Verbose/repetitive
    "Overly verbose with repeated information in multiple sections. The same shipment details appear in a 'Summary' section, a 'Details' section, and a 'Confirmation' section. Include 1500+ words. Add irrelevant subsections like 'Company History' or 'Service Level Agreement'.",

    # Cross-border
    "Cross-border shipment (US-Mexico or US-Canada): mix English and Spanish/French in headers and values. Include customs broker info, border crossing point, duty/tariff references. Use international phone formats. Some addresses in non-US format.",

    # Contradictory
    "Include CONTRADICTORY information: two different rates mentioned in different sections, pickup date in header doesn't match pickup date in stops section, weight listed differently in two places. This tests whether the system handles ambiguity.",
]

FORMAT_STYLES = [
    "Standard business document with clear sections and headers",
    "Tabular format with pipe-delimited columns",
    "Freeform text paragraphs with embedded field values",
    "Email-style with To/From/Subject headers followed by body",
    "Form-style with field labels and fill-in values on same line",
    "Mixed: some sections as tables, others as paragraphs",
]


def generate_seed_fields(doc_type: str, rng: random.Random) -> dict:
    """Generate random but realistic field values for a document."""
    origin = rng.choice(CITIES)
    dest = rng.choice([c for c in CITIES if c != origin])

    # Base fields present in most docs
    fields = {
        "shipment_id": f"LD{rng.randint(10000, 99999)}",
        "shipper": rng.choice(SHIPPER_NAMES),
        "consignee": rng.choice(CONSIGNEE_NAMES),
        "origin_city": f"{origin[0]}, {origin[1]} {origin[2]}",
        "dest_city": f"{dest[0]}, {dest[1]} {dest[2]}",
        "carrier_name": rng.choice(CARRIER_NAMES),
        "equipment_type": rng.choice(EQUIPMENT_TYPES),
        "mode": rng.choice(MODES),
        "currency": rng.choice(CURRENCIES),
    }

    # Rate — sometimes missing
    if rng.random() > 0.15:
        fields["rate"] = round(rng.uniform(200, 8000), 2)
    else:
        fields["rate"] = "MISSING"

    # Weight — sometimes missing
    if rng.random() > 0.2:
        unit = rng.choice(WEIGHT_UNITS)
        if unit == "lbs":
            fields["weight"] = f"{rng.randint(5000, 80000)} {unit}"
        elif unit == "kg":
            fields["weight"] = f"{rng.randint(2000, 36000)} {unit}"
        else:
            fields["weight"] = f"{rng.uniform(2.5, 40):.1f} {unit}"
    else:
        fields["weight"] = "MISSING"

    # Dates — sometimes missing or partial
    base_date = datetime(2026, rng.randint(1, 12), rng.randint(1, 28))
    if rng.random() > 0.1:
        fields["pickup_datetime"] = base_date.strftime(rng.choice([
            "%m/%d/%Y", "%Y-%m-%d", "%d-%b-%Y", "%B %d, %Y", "%m.%d.%Y",
        ]))
    else:
        fields["pickup_datetime"] = "MISSING"

    if rng.random() > 0.15:
        delivery = base_date + timedelta(days=rng.randint(1, 5))
        fields["delivery_datetime"] = delivery.strftime(rng.choice([
            "%m/%d/%Y", "%Y-%m-%d", "%d-%b-%Y", "%B %d, %Y", "%m.%d.%Y",
        ]))
    else:
        fields["delivery_datetime"] = "MISSING"

    # MC number for carriers
    fields["mc_number"] = f"MC-{rng.randint(100000, 9999999)}"

    # PO/Reference numbers
    fields["po_number"] = f"PO-{rng.randint(1000, 99999)}"

    return fields


def format_seed_fields_for_prompt(fields: dict) -> str:
    """Format seed fields as a readable string for the prompt."""
    lines = []
    for k, v in fields.items():
        if v == "MISSING":
            lines.append(f"- {k}: [INTENTIONALLY MISSING — do NOT include this field]")
        else:
            lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def choose_output_format(rng: random.Random) -> str:
    """Pick a random output file format."""
    r = rng.random()
    if r < 0.50:
        return "txt"
    elif r < 0.80:
        return "docx"
    else:
        return "pdf"


def save_as_txt(text: str, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_as_docx(text: str, path: str):
    try:
        from docx import Document
        doc = Document()
        for para in text.split("\n"):
            doc.add_paragraph(para)
        doc.save(path)
    except ImportError:
        # Fallback to txt if python-docx fails
        save_as_txt(text, path.replace(".docx", ".txt"))


def save_as_pdf(text: str, path: str):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(path, pagesize=letter)
        width, height = letter
        y = height - 40
        for line in text.split("\n"):
            if y < 40:
                c.showPage()
                y = height - 40
            # Handle encoding issues
            safe_line = line.encode("ascii", errors="replace").decode("ascii")
            c.drawString(40, y, safe_line[:100])  # truncate long lines
            y -= 14
        c.save()
    except ImportError:
        # Fallback to txt if reportlab not installed
        save_as_txt(text, path.replace(".pdf", ".txt"))


def generate_document(provider, doc_index: int, doc_type: str, rng: random.Random) -> dict:
    """Generate a single synthetic document + ground truth."""
    seed_fields = generate_seed_fields(doc_type, rng)
    messiness = rng.choice(MESSINESS_PROFILES)
    format_style = rng.choice(FORMAT_STYLES)
    output_format = choose_output_format(rng)

    # Step 1: Generate document text
    doc_prompt = registry.get("synth_document")
    rendered_doc = doc_prompt.render(
        doc_type=doc_type,
        messiness_profile=messiness,
        seed_fields=format_seed_fields_for_prompt(seed_fields),
        format_style=format_style,
    )

    doc_response = provider.generate(
        messages=[{"role": "user", "content": rendered_doc}],
        model=provider.fast_model,
        max_tokens=2000,
    )
    doc_text = doc_response.content.strip()

    # Step 2: Generate Q&A + extraction ground truth
    qa_prompt = registry.get("synth_qa_pairs")
    rendered_qa = qa_prompt.render(
        document_text=doc_text,
        doc_type=doc_type,
        seed_fields=format_seed_fields_for_prompt(seed_fields),
    )

    qa_response = provider.generate(
        messages=[{"role": "user", "content": rendered_qa}],
        model=provider.fast_model,
        max_tokens=1500,
        temperature=0.0,
    )

    # Parse ground truth JSON
    qa_text = qa_response.content.strip()
    # Strip markdown code fences if present
    if qa_text.startswith("```"):
        qa_text = qa_text.split("\n", 1)[1] if "\n" in qa_text else qa_text[3:]
        if qa_text.endswith("```"):
            qa_text = qa_text[:-3]
        qa_text = qa_text.strip()

    try:
        test_cases = json.loads(qa_text)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse QA JSON for doc_{doc_index:03d}, skipping ground truth")
        test_cases = []

    # Step 3: Save document file
    filename = f"doc_{doc_index:03d}.{output_format}"
    filepath = os.path.join(OUTPUT_DIR, filename)

    if output_format == "txt":
        save_as_txt(doc_text, filepath)
    elif output_format == "docx":
        save_as_docx(doc_text, filepath)
    elif output_format == "pdf":
        save_as_pdf(doc_text, filepath)

    # Tag test cases with filename
    for tc in test_cases:
        tc["doc_file"] = filename

    return {
        "filename": filename,
        "doc_type": doc_type,
        "format": output_format,
        "messiness": messiness[:50] + "...",
        "seed_fields": seed_fields,
        "test_cases": test_cases,
        "generation_tokens": doc_response.tokens_in + doc_response.tokens_out + qa_response.tokens_in + qa_response.tokens_out,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic logistics documents using LLM")
    parser.add_argument("--count", type=int, default=160, help="Number of documents to generate (default: 160)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--seed-to-langfuse", action="store_true", help="Also seed synthetic prompts to Langfuse")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build document type distribution
    doc_plan = []
    for doc_type, count in DOC_TYPES:
        # Scale counts proportionally if total != 160
        scaled = max(1, round(count * args.count / 160))
        doc_plan.extend([doc_type] * scaled)
    rng.shuffle(doc_plan)
    doc_plan = doc_plan[:args.count]

    print(f"Generating {len(doc_plan)} synthetic documents...")
    print(f"Distribution: {dict((t, doc_plan.count(t)) for t in set(doc_plan))}")
    print()

    provider = get_provider()
    all_test_cases = []
    stats = {
        "total_docs": 0,
        "total_test_cases": 0,
        "total_tokens": 0,
        "by_type": {},
        "by_format": {},
        "errors": 0,
    }

    for i, doc_type in enumerate(doc_plan, start=1):
        try:
            result = generate_document(provider, i, doc_type, rng)
            all_test_cases.extend(result["test_cases"])

            stats["total_docs"] += 1
            stats["total_test_cases"] += len(result["test_cases"])
            stats["total_tokens"] += result["generation_tokens"]
            stats["by_type"][doc_type] = stats["by_type"].get(doc_type, 0) + 1
            stats["by_format"][result["format"]] = stats["by_format"].get(result["format"], 0) + 1

            print(f"  [{i:3d}/{len(doc_plan)}] {result['filename']:20s} | {doc_type:25s} | {len(result['test_cases'])} test cases")

        except Exception as e:
            stats["errors"] += 1
            logger.error(f"Failed to generate doc_{i:03d}: {e}")
            print(f"  [{i:3d}/{len(doc_plan)}] ERROR: {e}")

    # Save ground truth
    with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as f:
        json.dump(all_test_cases, f, indent=2, ensure_ascii=False)

    # Optionally seed prompts to Langfuse
    if args.seed_to_langfuse:
        try:
            from app.config import settings
            from langfuse import Langfuse
            client = Langfuse()
            for name in ["synth_document", "synth_qa_pairs"]:
                template = registry.get(name)
                client.create_prompt(
                    name=name,
                    prompt=template.template,
                    type="text",
                    labels=["production", "latest"],
                )
                print(f"  Seeded to Langfuse: {name}")
            client.flush()
        except Exception as e:
            print(f"  Langfuse seeding failed: {e}")

    # Print summary
    print()
    print("=" * 60)
    print("SYNTHETIC DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Documents generated: {stats['total_docs']}")
    print(f"  Test cases created:  {stats['total_test_cases']}")
    print(f"  Total tokens used:   {stats['total_tokens']}")
    print(f"  Errors:              {stats['errors']}")
    print()
    print(f"  By type:   {json.dumps(stats['by_type'], indent=4)}")
    print(f"  By format: {json.dumps(stats['by_format'], indent=4)}")
    print()
    print(f"  Documents:    {OUTPUT_DIR}/")
    print(f"  Ground truth: {GROUND_TRUTH_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
