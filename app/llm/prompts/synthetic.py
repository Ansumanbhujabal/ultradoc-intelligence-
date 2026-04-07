"""Prompt templates for LLM-powered synthetic data generation."""

from app.llm.prompts.registry import registry

SYNTH_DOCUMENT_PROMPT = """Generate a realistic but MESSY logistics document of type: {doc_type}

MESSINESS PROFILE:
{messiness_profile}

USE THESE SEED VALUES (embed them naturally in the document):
{seed_fields}

FORMAT STYLE: {format_style}

RULES:
- Generate the FULL document text as it would appear in a real logistics system
- Apply ALL the messiness instructions above — typos, formatting issues, missing info, etc.
- The document should look like it was scanned, copy-pasted, or exported from a legacy TMS
- Include realistic addresses, phone numbers, reference numbers
- Do NOT include any markdown formatting — output raw document text only
- Do NOT explain what you're doing — just output the document
- Length: between 200 and 1500 words depending on document type

OUTPUT: Just the raw document text, nothing else."""

SYNTH_QA_PAIRS_PROMPT = """Given this logistics document, generate test cases for a Q&A and extraction system.

<document>
{document_text}
</document>

DOCUMENT TYPE: {doc_type}

KNOWN SEED VALUES (these were used to generate the document — use them as expected answers):
{seed_fields}

Generate a JSON array of test cases. Include:
1. 2-3 factual Q&A questions that can be answered from the document (use natural phrasing, not robotic)
2. 1 out-of-scope question (not related to logistics at all — weather, sports, politics, etc.)
3. 1 not-found question (asks about a specific logistics field that is NOT present in this document)
4. 1 extraction test case with expected field values (use null for fields not in the document)

FORMAT — return ONLY this JSON array, no explanation:
[
  {{"type": "qa", "question": "What is the carrier rate for this shipment?", "expected_answer": "1250.00", "expected_source_contains": "1250"}},
  {{"type": "qa", "question": "Who is the consignee?", "expected_answer": "ABC Logistics Inc", "expected_source_contains": "ABC"}},
  {{"type": "qa", "question": "What pickup date is listed?", "expected_answer": "February 8, 2026", "expected_source_contains": "Feb"}},
  {{"type": "qa", "question": "What is the current stock price of Apple?", "expected_answer": "__OUT_OF_SCOPE__", "expected_source_contains": ""}},
  {{"type": "qa", "question": "What is the insurance policy number?", "expected_answer": "__NOT_FOUND__", "expected_source_contains": ""}},
  {{"type": "extraction", "expected_fields": {{"shipment_id": "LD78901", "shipper": "Test Corp", "consignee": "ABC Logistics Inc", "pickup_datetime": null, "delivery_datetime": null, "equipment_type": "Flatbed", "mode": "FTL", "rate": 1250.00, "currency": "USD", "weight": "42000 lbs", "carrier_name": "Fast Freight LLC"}}}}
]

RULES:
- Use the SEED VALUES as the source of truth for expected answers
- If a seed value was intentionally omitted (marked as MISSING), use null in extraction and don't ask about it in Q&A
- Questions should be natural and varied — don't always use the same phrasing
- expected_source_contains should be a short distinctive substring from the document
- Return ONLY the JSON array"""

registry.register("synth_document", SYNTH_DOCUMENT_PROMPT)
registry.register("synth_qa_pairs", SYNTH_QA_PAIRS_PROMPT)
