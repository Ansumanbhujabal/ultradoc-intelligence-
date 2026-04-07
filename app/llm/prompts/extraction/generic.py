"""Generic extraction prompt for unknown document types."""

from app.llm.prompts.registry import registry

GENERIC_EXTRACTION_PROMPT = """Extract structured shipment data from this logistics document.

<document>
{document_text}
</document>

Extract the following fields into a JSON object. Use null for any field not found in the document.

IMPORTANT RULES:
- If a field contains only a dash (-), "N/A", "None", "TBD", or is blank, use null
- For rate: extract ONLY the freight/carrier rate. Do NOT use COD value, insurance value, declared value, or invoice totals as the rate. If freight charges say "Collect" or "Prepaid" without a numeric amount, use null
- For rate: return the number only, no currency symbols or codes (e.g., 400.00 not "$400.00 USD")
- For dates: use ISO format YYYY-MM-DDTHH:MM:SS. If only a date is available, append T00:00:00

Fields:
- shipment_id: Any reference number, load ID, BOL number, or tracking number
- shipper: Shipper/sender name and address
- consignee: Consignee/receiver name and address
- pickup_datetime: Pickup date/time in ISO format (YYYY-MM-DDTHH:MM:SS), null if not found
- delivery_datetime: Delivery date/time in ISO format, null if not found
- equipment_type: Equipment or trailer type, null if not found
- mode: Shipping mode (FTL, LTL, etc.), null if not found
- rate: Freight rate or carrier charge amount as a number only, null if not found
- currency: Currency code, null if not found
- weight: Weight with unit, null if not found
- carrier_name: Carrier or transportation company name, null if not found

Respond with ONLY the JSON object. No explanation or markdown formatting."""

registry.register("extraction_generic", GENERIC_EXTRACTION_PROMPT)
