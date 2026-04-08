"""Rate Confirmation extraction prompt."""

from app.llm.prompts.registry import registry

RC_EXTRACTION_PROMPT = """Extract structured shipment data from this Rate Confirmation document.

<document>
{document_text}
</document>

Extract the following fields into a JSON object. Use null for any field not found in the document.

Rate Confirmations typically contain:
- Reference/Load ID in the header section
- Carrier details section with carrier name, MC number, equipment
- Stops section with pickup (shipper) and drop (consignee) locations, dates, times
- Rate Breakdown section with charges and totals
- Customer or shipper details

IMPORTANT RULES:
- If a field contains only a dash (-), "N/A", "None", "TBD", or is blank, use null
- For rate: return the number only, no currency symbols or codes (e.g., 400.00 not "$400.00 USD")
- For dates: if a time window is given (e.g., "09:00 - 17:00"), use the start time. If an appointment time exists, prefer it over the window
- For dates: use ISO format YYYY-MM-DDTHH:MM:SS

Fields:
- shipment_id: The Reference ID, Load ID, or booking reference
- shipper: Shipper/pickup company or entity name only, no address (from the Pickup stop)
- consignee: Consignee/drop company or entity name only, no address (from the Drop stop)
- pickup_datetime: Shipping/pickup date and time in ISO format (YYYY-MM-DDTHH:MM:SS). Combine Shipping Date + Shipping Time or Appointment time
- delivery_datetime: Delivery date and time in ISO format. Combine Delivery Date + Delivery Time
- equipment_type: Equipment type from Carrier Details (e.g., Flatbed, Dry Van)
- mode: Load Type (e.g., FTL, LTL)
- rate: Total rate/agreed amount as a number (from Rate Breakdown total or Agreed Amount)
- currency: Currency code (e.g., USD)
- weight: Weight with unit from commodity details (e.g., "56000 lbs")
- carrier_name: Carrier company name from Carrier Details section

Example output:
{{"shipment_id": "LD53657", "shipper": "AAA", "consignee": "xyz", "pickup_datetime": "2026-02-08T09:00:00", "delivery_datetime": "2026-02-08T09:00:00", "equipment_type": "Flatbed", "mode": "FTL", "rate": 400.00, "currency": "USD", "weight": "56000.00 lbs", "carrier_name": "SWIFT SHIFT LOGISTICS LLC"}}

Respond with ONLY the JSON object. No explanation or markdown formatting."""

registry.register("extraction_rc", RC_EXTRACTION_PROMPT)
