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

Fields:
- shipment_id: The Reference ID, Load ID, or booking reference
- shipper: Full shipper/pickup location name and address (from the Pickup stop)
- consignee: Full consignee/drop location name and address (from the Drop stop)
- pickup_datetime: Shipping/pickup date and time in ISO format (YYYY-MM-DDTHH:MM:SS). Combine Shipping Date + Shipping Time or Appointment time
- delivery_datetime: Delivery date and time in ISO format. Combine Delivery Date + Delivery Time
- equipment_type: Equipment type from Carrier Details (e.g., Flatbed, Dry Van)
- mode: Load Type (e.g., FTL, LTL)
- rate: Total rate/agreed amount as a number (from Rate Breakdown total or Agreed Amount)
- currency: Currency code (e.g., USD)
- weight: Weight with unit from commodity details (e.g., "56000 lbs")
- carrier_name: Carrier company name from Carrier Details section

Example output:
{{"shipment_id": "LD53657", "shipper": "AAA, Los Angeles International Airport (LAX), World Way, Los Angeles, CA, USA", "consignee": "xyz, 7470 Cherry Avenue, Fontana, CA 92336, USA", "pickup_datetime": "2026-02-08T09:00:00", "delivery_datetime": "2026-02-08T09:00:00", "equipment_type": "Flatbed", "mode": "FTL", "rate": 400.00, "currency": "USD", "weight": "56000.00 lbs", "carrier_name": "SWIFT SHIFT LOGISTICS LLC"}}

Respond with ONLY the JSON object. No explanation or markdown formatting."""

registry.register("extraction_rc", RC_EXTRACTION_PROMPT)
