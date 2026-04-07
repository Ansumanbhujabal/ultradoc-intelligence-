"""Bill of Lading extraction prompt."""

from app.llm.prompts.registry import registry

BOL_EXTRACTION_PROMPT = """Extract structured shipment data from this Bill of Lading document.

<document>
{document_text}
</document>

Extract the following fields into a JSON object. Use null for any field not found in the document.

Fields:
- shipment_id: The Load ID, reference number, or BOL number
- shipper: Full shipper name and address
- consignee: Full consignee/receiver name and address
- pickup_datetime: Pickup or ship date/time in ISO format (YYYY-MM-DDTHH:MM:SS). If only date is available, use T00:00:00
- delivery_datetime: Delivery date/time in ISO format. If only date is available, use T00:00:00
- equipment_type: Equipment type (e.g., Flatbed, Dry Van, Reefer)
- mode: Shipping mode (e.g., FTL, LTL)
- rate: Freight rate or carrier charge amount (number only, no currency symbol). Do NOT use COD value, insurance value, or declared value as the rate. If freight charges say "Collect" or "Prepaid" without a number, use null
- currency: Currency code (e.g., USD, CAD)
- weight: Weight with unit (e.g., "56000 lbs")
- carrier_name: Carrier or transportation company name

Example output:
{{"shipment_id": "LD12345", "shipper": "ABC Corp, 123 Main St, City, ST 12345", "consignee": "XYZ Inc, 456 Oak Ave, Town, ST 67890", "pickup_datetime": "2026-01-15T09:00:00", "delivery_datetime": "2026-01-16T14:00:00", "equipment_type": "Dry Van", "mode": "FTL", "rate": 1500.00, "currency": "USD", "weight": "42000 lbs", "carrier_name": "Fast Freight LLC"}}

Respond with ONLY the JSON object. No explanation or markdown formatting."""

registry.register("extraction_bol", BOL_EXTRACTION_PROMPT)
