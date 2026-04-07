"""Generic extraction prompt for unknown document types."""

from app.llm.prompts.registry import registry

GENERIC_EXTRACTION_PROMPT = """Extract structured shipment data from this logistics document.

<document>
{document_text}
</document>

Extract the following fields into a JSON object. Use null for any field not found in the document.

Fields:
- shipment_id: Any reference number, load ID, BOL number, or tracking number
- shipper: Shipper/sender name and address
- consignee: Consignee/receiver name and address
- pickup_datetime: Pickup date/time in ISO format (YYYY-MM-DDTHH:MM:SS), null if not found
- delivery_datetime: Delivery date/time in ISO format, null if not found
- equipment_type: Equipment or trailer type, null if not found
- mode: Shipping mode (FTL, LTL, etc.), null if not found
- rate: Rate or charge amount as a number, null if not found
- currency: Currency code, null if not found
- weight: Weight with unit, null if not found
- carrier_name: Carrier or transportation company name, null if not found

Respond with ONLY the JSON object. No explanation or markdown formatting."""

registry.register("extraction_generic", GENERIC_EXTRACTION_PROMPT)
