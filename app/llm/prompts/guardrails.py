"""Out-of-scope detection — keyword-based, no LLM call needed."""

LOGISTICS_KEYWORDS = {
    "document", "doc", "bol", "bill of lading", "rate confirmation", "invoice",
    "shipment", "load", "order", "reference",
    "shipper", "consignee", "carrier", "driver", "dispatcher", "receiver",
    "sender", "vendor",
    "pickup", "delivery", "drop", "stop", "route", "transit",
    "freight", "cargo", "commodity", "weight", "quantity",
    "equipment", "trailer", "truck", "flatbed", "dry van", "reefer",
    "ftl", "ltl",
    "rate", "charge", "cost", "price", "pay", "total", "amount",
    "currency", "usd", "billing", "cod",
    "schedule", "appointment", "deadline",
    "address", "location", "city", "zip",
    "extract", "po",
    "section", "field",
}


def is_in_scope(question: str) -> bool:
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in LOGISTICS_KEYWORDS)
