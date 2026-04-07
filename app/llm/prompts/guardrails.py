"""Out-of-scope detection — keyword-based, no LLM call needed."""

LOGISTICS_KEYWORDS = {
    "document", "doc", "bol", "bill of lading", "rate confirmation", "invoice",
    "shipment", "load", "order", "reference",
    "shipper", "consignee", "carrier", "driver", "dispatcher", "receiver",
    "sender", "customer", "vendor",
    "pickup", "delivery", "drop", "stop", "route", "transit",
    "freight", "cargo", "commodity", "weight", "units", "quantity",
    "equipment", "trailer", "truck", "flatbed", "dry van", "reefer",
    "ftl", "ltl", "mode",
    "rate", "charge", "cost", "price", "pay", "total", "amount",
    "currency", "usd", "invoice", "billing", "cod",
    "date", "time", "when", "schedule", "appointment", "deadline",
    "address", "location", "city", "state", "zip", "where",
    "what", "who", "how much", "how many", "which", "list", "show",
    "find", "tell", "extract", "id", "number", "name", "po",
    "page", "section", "field", "detail", "information", "data",
    "mention", "say", "state", "contain", "include", "describe",
}


def is_in_scope(question: str) -> bool:
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in LOGISTICS_KEYWORDS)
