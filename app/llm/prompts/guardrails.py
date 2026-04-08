"""Out-of-scope detection — keyword-based, no LLM call needed."""

import re

# Multi-word phrases checked via substring (specific enough to avoid false positives)
LOGISTICS_PHRASES = {
    "bill of lading", "rate confirmation", "dry van", "rate breakdown",
    "shipment id", "load id", "pickup date", "delivery date", "ship date",
    "carrier details", "freight charge", "shipping instructions",
    "purchase order", "tracking number", "reference number",
    "cod value", "cash on delivery", "collect on delivery",
    "total charge", "total rate", "total cost", "total amount",
}

# Single words checked via word boundary regex (avoids "doc" matching in "doctor")
LOGISTICS_WORDS = {
    "bol", "shipment", "load", "consignee", "shipper", "carrier",
    "freight", "cargo", "commodity", "pickup", "delivery", "transit",
    "trailer", "flatbed", "reefer", "ftl", "ltl", "drayage",
    "consignor", "dispatcher", "waybill", "invoice",
    "equipment", "pallets", "weight", "tonnage",
    "rate", "tariff", "surcharge", "detention", "demurrage",
    "customs", "broker", "intermodal",
    "driver", "customer", "origin", "destination", "route",
    "cod", "insurance", "hazmat", "temperature", "seal",
    "dock", "warehouse", "port", "terminal", "chassis",
    "po", "pro", "sku", "pallet", "crate", "container",
}

# Generic document questions that should pass through when doc is already classified
# Only match patterns that explicitly reference document content
DOCUMENT_QUERY_PATTERNS = [
    r"\bin the document\b", r"\bin this document\b", r"\bfrom the document\b",
    r"\bmentioned\b", r"\baccording to\b", r"\bstated\b", r"\blisted\b",
    r"\bin the bol\b", r"\bin the invoice\b", r"\bon the form\b",
]


OFF_TOPIC_PATTERNS = [
    r"\b(weather|sports?|score|football|basketball|baseball|soccer|cricket)\b",
    r"\b(recipe|cook|movie|music|song|celebrity|gossip)\b",
    r"\b(stock market|bitcoin|crypto|investment advice)\b",
    r"\b(tell me a joke|write a poem|sing|dance)\b",
    r"\b(meaning of life|political|election|president)\b",
]


def is_obviously_off_topic(question: str) -> bool:
    """Hard check for clearly non-logistics questions (weather, sports, etc.)."""
    question_lower = question.lower()
    return any(re.search(pat, question_lower) for pat in OFF_TOPIC_PATTERNS)


def is_in_scope(question: str) -> bool:
    question_lower = question.lower()

    # Check multi-word phrases via substring (they're specific enough)
    if any(phrase in question_lower for phrase in LOGISTICS_PHRASES):
        return True

    # Check single words via word boundary to avoid partial matches
    words_in_question = set(re.findall(r"\b\w+\b", question_lower))
    if words_in_question & LOGISTICS_WORDS:
        return True

    # Check if it's a generic document query (e.g., "What is the ship date?")
    if any(re.search(pat, question_lower) for pat in DOCUMENT_QUERY_PATTERNS):
        return True

    return False
