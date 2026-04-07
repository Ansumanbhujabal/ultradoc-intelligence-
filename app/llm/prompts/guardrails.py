"""Out-of-scope detection — keyword-based, no LLM call needed."""

import re

# Multi-word phrases checked via substring (specific enough to avoid false positives)
LOGISTICS_PHRASES = {
    "bill of lading", "rate confirmation", "dry van", "rate breakdown",
    "shipment id", "load id", "pickup date", "delivery date",
    "carrier details", "freight charge", "shipping instructions",
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
}


def is_in_scope(question: str) -> bool:
    question_lower = question.lower()

    # Check multi-word phrases via substring (they're specific enough)
    if any(phrase in question_lower for phrase in LOGISTICS_PHRASES):
        return True

    # Check single words via word boundary to avoid partial matches
    words_in_question = set(re.findall(r"\b\w+\b", question_lower))
    return bool(words_in_question & LOGISTICS_WORDS)
