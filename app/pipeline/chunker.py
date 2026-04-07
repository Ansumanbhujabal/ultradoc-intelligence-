"""Section-based document chunker that keeps tables intact."""
import re

SECTION_PATTERNS = [
    r"^(Bill of Lading|BOL)\s*$",
    r"^(Carrier|Customer)\s+(Details|Information)",
    r"^(Carrier|Customer)\s+Rate\s+and\s+Load\s+Confirmation",
    r"^(Shipper|Consignee|3rd Party Billing|Transportation Company)",
    r"^(Stops|Stop\s+\d+|Pickup|Drop|Delivery)",
    r"^(Rate\s+Breakdown|Carrier\s+Pay|Rate\s+Details)",
    r"^(Standing\s+Instructions|Special\s+Instructions)",
    r"^(Shipper\s*&?\s*Carrier\s+Instructions)",
    r"^(Driver\s+Details|Driver\s+Information)",
    r"^(Notes|Comments|Description|Commodity)",
    r"^(#\s+Of\s+Units|Weight|COD\s+Value)",
    r"^(Consignor|Consignee)\s+name",
    r"^(Test\s+RC\s+Instructions)",
    r"^---\s*Page\s*Break\s*---$",
]

SECTION_REGEX = re.compile("|".join(SECTION_PATTERNS), re.IGNORECASE | re.MULTILINE)


def chunk_document(text: str, max_chunk_size: int = 1000) -> list[dict]:
    if not text.strip():
        return []
    sections = _split_into_sections(text)
    chunks = []
    for section_name, section_text in sections:
        if not section_text.strip():
            continue
        if len(section_text) <= max_chunk_size:
            chunks.append({"text": section_text.strip(), "section": section_name, "index": len(chunks)})
        else:
            sub_chunks = _split_section(section_text, max_chunk_size)
            for sub in sub_chunks:
                if sub.strip():
                    chunks.append({"text": sub.strip(), "section": section_name, "index": len(chunks)})
    return chunks


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    lines = text.split("\n")
    sections = []
    current_section = "content"
    current_lines = []
    for line in lines:
        match = SECTION_REGEX.search(line.strip())
        if match and len(line.strip()) < 80:
            if current_lines:
                sections.append((current_section, "\n".join(current_lines)))
                current_lines = []
            current_section = _normalize_section_name(line.strip())
            current_lines.append(line)
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_section, "\n".join(current_lines)))
    return sections


def _normalize_section_name(header: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9\s]", "", header).strip().lower()
    name = re.sub(r"\s+", "_", name)
    return name or "content"


def _split_section(text: str, max_size: int) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_size:
            current = current + "\n\n" + para if current else para
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return chunks
