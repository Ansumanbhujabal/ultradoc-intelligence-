"""Q&A prompt templates with hidden Chain-of-Thought."""

from app.llm.prompts.registry import registry

QA_PROMPT = """Based on the following logistics document, answer the user's question.

<document>
{document_text}
</document>

<relevant_sections>
{source_chunks}
</relevant_sections>

<question>
{question}
</question>

Instructions:
1. First, determine if the question is actually answerable from this document.
2. If the question asks about something NOT covered in the document (e.g., insurance when no insurance info exists, tracking numbers not mentioned, hazmat classifications not listed), you MUST respond with ANSWER: Not found in document. and CONFIDENCE: LOW.
3. If the question is completely unrelated to logistics or the document content (e.g., weather, sports scores, politics, general knowledge), you MUST respond with ANSWER: Not found in document. and CONFIDENCE: LOW.
4. For summary or overview questions (e.g., "What is this document about?", "Give me a summary", "What are all the contact details?"), provide a helpful answer by synthesizing information from the full document. These are valid questions.
5. Only if the answer IS in the document: identify the section, provide a clear and concise answer, and assess confidence.
6. Assess your confidence: HIGH (answer is clearly and explicitly stated in document), MEDIUM (answer requires some interpretation or synthesis of document content), LOW (answer is not found, uncertain, or the question is unrelated to the document).

CRITICAL: Do NOT guess, infer, or provide general knowledge answers. If the specific information is not explicitly in the document, always say "Not found in document." But DO answer questions that can be answered by reading and synthesizing what IS in the document.

Respond in this exact format:
SECTION: [section name or "N/A" if not found]
ANSWER: [your answer, or "Not found in document." if the information is not present]
CONFIDENCE: [HIGH/MEDIUM/LOW]"""

registry.register("qa_with_cot", QA_PROMPT)
