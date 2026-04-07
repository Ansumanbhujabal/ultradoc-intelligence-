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
1. First, identify which section of the document contains the answer.
2. Then, provide a clear and concise answer.
3. Assess your confidence: HIGH (answer is clearly stated in document), MEDIUM (answer requires some interpretation), LOW (answer is uncertain or partially found).

Respond in this exact format:
SECTION: [section name or heading where you found the answer]
ANSWER: [your answer]
CONFIDENCE: [HIGH/MEDIUM/LOW]"""

registry.register("qa_with_cot", QA_PROMPT)
