"""Eval prompt templates — LLM-as-judge for semantic equivalence."""

from app.llm.prompts.registry import registry

EVAL_QA_JUDGE = """You are an eval judge. Determine if the ACTUAL answer is semantically equivalent to the EXPECTED answer for the given question. The actual answer may contain extra details (like addresses, titles) — that's fine as long as the core expected value is present and correct.

Question: {question}
Expected answer: {expected}
Actual answer: {actual}

Reply with ONLY "YES" or "NO"."""

registry.register("eval_qa_judge", EVAL_QA_JUDGE)
