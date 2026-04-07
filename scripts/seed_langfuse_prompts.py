"""Seed local prompt templates to Langfuse for prompt management.

Usage:
    uv run python scripts/seed_langfuse_prompts.py

Requires LANGFUSE_ENABLED=true and valid keys in .env.
"""

from app.config import settings
from langfuse import Langfuse

# Import prompt modules to register them locally
from app.llm.prompts.system import SYSTEM_PROMPT, CLASSIFICATION_PROMPT, QUERY_REWRITE_PROMPT
from app.llm.prompts.qa import QA_PROMPT
from app.llm.prompts.extraction.bol import BOL_EXTRACTION_PROMPT
from app.llm.prompts.extraction.rate_confirm import RC_EXTRACTION_PROMPT
from app.llm.prompts.extraction.generic import GENERIC_EXTRACTION_PROMPT

PROMPTS = {
    "system": SYSTEM_PROMPT,
    "classification": CLASSIFICATION_PROMPT,
    "query_rewrite": QUERY_REWRITE_PROMPT,
    "qa_with_cot": QA_PROMPT,
    "extraction_bol": BOL_EXTRACTION_PROMPT,
    "extraction_rc": RC_EXTRACTION_PROMPT,
    "extraction_generic": GENERIC_EXTRACTION_PROMPT,
}


def main():
    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )

    for name, template in PROMPTS.items():
        try:
            client.create_prompt(
                name=name,
                prompt=template,
                type="text",
                labels=["production", "latest"],
            )
            print(f"  Seeded: {name}")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"  Exists: {name} (skipped)")
            else:
                print(f"  Error:  {name} — {e}")

    client.flush()
    print("\nDone. You can now set LANGFUSE_PROMPT_MANAGEMENT=true in .env")


if __name__ == "__main__":
    main()
