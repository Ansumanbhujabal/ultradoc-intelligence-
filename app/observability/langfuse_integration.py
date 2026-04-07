"""Langfuse integration for LLM tracing and observability."""

from typing import Optional
from app.config import settings


def get_langfuse_handler(trace_name: str = "", metadata: dict = None):
    """Returns Langfuse LangChain callback handler if enabled, else None."""
    if not settings.langfuse_enabled:
        return None
    try:
        from langfuse.langchain import CallbackHandler
        kwargs = {}
        if trace_name:
            kwargs["trace_name"] = trace_name
        if metadata:
            kwargs["metadata"] = metadata
        return CallbackHandler(**kwargs)
    except Exception:
        return None


def get_langfuse_client():
    """Returns Langfuse client for prompt management. None if disabled."""
    if not settings.langfuse_enabled:
        return None
    try:
        from langfuse import Langfuse
        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception:
        return None
