"""Langfuse integration for LLM tracing and observability."""

import os
from app.config import settings


def _ensure_langfuse_env():
    """Set Langfuse env vars so the SDK auto-configures."""
    if settings.langfuse_enabled:
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
        os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)


def get_langfuse_handler():
    """Returns Langfuse LangChain callback handler if enabled, else None."""
    if not settings.langfuse_enabled:
        return None
    try:
        _ensure_langfuse_env()
        from langfuse.langchain import CallbackHandler
        return CallbackHandler()
    except Exception:
        return None


def get_langfuse_client():
    """Returns Langfuse client for prompt management. None if disabled."""
    if not settings.langfuse_enabled:
        return None
    try:
        _ensure_langfuse_env()
        from langfuse import Langfuse
        return Langfuse()
    except Exception:
        return None
