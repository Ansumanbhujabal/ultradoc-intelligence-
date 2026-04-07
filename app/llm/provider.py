"""Provider-agnostic LLM abstraction using LangChain."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings


@dataclass
class LLMResponse:
    content: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    model_name: str = ""


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, messages: list[dict], model: Optional[str] = None, **kwargs) -> LLMResponse:
        ...

    @abstractmethod
    def generate_structured(self, messages: list[dict], schema: dict, model: Optional[str] = None, **kwargs) -> LLMResponse:
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class LangChainProvider(LLMProvider):
    def __init__(self):
        common_kwargs = dict(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
        self.llm = AzureChatOpenAI(
            azure_deployment=settings.azure_openai_model,
            **common_kwargs,
        )
        self.fast_llm = AzureChatOpenAI(
            azure_deployment=settings.azure_openai_fast_model,
            **common_kwargs,
        )
        self.default_model = settings.azure_openai_model
        self.fast_model = settings.azure_openai_fast_model

        # Configurable embeddings
        if settings.embedding_provider == "azure":
            self.embeddings = AzureOpenAIEmbeddings(
                model=settings.azure_openai_embedding_model,
                **common_kwargs,
            )
        else:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def _select_llm(self, model: Optional[str] = None):
        if model and model == self.fast_model:
            return self.fast_llm
        return self.llm

    def _to_langchain_messages(self, messages: list[dict]) -> list:
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))
        return lc_messages

    def _get_callbacks(self):
        """Get Langfuse callback if enabled."""
        try:
            from app.observability.langfuse_integration import get_langfuse_handler
            handler = get_langfuse_handler()
            return [handler] if handler else []
        except ImportError:
            return []

    def generate(self, messages: list[dict], model: Optional[str] = None, **kwargs) -> LLMResponse:
        llm = self._select_llm(model)
        lc_messages = self._to_langchain_messages(messages)
        callbacks = self._get_callbacks()

        start = time.perf_counter()
        response = llm.invoke(
            lc_messages,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 2000),
            config={"callbacks": callbacks} if callbacks else None,
        )
        latency = round((time.perf_counter() - start) * 1000)

        usage = response.usage_metadata or {}
        return LLMResponse(
            content=response.content or "",
            tokens_in=usage.get("input_tokens", 0),
            tokens_out=usage.get("output_tokens", 0),
            latency_ms=latency,
            model_name=model or self.default_model,
        )

    def generate_structured(self, messages: list[dict], schema: dict, model: Optional[str] = None, **kwargs) -> LLMResponse:
        llm = self._select_llm(model)
        llm_json = llm.bind(response_format={"type": "json_object"})
        lc_messages = self._to_langchain_messages(messages)
        callbacks = self._get_callbacks()

        start = time.perf_counter()
        response = llm_json.invoke(
            lc_messages,
            config={"callbacks": callbacks} if callbacks else None,
        )
        latency = round((time.perf_counter() - start) * 1000)

        usage = response.usage_metadata or {}
        return LLMResponse(
            content=response.content or "",
            tokens_in=usage.get("input_tokens", 0),
            tokens_out=usage.get("output_tokens", 0),
            latency_ms=latency,
            model_name=model or self.default_model,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(texts)


_provider_instance: Optional[LLMProvider] = None


def get_provider() -> LLMProvider:
    global _provider_instance
    if _provider_instance is None:
        if settings.llm_provider == "azure_openai":
            _provider_instance = LangChainProvider()
        else:
            raise ValueError(f"Unsupported provider: {settings.llm_provider}")
    return _provider_instance
