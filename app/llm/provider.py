"""Provider-agnostic LLM abstraction with Azure OpenAI primary."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from openai import AzureOpenAI

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


class AzureOpenAIProvider(LLMProvider):
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
        self.default_model = settings.azure_openai_model
        self.fast_model = settings.azure_openai_fast_model
        self.embedding_model = settings.azure_openai_embedding_model

    def generate(self, messages: list[dict], model: Optional[str] = None, **kwargs) -> LLMResponse:
        model = model or self.default_model
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 2000),
        )
        latency = round((time.perf_counter() - start) * 1000)
        usage = response.usage
        return LLMResponse(
            content=response.choices[0].message.content or "",
            tokens_in=usage.prompt_tokens if usage else 0,
            tokens_out=usage.completion_tokens if usage else 0,
            latency_ms=latency,
            model_name=model,
        )

    def generate_structured(self, messages: list[dict], schema: dict, model: Optional[str] = None, **kwargs) -> LLMResponse:
        model = model or self.default_model
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=kwargs.get("max_tokens", 2000),
            response_format={"type": "json_object"},
        )
        latency = round((time.perf_counter() - start) * 1000)
        usage = response.usage
        return LLMResponse(
            content=response.choices[0].message.content or "",
            tokens_in=usage.prompt_tokens if usage else 0,
            tokens_out=usage.completion_tokens if usage else 0,
            latency_ms=latency,
            model_name=model,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]


_provider_instance: Optional[LLMProvider] = None


def get_provider() -> LLMProvider:
    global _provider_instance
    if _provider_instance is None:
        if settings.llm_provider == "azure_openai":
            _provider_instance = AzureOpenAIProvider()
        else:
            raise ValueError(f"Unsupported provider: {settings.llm_provider}")
    return _provider_instance
