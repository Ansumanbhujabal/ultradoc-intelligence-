"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Azure OpenAI
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_api_version: str = "2024-02-15-preview"
    azure_openai_model: str = "gpt-4o"
    azure_openai_fast_model: str = "gpt-4.1-mini"
    azure_openai_embedding_model: str = "text-embedding-3-small"

    # Provider
    llm_provider: str = "azure_openai"

    # Storage
    chroma_persist_dir: str = "./data/chroma"
    upload_dir: str = "./data/uploads"

    # Pipeline defaults
    default_confidence_threshold: float = 0.35
    default_retrieval_mode: str = "hybrid"
    grounding_overlap_threshold: float = 0.4
    low_confidence_refusal_threshold: float = 0.3
    retrieval_top_k: int = 5
    final_top_k: int = 3

    # Embedding provider
    embedding_provider: str = "local"  # "local" | "azure"

    # Langfuse
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://us.cloud.langfuse.com"
    langfuse_prompt_management: bool = False

    # Server
    api_port: int = 7860

    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/ultradoc.log"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
