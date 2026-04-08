"""Tests for LangChain-based LLM provider and config."""
import pytest
from unittest.mock import patch, MagicMock
from app.config import Settings
from app.llm.provider import LLMResponse, LangChainProvider


class TestConfig:
    """Test configuration loading."""

    def test_default_embedding_provider(self):
        """Default embedding provider should be local."""
        s = Settings(
            azure_openai_api_key="test",
            azure_openai_endpoint="https://test.openai.azure.com/",
            _env_file=None,
        )
        assert s.embedding_provider == "local"

    def test_default_langfuse_disabled(self):
        s = Settings(
            azure_openai_api_key="test",
            azure_openai_endpoint="https://test.openai.azure.com/",
            _env_file=None,
        )
        assert s.langfuse_enabled is False
        assert s.langfuse_prompt_management is False

    def test_default_port(self):
        s = Settings(
            azure_openai_api_key="test",
            azure_openai_endpoint="https://test.openai.azure.com/",
            _env_file=None,
        )
        assert s.api_port == 7860


class TestLLMResponse:
    """Test LLMResponse dataclass."""

    def test_defaults(self):
        r = LLMResponse(content="hello")
        assert r.content == "hello"
        assert r.tokens_in == 0
        assert r.tokens_out == 0
        assert r.latency_ms == 0
        assert r.model_name == ""

    def test_all_fields(self):
        r = LLMResponse(content="answer", tokens_in=10, tokens_out=20, latency_ms=500, model_name="gpt-4o")
        assert r.tokens_in == 10
        assert r.tokens_out == 20


class TestLangChainProvider:
    """Test provider with mocked LangChain calls."""

    @patch("langchain_community.embeddings.HuggingFaceEmbeddings")
    @patch("app.llm.provider.AzureChatOpenAI")
    @patch("app.llm.provider.settings")
    def test_generate_returns_llm_response(self, mock_settings, mock_azure, mock_hf):
        mock_settings.azure_openai_model = "gpt-4o"
        mock_settings.azure_openai_fast_model = "gpt-4.1-mini"
        mock_settings.azure_openai_endpoint = "https://test.openai.azure.com/"
        mock_settings.azure_openai_api_key = "test-key"
        mock_settings.azure_openai_api_version = "2024-02-15-preview"
        mock_settings.azure_openai_embedding_model = "text-embedding-3-small"
        mock_settings.embedding_provider = "local"
        mock_settings.langfuse_enabled = False

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "The rate is $400"
        mock_response.usage_metadata = {"input_tokens": 50, "output_tokens": 10}
        mock_llm.invoke.return_value = mock_response
        mock_azure.return_value = mock_llm

        provider = LangChainProvider()
        result = provider.generate(
            messages=[{"role": "user", "content": "What is the rate?"}]
        )

        assert isinstance(result, LLMResponse)
        assert result.content == "The rate is $400"
        assert result.tokens_in == 50
        assert result.tokens_out == 10

    @patch("langchain_community.embeddings.HuggingFaceEmbeddings")
    @patch("app.llm.provider.AzureChatOpenAI")
    @patch("app.llm.provider.settings")
    def test_generate_structured_uses_json_mode(self, mock_settings, mock_azure, mock_hf):
        mock_settings.azure_openai_model = "gpt-4o"
        mock_settings.azure_openai_fast_model = "gpt-4.1-mini"
        mock_settings.azure_openai_endpoint = "https://test.openai.azure.com/"
        mock_settings.azure_openai_api_key = "test-key"
        mock_settings.azure_openai_api_version = "2024-02-15-preview"
        mock_settings.azure_openai_embedding_model = "text-embedding-3-small"
        mock_settings.embedding_provider = "local"
        mock_settings.langfuse_enabled = False

        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"shipment_id": "LD123"}'
        mock_response.usage_metadata = {"input_tokens": 100, "output_tokens": 20}
        mock_bound.invoke.return_value = mock_response
        mock_llm.bind.return_value = mock_bound
        mock_azure.return_value = mock_llm

        provider = LangChainProvider()
        result = provider.generate_structured(
            messages=[{"role": "user", "content": "Extract data"}],
            schema={"type": "object"},
        )

        assert isinstance(result, LLMResponse)
        assert "LD123" in result.content
        mock_llm.bind.assert_called_once_with(response_format={"type": "json_object"})

    @patch("langchain_community.embeddings.HuggingFaceEmbeddings")
    @patch("app.llm.provider.AzureChatOpenAI")
    @patch("app.llm.provider.settings")
    def test_fast_model_selection(self, mock_settings, mock_azure, mock_hf):
        mock_settings.azure_openai_model = "gpt-4o"
        mock_settings.azure_openai_fast_model = "gpt-4.1-mini"
        mock_settings.azure_openai_endpoint = "https://test.openai.azure.com/"
        mock_settings.azure_openai_api_key = "test-key"
        mock_settings.azure_openai_api_version = "2024-02-15-preview"
        mock_settings.azure_openai_embedding_model = "text-embedding-3-small"
        mock_settings.embedding_provider = "local"
        mock_settings.langfuse_enabled = False

        mock_llm = MagicMock()
        mock_fast_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "classified"
        mock_response.usage_metadata = {"input_tokens": 5, "output_tokens": 1}

        # Return different mocks for each AzureChatOpenAI() call
        mock_azure.side_effect = [mock_llm, mock_fast_llm]
        mock_fast_llm.invoke.return_value = mock_response

        provider = LangChainProvider()
        result = provider.generate(
            messages=[{"role": "user", "content": "classify"}],
            model="gpt-4.1-mini",
        )

        mock_fast_llm.invoke.assert_called_once()
        assert result.content == "classified"

    @patch("langchain_community.embeddings.HuggingFaceEmbeddings")
    @patch("app.llm.provider.AzureChatOpenAI")
    @patch("app.llm.provider.settings")
    def test_embed_local(self, mock_settings, mock_azure, mock_hf):
        mock_settings.azure_openai_model = "gpt-4o"
        mock_settings.azure_openai_fast_model = "gpt-4.1-mini"
        mock_settings.azure_openai_endpoint = "https://test.openai.azure.com/"
        mock_settings.azure_openai_api_key = "test-key"
        mock_settings.azure_openai_api_version = "2024-02-15-preview"
        mock_settings.embedding_provider = "local"
        mock_settings.langfuse_enabled = False

        mock_azure.return_value = MagicMock()
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_hf.return_value = mock_embeddings

        provider = LangChainProvider()
        result = provider.embed(["test sentence"])
        assert len(result) == 1
        assert len(result[0]) > 0  # embedding vector
        assert isinstance(result[0][0], float)

    @patch("langchain_community.embeddings.HuggingFaceEmbeddings")
    @patch("app.llm.provider.AzureChatOpenAI")
    @patch("app.llm.provider.settings")
    def test_message_conversion(self, mock_settings, mock_azure, mock_hf):
        mock_settings.azure_openai_model = "gpt-4o"
        mock_settings.azure_openai_fast_model = "gpt-4.1-mini"
        mock_settings.azure_openai_endpoint = "https://test.openai.azure.com/"
        mock_settings.azure_openai_api_key = "test-key"
        mock_settings.azure_openai_api_version = "2024-02-15-preview"
        mock_settings.embedding_provider = "local"
        mock_settings.langfuse_enabled = False
        mock_azure.return_value = MagicMock()

        provider = LangChainProvider()
        from langchain_core.messages import SystemMessage, HumanMessage
        msgs = provider._to_langchain_messages([
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ])
        assert len(msgs) == 2
        assert isinstance(msgs[0], SystemMessage)
        assert isinstance(msgs[1], HumanMessage)
