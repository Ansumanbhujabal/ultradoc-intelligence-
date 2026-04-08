"""Tests for Langfuse integration and prompt registry fallback."""

import pytest
from unittest.mock import patch, MagicMock

from app.llm.prompts.registry import PromptRegistry, PromptTemplate


class TestLangfuseIntegration:
    """Test Langfuse handler and client factories."""

    @patch("app.observability.langfuse_integration.settings")
    def test_handler_returns_none_when_disabled(self, mock_settings):
        mock_settings.langfuse_enabled = False
        from app.observability.langfuse_integration import get_langfuse_handler

        assert get_langfuse_handler() is None

    @patch("app.observability.langfuse_integration.settings")
    def test_client_returns_none_when_disabled(self, mock_settings):
        mock_settings.langfuse_enabled = False
        from app.observability.langfuse_integration import get_langfuse_client

        assert get_langfuse_client() is None

    @patch("app.observability.langfuse_integration.settings")
    def test_handler_returns_none_on_import_error(self, mock_settings):
        """If CallbackHandler construction fails, handler should gracefully return None."""
        mock_settings.langfuse_enabled = True
        mock_settings.langfuse_public_key = "pk-test"
        mock_settings.langfuse_secret_key = "sk-test"
        mock_settings.langfuse_host = "https://test.langfuse.com"
        import app.observability.langfuse_integration as mod

        with patch.dict("sys.modules", {"langfuse": MagicMock(), "langfuse.langchain": MagicMock()}):
            with patch(
                "langfuse.langchain.CallbackHandler",
                side_effect=Exception("connection failed"),
            ):
                result = mod.get_langfuse_handler()
                assert result is None

    @patch("app.observability.langfuse_integration.settings")
    def test_handler_returns_instance_when_enabled(self, mock_settings):
        mock_settings.langfuse_enabled = True
        mock_settings.langfuse_public_key = "pk-test"
        mock_settings.langfuse_secret_key = "sk-test"
        mock_settings.langfuse_host = "https://test.langfuse.com"

        mock_cb = MagicMock()
        mock_module = MagicMock()
        mock_module.CallbackHandler.return_value = mock_cb

        with patch.dict("sys.modules", {"langfuse": MagicMock(), "langfuse.langchain": mock_module}):
            from app.observability.langfuse_integration import get_langfuse_handler

            handler = get_langfuse_handler()
            assert handler is mock_cb
            mock_module.CallbackHandler.assert_called_once_with()

    @patch("app.observability.langfuse_integration.settings")
    def test_client_returns_none_on_error(self, mock_settings):
        mock_settings.langfuse_enabled = True
        mock_settings.langfuse_public_key = "pk-test"
        mock_settings.langfuse_secret_key = "sk-test"
        mock_settings.langfuse_host = "https://test.langfuse.com"

        mock_langfuse_mod = MagicMock()
        mock_langfuse_mod.Langfuse.side_effect = Exception("connection failed")

        with patch.dict("sys.modules", {"langfuse": mock_langfuse_mod}):
            from app.observability.langfuse_integration import get_langfuse_client

            assert get_langfuse_client() is None

    @patch("app.observability.langfuse_integration.settings")
    def test_client_returns_instance_when_enabled(self, mock_settings):
        mock_settings.langfuse_enabled = True
        mock_settings.langfuse_public_key = "pk-test"
        mock_settings.langfuse_secret_key = "sk-test"
        mock_settings.langfuse_host = "https://test.langfuse.com"

        mock_client = MagicMock()
        mock_langfuse_mod = MagicMock()
        mock_langfuse_mod.Langfuse.return_value = mock_client

        with patch.dict("sys.modules", {"langfuse": mock_langfuse_mod}):
            from app.observability.langfuse_integration import get_langfuse_client

            result = get_langfuse_client()
            assert result is mock_client
            mock_langfuse_mod.Langfuse.assert_called_once_with()


class TestPromptRegistry:
    """Test prompt registry with Langfuse fallback."""

    @patch("app.llm.prompts.registry.settings")
    def test_local_registration_and_get(self, mock_settings):
        mock_settings.langfuse_prompt_management = False
        reg = PromptRegistry()
        reg.register("test_prompt", "Hello {name}")
        result = reg.get("test_prompt")
        assert isinstance(result, PromptTemplate)
        assert result.render(name="World") == "Hello World"

    @patch("app.llm.prompts.registry.settings")
    def test_missing_prompt_raises(self, mock_settings):
        mock_settings.langfuse_prompt_management = False
        reg = PromptRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    @patch("app.llm.prompts.registry.settings")
    def test_missing_version_raises(self, mock_settings):
        mock_settings.langfuse_prompt_management = False
        reg = PromptRegistry()
        reg.register("test", "hello", version="v1")
        with pytest.raises(KeyError):
            reg.get("test", version="v99")

    @patch("app.llm.prompts.registry.settings")
    def test_list_templates(self, mock_settings):
        mock_settings.langfuse_prompt_management = False
        reg = PromptRegistry()
        reg.register("a", "template_a")
        reg.register("b", "template_b", version="v2")
        templates = reg.list_templates()
        assert len(templates) == 2
        names = [t["name"] for t in templates]
        assert "a" in names
        assert "b" in names

    @patch("app.llm.prompts.registry.settings")
    def test_langfuse_prompt_fetch_when_enabled(self, mock_settings):
        mock_settings.langfuse_prompt_management = True
        reg = PromptRegistry()
        reg.register("my_prompt", "local version")

        mock_client = MagicMock()
        mock_langfuse_prompt = MagicMock()
        mock_langfuse_prompt.prompt = "langfuse version"
        mock_client.get_prompt.return_value = mock_langfuse_prompt

        with patch(
            "app.observability.langfuse_integration.get_langfuse_client",
            return_value=mock_client,
        ):
            result = reg.get("my_prompt")
            assert result.template == "langfuse version"

    @patch("app.llm.prompts.registry.settings")
    def test_falls_back_to_local_on_langfuse_error(self, mock_settings):
        mock_settings.langfuse_prompt_management = True
        reg = PromptRegistry()
        reg.register("my_prompt", "local fallback")

        with patch(
            "app.observability.langfuse_integration.get_langfuse_client",
            side_effect=Exception("connection failed"),
        ):
            result = reg.get("my_prompt")
            assert result.template == "local fallback"

    @patch("app.llm.prompts.registry.settings")
    def test_skips_langfuse_when_disabled(self, mock_settings):
        mock_settings.langfuse_prompt_management = False
        reg = PromptRegistry()
        reg.register("my_prompt", "local only")
        result = reg.get("my_prompt")
        assert result.template == "local only"

    @patch("app.llm.prompts.registry.settings")
    def test_langfuse_client_none_falls_back_to_local(self, mock_settings):
        """When Langfuse is enabled but client returns None, use local."""
        mock_settings.langfuse_prompt_management = True
        reg = PromptRegistry()
        reg.register("my_prompt", "local version")

        with patch(
            "app.observability.langfuse_integration.get_langfuse_client",
            return_value=None,
        ):
            result = reg.get("my_prompt")
            assert result.template == "local version"

    def test_prompt_template_render(self):
        pt = PromptTemplate("test", "Hi {user}, welcome to {place}")
        assert pt.render(user="Alice", place="Wonderland") == "Hi Alice, welcome to Wonderland"
        assert pt.name == "test"
        assert pt.version == "v1"

    def test_prompt_template_custom_version(self):
        pt = PromptTemplate("test", "v2 template", version="v2")
        assert pt.version == "v2"
