"""Prompt registry — load templates by name and version."""

from app.config import settings


class PromptTemplate:
    def __init__(self, name: str, template: str, version: str = "v1"):
        self.name = name
        self.template = template
        self.version = version

    def render(self, **kwargs) -> str:
        return self.template.format(**kwargs)


class PromptRegistry:
    def __init__(self):
        self._templates: dict[str, dict[str, PromptTemplate]] = {}

    def register(self, name: str, template: str, version: str = "v1"):
        if name not in self._templates:
            self._templates[name] = {}
        self._templates[name][version] = PromptTemplate(name, template, version)

    def get(self, name: str, version: str = "v1") -> PromptTemplate:
        # Try Langfuse first if enabled
        if settings.langfuse_prompt_management:
            try:
                from app.observability.langfuse_integration import get_langfuse_client
                client = get_langfuse_client()
                if client:
                    prompt = client.get_prompt(name)
                    # Return a PromptTemplate wrapping the Langfuse template
                    return PromptTemplate(name, prompt.prompt, version)
            except Exception:
                pass  # fall through to local

        # Local fallback
        if name not in self._templates:
            raise KeyError(f"Prompt template '{name}' not found")
        versions = self._templates[name]
        if version not in versions:
            raise KeyError(f"Version '{version}' not found for prompt '{name}'")
        return versions[version]

    def list_templates(self) -> list[dict]:
        result = []
        for name, versions in self._templates.items():
            for version in versions:
                result.append({"name": name, "version": version})
        return result


registry = PromptRegistry()
