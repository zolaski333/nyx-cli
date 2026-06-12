"""OpenAI provider."""
from __future__ import annotations


from .openrouter import OpenRouterProvider


class OpenAIProvider(OpenRouterProvider):
    """Provider for OpenAI's API (compatible, same wire protocol)."""

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.get_api_key()}",
            "Content-Type": "application/json",
        }
