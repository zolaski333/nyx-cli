"""LLM provider abstraction layer. Each provider wraps a different API."""
from __future__ import annotations

from nyx.config import Config

from .base import BaseLLMProvider, LLMResponse as LLMResponse
from .openrouter import OpenRouterProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider


def get_provider(config: Config) -> BaseLLMProvider:
    """Factory: returns the right provider for the current config."""
    registry: dict[str, type[BaseLLMProvider]] = {
        "openrouter": OpenRouterProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }
    provider_cls = registry.get(config.provider)
    if not provider_cls:
        available = ", ".join(registry)
        raise ValueError(f"Unknown provider '{config.provider}'. Available: {available}")
    return provider_cls(config)
