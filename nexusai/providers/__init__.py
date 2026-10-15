from __future__ import annotations

from ..config import Settings
from ..pricing import CATALOG
from .anthropic_provider import AnthropicProvider
from .base import ChatRequest, ChatResponse, Message, Provider, ProviderError
from .gemini_provider import GeminiProvider
from .mock_provider import MockProvider
from .openai_provider import OpenAIProvider

__all__ = [
    "AnthropicProvider",
    "ChatRequest",
    "ChatResponse",
    "GeminiProvider",
    "Message",
    "MockProvider",
    "OpenAIProvider",
    "Provider",
    "ProviderError",
    "build_providers",
]


def build_providers(settings: Settings) -> dict[str, Provider]:
    """Create the provider set for the configured mode.

    Mock mode registers simulated providers under real provider names (from
    MOCK_PROVIDERS), so routing and failover behave exactly as they would live.
    Live mode enables every provider that has an API key.
    """
    if settings.provider_mode == "mock":
        names = [n.strip() for n in settings.mock_providers.split(",") if n.strip()]
        known = {m.provider for m in CATALOG.values()}
        unknown = set(names) - known
        if not names or unknown:
            raise ValueError(f"MOCK_PROVIDERS must list providers from {sorted(known)}")
        return {
            name: MockProvider(name, settings.mock_failure_rate, settings.mock_latency_ms, seed=i + 1)
            for i, name in enumerate(names)
        }

    providers: dict[str, Provider] = {}
    if settings.gemini_api_key:
        providers["gemini"] = GeminiProvider(
            settings.gemini_api_key, settings.gemini_base_url, settings.request_timeout_s,
            settings.gemini_reasoning_effort,
        )
    if settings.openai_api_key:
        providers["openai"] = OpenAIProvider(settings.openai_api_key, settings.openai_base_url, settings.request_timeout_s)
    if settings.anthropic_api_key:
        providers["anthropic"] = AnthropicProvider(
            settings.anthropic_api_key, settings.anthropic_base_url, settings.request_timeout_s
        )
    if not providers:
        raise RuntimeError("Live mode needs at least one of GEMINI_API_KEY, OPENAI_API_KEY or ANTHROPIC_API_KEY")
    return providers
