"""Model catalog: which provider serves each model, its tier, and its price.

Prices are USD per 1M tokens at each provider's paid list price (Gemini models
are also free on Google's free tier; costs are still reported at list price so
savings stay meaningful). Prices change, so verify them against the providers'
pricing pages before reporting cost numbers. Gemini prices are as listed in
October 2026; 3.6/3.8 Flash list prices double on Jan 1, 2027.

The router only uses models whose provider is configured, so the catalog can
list every provider. Add or edit models here without touching routing logic.
"""
from __future__ import annotations

from dataclasses import dataclass

TIERS = ("economy", "standard", "premium")


@dataclass(frozen=True)
class ModelInfo:
    name: str
    provider: str
    tier: str
    input_per_mtok: float
    output_per_mtok: float

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.input_per_mtok + completion_tokens * self.output_per_mtok) / 1_000_000


CATALOG: dict[str, ModelInfo] = {
    m.name: m
    for m in [
        # Gemini: free-tier models only. Gemini 3.1 Pro is paid-only; add it as
        # ModelInfo("gemini-3.1-pro-preview", "gemini", "premium", 2.00, 12.00) on a paid key.
        ModelInfo("gemini-3.1-flash-lite", "gemini", "economy", 0.25, 1.50),
        ModelInfo("gemini-3.5-flash-lite", "gemini", "economy", 0.30, 2.50),
        ModelInfo("gemini-3.6-flash", "gemini", "standard", 0.75, 3.75),
        ModelInfo("gemini-3.8-flash", "gemini", "premium", 0.75, 3.75),
        ModelInfo("gpt-4o-mini", "openai", "economy", 0.15, 0.60),
        ModelInfo("claude-haiku-4-5", "anthropic", "economy", 1.00, 5.00),
        ModelInfo("gpt-4.1", "openai", "standard", 2.00, 8.00),
        ModelInfo("claude-sonnet-4-5", "anthropic", "standard", 3.00, 15.00),
        ModelInfo("gpt-5", "openai", "premium", 1.25, 10.00),
        ModelInfo("claude-opus-4-1", "anthropic", "premium", 15.00, 75.00),
    ]
}


def get_model(name: str) -> ModelInfo:
    try:
        return CATALOG[name]
    except KeyError as exc:
        raise KeyError(f"Unknown model '{name}'. Known models: {', '.join(CATALOG)}") from exc


def blended_price(m: ModelInfo) -> float:
    """Price used to rank models: output tokens weighted 3x, since answers cost more than prompts."""
    return m.input_per_mtok + 3 * m.output_per_mtok


def models_in_tier(tier: str, providers: set[str] | None = None) -> list[ModelInfo]:
    """Models in a tier, cheapest first, optionally limited to configured providers."""
    models = [m for m in CATALOG.values() if m.tier == tier and (providers is None or m.provider in providers)]
    return sorted(models, key=blended_price)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (about 4 characters per token for English)."""
    return max(1, len(text) // 4)
