"""Cost-aware routing.

Routing rules are evaluated in priority order (lowest number first); the first
rule whose conditions all match picks a tier, or pins a specific model. The
router then builds an ordered candidate list for failover:

    1. models in the chosen tier, cheapest first
    2. models in the next tier up, cheapest first (escalation)

so a request is served by the cheapest healthy model that is good enough,
and an outage at one provider falls over to the other.

Rules are plain JSON so the admin UI can edit them at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass

from .pricing import CATALOG, TIERS, ModelInfo, blended_price, estimate_tokens, get_model, models_in_tier
from .providers.base import ChatRequest

DEFAULT_RULES: list[dict] = [
    {
        "name": "rag-answers",
        "priority": 10,
        "match": {"task": "rag"},
        "tier": "standard",
        "description": "Grounded answers with citations need a capable model.",
    },
    {
        "name": "complex-reasoning",
        "priority": 20,
        "match": {"keywords_any": ["step by step", "prove", "debug", "architecture", "trade-off", "refactor"]},
        "tier": "premium",
        "description": "Reasoning-heavy prompts go to the premium tier.",
    },
    {
        "name": "long-context",
        "priority": 30,
        "match": {"min_prompt_tokens": 2000},
        "tier": "standard",
        "description": "Long prompts get a standard-tier model.",
    },
    {
        "name": "default-economy",
        "priority": 100,
        "match": {},
        "tier": "economy",
        "description": "Everything else goes to the cheapest tier.",
    },
]

_ALLOWED_MATCH_KEYS = {"task", "keywords_any", "min_prompt_tokens", "max_prompt_tokens"}


class RuleValidationError(ValueError):
    pass


def validate_rules(rules: list[dict]) -> list[dict]:
    """Check rules from the admin UI before saving them."""
    if not isinstance(rules, list) or not rules:
        raise RuleValidationError("rules must be a non-empty list")
    names = set()
    for i, rule in enumerate(rules):
        where = f"rule #{i + 1}"
        if not isinstance(rule, dict):
            raise RuleValidationError(f"{where}: must be an object")
        name = rule.get("name")
        if not name or not isinstance(name, str):
            raise RuleValidationError(f"{where}: 'name' is required")
        if name in names:
            raise RuleValidationError(f"{where}: duplicate name '{name}'")
        names.add(name)
        if not isinstance(rule.get("priority"), int):
            raise RuleValidationError(f"{where} ({name}): 'priority' must be an integer")
        match = rule.get("match", {})
        if not isinstance(match, dict):
            raise RuleValidationError(f"{where} ({name}): 'match' must be an object")
        unknown = set(match) - _ALLOWED_MATCH_KEYS
        if unknown:
            raise RuleValidationError(f"{where} ({name}): unknown match keys {sorted(unknown)}")
        if "keywords_any" in match and not isinstance(match["keywords_any"], list):
            raise RuleValidationError(f"{where} ({name}): 'keywords_any' must be a list")
        has_tier, has_model = "tier" in rule, "model" in rule
        if has_tier == has_model:
            raise RuleValidationError(f"{where} ({name}): set exactly one of 'tier' or 'model'")
        if has_tier and rule["tier"] not in TIERS:
            raise RuleValidationError(f"{where} ({name}): tier must be one of {TIERS}")
        if has_model and rule["model"] not in CATALOG:
            raise RuleValidationError(f"{where} ({name}): unknown model '{rule['model']}'")
    if not any(not r.get("match") for r in rules):
        raise RuleValidationError("add a catch-all rule with an empty 'match' so every request is routed")
    return sorted(rules, key=lambda r: r["priority"])


@dataclass
class RoutePlan:
    rule: str
    tier: str
    candidates: list[ModelInfo]


class Router:
    def __init__(self, rules: list[dict] | None = None, providers: set[str] | None = None):
        """`providers` limits routing to configured providers (None = every provider in the catalog)."""
        self.rules = validate_rules(rules or DEFAULT_RULES)
        self.providers = set(providers) if providers is not None else None

    def set_rules(self, rules: list[dict]) -> None:
        self.rules = validate_rules(rules)

    @staticmethod
    def _matches(match: dict, request: ChatRequest) -> bool:
        if "task" in match and match["task"] != request.task:
            return False
        tokens = estimate_tokens(request.prompt_text)
        if "min_prompt_tokens" in match and tokens < match["min_prompt_tokens"]:
            return False
        if "max_prompt_tokens" in match and tokens > match["max_prompt_tokens"]:
            return False
        if "keywords_any" in match:
            text = request.last_user_message.lower()
            if not any(k.lower() in text for k in match["keywords_any"]):
                return False
        return True

    def plan(self, request: ChatRequest) -> RoutePlan:
        # An explicit model or tier from the caller overrides the rules.
        if request.model and request.model != "auto":
            pinned = get_model(request.model)
            if not self._available(pinned):
                raise ValueError(f"model '{pinned.name}' needs the {pinned.provider} provider, which is not configured")
            return RoutePlan("caller-model", pinned.tier, self._candidates(pinned.tier, first=pinned))
        if request.tier:
            if request.tier not in TIERS:
                raise ValueError(f"tier must be one of {TIERS}")
            return RoutePlan("caller-tier", request.tier, self._candidates(request.tier))

        for rule in self.rules:
            if self._matches(rule.get("match", {}), request):
                if "model" in rule:
                    pinned = get_model(rule["model"])
                    first = pinned if self._available(pinned) else None
                    return RoutePlan(rule["name"], pinned.tier, self._candidates(pinned.tier, first=first))
                return RoutePlan(rule["name"], rule["tier"], self._candidates(rule["tier"]))
        raise RuntimeError("no routing rule matched")  # unreachable: validate_rules requires a catch-all

    def _available(self, model: ModelInfo) -> bool:
        return self.providers is None or model.provider in self.providers

    def _candidates(self, tier: str, first: ModelInfo | None = None) -> list[ModelInfo]:
        """Ordered failover list:
          1. the chosen tier, cheapest first
          2. one tier up (escalate quality)
          3. lower tiers, nearest first (degrade quality rather than fail)
        Step 3 matters on free tiers: when the Flash models' small quotas run out,
        a Flash-Lite answer beats an error. Only configured providers are listed.
        """
        ordered: list[ModelInfo] = [first] if first else []
        start = TIERS.index(tier)
        order = list(TIERS[start:start + 2]) + list(reversed(TIERS[:start]))
        for t in order:
            ordered.extend(m for m in models_in_tier(t, self.providers) if m not in ordered)
        if not ordered:  # configured providers have no model in those tiers: use whatever is closest
            available = [m for m in CATALOG.values() if self._available(m)]
            ordered = sorted(available, key=lambda m: (abs(TIERS.index(m.tier) - start), blended_price(m)))
        return ordered
