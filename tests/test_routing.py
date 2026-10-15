import pytest

from nexusai.providers.base import ChatRequest, Message
from nexusai.routing import DEFAULT_RULES, RuleValidationError, validate_rules
from nexusai.routing import Router as _Router


def Router(rules=None, providers=("openai", "anthropic")):
    return _Router(rules, providers=set(providers))


def req(text, **kw):
    return ChatRequest(messages=[Message("user", text)], **kw)


def test_simple_prompt_goes_to_cheapest_economy_model():
    plan = Router().plan(req("What is a B-tree?"))
    assert plan.rule == "default-economy"
    assert plan.tier == "economy"
    assert plan.candidates[0].name == "gpt-4o-mini"


def test_candidates_cover_both_providers_then_escalate_one_tier():
    plan = Router().plan(req("hello"))
    tiers = [m.tier for m in plan.candidates]
    assert tiers == ["economy", "economy", "standard", "standard"]
    assert {m.provider for m in plan.candidates[:2]} == {"openai", "anthropic"}


def test_keyword_rule_routes_to_premium():
    plan = Router().plan(req("Please debug this race condition"))
    assert plan.rule == "complex-reasoning" and plan.tier == "premium"
    tiers = [m.tier for m in plan.candidates]
    assert tiers[:2] == ["premium", "premium"]  # no tier above premium...
    assert tiers[2:] == ["standard", "standard", "economy", "economy"]  # ...so degrade, nearest first


def test_rag_task_routes_to_standard():
    assert Router().plan(req("q", task="rag")).tier == "standard"


def test_long_prompt_routes_to_standard():
    assert Router().plan(req("x " * 5000)).rule == "long-context"


def test_caller_model_and_tier_override_rules():
    plan = Router().plan(req("debug this", model="claude-haiku-4-5"))
    assert plan.rule == "caller-model" and plan.candidates[0].name == "claude-haiku-4-5"
    assert Router().plan(req("hi", tier="premium")).tier == "premium"


def test_rule_can_pin_a_model():
    rules = [{"name": "pin", "priority": 1, "match": {}, "model": "claude-sonnet-4-5"}]
    assert Router(rules).plan(req("hi")).candidates[0].name == "claude-sonnet-4-5"


@pytest.mark.parametrize("bad, message", [
    ([], "non-empty"),
    ([{"name": "a", "priority": 1, "match": {"task": "chat"}, "tier": "economy"}], "catch-all"),
    ([{"name": "a", "priority": 1, "match": {}, "tier": "gold"}], "tier must be"),
    ([{"name": "a", "priority": 1, "match": {}, "tier": "economy", "model": "gpt-4.1"}], "exactly one"),
    ([{"name": "a", "priority": 1, "match": {"colour": "red"}, "tier": "economy"}], "unknown match keys"),
    ([{"name": "a", "priority": 1, "match": {}, "tier": "economy"},
      {"name": "a", "priority": 2, "match": {}, "tier": "economy"}], "duplicate"),
    ([{"name": "a", "priority": "1", "match": {}, "tier": "economy"}], "priority"),
])
def test_invalid_rules_are_rejected(bad, message):
    with pytest.raises(RuleValidationError, match=message):
        validate_rules(bad)


def test_default_rules_are_valid_and_sorted():
    rules = validate_rules(DEFAULT_RULES)
    assert [r["priority"] for r in rules] == sorted(r["priority"] for r in rules)


def test_gemini_only_routes_to_gemini_models():
    r = Router(providers=["gemini"])
    economy = r.plan(req("What is a B-tree?"))
    assert [m.name for m in economy.candidates] == [
        "gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.6-flash"]
    assert r.plan(req("q", task="rag")).candidates[0].name == "gemini-3.6-flash"
    assert r.plan(req("Please debug this")).candidates[0].name == "gemini-3.8-flash"


def test_tier_with_no_configured_model_falls_back_to_nearest_tier(monkeypatch):
    from nexusai import pricing

    catalog = {k: v for k, v in pricing.CATALOG.items() if k != "gemini-3.8-flash"}
    monkeypatch.setattr(pricing, "CATALOG", catalog)
    import nexusai.routing as routing
    monkeypatch.setattr(routing, "CATALOG", catalog)

    plan = Router(providers=["gemini"]).plan(req("hi", tier="premium"))  # no Gemini premium model now
    assert [m.name for m in plan.candidates][:2] == ["gemini-3.6-flash", "gemini-3.1-flash-lite"]


def test_rule_pinned_to_unconfigured_model_uses_configured_ones():
    rules = [{"name": "pin", "priority": 1, "match": {}, "model": "claude-opus-4-1"}]
    plan = Router(rules, providers=["gemini"]).plan(req("hi"))
    assert plan.candidates[0].name == "gemini-3.8-flash"


def test_caller_model_from_unconfigured_provider_is_rejected():
    with pytest.raises(ValueError, match="not configured"):
        Router(providers=["gemini"]).plan(req("hi", model="gpt-4.1"))


def test_rag_on_gemini_degrades_to_flash_lite_last():
    plan = Router(providers=["gemini"]).plan(req("q", task="rag"))
    assert [m.name for m in plan.candidates] == [
        "gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]
