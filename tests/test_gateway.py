import pytest

from nexusai.gateway import GatewayError
from nexusai.providers.base import ChatRequest, Message, ProviderError
from nexusai.resilience import CircuitBreaker


def ask(services, text, **kw):
    return services.gateway.chat(ChatRequest(messages=[Message("user", text)], **kw))


def test_cache_miss_then_exact_hit(services, providers):
    first = ask(services, "What is a B-tree?")
    second = ask(services, "What is a B-tree?")
    assert first.cache == "miss" and first.model == "gpt-4o-mini"
    assert second.cache == "exact" and second.content == first.content
    assert second.cost_usd == 0 and second.baseline_cost_usd > 0
    assert providers["openai"].calls == 1


def test_semantic_hit_on_rephrasing(services, providers):
    ask(services, "What is consistent hashing in distributed systems?")
    hit = ask(services, "what is consistent hashing in distributed systems")
    assert hit.cache == "semantic" and hit.similarity >= services.settings.semantic_cache_threshold
    assert providers["openai"].calls == 1


def test_different_question_is_not_a_semantic_hit(services):
    ask(services, "What is consistent hashing in distributed systems?")
    assert ask(services, "How do Bloom filters handle false positives?").cache == "miss"


def test_system_prompt_scopes_the_semantic_cache(services):
    sys_a = [Message("system", "Answer like a pirate."), Message("user", "What is TCP?")]
    sys_b = [Message("system", "Answer formally."), Message("user", "What is TCP?")]
    services.gateway.chat(ChatRequest(messages=sys_a))
    assert services.gateway.chat(ChatRequest(messages=sys_b)).cache == "miss"


def test_high_temperature_is_not_cached(services):
    ask(services, "Write a poem about Redis", temperature=0.9)
    assert ask(services, "Write a poem about Redis", temperature=0.9).cache == "miss"


def test_failover_to_other_provider_during_outage(services, providers):
    providers["openai"].down = True
    resp = ask(services, "What is a load balancer?")
    assert resp.provider == "anthropic" and resp.failover
    assert resp.attempts == services.settings.max_retries + 2  # 3 failed OpenAI attempts + 1 success


def test_escalates_tier_when_both_economy_models_fail(services, providers):
    services.settings.max_retries = 0
    calls = {"n": 0}
    original = providers["anthropic"].chat

    def flaky(request, model):
        if model == "claude-haiku-4-5":
            calls["n"] += 1
            raise ProviderError("overloaded", provider="anthropic", status=529)
        return original(request, model)

    providers["openai"].down = True
    providers["anthropic"].chat = flaky
    resp = ask(services, "What is CORS?")
    assert resp.tier == "standard" and resp.model == "claude-sonnet-4-5"


def test_non_retryable_error_fails_fast(services, providers):
    def bad_request(request, model):
        raise ProviderError("bad request", provider="openai", status=400, retryable=False)

    providers["openai"].chat = bad_request
    with pytest.raises(GatewayError) as exc:
        ask(services, "hello there")
    assert exc.value.status == 400
    assert providers["anthropic"].calls == 0
    assert services.store.stats()["failed"] == 1


def test_all_providers_down_returns_error_and_is_logged(services, providers):
    for p in providers.values():
        p.down = True
    with pytest.raises(GatewayError):
        ask(services, "anyone home?")
    stats = services.store.stats()
    assert stats["requests"] == 1 and stats["success_rate"] == 0


def test_open_breaker_skips_provider(services, providers):
    for _ in range(services.settings.breaker_failure_threshold):
        services.gateway.breakers["openai"].record_failure()
    resp = ask(services, "What is gRPC?")
    assert resp.provider == "anthropic"
    assert providers["openai"].calls == 0


def test_breaker_half_open_recovery():
    now = {"t": 0.0}
    breaker = CircuitBreaker(failure_threshold=2, reset_timeout_s=10, clock=lambda: now["t"])
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "open" and not breaker.allow_request()
    now["t"] = 11
    assert breaker.allow_request() and breaker.state == "half_open"
    breaker.record_failure()
    assert breaker.state == "open"
    now["t"] = 22
    breaker.allow_request()
    breaker.record_success()
    assert breaker.state == "closed"


def test_stats_report_savings_and_call_reduction(services):
    for _ in range(3):
        ask(services, "What is a B-tree?")
    stats = services.store.stats()
    assert stats["requests"] == 3
    assert stats["api_call_reduction"] == pytest.approx(2 / 3, abs=1e-3)
    assert 0 < stats["cost_savings"] < 1


def test_cache_miss_embeds_the_prompt_only_once(services):
    calls = {"n": 0}
    original = services.gateway.embedder.embed

    def counting(texts, task="document"):
        calls["n"] += len(texts)
        return original(texts, task)

    services.gateway.embedder.embed = counting
    ask(services, "How do vector databases index embeddings?")
    assert calls["n"] == 1


def test_cut_off_answers_are_flagged_and_not_cached(services, providers):
    first = ask(services, "Tell me everything about B-trees", max_tokens=3)
    assert first.truncated
    again = ask(services, "Tell me everything about B-trees", max_tokens=3)
    assert again.cache == "miss"  # a cut-off answer is never replayed from the cache
    assert providers["openai"].calls == 2
