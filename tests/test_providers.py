"""Provider adapters tested against a mocked HTTP transport (no network)."""
import json

import httpx
import numpy as np
import pytest

from nexusai.embeddings import HashEmbedder, OpenAIEmbedder
from nexusai.gateway import GatewayError
from nexusai.providers import AnthropicProvider, OpenAIProvider
from nexusai.providers.base import ChatRequest, Message, ProviderError


def request():
    return ChatRequest(messages=[Message("system", "Be brief."), Message("user", "Hi")], temperature=0.2, max_tokens=50)


def transport(status=200, body=None, capture=None):
    def handler(req: httpx.Request):
        if capture is not None:
            capture.append({"url": str(req.url), "headers": dict(req.headers), "json": json.loads(req.content)})
        return httpx.Response(status, json=body or {})
    return httpx.MockTransport(handler)


def test_openai_request_shape_and_parsing():
    sent = []
    body = {"model": "gpt-4o-mini-2024", "choices": [{"message": {"content": "Hello!"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3}}
    p = OpenAIProvider("sk-test", transport=transport(body=body, capture=sent))
    resp = p.chat(request(), "gpt-4o-mini")
    assert (resp.content, resp.prompt_tokens, resp.completion_tokens) == ("Hello!", 12, 3)
    call = sent[0]
    assert call["url"].endswith("/chat/completions")
    assert call["headers"]["authorization"] == "Bearer sk-test"
    assert call["json"]["messages"][0] == {"role": "system", "content": "Be brief."}
    assert call["json"]["temperature"] == 0.2 and call["json"]["max_completion_tokens"] == 50


def test_openai_reasoning_models_omit_temperature():
    sent = []
    body = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
    OpenAIProvider("sk-test", transport=transport(body=body, capture=sent)).chat(request(), "gpt-5")
    assert "temperature" not in sent[0]["json"]


def test_anthropic_moves_system_prompt_to_top_level():
    sent = []
    body = {"model": "claude-haiku-4-5", "content": [{"type": "text", "text": "Hey"}],
            "usage": {"input_tokens": 9, "output_tokens": 2}}
    p = AnthropicProvider("sk-ant", transport=transport(body=body, capture=sent))
    resp = p.chat(request(), "claude-haiku-4-5")
    assert (resp.content, resp.prompt_tokens, resp.completion_tokens) == ("Hey", 9, 2)
    call = sent[0]
    assert call["url"].endswith("/messages")
    assert call["headers"]["x-api-key"] == "sk-ant" and call["headers"]["anthropic-version"] == "2023-06-01"
    assert call["json"]["system"] == "Be brief."
    assert all(m["role"] != "system" for m in call["json"]["messages"])


@pytest.mark.parametrize("status, retryable", [(429, True), (500, True), (503, True), (400, False), (401, False)])
def test_http_errors_map_to_retryable_flag(status, retryable):
    p = OpenAIProvider("sk-test", transport=transport(status=status, body={"error": "x"}))
    with pytest.raises(ProviderError) as exc:
        p.chat(request(), "gpt-4o-mini")
    assert exc.value.status == status and exc.value.retryable is retryable


def test_anthropic_overloaded_529_is_retryable():
    p = AnthropicProvider("sk-ant", transport=transport(status=529, body={"error": "overloaded"}))
    with pytest.raises(ProviderError) as exc:
        p.chat(request(), "claude-haiku-4-5")
    assert exc.value.retryable


def test_timeouts_are_retryable():
    def handler(req):
        raise httpx.ReadTimeout("slow", request=req)
    p = OpenAIProvider("sk-test", transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError) as exc:
        p.chat(request(), "gpt-4o-mini")
    assert exc.value.retryable


def test_missing_key_is_rejected():
    with pytest.raises(ValueError):
        OpenAIProvider("")
    with pytest.raises(ValueError):
        AnthropicProvider("")


def test_bad_key_at_one_provider_fails_over_to_the_other(services, providers):
    def unauthorized(req, model):
        raise ProviderError("invalid key", provider="openai", status=401, retryable=False)

    providers["openai"].chat = unauthorized
    resp = services.gateway.chat(ChatRequest(messages=[Message("user", "hello")]))
    assert resp.provider == "anthropic" and resp.failover


def test_bad_request_still_fails_fast(services, providers):
    def invalid(req, model):
        raise ProviderError("bad request", provider="openai", status=400, retryable=False)

    providers["openai"].chat = invalid
    with pytest.raises(GatewayError):
        services.gateway.chat(ChatRequest(messages=[Message("user", "hello")]))
    assert providers["anthropic"].calls == 0


def test_openai_embedder_batches_and_normalizes():
    calls = []

    def handler(req):
        payload = json.loads(req.content)
        calls.append(len(payload["input"]))
        data = [{"index": i, "embedding": [3.0, 4.0]} for i in range(len(payload["input"]))]
        return httpx.Response(200, json={"data": list(reversed(data))})

    emb = OpenAIEmbedder("sk-test", dim=2, batch_size=2, transport=httpx.MockTransport(handler))
    vectors = emb.embed(["a", "b", "c"])
    assert calls == [2, 1]
    assert np.allclose(vectors[0], [0.6, 0.8])


def test_hash_embedder_similarity_tracks_overlap():
    e = HashEmbedder(1024)
    a, b, c = e.embed(["circuit breaker retries", "circuit breakers retried", "chocolate cake recipe"])
    assert np.dot(a, b) > 0.8 and abs(np.dot(a, c)) < 0.2


# ---- Gemini ---------------------------------------------------------------
from nexusai.embeddings import EmbeddingError, GeminiEmbedder  # noqa: E402
from nexusai.providers import GeminiProvider  # noqa: E402
from nexusai.providers.gemini_provider import THINKING_HEADROOM_TOKENS  # noqa: E402


def test_gemini_uses_openai_compatible_endpoint():
    sent = []
    body = {"model": "gemini-3.1-flash-lite", "choices": [{"message": {"content": "Hi!"}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 2}}
    p = GeminiProvider("gm-key", transport=transport(body=body, capture=sent))
    resp = p.chat(request(), "gemini-3.1-flash-lite")
    assert (resp.provider, resp.content, resp.prompt_tokens) == ("gemini", "Hi!", 8)
    call = sent[0]
    assert call["url"] == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    assert call["headers"]["authorization"] == "Bearer gm-key"
    # The answer gets its 50 tokens plus room for Gemini's hidden thinking, which counts against max_tokens.
    assert call["json"]["max_tokens"] == 50 + THINKING_HEADROOM_TOKENS and "max_completion_tokens" not in call["json"]
    assert call["json"]["temperature"] == 0.2 and "reasoning_effort" not in call["json"]


def test_gemini_reasoning_effort_is_opt_in_and_validated():
    sent = []
    body = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
    GeminiProvider("k", reasoning_effort="low", transport=transport(body=body, capture=sent)).chat(request(), "gemini-3.8-flash")
    assert sent[0]["json"]["reasoning_effort"] == "low"
    with pytest.raises(ValueError):
        GeminiProvider("k", reasoning_effort="max")


def test_gemini_errors_are_labeled():
    p = GeminiProvider("k", transport=transport(status=503, body={"error": "busy"}))
    with pytest.raises(ProviderError, match="Gemini HTTP 503") as exc:
        p.chat(request(), "gemini-3.6-flash")
    assert exc.value.provider == "gemini" and exc.value.retryable


def gemini_embed_transport(capture, dim=4, fail_first=0, status=429, headers=None):
    state = {"n": 0}

    def handler(req):
        state["n"] += 1
        if state["n"] <= fail_first:
            return httpx.Response(status, json={"error": "quota"}, headers=headers or {})
        payload = json.loads(req.content)
        capture.append({"url": str(req.url), "headers": dict(req.headers), "json": payload})
        return httpx.Response(200, json={"embeddings": [{"values": [3.0, 4.0] + [0.0] * (dim - 2)}
                                                        for _ in payload["requests"]]})
    return httpx.MockTransport(handler)


def test_gemini_embedder_request_shape_batching_and_normalization():
    sent = []
    emb = GeminiEmbedder("gm-key", dim=128, batch_size=2, transport=gemini_embed_transport(sent, dim=128),
                         sleep=lambda s: None)
    vectors = emb.embed(["a", "b", "c"], task="query")
    assert len(sent) == 2 and len(sent[0]["json"]["requests"]) == 2
    call = sent[0]
    assert call["url"].endswith("/v1beta/models/gemini-embedding-001:batchEmbedContents")
    assert call["headers"]["x-goog-api-key"] == "gm-key"
    first = call["json"]["requests"][0]
    assert first["taskType"] == "RETRIEVAL_QUERY" and first["outputDimensionality"] == 128
    assert first["model"] == "models/gemini-embedding-001" and first["content"]["parts"][0]["text"] == "a"
    assert np.isclose(np.linalg.norm(vectors[0]), 1.0)


def test_gemini_embedder_task_types():
    sent = []
    emb = GeminiEmbedder("k", dim=128, transport=gemini_embed_transport(sent, dim=128), sleep=lambda s: None)
    emb.embed_one("x", task="document")
    emb.embed_one("x", task="similarity")
    assert [c["json"]["requests"][0]["taskType"] for c in sent] == ["RETRIEVAL_DOCUMENT", "SEMANTIC_SIMILARITY"]


def test_gemini_embedder_retries_rate_limits_then_succeeds():
    sent, slept = [], []
    emb = GeminiEmbedder("k", dim=128, transport=gemini_embed_transport(sent, dim=128, fail_first=2,
                         headers={"retry-after": "3"}), sleep=slept.append)
    assert len(emb.embed(["a"])) == 1
    assert slept == [3.0, 3.0]


def test_gemini_embedder_does_not_retry_bad_requests():
    sent = []
    emb = GeminiEmbedder("k", dim=128, transport=gemini_embed_transport(sent, fail_first=9, status=400),
                         sleep=lambda s: None)
    with pytest.raises(EmbeddingError, match="HTTP 400"):
        emb.embed(["a"])


def test_gemini_embedder_gives_up_after_max_attempts():
    emb = GeminiEmbedder("k", dim=128, transport=gemini_embed_transport([], fail_first=99), sleep=lambda s: None)
    with pytest.raises(EmbeddingError, match="after 5 attempts"):
        emb.embed(["a"])


def test_gemini_embedder_rejects_unindexable_dimensions():
    with pytest.raises(ValueError, match="2,000"):
        GeminiEmbedder("k", dim=3072)


# ---- gateway behaviour around rate limits and embedding outages ---------------

def test_rate_limit_moves_to_next_model_without_tripping_breaker(services, providers):
    services.settings.breaker_failure_threshold = 1

    def limited(req, model):
        raise ProviderError("quota", provider="openai", status=429)

    providers["openai"].chat = limited
    for _ in range(3):
        resp = services.gateway.chat(ChatRequest(messages=[Message("user", f"q{_}")]))
        assert resp.provider == "anthropic" and resp.failover
    assert services.gateway.breakers["openai"].state == "closed"


def test_embedding_outage_degrades_to_cache_miss(services):
    def broken(texts, task="document"):
        raise EmbeddingError("embedding API down")

    services.gateway.embedder.embed = broken
    first = services.gateway.chat(ChatRequest(messages=[Message("user", "What is CORS?")]))
    second = services.gateway.chat(ChatRequest(messages=[Message("user", "What is CORS?")]))
    assert first.cache == "miss" and second.cache == "exact"  # exact cache still works


def test_rag_returns_503_when_embeddings_are_down(client, services):
    def broken(texts, task="document"):
        raise EmbeddingError("embedding API down")

    services.retriever.embedder.embed = broken
    resp = client.post("/v1/rag/query", json={"question": "anything"})
    assert resp.status_code == 503 and "embedding" in resp.get_json()["error"]


def test_mock_mode_can_simulate_gemini_only(settings):
    from nexusai.app import build_services

    settings.mock_providers = "gemini"
    services = build_services(settings, sleep=lambda s: None)
    resp = services.gateway.chat(ChatRequest(messages=[Message("user", "hello")]))
    assert resp.provider == "gemini" and resp.model == "gemini-3.1-flash-lite"


def test_settings_pick_gemini_when_it_is_the_only_key(monkeypatch):
    from nexusai.config import Settings

    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NEXUS_EMBEDDINGS", "NEXUS_EMBEDDING_DIM",
                "NEXUS_BASELINE_MODEL", "NEXUS_SEMANTIC_THRESHOLD", "RAG_MIN_SCORE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NEXUS_PROVIDER_MODE", "live")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    s = Settings()
    assert (s.embedding_backend, s.embedding_dim, s.baseline_model) == ("gemini", 768, "gemini-3.8-flash")


def test_rate_limited_last_model_gets_a_second_chance(services, providers):
    providers["openai"].down = True
    calls = {"n": 0}
    original = providers["anthropic"].chat

    def limited_once(req, model):
        calls["n"] += 1
        if calls["n"] <= 2:  # both anthropic candidates rate-limited on the first pass
            raise ProviderError("quota", provider="anthropic", status=429)
        return original(req, model)

    providers["anthropic"].chat = limited_once
    resp = services.gateway.chat(ChatRequest(messages=[Message("user", "hello")]))
    assert resp.provider == "anthropic"


def test_rate_limited_everywhere_returns_429(client, providers):
    def limited(req, model):
        raise ProviderError("quota", provider="x", status=429)

    for p in providers.values():
        p.chat = limited
    resp = client.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 429


def test_truncated_answers_are_flagged():
    def handler(request):
        return httpx.Response(200, json={
            "model": "gpt-4o-mini", "usage": {"prompt_tokens": 5, "completion_tokens": 9},
            "choices": [{"message": {"content": "partial"}, "finish_reason": "length"}],
        })
    p = OpenAIProvider("sk-test", transport=httpx.MockTransport(handler))
    req = ChatRequest(messages=[Message("user", "hi")])
    assert p.chat(req, "gpt-4o-mini").truncated
