"""The gateway: cache lookup, cost-aware routing, failover and logging.

Request flow for `Gateway.chat`:

    exact cache ──hit──▶ return
        │ miss
    semantic cache ──hit──▶ return (and warm the exact cache)
        │ miss
    router.plan() ──▶ candidates, cheapest-first, with escalation
        │
    for each candidate whose circuit breaker allows it:
        try up to (1 + max_retries) times with exponential backoff
        success ──▶ cache, log, return
        non-retryable error (e.g. 400) ──▶ fail fast
    all candidates failed ──▶ GatewayError (logged as status=error)
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict, dataclass, field

from .cache import ExactCache, is_cacheable, semantic_namespace
from .config import Settings
from .embeddings import Embedder
from .pricing import get_model
from .providers.base import PROVIDER_SPECIFIC_STATUSES, ChatRequest, Provider, ProviderError
from .resilience import CircuitBreaker, backoff_delay
from .routing import Router
from .storage.base import Store
from .telemetry import (
    cost_counter,
    latency_histogram,
    provider_attempts,
    requests_counter,
    tokens_counter,
    tracer,
)

log = logging.getLogger("nexusai.gateway")


def route_summary(attempts: list[dict], answered_by: str | None = None, answered_ms: float = 0.0) -> list[dict]:
    """Collapse raw attempt records into readable hops: one entry per model and outcome.

    outcome is "answered", "rate_limited", "failed" or "skipped"; `count` is how
    many times that happened in a row (retries), `detail` a short reason, and
    `ms` the time spent on that hop (summed across retries).
    """
    hops: list[dict] = []
    for a in attempts:
        if a.get("skipped"):
            outcome, detail = "skipped", "provider marked unhealthy (circuit open)"
        elif a.get("status") == 429:
            outcome, detail = "rate_limited", "quota or rate limit reached"
        else:
            status = a.get("status")
            outcome, detail = "failed", (f"HTTP {status}" if status else "timeout or connection error")
        ms = round(a.get("ms", 0.0), 2)
        if hops and hops[-1]["model"] == a["model"] and hops[-1]["outcome"] == outcome:
            hops[-1]["count"] += 1
            hops[-1]["ms"] = round(hops[-1]["ms"] + ms, 2)
        else:
            hops.append({"model": a["model"], "outcome": outcome, "detail": detail, "count": 1, "ms": ms})
    if answered_by:
        hops.append({"model": answered_by, "outcome": "answered", "detail": "", "count": 1,
                     "ms": round(answered_ms, 2)})
    return hops


class GatewayError(Exception):
    def __init__(self, message: str, *, status: int = 502, attempts: list[dict] | None = None):
        super().__init__(message)
        self.status = status
        self.attempts = attempts or []


@dataclass
class GatewayResponse:
    id: str
    content: str
    provider: str
    model: str
    tier: str
    rule: str
    cache: str  # "miss" | "exact" | "semantic"
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    uncached_cost_usd: float  # what this request would have cost without the cache
    baseline_cost_usd: float  # what it would have cost on the baseline model, no cache or routing
    latency_ms: float
    attempts: int
    failover: bool
    similarity: float | None = None
    # Each model the router tried, in order, and what happened: shown in the admin UI.
    route: list[dict] = field(default_factory=list)
    cache_ms: float = 0.0  # time spent checking the exact and semantic caches
    truncated: bool = False  # the answer hit max_tokens and was cut off

    def to_dict(self) -> dict:
        return asdict(self)


class Gateway:
    def __init__(self, settings: Settings, providers: dict[str, Provider], store: Store,
                 exact_cache: ExactCache, embedder: Embedder, router: Router | None = None,
                 sleep=time.sleep):
        self.settings = settings
        self.providers = providers
        self.store = store
        self.exact_cache = exact_cache
        self.embedder = embedder
        self.router = router or Router(store.get_rules(), providers=set(providers))
        self.breakers = {
            name: CircuitBreaker(settings.breaker_failure_threshold, settings.breaker_reset_timeout_s)
            for name in providers
        }
        self._sleep = sleep

    # public API -----------------------------------------------------------
    def chat(self, request: ChatRequest) -> GatewayResponse:
        started = time.perf_counter()
        request_id = f"req_{uuid.uuid4().hex[:16]}"
        with tracer.start_as_current_span("gateway.chat") as span:
            span.set_attribute("nexus.request_id", request_id)
            span.set_attribute("nexus.task", request.task)

            cache_started = time.perf_counter()
            cached, embedding = self._cache_lookup(request)
            cache_ms = (time.perf_counter() - cache_started) * 1000
            if cached:
                response = self._from_cache(request_id, cached, started)
                response.cache_ms = round(cache_ms, 2)
                span.set_attribute("nexus.cache", response.cache)
                self._record(request, response)
                return response

            plan = self.router.plan(request)
            span.set_attribute("nexus.rule", plan.rule)
            span.set_attribute("nexus.tier", plan.tier)
            try:
                result, model, attempts, failover, route = self._call_with_failover(request, plan.candidates)
            except GatewayError as exc:
                self._record_error(request_id, request, plan, exc, started)
                span.set_attribute("nexus.error", str(exc))
                raise

            info = get_model(model)
            baseline = get_model(self.settings.baseline_model)
            response = GatewayResponse(
                id=request_id,
                content=result.content,
                provider=info.provider,
                model=model,
                tier=info.tier,
                rule=plan.rule,
                cache="miss",
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                cost_usd=info.cost(result.prompt_tokens, result.completion_tokens),
                uncached_cost_usd=info.cost(result.prompt_tokens, result.completion_tokens),
                baseline_cost_usd=baseline.cost(result.prompt_tokens, result.completion_tokens),
                latency_ms=(time.perf_counter() - started) * 1000,
                attempts=attempts,
                failover=failover,
                route=route,
                cache_ms=round(cache_ms, 2),
                truncated=result.truncated,
            )
            span.set_attribute("nexus.model", model)
            span.set_attribute("nexus.cache", "miss")
            self._cache_store(request, response, embedding)
            self._record(request, response)
            return response

    def provider_health(self) -> dict:
        return {name: breaker.snapshot() for name, breaker in self.breakers.items()}

    # caching --------------------------------------------------------------
    def _cache_lookup(self, request: ChatRequest):
        """Return (cached response or None, prompt embedding or None).

        The embedding computed for the semantic lookup is handed back so a cache
        miss can store it without embedding the prompt a second time.
        """
        if not is_cacheable(request, self.settings.cache_max_temperature):
            return None, None
        with tracer.start_as_current_span("cache.lookup") as span:
            hit = self.exact_cache.get(request)
            if hit:
                span.set_attribute("nexus.cache", "exact")
                return {**hit, "cache": "exact"}, None
            if not self._semantic_enabled(request):
                return None, None
            try:
                embedding = self.embedder.embed_one(request.last_user_message, task="similarity")
                found = self.store.semantic_lookup(
                    semantic_namespace(request), embedding,
                    self.settings.semantic_cache_threshold, self.settings.semantic_cache_ttl_s,
                )
            except Exception as exc:  # noqa: BLE001 - the cache is an optimization, never a dependency
                # Embedding API down or rate-limited: treat as a cache miss and answer normally.
                log.warning("semantic cache lookup skipped: %s", exc)
                span.set_attribute("nexus.cache.error", str(exc)[:200])
                return None, None
            if not found:
                span.set_attribute("nexus.cache", "miss")
                return None, embedding
            cached, similarity = found
            span.set_attribute("nexus.cache", "semantic")
            span.set_attribute("nexus.cache.similarity", similarity)
            self.exact_cache.set(request, cached)  # next identical request is an exact hit
            return {**cached, "cache": "semantic", "similarity": similarity}, embedding

    def _semantic_enabled(self, request: ChatRequest) -> bool:
        # RAG prompts embed retrieved context in the user message, so two different
        # questions that retrieve the same passages would look near-identical.
        # RAG answers therefore only use the exact cache.
        # Long prompts are skipped for the same reason: shared context dominates the embedding.
        return (self.settings.semantic_cache_enabled and request.task != "rag"
                and len(request.last_user_message) <= self.settings.semantic_cache_max_chars)

    def _cache_store(self, request: ChatRequest, response: GatewayResponse, embedding=None) -> None:
        if not is_cacheable(request, self.settings.cache_max_temperature):
            return
        if response.truncated:
            return  # don't replay a cut-off answer; a retry with more tokens should get a fresh one
        payload = {
            "content": response.content,
            "provider": response.provider,
            "model": response.model,
            "tier": response.tier,
            "rule": response.rule,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
        }
        self.exact_cache.set(request, payload)
        if self._semantic_enabled(request):
            try:
                if embedding is None:
                    embedding = self.embedder.embed_one(request.last_user_message, task="similarity")
                self.store.semantic_store(semantic_namespace(request), request.last_user_message, embedding, payload)
            except Exception as exc:  # noqa: BLE001 - failing to cache must not fail the request
                log.warning("semantic cache store skipped: %s", exc)

    def _from_cache(self, request_id: str, cached: dict, started: float) -> GatewayResponse:
        baseline = get_model(self.settings.baseline_model)
        return GatewayResponse(
            id=request_id,
            content=cached["content"],
            provider=cached["provider"],
            model=cached["model"],
            tier=cached["tier"],
            rule=cached.get("rule", ""),
            cache=cached["cache"],
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0.0,
            uncached_cost_usd=get_model(cached["model"]).cost(cached["prompt_tokens"], cached["completion_tokens"]),
            baseline_cost_usd=baseline.cost(cached["prompt_tokens"], cached["completion_tokens"]),
            latency_ms=(time.perf_counter() - started) * 1000,
            attempts=0,
            failover=False,
            similarity=cached.get("similarity"),
        )

    # failover -------------------------------------------------------------
    def _call_with_failover(self, request, candidates):
        """Try candidates in order; return (result, model, upstream calls, failover).

        Pass 1 walks every candidate. A rate-limited model (429) is skipped
        immediately: quotas are often per model, so the next candidate usually
        succeeds, and 429s do not count against the circuit breaker. If every
        candidate fails, pass 2 waits once and retries only the rate-limited
        models, which recovers short per-minute limits.
        """
        attempts: list[dict] = []
        state = {"tried_any": False}
        primary = next((c for c in candidates if c.provider in self.providers), None)

        rate_limited: list = []
        for info in candidates:
            outcome = self._try_candidate(request, info, attempts, state, retries=self.settings.max_retries)
            if outcome == "rate_limited":
                rate_limited.append(info)
            elif outcome is not None:
                return self._success(outcome, info, attempts, primary, state)

        if rate_limited:
            self._sleep(backoff_delay(1, self.settings.retry_base_delay_s * 4))
            for info in rate_limited:
                outcome = self._try_candidate(request, info, attempts, state, retries=0)
                if outcome not in (None, "rate_limited"):
                    return self._success(outcome, info, attempts, primary, state)
            raise GatewayError("Rate limited by every available model", status=429, attempts=attempts)
        if not state["tried_any"]:
            raise GatewayError("No healthy provider available for this request", status=503, attempts=attempts)
        raise GatewayError("All providers failed for this request", status=502, attempts=attempts)

    @staticmethod
    def _success(result, info, attempts, primary, state):
        calls = sum(1 for a in attempts if "error" in a) + 1
        # Failover = served by a different model than the router's first choice.
        # A retry that succeeds on the same model is not a failover.
        route = route_summary(attempts, info.name, state.get("answer_ms", 0.0))
        return result, info.name, calls, info is not primary, route

    def _try_candidate(self, request, info, attempts, state, retries):
        """Call one candidate with retries. Returns the provider result, "rate_limited",
        or None (failed / skipped). Raises GatewayError for requests no provider can serve."""
        provider = self.providers.get(info.provider)
        breaker = self.breakers.get(info.provider)
        if provider is None or breaker is None:
            return None  # provider not configured (e.g. no API key)
        if not breaker.allow_request():
            attempts.append({"model": info.name, "skipped": "circuit_open"})
            return None
        for attempt in range(retries + 1):
            state["tried_any"] = True
            with tracer.start_as_current_span("provider.call") as span:
                span.set_attribute("nexus.provider", info.provider)
                span.set_attribute("nexus.model", info.name)
                span.set_attribute("nexus.attempt", attempt)
                call_started = time.perf_counter()
                try:
                    result = provider.chat(request, info.name)
                except ProviderError as exc:
                    span.set_attribute("nexus.error", str(exc))
                    attempts.append({"model": info.name, "error": str(exc), "status": exc.status,
                                     "ms": (time.perf_counter() - call_started) * 1000})
                    if exc.status == 429:
                        # A quota, not a sign the provider is unhealthy: don't trip the breaker.
                        provider_attempts.add(1, {"provider": info.provider, "outcome": "rate_limited"})
                        return "rate_limited"
                    breaker.record_failure()
                    provider_attempts.add(1, {"provider": info.provider, "outcome": "error"})
                    if exc.status in PROVIDER_SPECIFIC_STATUSES:
                        return None  # bad key or unknown model at this provider: try the next candidate
                    if not exc.retryable:
                        # The request itself is invalid (e.g. 400); every provider would reject it.
                        raise GatewayError(str(exc), status=exc.status or 400, attempts=attempts) from exc
                    if not breaker.allow_request():
                        return None  # breaker just opened; move to the next candidate
                    if attempt < retries:
                        self._sleep(backoff_delay(attempt, self.settings.retry_base_delay_s))
                    continue
                breaker.record_success()
                provider_attempts.add(1, {"provider": info.provider, "outcome": "ok"})
                state["answer_ms"] = (time.perf_counter() - call_started) * 1000
                return result
        return None

    # logging and metrics --------------------------------------------------
    def _record(self, request: ChatRequest, response: GatewayResponse) -> None:
        labels = {"task": request.task, "cache": response.cache, "model": response.model, "status": "ok"}
        requests_counter.add(1, labels)
        latency_histogram.record(response.latency_ms, {"cache": response.cache})
        if response.cache == "miss":
            tokens_counter.add(response.prompt_tokens, {"model": response.model, "direction": "input"})
            tokens_counter.add(response.completion_tokens, {"model": response.model, "direction": "output"})
            cost_counter.add(response.cost_usd, {"model": response.model})
        self.store.log_request({
            "id": response.id,
            "task": request.task,
            "rule": response.rule,
            "tier": response.tier,
            "provider": response.provider,
            "model": response.model,
            "status": "ok",
            "cache": response.cache,
            "attempts": response.attempts,
            "failover": response.failover,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "cost_usd": response.cost_usd,
            "uncached_cost_usd": response.uncached_cost_usd,
            "baseline_cost_usd": response.baseline_cost_usd,
            "latency_ms": response.latency_ms,
            "error": None,
        })

    def _record_error(self, request_id, request, plan, exc: GatewayError, started) -> None:
        requests_counter.add(1, {"task": request.task, "cache": "error", "status": "error"})
        self.store.log_request({
            "id": request_id,
            "task": request.task,
            "rule": plan.rule,
            "tier": plan.tier,
            "provider": None,
            "model": None,
            "status": "error",
            "cache": "error",
            "attempts": sum(1 for a in exc.attempts if "error" in a),
            "failover": False,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_usd": 0.0,
            "uncached_cost_usd": 0.0,
            "baseline_cost_usd": 0.0,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "error": str(exc)[:500],
        })
