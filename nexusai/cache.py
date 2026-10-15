"""Two-level response cache.

1. Exact cache (Redis, or in-memory): key = SHA-256 of the normalized request.
   Cheap, and catches identical repeat requests.
2. Semantic cache (pgvector, or in-memory): embeds the user prompt and returns
   a cached response when a previous prompt in the same namespace is at least
   `threshold` cosine-similar. Catches rephrasings of the same question.

The namespace is a hash of everything besides the user prompt that changes the
answer (system prompt, earlier turns, pinned model or tier, task), so a cached
answer is never reused for a request with a different context.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time

from .providers.base import ChatRequest


def _sha(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def exact_key(request: ChatRequest) -> str:
    return "nexus:exact:" + _sha({
        "messages": [[m.role, " ".join(m.content.split())] for m in request.messages],
        "model": request.model,
        "tier": request.tier,
        "task": request.task,
        "temperature": round(request.temperature, 2),
        "max_tokens": request.max_tokens,
    })


def semantic_namespace(request: ChatRequest) -> str:
    context = [[m.role, m.content] for m in request.messages[:-1]]  # everything but the last user turn
    return _sha({"context": context, "model": request.model, "tier": request.tier, "task": request.task})[:32]


def is_cacheable(request: ChatRequest, max_temperature: float) -> bool:
    return request.use_cache and request.temperature <= max_temperature


class MemoryKV:
    """Minimal Redis-like key/value store with TTL, used when REDIS_URL is unset."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            value, expires = item
            if expires < time.time():
                del self._data[key]
                return None
            return value

    def set(self, key: str, value: str, ex: int) -> None:
        with self._lock:
            self._data[key] = (value, time.time() + ex)

    def ping(self) -> bool:
        return True


class ExactCache:
    def __init__(self, kv, ttl_s: int):
        self.kv = kv
        self.ttl_s = ttl_s

    def get(self, request: ChatRequest) -> dict | None:
        raw = self.kv.get(exact_key(request))
        if raw is None:
            return None
        return json.loads(raw)

    def set(self, request: ChatRequest, response: dict) -> None:
        self.kv.set(exact_key(request), json.dumps(response), ex=self.ttl_s)


def build_kv(settings):
    if settings.redis_url:
        import redis

        return redis.Redis.from_url(settings.redis_url, decode_responses=True)
    return MemoryKV()
