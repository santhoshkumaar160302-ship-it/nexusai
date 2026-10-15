"""Simulated provider for offline development, tests and benchmarks.

It returns deterministic answers, estimates token usage from text length, and
can inject failures and latency so failover and retries can be exercised
without spending money on real API calls.
"""
from __future__ import annotations

import hashlib
import random
import re
import time

from ..pricing import estimate_tokens
from .base import ChatRequest, ChatResponse, Provider, ProviderError


class MockProvider(Provider):
    def __init__(self, name: str, failure_rate: float = 0.0, latency_ms: float = 0.0, seed: int | None = None):
        self.name = name
        self.failure_rate = failure_rate
        self.latency_ms = latency_ms
        self.calls = 0
        self.down = False  # set True to simulate a full outage
        self._rng = random.Random(seed)

    def chat(self, request: ChatRequest, model: str) -> ChatResponse:
        self.calls += 1
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000 * self._rng.uniform(0.5, 1.5))
        if self.down:
            raise ProviderError(f"{self.name} is down (simulated outage)", provider=self.name, status=503)
        if self._rng.random() < self.failure_rate:
            status = self._rng.choice([429, 500, 503])
            raise ProviderError(f"{self.name} simulated HTTP {status}", provider=self.name, status=status)

        content = self._answer(request)
        truncated = estimate_tokens(content) > request.max_tokens
        if truncated:  # behave like a real model that runs out of tokens
            content = content[: request.max_tokens * 4]
        return ChatResponse(
            content=content,
            provider=self.name,
            model=model,
            prompt_tokens=estimate_tokens(request.prompt_text),
            completion_tokens=estimate_tokens(content),
            truncated=truncated,
        )

    @staticmethod
    def _answer(request: ChatRequest) -> str:
        question = request.last_user_message
        if request.task == "rag":
            # Echo the first context passage so RAG answers carry a real citation.
            match = re.search(r"\[(\d+)\][^\n]*\n(.+?)(?:\n\n|\Z)", question, re.S)
            if match:
                sentence = match.group(2).strip().split(". ")[0][:240]
                return f"{sentence}. [{match.group(1)}]"
            return "I don't know based on the provided sources."
        digest = hashlib.sha256(question.encode()).hexdigest()[:8]
        return f"Simulated answer ({digest}) to: {question[:120]}"
