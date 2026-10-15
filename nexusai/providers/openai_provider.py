"""OpenAI Chat Completions adapter (plain HTTP via httpx, no SDK).

Also the base class for providers that expose an OpenAI-compatible endpoint
(Gemini, and others such as Groq or OpenRouter): they differ only in base URL,
API key and a few request fields.
"""
from __future__ import annotations

import httpx

from .base import ChatRequest, ChatResponse, Provider, ProviderError, is_retryable_status

# OpenAI reasoning models only accept the default temperature, so it must be omitted.
REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


class OpenAIProvider(Provider):
    name = "openai"
    label = "OpenAI"
    default_base_url = "https://api.openai.com/v1"
    max_tokens_field = "max_completion_tokens"

    def __init__(self, api_key: str, base_url: str | None = None, timeout_s: float = 30.0,
                 transport: httpx.BaseTransport | None = None):
        if not api_key:
            raise ValueError(f"An API key is required for the {self.label} provider")
        self._client = httpx.Client(
            base_url=(base_url or self.default_base_url).rstrip("/"),
            timeout=timeout_s,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    def build_payload(self, request: ChatRequest, model: str) -> dict:
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            self.max_tokens_field: request.max_tokens,
        }
        if not model.startswith(REASONING_MODEL_PREFIXES):
            payload["temperature"] = request.temperature
        return payload

    def chat(self, request: ChatRequest, model: str) -> ChatResponse:
        try:
            resp = self._client.post("/chat/completions", json=self.build_payload(request, model))
        except httpx.TimeoutException as exc:
            raise ProviderError(f"{self.label} timeout: {exc}", provider=self.name, retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.label} connection error: {exc}", provider=self.name, retryable=True) from exc

        if resp.status_code != 200:
            raise ProviderError(
                f"{self.label} HTTP {resp.status_code}: {resp.text[:300]}",
                provider=self.name,
                status=resp.status_code,
                retryable=is_retryable_status(resp.status_code),
            )
        data = resp.json()
        usage = data.get("usage") or {}
        choice = (data.get("choices") or [{}])[0]
        return ChatResponse(
            content=(choice.get("message") or {}).get("content") or "",
            provider=self.name,
            model=data.get("model", model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            truncated=choice.get("finish_reason") == "length",
            raw=data,
        )
