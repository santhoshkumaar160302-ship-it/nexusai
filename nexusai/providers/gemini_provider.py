"""Google Gemini adapter, via Gemini's OpenAI-compatible endpoint.

Gemini accepts OpenAI-style chat completions at
https://generativelanguage.googleapis.com/v1beta/openai/ with the API key as a
bearer token, so this adapter reuses the OpenAI one and changes only:

- the base URL and provider name,
- `max_tokens` instead of `max_completion_tokens`,
- an optional `reasoning_effort` ("low" | "medium" | "high"). Gemini 3 models
  always think and their thinking tokens are billed as output, so a low effort
  keeps latency and cost down for simple requests. It is off by default because
  not every model accepts it; set GEMINI_REASONING_EFFORT to enable it.

Free-tier quotas are per model, which is why the router lists several Gemini
models per tier: a 429 on one model moves the request to the next.
"""
from __future__ import annotations

import httpx

from .base import ChatRequest
from .openai_provider import OpenAIProvider

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


class GeminiProvider(OpenAIProvider):
    name = "gemini"
    label = "Gemini"
    default_base_url = GEMINI_OPENAI_BASE_URL
    max_tokens_field = "max_tokens"

    def __init__(self, api_key: str, base_url: str | None = None, timeout_s: float = 30.0,
                 reasoning_effort: str = "", transport: httpx.BaseTransport | None = None):
        if reasoning_effort and reasoning_effort not in ("low", "medium", "high"):
            raise ValueError("GEMINI_REASONING_EFFORT must be low, medium, high or empty")
        super().__init__(api_key, base_url, timeout_s, transport)
        self.reasoning_effort = reasoning_effort

    def build_payload(self, request: ChatRequest, model: str) -> dict:
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        return payload
