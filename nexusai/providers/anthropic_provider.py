"""Anthropic Messages API adapter (plain HTTP via httpx, no SDK)."""
from __future__ import annotations

import httpx

from .base import ChatRequest, ChatResponse, Provider, ProviderError, is_retryable_status

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com/v1", timeout_s: float = 30.0,
                 transport: httpx.BaseTransport | None = None):
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for the Anthropic provider")
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout_s,
            transport=transport,
            headers={
                "x-api-key": api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
        )

    def chat(self, request: ChatRequest, model: str) -> ChatResponse:
        # Anthropic takes the system prompt as a top-level field, not a message.
        messages = [{"role": m.role, "content": m.content} for m in request.messages if m.role != "system"]
        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.system_prompt:
            payload["system"] = request.system_prompt

        try:
            resp = self._client.post("/messages", json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderError(f"Anthropic timeout: {exc}", provider=self.name, retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Anthropic connection error: {exc}", provider=self.name, retryable=True) from exc

        if resp.status_code != 200:
            raise ProviderError(
                f"Anthropic HTTP {resp.status_code}: {resp.text[:300]}",
                provider=self.name,
                status=resp.status_code,
                retryable=is_retryable_status(resp.status_code) or resp.status_code == 529,
            )
        data = resp.json()
        text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        usage = data.get("usage", {})
        return ChatResponse(
            content=text,
            provider=self.name,
            model=data.get("model", model),
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            truncated=data.get("stop_reason") == "max_tokens",
            raw=data,
        )
