"""Provider-neutral request/response types and the Provider interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class ChatRequest:
    messages: list[Message]
    model: str = "auto"  # "auto" lets the router choose
    temperature: float = 0.0
    max_tokens: int = 512
    task: str = "chat"  # "chat" | "rag"; routing rules can match on it
    tier: str | None = None  # force a tier ("economy" | "standard" | "premium")
    use_cache: bool = True

    @property
    def system_prompt(self) -> str:
        return "\n".join(m.content for m in self.messages if m.role == "system")

    @property
    def last_user_message(self) -> str:
        for m in reversed(self.messages):
            if m.role == "user":
                return m.content
        return ""

    @property
    def prompt_text(self) -> str:
        return "\n".join(f"{m.role}: {m.content}" for m in self.messages)


@dataclass
class ChatResponse:
    content: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    raw: dict = field(default_factory=dict, repr=False)


class ProviderError(Exception):
    """Raised by providers. `retryable` marks transient failures (timeouts, 429, 5xx)."""

    def __init__(self, message: str, *, provider: str, status: int | None = None, retryable: bool = True):
        super().__init__(message)
        self.provider = provider
        self.status = status
        self.retryable = retryable


# Errors tied to one provider (its API key, permissions or model catalog): retrying the
# same provider is pointless, but another provider may well succeed.
PROVIDER_SPECIFIC_STATUSES = frozenset({401, 403, 404})


def is_retryable_status(status: int) -> bool:
    return status == 408 or status == 409 or status == 429 or status >= 500


class Provider(ABC):
    name: str

    @abstractmethod
    def chat(self, request: ChatRequest, model: str) -> ChatResponse:
        """Send a chat completion to `model` and return a normalized response."""
