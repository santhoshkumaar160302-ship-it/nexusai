"""Circuit breaker and retry-with-backoff helpers."""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field


@dataclass
class CircuitBreaker:
    """Per-provider circuit breaker.

    closed    -> requests flow; consecutive failures are counted
    open      -> requests are skipped until `reset_timeout_s` passes
    half_open -> one trial request is allowed; success closes, failure re-opens

    State is in-process. With several gateway replicas, move it to Redis so
    all replicas see the same provider health.
    """

    failure_threshold: int = 5
    reset_timeout_s: float = 30.0
    clock: callable = field(default=time.monotonic, repr=False)
    state: str = "closed"
    consecutive_failures: int = 0
    opened_at: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def allow_request(self) -> bool:
        with self._lock:
            if self.state == "open" and self.clock() - self.opened_at >= self.reset_timeout_s:
                self.state = "half_open"
                return True
            return self.state in ("closed", "half_open")

    def record_success(self) -> None:
        with self._lock:
            self.state = "closed"
            self.consecutive_failures = 0

    def record_failure(self) -> None:
        with self._lock:
            self.consecutive_failures += 1
            if self.state == "half_open" or self.consecutive_failures >= self.failure_threshold:
                self.state = "open"
                self.opened_at = self.clock()

    def snapshot(self) -> dict:
        return {"state": self.state, "consecutive_failures": self.consecutive_failures}


def backoff_delay(attempt: int, base_delay_s: float, max_delay_s: float = 8.0) -> float:
    """Exponential backoff with full jitter: random in [0, base * 2^attempt]."""
    return random.uniform(0, min(max_delay_s, base_delay_s * (2 ** attempt)))
