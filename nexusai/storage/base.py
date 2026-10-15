"""Storage interface shared by the in-memory and PostgreSQL/pgvector backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


def context_label(title: str, heading: str) -> str:
    """"Title > Section > Subsection", without repeating the title when the heading path starts with it."""
    if heading and title and (heading == title or heading.startswith(f"{title} > ")):
        return heading
    return " > ".join(part for part in (title, heading) if part)


@dataclass
class Chunk:
    id: str
    document_id: str
    ordinal: int
    text: str
    heading: str = ""
    title: str = ""
    source: str = ""
    embedding: np.ndarray | None = field(default=None, repr=False)

    def public(self) -> dict:
        return {
            "chunk_id": self.id,
            "document_id": self.document_id,
            "title": self.title,
            "source": self.source,
            "heading": self.heading,
            "label": context_label(self.title, self.heading),
            "ordinal": self.ordinal,
        }


@dataclass
class Document:
    id: str
    title: str
    source: str
    content_hash: str
    chunk_count: int = 0
    metadata: dict = field(default_factory=dict)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), pct))


def cost_breakdown(cost: float, uncached: float, baseline: float) -> dict:
    """Split total savings into the part from caching and the part from routing.

    baseline  every request on the baseline model, no cache, no routing
    uncached  routed models, no cache
    cost      what was actually spent
    """
    return {
        "cost_usd": round(cost, 6),
        "uncached_cost_usd": round(uncached, 6),
        "baseline_cost_usd": round(baseline, 6),
        "cost_savings": round(1 - cost / baseline, 4) if baseline else 0.0,
        "savings_from_caching": round(1 - cost / uncached, 4) if uncached else 0.0,
        "savings_from_routing": round(1 - uncached / baseline, 4) if baseline else 0.0,
    }


def summarize(records: list[dict]) -> dict:
    """Aggregate request-log records into the numbers the admin UI and benchmark report."""
    total = len(records)
    ok = [r for r in records if r["status"] == "ok"]
    exact = sum(1 for r in ok if r["cache"] == "exact")
    semantic = sum(1 for r in ok if r["cache"] == "semantic")
    rag_no_context = sum(1 for r in ok if r["cache"] == "no_context")
    upstream_requests = sum(1 for r in records if r["cache"] in ("miss", "error"))
    upstream_attempts = sum(r.get("attempts", 0) for r in records)
    cost = sum(r.get("cost_usd", 0.0) for r in records)
    uncached = sum(r.get("uncached_cost_usd", 0.0) for r in records)
    baseline = sum(r.get("baseline_cost_usd", 0.0) for r in records)
    by_model: dict[str, int] = {}
    for r in ok:
        if r.get("model"):
            by_model[r["model"]] = by_model.get(r["model"], 0) + 1
    latencies = [r["latency_ms"] for r in ok]
    return {
        "requests": total,
        "succeeded": len(ok),
        "failed": total - len(ok),
        "success_rate": round(len(ok) / total, 4) if total else 0.0,
        "cache_hits_exact": exact,
        "cache_hits_semantic": semantic,
        "rag_answered_without_llm": rag_no_context,
        "cache_hit_rate": round((exact + semantic) / total, 4) if total else 0.0,
        "upstream_requests": upstream_requests,
        "upstream_attempts": upstream_attempts,
        "api_call_reduction": round(1 - upstream_requests / total, 4) if total else 0.0,
        "failovers": sum(1 for r in records if r.get("failover")),
        **cost_breakdown(cost, uncached, baseline),
        "latency_p50_ms": round(percentile(latencies, 50), 2),
        "latency_p95_ms": round(percentile(latencies, 95), 2),
        "requests_by_model": dict(sorted(by_model.items(), key=lambda kv: -kv[1])),
    }


class Store(ABC):
    """Persistence for routing rules, request logs, the semantic cache and RAG chunks."""

    # routing rules
    @abstractmethod
    def get_rules(self) -> list[dict] | None: ...

    @abstractmethod
    def save_rules(self, rules: list[dict]) -> None: ...

    # request log
    @abstractmethod
    def log_request(self, record: dict) -> None: ...

    @abstractmethod
    def stats(self) -> dict: ...

    @abstractmethod
    def recent_requests(self, limit: int = 50) -> list[dict]: ...

    @abstractmethod
    def reset_stats(self) -> None: ...

    # semantic cache
    @abstractmethod
    def semantic_lookup(self, namespace: str, embedding: np.ndarray, threshold: float, ttl_s: int) -> tuple[dict, float] | None: ...

    @abstractmethod
    def semantic_store(self, namespace: str, prompt: str, embedding: np.ndarray, response: dict) -> None: ...

    # documents and chunks (RAG)
    @abstractmethod
    def get_document(self, document_id: str) -> Document | None: ...

    @abstractmethod
    def upsert_document(self, document: Document, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    def list_documents(self) -> list[Document]: ...

    @abstractmethod
    def delete_document(self, document_id: str) -> bool: ...

    @abstractmethod
    def search_chunks(self, embedding: np.ndarray, k: int) -> list[tuple[Chunk, float]]: ...
