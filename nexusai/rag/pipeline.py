"""RAG question answering: retrieve, ground, answer with citations.

The answer is generated through the gateway (task="rag"), so it gets the same
routing, failover, caching and cost tracking as any other request. If no
chunk clears the relevance bar, the pipeline answers "not found" without
calling an LLM at all, which saves cost and avoids ungrounded answers.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field

from ..gateway import Gateway
from ..pricing import get_model
from ..providers.base import ChatRequest, Message
from ..storage.base import Store
from ..telemetry import tracer
from .ingest import context_label
from .retriever import RetrievedChunk, Retriever

SYSTEM_PROMPT = """You answer questions using only the numbered sources provided.
Rules:
- Cite every claim with the source number in square brackets, like [1] or [2][3].
- If the sources do not contain the answer, reply exactly: "I don't know based on the provided sources."
- Do not use outside knowledge. Be concise."""

NO_CONTEXT_ANSWER = "I don't know based on the provided sources."
_CITATION_RE = re.compile(r"\[(\d+)\]")


def build_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, item in enumerate(chunks, start=1):
        c = item.chunk
        label = context_label(c.title, c.heading)
        blocks.append(f"[{i}] {label} ({c.source})\n{c.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[Message]:
    user = f"Sources:\n\n{build_context(chunks)}\n\nQuestion: {question}"
    return [Message("system", SYSTEM_PROMPT), Message("user", user)]


def extract_citations(answer: str, n_sources: int) -> list[int]:
    """Citation numbers used in the answer, in first-use order, ignoring out-of-range ones."""
    seen: list[int] = []
    for match in _CITATION_RE.finditer(answer):
        n = int(match.group(1))
        if 1 <= n <= n_sources and n not in seen:
            seen.append(n)
    return seen


@dataclass
class RAGAnswer:
    answer: str
    citations: list[dict]
    sources: list[dict]
    grounded: bool
    model: str | None = None
    provider: str | None = None
    cache: str | None = None
    cost_usd: float = 0.0
    retrieval_ms: float = 0.0
    total_ms: float = 0.0
    request_id: str | None = None
    rule: str | None = None
    tier: str | None = None
    route: list = field(default_factory=list)
    cache_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    truncated: bool = False
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class RAGPipeline:
    def __init__(self, gateway: Gateway, retriever: Retriever, store: Store, default_top_k: int = 5):
        self.gateway = gateway
        self.retriever = retriever
        self.store = store
        self.default_top_k = default_top_k

    def answer(self, question: str, top_k: int | None = None, tier: str | None = None) -> RAGAnswer:
        question = question.strip()
        if not question:
            raise ValueError("question is required")
        started = time.perf_counter()
        with tracer.start_as_current_span("rag.answer") as span:
            chunks = self.retriever.retrieve(question, top_k or self.default_top_k)
            retrieval_ms = (time.perf_counter() - started) * 1000
            sources = [item.public() for item in chunks]
            span.set_attribute("nexus.rag.sources", len(chunks))

            if not chunks:
                return self._no_context(question, retrieval_ms, started)

            request = ChatRequest(messages=build_messages(question, chunks), task="rag", tier=tier, max_tokens=1024)
            response = self.gateway.chat(request)
            cited = extract_citations(response.content, len(chunks))
            citations = [{"n": n, **chunks[n - 1].chunk.public()} for n in cited]
            grounded = bool(cited) and NO_CONTEXT_ANSWER.lower() not in response.content.lower()
            span.set_attribute("nexus.rag.grounded", grounded)
            return RAGAnswer(
                answer=response.content,
                citations=citations,
                sources=sources,
                grounded=grounded,
                model=response.model,
                provider=response.provider,
                cache=response.cache,
                cost_usd=response.cost_usd,
                retrieval_ms=round(retrieval_ms, 2),
                total_ms=round((time.perf_counter() - started) * 1000, 2),
                request_id=response.id,
                rule=response.rule,
                tier=response.tier,
                route=response.route,
                cache_ms=response.cache_ms,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                truncated=response.truncated,
            )

    def _no_context(self, question: str, retrieval_ms: float, started: float) -> RAGAnswer:
        """Nothing relevant was retrieved: answer without spending an LLM call."""
        request_id = f"req_{uuid.uuid4().hex[:16]}"
        baseline = get_model(self.gateway.settings.baseline_model)
        total_ms = (time.perf_counter() - started) * 1000
        self.store.log_request({
            "id": request_id, "task": "rag", "rule": "no-context", "tier": None, "provider": None,
            "model": None, "status": "ok", "cache": "no_context", "attempts": 0, "failover": False,
            "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0,
            # A naive pipeline would still have called the model with the question.
            "uncached_cost_usd": baseline.cost(len(question) // 4 + 150, 20),
            "baseline_cost_usd": baseline.cost(len(question) // 4 + 150, 20),
            "latency_ms": total_ms, "error": None,
        })
        return RAGAnswer(
            answer=NO_CONTEXT_ANSWER, citations=[], sources=[], grounded=False, cache="no_context",
            retrieval_ms=round(retrieval_ms, 2), total_ms=round(total_ms, 2), request_id=request_id,
        )
