"""RAG question answering: retrieve, ground, answer with citations.

The answer is generated through the gateway (task="rag"), so it gets the same
routing, failover, caching and cost tracking as any other request. If no
chunk clears the relevance bar, the pipeline answers "not found" without
calling an LLM at all, which saves cost and avoids ungrounded answers.

Follow-up questions: callers can pass the earlier turns as `history`. A follow-up
like "what about the half-open state?" is useless as a search query on its own,
so it's first rewritten into a standalone question (one cheap LLM call, routed to
the economy tier) and that is what gets searched. "Continue" is handled without
an LLM call: it searches the previous question again, which returns the same
sources, and asks the model to pick up where its last answer stopped.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field

from ..gateway import Gateway, GatewayError
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

MAX_HISTORY_MESSAGES = 6  # the last three question/answer pairs
MAX_HISTORY_CHARS = 4000  # per message, so one huge answer can't crowd out everything else
CONTINUE_RE = re.compile(
    r"^\s*(please\s+)?(continue|go on|keep going|carry on|more|the rest|finish( it| that| the answer)?)"
    r"(\s+please)?\s*[.!?]*\s*$", re.I)

REWRITE_PROMPT = """Rewrite the user's latest message as one standalone question about the documents, \
using the conversation for context (resolve words like "it", "that" or "the second one").
Reply with the question only: no quotes, no explanation. If it is already standalone, repeat it unchanged."""


def clean_history(history) -> list[Message]:
    """Validate caller-supplied turns: user/assistant only, text only, most recent kept."""
    if not history:
        return []
    if not isinstance(history, list):
        raise ValueError("'history' must be a list of {role, content} messages")
    out = []
    for i, m in enumerate(history):
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant") or not isinstance(m.get("content"), str):
            raise ValueError(f"history[{i}] needs a role (user/assistant) and string content")
        if m["content"].strip():
            out.append(Message(m["role"], m["content"][:MAX_HISTORY_CHARS]))
    return out[-MAX_HISTORY_MESSAGES:]


def strip_citations(text: str) -> str:
    """Old answers cite *their* sources; drop the numbers so they don't clash with the new ones."""
    return _CITATION_RE.sub("", text)


def build_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, item in enumerate(chunks, start=1):
        c = item.chunk
        label = context_label(c.title, c.heading)
        blocks.append(f"[{i}] {label} ({c.source})\n{c.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: list[RetrievedChunk], history: list[Message] | None = None,
                   meaning: str | None = None, continuing: bool = False) -> list[Message]:
    """System prompt, then any earlier turns, then the sources and the question."""
    user = f"Sources:\n\n{build_context(chunks)}\n\nQuestion: {question}"
    if continuing:
        user += ("\n\nYour previous answer stopped before it was finished. Continue exactly where it "
                 "stopped, without repeating what you already wrote. Keep citing the sources.")
    elif meaning and meaning != question:
        user += f"\n(In this conversation, that means: {meaning})"
    return [Message("system", SYSTEM_PROMPT), *(history or []), Message("user", user)]


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
    # How the question was understood: "new", "follow_up" (rewritten) or "continue".
    mode: str = "new"
    search_query: str | None = None  # what was actually searched for
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class RAGPipeline:
    def __init__(self, gateway: Gateway, retriever: Retriever, store: Store, default_top_k: int = 5):
        self.gateway = gateway
        self.retriever = retriever
        self.store = store
        self.default_top_k = default_top_k

    def answer(self, question: str, top_k: int | None = None, tier: str | None = None,
               history: list | None = None) -> RAGAnswer:
        question = question.strip()
        if not question:
            raise ValueError("question is required")
        turns = clean_history(history)
        started = time.perf_counter()
        with tracer.start_as_current_span("rag.answer") as span:
            mode, search_query, extra_cost = "new", question, 0.0
            previous_question = next((m.content for m in reversed(turns) if m.role == "user"), None)
            if previous_question and CONTINUE_RE.match(question):
                mode, search_query = "continue", previous_question  # same search, same sources
            elif turns:
                search_query, extra_cost = self._standalone(question, turns, previous_question)
                mode = "follow_up"
            span.set_attribute("nexus.rag.mode", mode)

            chunks = self.retriever.retrieve(search_query, top_k or self.default_top_k)
            retrieval_ms = (time.perf_counter() - started) * 1000
            sources = [item.public() for item in chunks]
            span.set_attribute("nexus.rag.sources", len(chunks))

            if not chunks:
                result = self._no_context(question, retrieval_ms, started)
                result.mode, result.search_query = mode, search_query
                return result

            if mode == "continue":
                messages = build_messages(previous_question, chunks, turns, continuing=True)
            else:
                context = [Message(m.role, strip_citations(m.content)) for m in turns]
                messages = build_messages(question, chunks, context, meaning=search_query)
            request = ChatRequest(messages=messages, task="rag", tier=tier, max_tokens=1500)
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
                cost_usd=response.cost_usd + extra_cost,
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
                mode=mode,
                search_query=search_query,
            )

    def _standalone(self, question: str, turns: list[Message], previous_question: str | None) -> tuple[str, float]:
        """Turn a follow-up into a question that makes sense on its own. Returns (question, cost)."""
        transcript = "\n".join(f"{m.role.title()}: {strip_citations(m.content)[:600]}" for m in turns)
        request = ChatRequest(
            messages=[Message("system", REWRITE_PROMPT),
                      Message("user", f"Conversation:\n{transcript}\n\nLatest message: {question}")],
            # No caching: the transcript dominates this prompt, so two different follow-ups in similar
            # conversations could look alike to the semantic cache and get each other's rewrite.
            task="rewrite", temperature=0.0, max_tokens=200, use_cache=False)
        try:
            response = self.gateway.chat(request)
        except GatewayError:
            # No model available: searching with the previous question as context still beats nothing.
            return (f"{previous_question} {question}" if previous_question else question), 0.0
        rewritten = response.content.strip().splitlines()[0].strip().strip('"').strip() if response.content else ""
        if not rewritten or len(rewritten) > 500:
            return question, response.cost_usd
        return rewritten, response.cost_usd

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
