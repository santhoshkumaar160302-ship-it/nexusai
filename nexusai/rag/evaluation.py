"""Offline evaluation for the RAG pipeline.

Eval set format (JSON Lines), one question per line:

    {"question": "...", "relevant_sources": ["docs/caching.md"], "answer_keywords": ["0.92"]}

`relevant_sources` lists documents that contain the answer. Use an empty list
for questions the corpus cannot answer; the pipeline should abstain on those.

Retrieval metrics (computed per question, then averaged):
  hit@k      at least one relevant document is in the top k
  recall@k   share of relevant documents found in the top k
  MRR        1 / rank of the first relevant document (0 if none)

Answer metrics (only with --answers, since they call the LLM):
  citation precision   share of cited sources that are relevant
  keyword accuracy     share of answers containing every expected keyword
  abstention accuracy  share of unanswerable questions answered "I don't know"
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .pipeline import RAGPipeline
from .retriever import Retriever


@dataclass
class EvalCase:
    question: str
    relevant_sources: list[str]
    answer_keywords: list[str] = field(default_factory=list)


def load_eval_set(path: str | Path) -> list[EvalCase]:
    cases = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
        cases.append(EvalCase(row["question"], row.get("relevant_sources", []), row.get("answer_keywords", [])))
    return cases


def retrieval_metrics(ranked_sources: list[str], relevant: set[str], k: int) -> dict:
    # A document can appear several times (several chunks); rank by first appearance.
    unique: list[str] = []
    for s in ranked_sources:
        if s not in unique:
            unique.append(s)
    top = unique[:k]
    found = relevant.intersection(top)
    first_rank = next((i for i, s in enumerate(unique, start=1) if s in relevant), None)
    return {
        "hit": 1.0 if found else 0.0,
        "recall": len(found) / len(relevant),
        "mrr": 1.0 / first_rank if first_rank else 0.0,
    }


def _answer_with_retry(pipeline, question, k, rate_limit_wait_s, max_rate_limit_retries, sleep):
    """Answer one question; on a 429 from every model, wait out the quota window and retry."""
    from ..gateway import GatewayError

    for attempt in range(max_rate_limit_retries + 1):
        try:
            return pipeline.answer(question, top_k=k), None
        except GatewayError as exc:
            detail = next((a.get("error", "") for a in reversed(exc.attempts) if a.get("error")), "")
            error = f"{exc} ({detail[:160]})" if detail else str(exc)
            if exc.status != 429 or attempt == max_rate_limit_retries:
                return None, error
            sleep(rate_limit_wait_s)
    return None, "unreachable"


def evaluate(cases: list[EvalCase], retriever: Retriever, k: int = 5,
             pipeline: RAGPipeline | None = None, delay_s: float = 0.0,
             rate_limit_wait_s: float = 60.0, max_rate_limit_retries: int = 2,
             progress=None, sleep=time.sleep) -> dict:
    """Score retrieval (and, with a pipeline, answers).

    A question whose answer fails (for example, quota exhausted on every model)
    is recorded in `answer_errors` and skipped, instead of aborting the run.
    `progress(i, n, question, status)` is called after each answered question.
    """
    answerable = [c for c in cases if c.relevant_sources]
    unanswerable = [c for c in cases if not c.relevant_sources]
    per_question = []
    totals = {"hit": 0.0, "recall": 0.0, "mrr": 0.0}

    for case in answerable:
        results = retriever.retrieve(case.question, k)
        ranked = [r.chunk.source for r in results]
        m = retrieval_metrics(ranked, set(case.relevant_sources), k)
        for key in totals:
            totals[key] += m[key]
        per_question.append({"question": case.question, **m, "retrieved": list(dict.fromkeys(ranked))})

    n = len(answerable) or 1
    report = {
        "questions": len(cases),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "k": k,
        f"hit@{k}": round(totals["hit"] / n, 4),
        f"recall@{k}": round(totals["recall"] / n, 4),
        "mrr": round(totals["mrr"] / n, 4),
    }

    if pipeline is not None:
        cited_total = cited_relevant = keyword_hits = keyword_cases = abstained = 0
        answered_unanswerable = 0
        errors: list[dict] = []
        models: dict[str, int] = {}
        for i, case in enumerate(cases):
            if delay_s and i:
                sleep(delay_s)  # stay under free-tier per-minute limits
            result, error = _answer_with_retry(pipeline, case.question, k, rate_limit_wait_s,
                                               max_rate_limit_retries, sleep)
            if progress:
                status = f"ERROR {error}" if error else (result.model or "no LLM call (nothing relevant)")
                progress(i + 1, len(cases), case.question, status)
            if error:
                errors.append({"question": case.question, "error": error})
                continue
            if result.model:
                models[result.model] = models.get(result.model, 0) + 1
            if case.relevant_sources:
                sources = [c["source"] for c in result.citations]
                cited_total += len(sources)
                cited_relevant += sum(1 for s in sources if s in case.relevant_sources)
                if case.answer_keywords:
                    keyword_cases += 1
                    text = result.answer.lower()
                    keyword_hits += all(kw.lower() in text for kw in case.answer_keywords)
            else:
                answered_unanswerable += 1
                abstained += not result.grounded
        report["citation_precision"] = round(cited_relevant / cited_total, 4) if cited_total else 0.0
        report["keyword_accuracy"] = round(keyword_hits / keyword_cases, 4) if keyword_cases else None
        report["abstention_accuracy"] = (round(abstained / answered_unanswerable, 4)
                                         if answered_unanswerable else None)
        report["answered"] = len(cases) - len(errors)
        report["answer_errors"] = errors
        report["answer_models"] = models

    report["per_question"] = per_question
    return report
