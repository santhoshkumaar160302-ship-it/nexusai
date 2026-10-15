"""Evaluate RAG retrieval (and optionally answers) on a labeled question set.

Examples:
  # retrieval only: hit@k, recall@k, MRR (cheap: embeddings only, no LLM calls)
  python scripts/eval_rag.py --docs data/sample_docs --questions data/eval_questions.jsonl

  # also generate answers: citation precision, keyword accuracy, abstention accuracy
  python scripts/eval_rag.py --docs data/sample_docs --questions data/eval_questions.jsonl --answers

Uses the environment's settings: with NEXUS_PROVIDER_MODE=live and an OpenAI
key it embeds with text-embedding-3-small and answers with real models;
otherwise it uses the offline hashing embedder and simulated providers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexusai.app import build_services  # noqa: E402
from nexusai.config import Settings  # noqa: E402
from nexusai.rag import evaluate, load_eval_set  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--docs", default="data/sample_docs", help="directory of .md/.txt documents to ingest")
    p.add_argument("--questions", default="data/eval_questions.jsonl")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--answers", action="store_true", help="also generate and score answers (calls the LLM)")
    p.add_argument("--delay", type=float, default=0.0,
                   help="seconds to wait between answers with --answers (e.g. 4 for free-tier rate limits)")
    p.add_argument("--rate-limit-wait", type=float, default=60.0,
                   help="seconds to wait before retrying a question that every model rate-limited")
    p.add_argument("--verbose", action="store_true", help="print per-question retrieval results")
    p.add_argument("--json", action="store_true")
    p.add_argument("--min-hit", type=float, default=None,
                   help="quality gate: exit 1 if hit@k is below this (e.g. 0.9 in CI)")
    p.add_argument("--min-mrr", type=float, default=None, help="quality gate: exit 1 if MRR is below this")
    args = p.parse_args()

    services = build_services(Settings())
    s = services.settings
    print(f"Mode: {s.provider_mode} | providers: {', '.join(services.gateway.providers)} | "
          f"embeddings: {s.embedding_backend}", file=sys.stderr, flush=True)
    if s.provider_mode != "live":
        print("  (simulated: set NEXUS_PROVIDER_MODE=live and GEMINI_API_KEY for real results)",
              file=sys.stderr, flush=True)
    print(f"Embedding documents from {args.docs} ...", file=sys.stderr, flush=True)
    ingested = services.ingestor.ingest_directory(args.docs)
    chunks = sum(r.chunks for r in ingested)
    cases = load_eval_set(args.questions)
    print(f"Scoring retrieval for {len(cases)} questions ...", file=sys.stderr, flush=True)

    def progress(i, n, question, status):
        print(f"  [{i:>2}/{n}] {status[:60]:<60} {question[:60]}", file=sys.stderr, flush=True)

    if args.answers:
        print(f"Generating answers ({args.delay:g}s between questions) ...", file=sys.stderr, flush=True)
    report = evaluate(cases, services.retriever, k=args.k, pipeline=services.rag if args.answers else None,
                      delay_s=args.delay, rate_limit_wait_s=args.rate_limit_wait, progress=progress)
    report["documents"] = len(ingested)
    report["chunks"] = chunks
    report["embeddings"] = services.settings.embedding_backend

    if args.json:
        print(json.dumps(report, indent=2))
        sys.exit(gate(report, args))

    k = args.k
    print(f"\nRAG evaluation: {report['documents']} documents, {chunks} chunks, "
          f"{report['embeddings']} embeddings")
    print(f"Questions:  {report['questions']} ({report['answerable']} answerable, "
          f"{report['unanswerable']} unanswerable)")
    print(f"hit@{k}:     {report[f'hit@{k}']:.2%}")
    print(f"recall@{k}:  {report[f'recall@{k}']:.2%}")
    print(f"MRR:        {report['mrr']:.3f}")
    if args.answers:
        print(f"Answered:             {report['answered']}/{report['questions']}"
              f"  (models: {report['answer_models']})")
        print(f"Citation precision:   {report['citation_precision']:.2%}")
        if report["keyword_accuracy"] is not None:
            print(f"Keyword accuracy:     {report['keyword_accuracy']:.2%}")
        if report["abstention_accuracy"] is not None:
            print(f"Abstention accuracy:  {report['abstention_accuracy']:.2%}")
    if args.answers and report["answer_errors"]:
        print(f"\nQuestions that could not be answered ({len(report['answer_errors'])}):")
        for e in report["answer_errors"]:
            print(f"  - {e['question']}\n      {e['error'][:200]}")
    misses = [q for q in report["per_question"] if q["recall"] < 1]
    if misses:
        print(f"\nQuestions with incomplete recall ({len(misses)}):")
        for q in misses:
            print(f"  - {q['question']}  -> retrieved {q['retrieved'][:3]}")
    if args.verbose:
        for q in report["per_question"]:
            print(f"  {q['mrr']:.2f}  {q['question']}  {q['retrieved'][:3]}")
    sys.exit(gate(report, args))


def gate(report: dict, args) -> int:
    """Return a non-zero exit code when a quality gate fails (used by CI)."""
    failures = []
    hit = report[f"hit@{args.k}"]
    if args.min_hit is not None and hit < args.min_hit:
        failures.append(f"hit@{args.k} {hit:.2%} < {args.min_hit:.2%}")
    if args.min_mrr is not None and report["mrr"] < args.min_mrr:
        failures.append(f"MRR {report['mrr']:.3f} < {args.min_mrr:.3f}")
    for f in failures:
        print(f"QUALITY GATE FAILED: {f}", file=sys.stderr)
    if (args.min_hit is not None or args.min_mrr is not None) and not failures:
        print("Quality gates passed.", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    main()
