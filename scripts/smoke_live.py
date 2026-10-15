"""Live check of the Gemini setup: run this once with your own GEMINI_API_KEY.

    export GEMINI_API_KEY=...            # from https://aistudio.google.com/apikey
    python scripts/smoke_live.py

It makes roughly 15 API calls (well inside free-tier limits) and:
  1. calls every Gemini chat model in the catalog and reports latency and tokens,
  2. measures embedding similarities to suggest NEXUS_SEMANTIC_THRESHOLD,
  3. ingests the sample docs and compares retrieval scores for answerable and
     unanswerable questions to suggest RAG_MIN_SCORE,
  4. sends one request through the full gateway and one RAG question end to end.
Nothing is written to your databases: it uses the in-memory store.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]

PARAPHRASES = [
    ("What is a Bloom filter?", "Can you explain what a Bloom filter is?"),
    ("How do I reverse a list in Python?", "What's the way to reverse a Python list?"),
    ("What causes a TCP connection timeout?", "Why do TCP connections time out?"),
]
DIFFERENT_QUESTIONS = [
    ("What is a Bloom filter?", "What is a B-tree?"),
    ("Explain Kubernetes pods for a beginner.", "Explain Kubernetes pods for a senior engineer."),
    ("How do I reverse a list in Python?", "How do I reverse a list in Go?"),
]
ANSWERABLE = [
    "How many consecutive failures open a provider's circuit breaker?",
    "What chunk size does the RAG chunker use?",
    "Where do I send traces so they show up in Jaeger?",
]
UNANSWERABLE = [
    "What is the monthly subscription price for enterprise support?",
    "Who won the 2022 football world cup?",
    "How do I bake sourdough bread?",
]


def header(text: str) -> None:
    print(f"\n== {text} " + "=" * max(0, 60 - len(text)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--delay", type=float, default=4.0, help="seconds between chat calls (free-tier friendly)")
    p.add_argument("--skip-chat", action="store_true", help="skip the per-model chat checks")
    args = p.parse_args()

    if not os.getenv("GEMINI_API_KEY"):
        sys.exit("Set GEMINI_API_KEY first (create one at https://aistudio.google.com/apikey).")
    os.environ["NEXUS_PROVIDER_MODE"] = "live"
    for var in ("DATABASE_URL", "REDIS_URL"):
        os.environ.pop(var, None)

    from nexusai.app import build_services
    from nexusai.config import Settings
    from nexusai.pricing import CATALOG
    from nexusai.providers.base import ChatRequest, Message, ProviderError

    settings = Settings()
    services = build_services(settings)
    print(f"Providers: {', '.join(services.gateway.providers)} | embeddings: {settings.embedding_backend} "
          f"({settings.embedding_dim} dims) | baseline: {settings.baseline_model}")

    # 1. chat models ---------------------------------------------------------
    if not args.skip_chat:
        header("1. Chat models")
        gemini = services.gateway.providers.get("gemini")
        for model in [m for m in CATALOG.values() if m.provider == "gemini"]:
            req = ChatRequest(messages=[Message("user", "Reply with exactly the word OK.")], max_tokens=200)
            started = time.perf_counter()
            try:
                r = gemini.chat(req, model.name)
                ms = (time.perf_counter() - started) * 1000
                text = r.content.strip().replace("\n", " ")[:40] or "(empty: raise max_tokens?)"
                print(f"  ok    {model.name:<24} {ms:7.0f} ms  in={r.prompt_tokens:<4} out={r.completion_tokens:<5} {text!r}")
            except ProviderError as exc:
                hint = " (rate limited: wait a minute or check AI Studio quotas)" if exc.status == 429 else ""
                hint = " (model not available to this key)" if exc.status in (403, 404) else hint
                print(f"  FAIL  {model.name:<24} {str(exc)[:90]}{hint}")
            time.sleep(args.delay)

    # 2. semantic cache threshold -------------------------------------------
    header("2. Semantic cache threshold")
    emb = services.gateway.embedder

    def sims(pairs):
        flat = [t for pair in pairs for t in pair]
        vecs = emb.embed(flat, task="similarity")
        return [float(np.dot(vecs[i], vecs[i + 1])) for i in range(0, len(vecs), 2)]

    same, different = sims(PARAPHRASES), sims(DIFFERENT_QUESTIONS)
    for (a, b), s in zip(PARAPHRASES, same, strict=True):
        print(f"  same question   {s:.3f}  {a!r} / {b!r}")
    for (a, b), s in zip(DIFFERENT_QUESTIONS, different, strict=True):
        print(f"  different       {s:.3f}  {a!r} / {b!r}")
    lowest_same, highest_diff = min(same), max(different)
    if highest_diff < lowest_same:
        suggestion = round((lowest_same + highest_diff) / 2, 3)
        print(f"  -> clean separation. Suggested NEXUS_SEMANTIC_THRESHOLD={suggestion} "
              f"(current {settings.semantic_cache_threshold})")
    else:
        print(f"  -> overlap: some different questions ({highest_diff:.3f}) score above some rephrasings "
              f"({lowest_same:.3f}). Keep the threshold above {highest_diff:.3f} so wrong answers are never "
              f"served (fewer, safer hits).")

    # 3. RAG relevance floor -------------------------------------------------
    header("3. RAG relevance floor")
    results = services.ingestor.ingest_directory(ROOT / "data" / "sample_docs")
    print(f"  ingested {len(results)} documents, {sum(r.chunks for r in results)} chunks")

    def top_score(q):
        vec = emb.embed_one(q, task="query")
        hits = services.store.search_chunks(vec, 1)
        return (hits[0][1], hits[0][0].source) if hits else (0.0, "-")

    ans = [top_score(q) for q in ANSWERABLE]
    unans = [top_score(q) for q in UNANSWERABLE]
    for q, (score, src) in zip(ANSWERABLE, ans, strict=True):
        print(f"  answerable      {score:.3f}  {src:<20} {q}")
    for q, (score, src) in zip(UNANSWERABLE, unans, strict=True):
        print(f"  unanswerable    {score:.3f}  {src:<20} {q}")
    low_ans, high_unans = min(s for s, _ in ans), max(s for s, _ in unans)
    if high_unans < low_ans:
        print(f"  -> Suggested RAG_MIN_SCORE={round((low_ans + high_unans) / 2, 3)} (current {settings.rag_min_score})")
    else:
        print(f"  -> overlap ({high_unans:.3f} >= {low_ans:.3f}): keep RAG_MIN_SCORE below {low_ans:.3f} and let "
              "the model abstain on weak context.")

    # 4. end to end ----------------------------------------------------------
    header("4. End to end through the gateway")
    time.sleep(args.delay)
    r = services.gateway.chat(ChatRequest(messages=[Message("user", "In one sentence, what is a circuit breaker in software?")]))
    print(f"  chat: {r.model} ({r.rule}, {r.cache}) {r.latency_ms:.0f} ms, cost at list price ${r.cost_usd:.6f}")
    print(f"        {r.content.strip()[:200]}")
    time.sleep(args.delay)
    a = services.rag.answer(ANSWERABLE[0])
    print(f"  rag:  {a.model} grounded={a.grounded} citations={[c['source'] for c in a.citations]} {a.total_ms:.0f} ms")
    print(f"        {a.answer.strip()[:200]}")
    print("\nDone. Put any suggested values in a .env file in the project folder (or export them),\n"
          "for example:  RAG_MIN_SCORE=0.62  then run:\n"
          "  NEXUS_PROVIDER_MODE=live python scripts/eval_rag.py --answers --delay 7")


if __name__ == "__main__":
    main()
