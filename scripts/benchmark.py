"""Load benchmark for the gateway.

Runs a workload of chat requests and reports success rate, failovers, cache
hits, API call reduction and cost savings versus the baseline model.

Two modes:
  in-process (default)   builds the gateway with simulated providers, injects
                         random errors plus a full provider outage, and runs
                         with no network or API keys.
  --url http://host:8000 sends the workload to a running gateway over HTTP
                         (use this with NEXUS_PROVIDER_MODE=live for real numbers).

Workload: by default a synthetic mix of new questions, exact repeats and
light rephrasings. The cache numbers depend directly on that mix, so state it
when you report results, or replay real traffic with --prompts (a JSONL file
with one {"prompt": "..."} per line).

Examples:
  python scripts/benchmark.py --requests 100000
  python scripts/benchmark.py --requests 2000 --repeat-rate 0.2 --paraphrase-rate 0.1
  python scripts/benchmark.py --url http://localhost:8000 --requests 500 --api-key $NEXUS_API_KEY
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexusai.config import Settings  # noqa: E402

SUBJECTS = [
    "Python generators", "database indexing", "HTTP caching", "OAuth 2.0", "Kubernetes pods", "rate limiting",
    "message queues", "binary search", "TCP handshakes", "CSS grid", "React hooks", "SQL joins", "Docker volumes",
    "garbage collection", "JSON Web Tokens", "load balancers", "event sourcing", "unit testing", "CORS",
    "consistent hashing", "Bloom filters", "B-trees", "REST pagination", "WebSockets", "the CAP theorem",
    "feature flags", "blue-green deployments", "vector databases", "gradient descent", "transformers",
]
SIMPLE_TEMPLATES = [
    "What is {s} {c}?", "Give me a one-paragraph summary of {s} {c}.", "What are common mistakes with {s} {c}?",
    "When should I use {s} {c}?", "List three facts about {s} {c}.", "Explain {s} {c}.",
]
COMPLEX_TEMPLATES = [
    "Explain step by step how {s} works {c}.", "Debug this: my {s} setup fails under load {c}. What do I check?",
    "What is the trade-off between {s} and its main alternative {c}?", "Design an architecture that uses {s} {c}.",
]
CONTEXTS = [
    "in Python", "in Go", "in Java", "in Rust", "in TypeScript", "on AWS", "on Google Cloud", "on Azure",
    "in a microservices system", "in a monolith", "for a mobile app", "for a startup", "at a bank",
    "for a real-time chat app", "for an e-commerce site", "in a data pipeline", "in a multiplayer game",
    "for IoT devices", "with limited memory", "in a serverless app", "in healthcare software", "on Kubernetes",
    "for a search engine", "in an analytics dashboard", "for a payments API", "in an embedded system",
    "for a social network", "in a CI/CD pipeline", "for a video streaming service", "in a recommendation system",
    "for a ride-sharing app", "in a logging platform", "for an online exam system", "in a CRM",
    "for a booking system", "in a machine learning platform", "for a news website", "in a trading system",
    "for a smart home hub", "in a government portal",
]
AUDIENCES = ["", " for a beginner", " for a senior engineer", " for a job interview", " for a product manager",
             " in under 100 words", " with a code example", " as a bulleted list", " for a student",
             " with real-world examples", " for a team lead", " with diagrams described in text"]
PARAPHRASES = [
    lambda q: q.lower(),
    lambda q: q.rstrip("?.") + "?",
    lambda q: q.rstrip("?.") + ", please.",
    lambda q: q.replace("What is", "what's").replace("What are", "what're"),
    lambda q: q + " Thanks!",
]


def build_workload(n: int, repeat_rate: float, paraphrase_rate: float, complex_rate: float,
                   long_rate: float, seed: int) -> list[tuple[str, str]]:
    """Return (prompt, kind) pairs; kind is "new", "repeat" or "paraphrase".

    Every "new" prompt is a distinct (subject, question, context, audience)
    combination, so the kind is ground truth: a semantic cache hit on a "new"
    prompt means a different question received a cached answer.
    """
    rng = random.Random(seed)
    simple = [(s, t, c, a) for s in SUBJECTS for t in SIMPLE_TEMPLATES for c in CONTEXTS for a in AUDIENCES]
    complex_ = [(s, t, c, a) for s in SUBJECTS for t in COMPLEX_TEMPLATES for c in CONTEXTS for a in AUDIENCES]
    rng.shuffle(simple)
    rng.shuffle(complex_)
    history: list[str] = []
    workload: list[tuple[str, str]] = []
    for _ in range(n):
        roll = rng.random()
        pool = complex_ if rng.random() < complex_rate else simple
        if history and (roll < repeat_rate or not pool):
            # repeat (also used if this pool of distinct questions is exhausted, which keeps the mix honest)
            workload.append((rng.choice(history), "repeat"))
            continue
        if history and roll < repeat_rate + paraphrase_rate:
            workload.append((rng.choice(PARAPHRASES)(rng.choice(history)), "paraphrase"))
            continue
        subject, template, context, audience = pool.pop()
        prompt = template.format(s=subject, c=context).rstrip("?.") + audience + template.strip()[-1]
        if rng.random() < long_rate:
            prompt += "\n\nContext:\n" + ("Log line: request handled, status ok, latency nominal. " * 160)
        history.append(prompt)
        workload.append((prompt, "new"))
    return workload


def load_prompts(path: str) -> list[tuple[str, str]]:
    prompts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            prompts.append((json.loads(line)["prompt"], "unknown"))
    return prompts


def run_in_process(workload: list[tuple[str, str]], args, threshold: float | None = None) -> dict:
    from nexusai.app import build_services
    from nexusai.gateway import GatewayError
    from nexusai.providers.base import ChatRequest, Message

    settings = Settings()
    settings.provider_mode = "mock"
    settings.mock_failure_rate = args.failure_rate
    settings.database_url = args.database_url or ""
    settings.redis_url = args.redis_url or ""
    if threshold is not None:
        settings.semantic_cache_threshold = threshold
    # Simulated time: the run takes seconds of wall time, but circuit breakers must
    # see realistic elapsed time (requests arrive at --rps), or a breaker opened
    # during the outage would never reach its reset timeout.
    clock = {"t": 0.0}
    services = build_services(settings, sleep=lambda s: clock.__setitem__("t", clock["t"] + s))
    for breaker in services.gateway.breakers.values():
        breaker.clock = lambda: clock["t"]
    services.store.reset_stats()
    names = list(services.gateway.providers)
    outage_name = args.outage_provider if args.outage_provider is not None else (names[0] if len(names) > 1 else "")
    if outage_name and outage_name not in names:
        raise SystemExit(f"--outage-provider {outage_name!r} is not one of the simulated providers {names}")
    outage_provider = services.gateway.providers.get(outage_name) if outage_name else None

    outage_start = int(len(workload) * args.outage_start)
    outage_end = outage_start + int(len(workload) * args.outage_length)
    false_semantic_hits = 0
    kinds = {k: sum(1 for _, kk in workload if kk == k) for k in ("new", "repeat", "paraphrase")}
    started = time.perf_counter()
    for i, (prompt, kind) in enumerate(workload):
        clock["t"] += 1.0 / args.rps
        if outage_provider is not None:  # simulated full outage of one provider
            outage_provider.down = outage_start <= i < outage_end
        try:
            response = services.gateway.chat(ChatRequest(messages=[Message("user", prompt)]))
        except GatewayError:
            continue  # recorded in the request log as a failure
        if kind == "new" and response.cache == "semantic":
            false_semantic_hits += 1  # a brand-new question got someone else's cached answer
        if args.progress and (i + 1) % args.progress == 0:
            print(f"  {i + 1:,}/{len(workload):,} requests", file=sys.stderr)
    elapsed = time.perf_counter() - started
    stats = services.store.stats()
    stats["wall_time_s"] = round(elapsed, 2)
    stats["simulated_providers"] = names
    stats["simulated_outage"] = (
        f"{outage_name} fully down for requests {outage_start:,}-{outage_end:,} "
        f"({(outage_end - outage_start) / args.rps:,.0f} s at {args.rps:g} req/s)"
        if outage_provider is not None else "none (a single provider cannot survive its own outage)")
    stats["injected_error_rate"] = args.failure_rate
    stats["false_semantic_hits"] = false_semantic_hits
    stats["actual_mix"] = kinds
    stats["baseline_model"] = settings.baseline_model
    stats["semantic_threshold"] = settings.semantic_cache_threshold
    return stats


def run_http(workload: list[str], args) -> dict:
    import httpx

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    with httpx.Client(base_url=args.url, timeout=60, headers=headers) as client:
        if args.admin_key:
            client.post("/admin/api/stats/reset", headers={"X-API-Key": args.admin_key})
        started = time.perf_counter()
        client_errors = 0
        for i, prompt in enumerate(workload):
            try:
                resp = client.post("/v1/chat", json={"messages": [{"role": "user", "content": prompt[0]}]})
                client_errors += resp.status_code >= 500
            except httpx.HTTPError:
                client_errors += 1
            if args.progress and (i + 1) % args.progress == 0:
                print(f"  {i + 1:,}/{len(workload):,} requests", file=sys.stderr)
        elapsed = time.perf_counter() - started
        stats = client.get("/admin/api/stats", headers={"X-API-Key": args.admin_key or args.api_key}).json()
    stats["wall_time_s"] = round(elapsed, 2)
    stats["client_observed_errors"] = client_errors
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--requests", type=int, default=10_000)
    p.add_argument("--prompts", help="JSONL file of real prompts to replay instead of the synthetic mix")
    p.add_argument("--repeat-rate", type=float, default=0.18, help="share of requests that repeat an earlier prompt")
    p.add_argument("--paraphrase-rate", type=float, default=0.12, help="share that lightly rephrase an earlier prompt")
    p.add_argument("--complex-rate", type=float, default=0.2, help="share of new prompts that are reasoning-heavy")
    p.add_argument("--long-rate", type=float, default=0.05, help="share of new prompts with long context")
    p.add_argument("--failure-rate", type=float, default=0.03, help="random provider error rate (in-process only)")
    p.add_argument("--outage-provider", default=None,
                   help="provider to take fully down (default: the first simulated provider when there are "
                        "several; none when there is one). Simulated providers come from MOCK_PROVIDERS.")
    p.add_argument("--outage-start", type=float, default=0.4, help="when the simulated OpenAI outage starts (fraction)")
    p.add_argument("--outage-length", type=float, default=0.05, help="how long the outage lasts (fraction)")
    p.add_argument("--rps", type=float, default=20, help="simulated arrival rate (in-process only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--url", help="benchmark a running gateway over HTTP instead of in-process")
    p.add_argument("--api-key", default="")
    p.add_argument("--admin-key", default="")
    p.add_argument("--database-url", default="", help="in-process: use PostgreSQL instead of memory")
    p.add_argument("--redis-url", default="", help="in-process: use Redis instead of memory")
    p.add_argument("--progress", type=int, default=0, help="print progress every N requests")
    p.add_argument("--thresholds", help="comma-separated semantic thresholds to compare, e.g. 0.92,0.95,0.97")
    p.add_argument("--json", action="store_true", help="print raw JSON only")
    p.add_argument("--min-success-rate", type=float, default=None,
                   help="quality gate: exit 1 if the success rate is below this (e.g. 0.999 in CI)")
    p.add_argument("--max-wrong-hits", type=int, default=None,
                   help="quality gate: exit 1 if more semantic cache hits than this answered the wrong question")
    args = p.parse_args()

    if args.prompts:
        workload = load_prompts(args.prompts)[: args.requests]
        mix = f"replayed {len(workload):,} prompts from {args.prompts}"
    else:
        workload = build_workload(args.requests, args.repeat_rate, args.paraphrase_rate, args.complex_rate,
                                  args.long_rate, args.seed)
        mix = (f"synthetic: {args.repeat_rate:.0%} exact repeats, {args.paraphrase_rate:.0%} rephrasings, "
               f"{args.complex_rate:.0%} of new prompts reasoning-heavy, {args.long_rate:.0%} long-context")

    if args.thresholds:
        if args.url:
            p.error("--thresholds runs in-process only")
        print(f"Workload: {mix}\n")
        print(f"{'threshold':>9}  {'semantic hits':>13}  {'wrong hits':>10}  {'cache hit rate':>14}  {'saved by caching':>16}")
        for t in (float(x) for x in args.thresholds.split(",")):
            r = run_in_process(workload, args, threshold=t)
            print(f"{t:>9.2f}  {r['cache_hits_semantic']:>13,}  {r['false_semantic_hits']:>10,}  "
                  f"{r['cache_hit_rate']:>14.2%}  {r['savings_from_caching']:>16.2%}")
        return

    stats = run_http(workload, args) if args.url else run_in_process(workload, args)
    stats["workload"] = mix

    if args.json:
        print(json.dumps(stats, indent=2))
        sys.exit(gate(stats, args))
    sim = f"in-process, simulated providers: {', '.join(stats.get('simulated_providers', []))}"
    print(f"\nNexusAI benchmark ({'HTTP ' + args.url if args.url else sim})")
    print(f"Workload:            {mix}")
    if "actual_mix" in stats:
        m = stats["actual_mix"]
        print(f"Actual mix:          {m['new']:,} new, {m['repeat']:,} exact repeats, {m['paraphrase']:,} rephrasings")
    if "simulated_outage" in stats:
        print(f"Fault injection:     {stats['injected_error_rate']:.0%} random errors, {stats['simulated_outage']}")
    print(f"Requests:            {stats['requests']:,}")
    print(f"Success rate:        {stats['success_rate']:.2%}  ({stats['failed']:,} failed, {stats['failovers']:,} failovers)")
    print(f"Cache hit rate:      {stats['cache_hit_rate']:.2%}  ({stats['cache_hits_exact']:,} exact, "
          f"{stats['cache_hits_semantic']:,} semantic)")
    if "false_semantic_hits" in stats:
        print(f"Wrong semantic hits: {stats['false_semantic_hits']:,}  (new questions answered from another question's "
              f"cache, threshold {stats['semantic_threshold']})")
    print(f"API call reduction:  {stats['api_call_reduction']:.2%}  ({stats['upstream_requests']:,} requests reached a provider)")
    print(f"Cost:                ${stats['cost_usd']:.4f} spent, ${stats['uncached_cost_usd']:.4f} without cache, "
          f"${stats['baseline_cost_usd']:.4f} all on {stats.get('baseline_model', Settings().baseline_model)}")
    print(f"  saved by caching:  {stats['savings_from_caching']:.2%}  (vs the same routing with no cache)")
    print(f"  saved by routing:  {stats['savings_from_routing']:.2%}  (vs every request on the baseline model)")
    print(f"  total savings:     {stats['cost_savings']:.2%}")
    print(f"Latency:             p50 {stats['latency_p50_ms']} ms, p95 {stats['latency_p95_ms']} ms")
    print(f"Requests by model:   {stats['requests_by_model']}")
    print(f"Wall time:           {stats['wall_time_s']} s")
    sys.exit(gate(stats, args))


def gate(stats: dict, args) -> int:
    """Return a non-zero exit code when a quality gate fails (used by CI)."""
    failures = []
    if args.min_success_rate is not None and stats["success_rate"] < args.min_success_rate:
        failures.append(f"success rate {stats['success_rate']:.2%} < {args.min_success_rate:.2%}")
    wrong = stats.get("false_semantic_hits")
    if args.max_wrong_hits is not None and wrong is not None and wrong > args.max_wrong_hits:
        failures.append(f"{wrong} wrong semantic hits > {args.max_wrong_hits}")
    for f in failures:
        print(f"QUALITY GATE FAILED: {f}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    main()
