# Observability

NexusAI emits OpenTelemetry traces and metrics, and keeps a request log in PostgreSQL.

## Traces

Each request produces a gateway.chat span with child spans for cache.lookup and every provider.call attempt. RAG requests add rag.answer and rag.retrieve spans. Spans carry attributes such as nexus.model, nexus.tier, nexus.rule, nexus.cache and nexus.attempt, so a slow request can be traced to the exact provider attempt that was slow.

## Metrics

The gateway records these metrics:

- nexus.requests: requests by task, cache result, model and status
- nexus.provider.attempts: upstream calls by provider and outcome
- nexus.tokens: tokens by model and direction (input or output)
- nexus.cost_usd: estimated spend by model
- nexus.request.latency: end-to-end latency histogram in milliseconds

## Exporting

Set OTEL_EXPORTER_OTLP_ENDPOINT to send traces and metrics to an OTLP collector such as Jaeger, Grafana Tempo or Honeycomb. Set NEXUS_OTEL_CONSOLE=1 to print them to the console during development. With neither set, telemetry is collected but not exported.

## Request log and stats

Every request is written to the request_log table with its model, cache result, attempts, token counts, cost and latency. GET /admin/api/stats aggregates the log into success rate, cache hit rate, API call reduction, cost savings, failovers and p50 and p95 latency.

Cost savings compare actual spend with a baseline: what every request would have cost on NEXUS_BASELINE_MODEL, claude-sonnet-4-5 by default, with no caching or routing. API call reduction is the share of requests that never reached a provider, because they were served from cache or answered without an LLM call.
