# Failover and Retries

NexusAI keeps answering when a provider has an outage, rate-limits the gateway or times out.

## Retryable errors

Timeouts, connection errors, HTTP 408, 409, 429 and all 5xx responses are treated as transient. Anthropic's HTTP 529 "overloaded" response is also retryable. HTTP 400 (bad request) is not retried and is returned to the caller immediately, because every provider would reject the same invalid request. HTTP 401, 403 and 404 are specific to one provider: a bad API key, missing permissions, or a model that provider does not serve. The gateway does not retry them, but moves straight to the next candidate, which is usually the other provider.

## Retry policy

Each candidate model gets up to NEXUS_MAX_RETRIES retries, 2 by default, so at most three attempts. Between attempts the gateway waits using exponential backoff with full jitter: a random delay between zero and NEXUS_RETRY_BASE_DELAY_S times two to the power of the attempt number, capped at 8 seconds. The default base delay is 0.25 seconds. Jitter spreads retries out so many clients do not hit a recovering provider at the same moment.

## Circuit breakers

Every provider has a circuit breaker with three states. In the closed state requests flow normally. After NEXUS_BREAKER_THRESHOLD consecutive failures (5 by default) the breaker opens, and the gateway skips that provider entirely instead of waiting on timeouts. After NEXUS_BREAKER_RESET_S seconds (30 by default) the breaker becomes half-open and lets one trial request through. If the trial succeeds the breaker closes again; if it fails the breaker reopens.

When a breaker opens in the middle of a retry loop, the gateway stops retrying that provider and moves straight to the next candidate.

## Rate limits

HTTP 429 means a quota was exhausted, not that the provider is unhealthy, so it never counts against the circuit breaker. Quotas are often per model, as on Gemini's free tier, so the gateway moves straight to the next candidate model instead of retrying. If every candidate fails, the gateway waits once and retries only the rate-limited models, which recovers short per-minute limits. If all of them are still rate limited, the caller receives HTTP 429.

## Moving between providers

When several providers are configured, each tier contains models from more than one of them, so a failure at one provider fails over to another provider in the same tier before the request escalates to a higher tier. With a single provider, the gateway can still fail over between that provider's models, but it cannot survive a full outage of that provider. A response that needed more than one attempt or skipped a candidate is marked failover: true in its response and in the request log.

Breaker state is kept in process memory. With several gateway replicas behind a load balancer, store breaker state in Redis so every replica sees the same provider health.
