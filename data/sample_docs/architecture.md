# Gateway Architecture

NexusAI is a single HTTP service that sits between applications and LLM providers. Applications send one request format to NexusAI, and the gateway decides which provider and model serve it. The gateway supports Google Gemini, OpenAI and Anthropic, and uses whichever providers have an API key configured.

## Request lifecycle

Every chat request goes through the same five stages. First the exact cache is checked in Redis. Then the semantic cache is checked in PostgreSQL with pgvector. On a cache miss, the router picks a tier and builds an ordered list of candidate models. The failover loop then calls candidates in order until one succeeds. Finally, the response is cached, the request is logged to the request_log table, and metrics are emitted through OpenTelemetry.

## Components

The provider adapters translate the gateway's neutral request format into each vendor's API. The OpenAI adapter calls the Chat Completions endpoint. The Gemini adapter reuses it against Gemini's OpenAI-compatible endpoint, sending max_tokens instead of max_completion_tokens. The Anthropic adapter calls the Messages endpoint and moves the system prompt into Anthropic's top-level system field, because Anthropic does not accept system messages in the message list.

The router holds the routing rules. The circuit breakers track the health of each provider. The store persists rules, request logs, the semantic cache and RAG chunks in PostgreSQL.

## Storage choices

PostgreSQL with the pgvector extension stores every vector in the system: semantic cache entries and RAG document chunks. Both tables use an HNSW index with cosine distance. Redis holds the exact-match cache because lookups there take well under a millisecond and keys expire automatically with a TTL.

For local development, both stores have in-memory implementations. Leaving DATABASE_URL and REDIS_URL empty selects them automatically, so the gateway runs with no external services.
