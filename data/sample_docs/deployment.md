# Deployment

## Docker Compose

docker compose up starts three containers: the gateway on port 8000, PostgreSQL 16 with pgvector, and Redis 7. The database schema is created automatically on startup, including the vector extension and HNSW indexes.

## Environment variables

NEXUS_PROVIDER_MODE selects "live" for real APIs or "mock" for simulated providers. Live mode needs at least one of GEMINI_API_KEY, OPENAI_API_KEY or ANTHROPIC_API_KEY. Gemini has a free tier, so a Gemini key alone is enough to run everything live. DATABASE_URL and REDIS_URL point at PostgreSQL and Redis.

NEXUS_API_KEY protects the /v1 endpoints and NEXUS_ADMIN_API_KEY protects the admin API. Clients send the key in an X-API-Key header or as a bearer token. When a key is empty, that group of endpoints is open, which is only appropriate for local development.

## Embeddings

In live mode with a Gemini key, embeddings use gemini-embedding-001 at 768 dimensions, with separate task types for documents, queries and cache prompts. pgvector's HNSW index supports at most 2,000 dimensions, so Gemini's default of 3,072 is reduced. With only an OpenAI key, embeddings use text-embedding-3-small with 1536 dimensions. Otherwise the gateway uses an offline hashing embedder with 1024 dimensions. The vector columns are created with the configured dimension, so changing embedding models requires dropping the semantic_cache and chunks tables and re-ingesting documents. The gateway refuses to start if the dimensions do not match.

## Production server

The container runs gunicorn with the wsgi:app entry point. Run several workers only with PostgreSQL and Redis configured, because the in-memory stores are per process.
