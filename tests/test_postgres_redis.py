"""Integration tests against real PostgreSQL + pgvector and Redis.

Skipped unless both URLs are set, for example:
  NEXUS_TEST_DATABASE_URL=postgresql://postgres@localhost:5432/nexus_test \
  NEXUS_TEST_REDIS_URL=redis://localhost:6379/15 pytest tests/test_postgres_redis.py
"""
from __future__ import annotations

import os

import pytest

from nexusai.app import build_services, create_app
from nexusai.embeddings import HashEmbedder
from nexusai.providers.base import ChatRequest, Message
from nexusai.storage import MemoryStore

DB_URL = os.getenv("NEXUS_TEST_DATABASE_URL")
REDIS_URL = os.getenv("NEXUS_TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(not (DB_URL and REDIS_URL), reason="set NEXUS_TEST_DATABASE_URL and NEXUS_TEST_REDIS_URL")


@pytest.fixture
def pg_services(settings, providers):
    import psycopg
    import redis

    with psycopg.connect(DB_URL, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks, documents, semantic_cache, request_log, routing_rules CASCADE")
    redis.Redis.from_url(REDIS_URL).flushdb()

    from nexusai.storage.postgres import PostgresStore

    settings.database_url, settings.redis_url = DB_URL, REDIS_URL
    store = PostgresStore(DB_URL, settings.embedding_dim)
    services = build_services(settings, providers=providers, store=store,
                              embedder=HashEmbedder(settings.embedding_dim), sleep=lambda s: None)
    yield services
    store.close()


def ask(services, text):
    return services.gateway.chat(ChatRequest(messages=[Message("user", text)]))


def test_exact_and_semantic_cache_with_redis_and_pgvector(pg_services, providers):
    assert ask(pg_services, "What is consistent hashing in distributed systems?").cache == "miss"
    assert ask(pg_services, "What is consistent hashing in distributed systems?").cache == "exact"
    assert ask(pg_services, "what is consistent hashing in distributed systems").cache == "semantic"
    assert providers["openai"].calls == 1
    assert type(pg_services.kv).__name__ == "Redis"


def test_stats_match_memory_store_on_same_log(pg_services, providers):
    for text in ["a question about Redis", "a question about Redis", "debug this architecture"]:
        ask(pg_services, text)
    providers["openai"].down = providers["anthropic"].down = True
    with pytest.raises(Exception):
        ask(pg_services, "will fail")

    memory = MemoryStore()
    for record in reversed(pg_services.store.recent_requests(100)):
        memory.log_request({k: v for k, v in record.items() if k != "created_at"})
    pg, mem = pg_services.store.stats(), memory.stats()
    for key in ("requests", "succeeded", "failed", "cache_hits_exact", "upstream_requests", "upstream_attempts",
                "failovers", "requests_by_model"):
        assert pg[key] == mem[key], key
    assert pg["cost_usd"] == pytest.approx(mem["cost_usd"])
    assert pg["latency_p95_ms"] == pytest.approx(mem["latency_p95_ms"], rel=1e-3)


def test_rag_ingest_search_delete_in_pgvector(pg_services, docs_dir):
    results = pg_services.ingestor.ingest_directory(docs_dir)
    assert all(r.status == "created" for r in results)
    assert all(r.status == "unchanged" for r in pg_services.ingestor.ingest_directory(docs_dir))

    top = pg_services.retriever.retrieve("How many consecutive failures open a provider's circuit breaker?", k=3)
    assert top[0].chunk.source == "failover.md"

    answer = pg_services.rag.answer("What chunk size and overlap does the chunker use?")
    assert answer.grounded and answer.citations[0]["source"] == "rag.md"

    doc = pg_services.store.list_documents()[0]
    assert pg_services.store.delete_document(doc.id)
    assert pg_services.store.get_document(doc.id) is None


def test_rules_persist_across_restart(pg_services):
    client = create_app(pg_services).test_client()
    rules = client.get("/admin/api/rules").get_json()["rules"]
    rules[-1]["tier"] = "standard"
    assert client.put("/admin/api/rules", json={"rules": rules}).status_code == 200
    assert pg_services.store.get_rules()[-1]["tier"] == "standard"


def test_dimension_mismatch_is_detected(pg_services):
    from nexusai.storage.postgres import PostgresStore

    with pytest.raises(RuntimeError, match="dimension"):
        PostgresStore(DB_URL, pg_services.settings.embedding_dim + 1)
