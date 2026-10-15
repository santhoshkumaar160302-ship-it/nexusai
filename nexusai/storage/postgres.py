"""PostgreSQL + pgvector store.

Tables:
  routing_rules   one row holding the active rule set (JSONB)
  request_log     one row per gateway request, for stats and cost reporting
  semantic_cache  prompt embeddings + cached responses, HNSW cosine index
  documents       ingested RAG documents
  chunks          document chunks + embeddings, HNSW cosine index
"""
from __future__ import annotations

import json

import numpy as np
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .base import Chunk, Document, Store, cost_breakdown

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS routing_rules (
    id          integer PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    rules       jsonb NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS request_log (
    id                 text PRIMARY KEY,
    created_at         timestamptz NOT NULL DEFAULT now(),
    task               text NOT NULL,
    rule               text,
    tier               text,
    provider           text,
    model              text,
    status             text NOT NULL,
    cache              text NOT NULL,
    attempts           integer NOT NULL DEFAULT 0,
    failover           boolean NOT NULL DEFAULT false,
    prompt_tokens      integer NOT NULL DEFAULT 0,
    completion_tokens  integer NOT NULL DEFAULT 0,
    cost_usd           double precision NOT NULL DEFAULT 0,
    uncached_cost_usd  double precision NOT NULL DEFAULT 0,
    baseline_cost_usd  double precision NOT NULL DEFAULT 0,
    latency_ms         double precision NOT NULL DEFAULT 0,
    error              text
);
ALTER TABLE request_log ADD COLUMN IF NOT EXISTS uncached_cost_usd double precision NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS request_log_created_at_idx ON request_log (created_at DESC);

CREATE TABLE IF NOT EXISTS semantic_cache (
    id          bigserial PRIMARY KEY,
    namespace   text NOT NULL,
    prompt      text NOT NULL,
    embedding   vector({dim}) NOT NULL,
    response    jsonb NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS semantic_cache_embedding_idx ON semantic_cache USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS semantic_cache_namespace_idx ON semantic_cache (namespace, created_at DESC);

CREATE TABLE IF NOT EXISTS documents (
    id            text PRIMARY KEY,
    title         text NOT NULL,
    source        text NOT NULL,
    content_hash  text NOT NULL,
    chunk_count   integer NOT NULL DEFAULT 0,
    metadata      jsonb NOT NULL DEFAULT '{{}}',
    updated_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           text PRIMARY KEY,
    document_id  text NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    ordinal      integer NOT NULL,
    heading      text NOT NULL DEFAULT '',
    text         text NOT NULL,
    embedding    vector({dim}) NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id, ordinal);
"""

MIGRATION_LOCK_ID = 7_316_402  # arbitrary, unique to this app's schema migration

_LOG_COLUMNS = (
    "id", "task", "rule", "tier", "provider", "model", "status", "cache", "attempts", "failover",
    "prompt_tokens", "completion_tokens", "cost_usd", "uncached_cost_usd", "baseline_cost_usd", "latency_ms", "error",
)


class PostgresStore(Store):
    def __init__(self, database_url: str, embedding_dim: int, pool_size: int = 10):
        self.dim = embedding_dim
        # Migrate before opening the pool: the pool's configure hook registers the
        # vector type, which needs the extension to exist.
        self._migrate(database_url)
        self._pool = ConnectionPool(
            database_url, min_size=1, max_size=pool_size, configure=register_vector, open=True,
            kwargs={"autocommit": True},
        )

    def close(self) -> None:
        self._pool.close()

    def _migrate(self, database_url: str) -> None:
        import psycopg

        with psycopg.connect(database_url, autocommit=True) as conn:
            # Several gateway workers start at once; an advisory lock makes them run
            # the schema migration one at a time instead of racing on CREATE statements.
            conn.execute("SELECT pg_advisory_lock(%s)", (MIGRATION_LOCK_ID,))
            try:
                conn.execute(SCHEMA.format(dim=self.dim))
                self._check_dimensions(conn)
            finally:
                conn.execute("SELECT pg_advisory_unlock(%s)", (MIGRATION_LOCK_ID,))

    def _check_dimensions(self, conn) -> None:
        """Fail fast if the vector columns were created for a different embedding model."""
        for table in ("semantic_cache", "chunks"):
            existing = conn.execute(
                "SELECT atttypmod FROM pg_attribute WHERE attrelid = %s::regclass AND attname = 'embedding'",
                (table,),
            ).fetchone()
            if existing and existing[0] != self.dim:
                raise RuntimeError(
                    f"{table}.embedding has dimension {existing[0]} but NEXUS_EMBEDDING_DIM is {self.dim}. "
                    "Drop the table (or the database) and re-ingest after changing embedding models."
                )

    # routing rules
    def get_rules(self):
        with self._pool.connection() as conn:
            row = conn.execute("SELECT rules FROM routing_rules WHERE id = 1").fetchone()
        return row[0] if row else None

    def save_rules(self, rules):
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO routing_rules (id, rules) VALUES (1, %s) "
                "ON CONFLICT (id) DO UPDATE SET rules = EXCLUDED.rules, updated_at = now()",
                (json.dumps(rules),),
            )

    # request log
    def log_request(self, record):
        values = [record.get(c) for c in _LOG_COLUMNS]
        placeholders = ", ".join(["%s"] * len(_LOG_COLUMNS))
        with self._pool.connection() as conn:
            conn.execute(f"INSERT INTO request_log ({', '.join(_LOG_COLUMNS)}) VALUES ({placeholders})", values)

    def stats(self):
        with self._pool.connection() as conn:
            row = conn.execute(
                """
                SELECT
                  count(*)                                                        AS requests,
                  count(*) FILTER (WHERE status = 'ok')                           AS succeeded,
                  count(*) FILTER (WHERE status = 'ok' AND cache = 'exact')       AS exact,
                  count(*) FILTER (WHERE status = 'ok' AND cache = 'semantic')    AS semantic,
                  count(*) FILTER (WHERE status = 'ok' AND cache = 'no_context')  AS no_context,
                  count(*) FILTER (WHERE cache IN ('miss', 'error'))              AS upstream_requests,
                  coalesce(sum(attempts), 0)                                      AS upstream_attempts,
                  count(*) FILTER (WHERE failover)                                AS failovers,
                  coalesce(sum(cost_usd), 0)                                      AS cost,
                  coalesce(sum(uncached_cost_usd), 0)                             AS uncached,
                  coalesce(sum(baseline_cost_usd), 0)                             AS baseline,
                  coalesce(percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms)
                           FILTER (WHERE status = 'ok'), 0)                       AS p50,
                  coalesce(percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)
                           FILTER (WHERE status = 'ok'), 0)                       AS p95
                FROM request_log
                """
            ).fetchone()
            by_model = conn.execute(
                "SELECT model, count(*) FROM request_log WHERE status = 'ok' AND model IS NOT NULL "
                "GROUP BY model ORDER BY count(*) DESC"
            ).fetchall()
        (total, ok, exact, semantic, no_context, upstream, attempts, failovers, cost, uncached, baseline, p50, p95) = row
        return {
            "requests": total,
            "succeeded": ok,
            "failed": total - ok,
            "success_rate": round(ok / total, 4) if total else 0.0,
            "cache_hits_exact": exact,
            "cache_hits_semantic": semantic,
            "rag_answered_without_llm": no_context,
            "cache_hit_rate": round((exact + semantic) / total, 4) if total else 0.0,
            "upstream_requests": upstream,
            "upstream_attempts": int(attempts),
            "api_call_reduction": round(1 - upstream / total, 4) if total else 0.0,
            "failovers": failovers,
            **cost_breakdown(cost, uncached, baseline),
            "latency_p50_ms": round(float(p50), 2),
            "latency_p95_ms": round(float(p95), 2),
            "requests_by_model": {m: c for m, c in by_model},
        }

    def recent_requests(self, limit=50):
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                rows = cur.execute(
                    "SELECT * FROM request_log ORDER BY created_at DESC LIMIT %s", (limit,)
                ).fetchall()
        for r in rows:
            r["created_at"] = r["created_at"].isoformat()
        return rows

    def reset_stats(self):
        with self._pool.connection() as conn:
            conn.execute("TRUNCATE request_log")

    # semantic cache
    def semantic_lookup(self, namespace, embedding, threshold, ttl_s):
        vec = np.asarray(embedding, dtype=np.float32)
        with self._pool.connection() as conn:
            # Note: with pgvector < 0.8, a filtered HNSW scan can return fewer rows than exist
            # (filtering happens after the index scan). That shows up as a cache miss, never a
            # wrong hit. pgvector 0.8+ can use `SET hnsw.iterative_scan = relaxed_order`.
            row = conn.execute(
                """
                WITH q AS (SELECT %s::vector AS v)
                SELECT c.response, 1 - (c.embedding <=> q.v) AS similarity
                FROM semantic_cache c, q
                WHERE c.namespace = %s AND c.created_at > now() - make_interval(secs => %s)
                ORDER BY c.embedding <=> q.v
                LIMIT 1
                """,
                (vec, namespace, ttl_s),
            ).fetchone()
        if row and row[1] >= threshold:
            return row[0], float(row[1])
        return None

    def semantic_store(self, namespace, prompt, embedding, response):
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO semantic_cache (namespace, prompt, embedding, response) VALUES (%s, %s, %s, %s)",
                (namespace, prompt, np.asarray(embedding, dtype=np.float32), json.dumps(response)),
            )

    # documents and chunks
    def get_document(self, document_id):
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT id, title, source, content_hash, chunk_count, metadata FROM documents WHERE id = %s",
                (document_id,),
            ).fetchone()
        return Document(*row) if row else None

    def upsert_document(self, document, chunks):
        document.chunk_count = len(chunks)
        with self._pool.connection() as conn:
            with conn.transaction():
                conn.execute(
                    """
                    INSERT INTO documents (id, title, source, content_hash, chunk_count, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title, source = EXCLUDED.source,
                        content_hash = EXCLUDED.content_hash, chunk_count = EXCLUDED.chunk_count,
                        metadata = EXCLUDED.metadata, updated_at = now()
                    """,
                    (document.id, document.title, document.source, document.content_hash,
                     document.chunk_count, json.dumps(document.metadata)),
                )
                conn.execute("DELETE FROM chunks WHERE document_id = %s", (document.id,))
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO chunks (id, document_id, ordinal, heading, text, embedding) "
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        [(c.id, c.document_id, c.ordinal, c.heading, c.text,
                          np.asarray(c.embedding, dtype=np.float32)) for c in chunks],
                    )

    def list_documents(self):
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT id, title, source, content_hash, chunk_count, metadata FROM documents ORDER BY lower(title)"
            ).fetchall()
        return [Document(*r) for r in rows]

    def delete_document(self, document_id):
        with self._pool.connection() as conn:
            cur = conn.execute("DELETE FROM documents WHERE id = %s", (document_id,))
            return cur.rowcount > 0

    def search_chunks(self, embedding, k):
        vec = np.asarray(embedding, dtype=np.float32)
        with self._pool.connection() as conn:
            conn.execute(f"SET hnsw.ef_search = {int(max(40, k * 2))}")
            rows = conn.execute(
                """
                WITH q AS (SELECT %s::vector AS v)
                SELECT c.id, c.document_id, c.ordinal, c.text, c.heading, d.title, d.source, c.embedding,
                       1 - (c.embedding <=> q.v) AS score
                FROM chunks c JOIN documents d ON d.id = c.document_id, q
                ORDER BY c.embedding <=> q.v
                LIMIT %s
                """,
                (vec, k),
            ).fetchall()
        return [(Chunk(*r[:8]), float(r[8])) for r in rows]
