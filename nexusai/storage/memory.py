"""In-memory store for local development and tests (not durable)."""
from __future__ import annotations

import copy
import threading
import time

import numpy as np

from .base import Chunk, Document, Store, summarize


class MemoryStore(Store):
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._rules: list[dict] | None = None
        self._log: list[dict] = []
        # Semantic cache, per namespace: a growing embedding matrix plus parallel lists.
        self._cache: dict[str, dict] = {}
        self._docs: dict[str, Document] = {}
        self._chunks: dict[str, list[Chunk]] = {}

    def get_rules(self):
        with self._lock:
            return copy.deepcopy(self._rules)

    def save_rules(self, rules):
        with self._lock:
            self._rules = copy.deepcopy(rules)

    def log_request(self, record):
        with self._lock:
            self._log.append(dict(record))

    def stats(self):
        with self._lock:
            return summarize(list(self._log))

    def recent_requests(self, limit=50):
        with self._lock:
            return [dict(r) for r in self._log[-limit:][::-1]]

    def reset_stats(self):
        with self._lock:
            self._log.clear()

    def semantic_lookup(self, namespace, embedding, threshold, ttl_s):
        with self._lock:
            entry = self._cache.get(namespace)
            if not entry or entry["size"] == 0:
                return None
            n = entry["size"]
            sims = entry["matrix"][:n] @ np.asarray(embedding, dtype=np.float32)
            sims[np.asarray(entry["created"][:n]) < time.time() - ttl_s] = -1.0  # expired
            best = int(np.argmax(sims))
            if sims[best] >= threshold:
                return dict(entry["responses"][best]), float(sims[best])
        return None

    def semantic_store(self, namespace, prompt, embedding, response):
        vec = np.asarray(embedding, dtype=np.float32)
        with self._lock:
            entry = self._cache.setdefault(
                namespace, {"matrix": np.zeros((64, vec.shape[0]), dtype=np.float32), "size": 0,
                            "responses": [], "created": []},
            )
            if entry["size"] == entry["matrix"].shape[0]:  # grow geometrically
                entry["matrix"] = np.vstack([entry["matrix"], np.zeros_like(entry["matrix"])])
            entry["matrix"][entry["size"]] = vec
            entry["size"] += 1
            entry["responses"].append(dict(response))
            entry["created"].append(time.time())

    def get_document(self, document_id):
        with self._lock:
            return self._docs.get(document_id)

    def upsert_document(self, document, chunks):
        with self._lock:
            document.chunk_count = len(chunks)
            self._docs[document.id] = document
            self._chunks[document.id] = list(chunks)

    def list_documents(self):
        with self._lock:
            return sorted(self._docs.values(), key=lambda d: d.title.lower())

    def delete_document(self, document_id):
        with self._lock:
            self._chunks.pop(document_id, None)
            return self._docs.pop(document_id, None) is not None

    def search_chunks(self, embedding, k):
        with self._lock:
            chunks = [c for cs in self._chunks.values() for c in cs]
        if not chunks:
            return []
        matrix = np.stack([c.embedding for c in chunks])
        scores = matrix @ np.asarray(embedding, dtype=np.float32)
        order = np.argsort(-scores)[:k]
        return [(chunks[i], float(scores[i])) for i in order]
