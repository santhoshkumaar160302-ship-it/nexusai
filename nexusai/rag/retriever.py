"""Hybrid retrieval: vector search, then a lexical re-rank.

1. Fetch `k * candidate_multiplier` nearest chunks by cosine similarity.
2. Score each candidate on keyword overlap with the query (IDF-weighted over
   the candidate set), which rescues exact terms such as error codes, config
   keys and product names that embeddings tend to blur.
3. Final score = alpha * vector + (1 - alpha) * keyword.
4. Drop candidates under `min_score` (vector similarity) and keep at most
   `max_per_document` chunks per document so one long document cannot crowd
   out the others.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..embeddings import Embedder, tokenize
from ..storage.base import Chunk, Store
from ..telemetry import tracer


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    vector_score: float
    keyword_score: float

    def public(self) -> dict:
        return {
            **self.chunk.public(),
            "score": round(self.score, 4),
            "vector_score": round(self.vector_score, 4),
            "keyword_score": round(self.keyword_score, 4),
            "text": self.chunk.text,
        }


class Retriever:
    def __init__(self, store: Store, embedder: Embedder, *, alpha: float = 0.75, min_score: float = 0.2,
                 candidate_multiplier: int = 4, max_per_document: int = 3):
        self.store = store
        self.embedder = embedder
        self.alpha = alpha
        self.min_score = min_score
        self.candidate_multiplier = candidate_multiplier
        self.max_per_document = max_per_document

    @staticmethod
    def _keyword_scores(query: str, chunks: list[Chunk]) -> list[float]:
        query_terms = set(tokenize(query))
        if not query_terms or not chunks:
            return [0.0] * len(chunks)
        chunk_terms = [set(tokenize(f"{c.title} {c.heading} {c.text}")) for c in chunks]
        n = len(chunks)
        idf = {t: math.log(1 + n / (1 + sum(t in terms for terms in chunk_terms))) for t in query_terms}
        total = sum(idf.values()) or 1.0
        return [sum(idf[t] for t in query_terms if t in terms) / total for terms in chunk_terms]

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        with tracer.start_as_current_span("rag.retrieve") as span:
            embedding = self.embedder.embed_one(query, task="query")
            candidates = self.store.search_chunks(embedding, k * self.candidate_multiplier)
            candidates = [(c, s) for c, s in candidates if s >= self.min_score]
            keyword = self._keyword_scores(query, [c for c, _ in candidates])
            scored = [
                RetrievedChunk(c, self.alpha * s + (1 - self.alpha) * kw, s, kw)
                for (c, s), kw in zip(candidates, keyword, strict=True)
            ]
            scored.sort(key=lambda r: r.score, reverse=True)

            results: list[RetrievedChunk] = []
            per_doc: dict[str, int] = {}
            for item in scored:
                doc = item.chunk.document_id
                if per_doc.get(doc, 0) >= self.max_per_document:
                    continue
                per_doc[doc] = per_doc.get(doc, 0) + 1
                results.append(item)
                if len(results) == k:
                    break
            span.set_attribute("nexus.rag.candidates", len(candidates))
            span.set_attribute("nexus.rag.returned", len(results))
            return results
