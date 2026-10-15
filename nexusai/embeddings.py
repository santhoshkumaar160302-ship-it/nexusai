"""Text embedders used by the semantic cache and the RAG pipeline.

- GeminiEmbedder: gemini-embedding-001 via Gemini's batchEmbedContents API,
  at 768 dimensions (pgvector's HNSW index supports at most 2,000), with
  task-specific embeddings for documents, queries and similarity.
- OpenAIEmbedder: text-embedding-3-small via the OpenAI embeddings endpoint.
- HashEmbedder: an offline embedder using the hashing trick over word
  unigrams and bigrams. It captures lexical overlap, not meaning, but lets the
  whole system (cache, RAG, evals) run with no API keys.

All return L2-normalized vectors, so cosine similarity is a dot product.

Every embed call names a task, because retrieval quality improves when
documents and queries are embedded for their role:
  "document"   chunks being indexed for RAG
  "query"      a question searched against those chunks
  "similarity" prompts compared with each other (semantic cache)
Embedders that have no notion of tasks ignore it.
"""
from __future__ import annotations

import hashlib
import random
import re
import time
from abc import ABC, abstractmethod

import httpx
import numpy as np

TASKS = ("document", "query", "similarity")

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from how i in is it of on or that the this to was what when where which who why with you your".split()
)


class EmbeddingError(Exception):
    """The embedding service failed after retries."""


def stem(token: str) -> str:
    """Tiny suffix stripper so "retried"/"retries"/"retry" and "caching"/"cache" share a feature."""
    if len(token) <= 4 or token.isdigit():
        return token
    for suffix, replacement in (("ies", "y"), ("ied", "y"), ("ing", ""), ("able", ""), ("ed", ""), ("es", ""), ("s", "")):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)] + replacement
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, texts: list[str], task: str = "document") -> list[np.ndarray]:
        ...

    def embed_one(self, text: str, task: str = "document") -> np.ndarray:
        return self.embed([text], task)[0]


class HashEmbedder(Embedder):
    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def embed(self, texts: list[str], task: str = "document") -> list[np.ndarray]:
        out = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            tokens = tokenize(text)
            features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
            for feature in features:
                idx, sign = self._bucket(feature)
                vec[idx] += sign * (0.5 if "_" in feature else 1.0)
            out.append(_normalize(vec))
        return out


class _RetryingHTTPEmbedder(Embedder):
    """Shared retry loop: embedding calls retry 429/5xx/timeouts with backoff.

    Free tiers rate-limit embeddings too, and ingestion sends many batches, so a
    short wait-and-retry is better than failing a whole ingest. Retry-After is
    honored when the server sends it.
    """

    max_attempts = 5
    base_delay_s = 1.0
    max_delay_s = 30.0

    def __init__(self, sleep=time.sleep):
        self._sleep = sleep

    def _post(self, client: httpx.Client, url: str, payload: dict) -> dict:
        last_error = ""
        for attempt in range(self.max_attempts):
            try:
                resp = client.post(url, json=payload)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code == 200:
                    return resp.json()
                last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if resp.status_code not in (408, 429) and resp.status_code < 500:
                    raise EmbeddingError(last_error)  # bad key, bad request: retrying won't help
                retry_after = resp.headers.get("retry-after", "")
                if retry_after.replace(".", "", 1).isdigit():
                    if attempt < self.max_attempts - 1:
                        self._sleep(min(float(retry_after), self.max_delay_s))
                    continue
            if attempt < self.max_attempts - 1:
                delay = min(self.max_delay_s, self.base_delay_s * 2 ** attempt)
                self._sleep(delay * random.uniform(0.5, 1.0))
        raise EmbeddingError(f"embedding failed after {self.max_attempts} attempts: {last_error}")


class GeminiEmbedder(_RetryingHTTPEmbedder):
    TASK_TYPES = {"document": "RETRIEVAL_DOCUMENT", "query": "RETRIEVAL_QUERY", "similarity": "SEMANTIC_SIMILARITY"}

    def __init__(self, api_key: str, model: str = "gemini-embedding-001", dim: int = 768,
                 base_url: str = "https://generativelanguage.googleapis.com/v1beta", timeout_s: float = 60.0,
                 batch_size: int = 100, transport: httpx.BaseTransport | None = None, sleep=time.sleep):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini embeddings")
        if not 128 <= dim <= 2000:
            raise ValueError("Gemini embedding dimension must be 128-2000 (pgvector HNSW indexes at most 2,000)")
        super().__init__(sleep)
        self.dim = dim
        self.model = model
        self.batch_size = batch_size  # Gemini accepts up to 100 texts per batch request
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout_s, transport=transport,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        )

    def embed(self, texts: list[str], task: str = "document") -> list[np.ndarray]:
        task_type = self.TASK_TYPES.get(task, "SEMANTIC_SIMILARITY")
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            payload = {"requests": [
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                    "outputDimensionality": self.dim,
                }
                for text in batch
            ]}
            data = self._post(self._client, f"/models/{self.model}:batchEmbedContents", payload)
            embeddings = data.get("embeddings") or []
            if len(embeddings) != len(batch):
                raise EmbeddingError(f"expected {len(batch)} embeddings, got {len(embeddings)}")
            # Gemini only normalizes full 3072-dim vectors; smaller ones must be normalized here.
            vectors.extend(_normalize(np.asarray(e["values"], dtype=np.float32)) for e in embeddings)
        return vectors


class OpenAIEmbedder(_RetryingHTTPEmbedder):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", dim: int = 1536,
                 base_url: str = "https://api.openai.com/v1", timeout_s: float = 30.0, batch_size: int = 96,
                 transport: httpx.BaseTransport | None = None, sleep=time.sleep):
        super().__init__(sleep)
        self.dim = dim
        self.model = model
        self.batch_size = batch_size
        self._client = httpx.Client(
            base_url=base_url, timeout=timeout_s, transport=transport,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def embed(self, texts: list[str], task: str = "document") -> list[np.ndarray]:
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            data = self._post(self._client, "/embeddings", {"model": self.model, "input": batch, "dimensions": self.dim})
            ordered = sorted(data["data"], key=lambda d: d["index"])
            vectors.extend(_normalize(np.asarray(d["embedding"], dtype=np.float32)) for d in ordered)
        return vectors


def build_embedder(settings) -> Embedder:
    if settings.embedding_backend == "gemini":
        return GeminiEmbedder(
            settings.gemini_api_key,
            settings.gemini_embedding_model,
            settings.embedding_dim,
            settings.gemini_native_base_url,
            settings.request_timeout_s,
        )
    if settings.embedding_backend == "openai":
        return OpenAIEmbedder(
            settings.openai_api_key,
            settings.openai_embedding_model,
            settings.embedding_dim,
            settings.openai_base_url,
            settings.request_timeout_s,
        )
    return HashEmbedder(settings.embedding_dim)
