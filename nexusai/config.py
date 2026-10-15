"""Runtime configuration, read from environment variables.

Every setting has a safe default so the gateway runs locally in "mock" mode
with no API keys, no Postgres and no Redis.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> list[str]:
    """Load KEY=VALUE lines from a .env file into os.environ.

    Variables already set in the environment win, so `export` still overrides
    the file. Supports comments, blank lines, `export KEY=...` and quoted values.
    Returns the names it set. Skipped when NEXUS_SKIP_DOTENV=1 (used by tests).
    """
    if os.getenv("NEXUS_SKIP_DOTENV") == "1":
        return []
    path = Path(path)
    if not path.is_file():
        return []
    loaded = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, value = (part.strip() for part in line.split("=", 1))
        if " #" in value and not value.startswith(("'", '"')):
            value = value.split(" #", 1)[0].strip()  # inline comment
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


# Load .env from the current directory, then from the project root, before any settings are read.
load_dotenv(Path.cwd() / ".env")
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


@dataclass
class Settings:
    # "live" calls real provider APIs; "mock" uses simulated providers.
    provider_mode: str = field(default_factory=lambda: os.getenv("NEXUS_PROVIDER_MODE", "mock"))
    # Any provider with a key is enabled in live mode; the router only uses enabled ones.
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    gemini_base_url: str = field(default_factory=lambda: os.getenv(
        "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"))
    gemini_native_base_url: str = field(default_factory=lambda: os.getenv(
        "GEMINI_NATIVE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"))
    # Optional: "low" | "medium" | "high" sent as reasoning_effort to Gemini models.
    gemini_reasoning_effort: str = field(default_factory=lambda: os.getenv("GEMINI_REASONING_EFFORT", ""))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    anthropic_base_url: str = field(default_factory=lambda: os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"))
    request_timeout_s: float = field(default_factory=lambda: _float("NEXUS_REQUEST_TIMEOUT_S", 30.0))

    # Storage: empty DATABASE_URL / REDIS_URL means in-memory backends.
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", ""))

    # Embeddings: "gemini" (gemini-embedding-001), "openai" (text-embedding-3-small)
    # or "hash" (offline, lexical). Empty picks the first available in that order.
    embedding_backend: str = field(default_factory=lambda: os.getenv("NEXUS_EMBEDDINGS", ""))
    gemini_embedding_model: str = field(default_factory=lambda: os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"))
    openai_embedding_model: str = field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    embedding_dim: int = field(default_factory=lambda: _int("NEXUS_EMBEDDING_DIM", 0))

    # Caching
    exact_cache_ttl_s: int = field(default_factory=lambda: _int("NEXUS_EXACT_CACHE_TTL_S", 3600))
    semantic_cache_enabled: bool = field(default_factory=lambda: _bool("NEXUS_SEMANTIC_CACHE", True))
    # Minimum cosine similarity for a semantic cache hit. Empty means a default per
    # embedder (0.95 for Gemini and OpenAI, 0.97 for hashing); tune it with
    # `scripts/benchmark.py --thresholds`, which counts wrong hits.
    semantic_cache_threshold: float = field(default_factory=lambda: _float("NEXUS_SEMANTIC_THRESHOLD", -1.0))
    # Long prompts are dominated by shared context (logs, pasted documents), so two
    # different questions can embed almost identically. Only short prompts use the
    # semantic cache; long ones still use the exact cache.
    semantic_cache_max_chars: int = field(default_factory=lambda: _int("NEXUS_SEMANTIC_MAX_CHARS", 2000))
    semantic_cache_ttl_s: int = field(default_factory=lambda: _int("NEXUS_SEMANTIC_CACHE_TTL_S", 86400))
    # Responses sampled above this temperature are not cached (they are meant to vary).
    cache_max_temperature: float = field(default_factory=lambda: _float("NEXUS_CACHE_MAX_TEMPERATURE", 0.3))

    # Resilience
    max_retries: int = field(default_factory=lambda: _int("NEXUS_MAX_RETRIES", 2))
    retry_base_delay_s: float = field(default_factory=lambda: _float("NEXUS_RETRY_BASE_DELAY_S", 0.25))
    breaker_failure_threshold: int = field(default_factory=lambda: _int("NEXUS_BREAKER_THRESHOLD", 5))
    breaker_reset_timeout_s: float = field(default_factory=lambda: _float("NEXUS_BREAKER_RESET_S", 30.0))

    # Mock provider behaviour (used for offline benchmarks and tests).
    # MOCK_PROVIDERS picks which providers to simulate, e.g. "gemini" to mirror a Gemini-only setup.
    mock_providers: str = field(default_factory=lambda: os.getenv("MOCK_PROVIDERS", "openai,anthropic"))
    mock_failure_rate: float = field(default_factory=lambda: _float("MOCK_FAILURE_RATE", 0.0))
    mock_latency_ms: float = field(default_factory=lambda: _float("MOCK_LATENCY_MS", 0.0))

    # RAG
    rag_chunk_size: int = field(default_factory=lambda: _int("RAG_CHUNK_SIZE", 800))
    rag_chunk_overlap: int = field(default_factory=lambda: _int("RAG_CHUNK_OVERLAP", 150))
    rag_top_k: int = field(default_factory=lambda: _int("RAG_TOP_K", 5))
    # Minimum vector similarity for a chunk to count as relevant. Empty means a
    # default suited to the embedder: 0.5 for Gemini, 0.25 for OpenAI, 0.15 for hashing.
    # These are starting points; scripts/smoke_live.py prints scores to calibrate them.
    rag_min_score: float = field(default_factory=lambda: _float("RAG_MIN_SCORE", -1.0))

    # Optional folder of .md/.txt documents to ingest at startup (idempotent). Useful for
    # demos on the in-memory store, which starts empty after every restart or deploy.
    ingest_on_start: str = field(default_factory=lambda: os.getenv("NEXUS_INGEST_ON_START", ""))

    # Auth: empty means auth disabled (local development).
    api_key: str = field(default_factory=lambda: os.getenv("NEXUS_API_KEY", ""))
    admin_api_key: str = field(default_factory=lambda: os.getenv("NEXUS_ADMIN_API_KEY", ""))

    # Baseline used to report cost savings: what every request would have cost
    # on this model with no caching or routing.
    # Empty means: gemini-3.8-flash when Gemini is the only provider (live or simulated),
    # else claude-sonnet-4-5. Savings are reported against this model.
    baseline_model: str = field(default_factory=lambda: os.getenv("NEXUS_BASELINE_MODEL", ""))

    def __post_init__(self) -> None:
        live = self.provider_mode == "live"
        if not self.embedding_backend:
            if live and self.gemini_api_key:
                self.embedding_backend = "gemini"
            elif live and self.openai_api_key:
                self.embedding_backend = "openai"
            else:
                self.embedding_backend = "hash"
        defaults = {  # backend: (dimension, semantic threshold, RAG min score)
            "gemini": (768, 0.95, 0.5),
            "openai": (1536, 0.95, 0.25),
            "hash": (1024, 0.97, 0.15),
        }
        if self.embedding_backend not in defaults:
            raise ValueError(f"NEXUS_EMBEDDINGS must be one of {sorted(defaults)}")
        dim, threshold, min_score = defaults[self.embedding_backend]
        if not self.embedding_dim:
            self.embedding_dim = dim
        if self.semantic_cache_threshold < 0:
            self.semantic_cache_threshold = threshold
        if self.rag_min_score < 0:
            self.rag_min_score = min_score
        if not self.baseline_model:
            if live:
                gemini_only = bool(self.gemini_api_key) and not (self.openai_api_key or self.anthropic_api_key)
            else:
                gemini_only = [p.strip() for p in self.mock_providers.split(",") if p.strip()] == ["gemini"]
            self.baseline_model = "gemini-3.8-flash" if gemini_only else "claude-sonnet-4-5"
