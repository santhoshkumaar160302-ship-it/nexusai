from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["NEXUS_SKIP_DOTENV"] = "1"  # tests must not pick up a developer's .env

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexusai.app import build_services, create_app  # noqa: E402
from nexusai.cache import MemoryKV  # noqa: E402
from nexusai.config import Settings  # noqa: E402
from nexusai.embeddings import HashEmbedder  # noqa: E402
from nexusai.providers import MockProvider  # noqa: E402
from nexusai.storage import MemoryStore  # noqa: E402

DOCS_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_docs"


@pytest.fixture
def settings(monkeypatch):
    for var in ("DATABASE_URL", "REDIS_URL", "NEXUS_API_KEY", "NEXUS_ADMIN_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    s = Settings()
    s.provider_mode = "mock"
    return s


@pytest.fixture
def providers():
    return {"openai": MockProvider("openai", seed=1), "anthropic": MockProvider("anthropic", seed=2)}


@pytest.fixture
def services(settings, providers):
    return build_services(
        settings, providers=providers, store=MemoryStore(), embedder=HashEmbedder(settings.embedding_dim),
        kv=MemoryKV(), sleep=lambda s: None,
    )


@pytest.fixture
def client(services):
    app = create_app(services)
    app.testing = True
    return app.test_client()


@pytest.fixture
def docs_dir():
    return DOCS_DIR
