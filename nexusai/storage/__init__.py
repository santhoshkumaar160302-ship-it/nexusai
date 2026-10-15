from __future__ import annotations

from .base import Chunk, Document, Store
from .memory import MemoryStore


def build_store(settings) -> Store:
    if settings.database_url:
        from .postgres import PostgresStore

        return PostgresStore(settings.database_url, settings.embedding_dim)
    return MemoryStore()


__all__ = ["Chunk", "Document", "MemoryStore", "Store", "build_store"]
