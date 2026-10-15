"""Document ingestion: chunk, embed, store.

Ingestion is idempotent. A document's id defaults to a hash of its source,
and re-ingesting unchanged content (same SHA-256) is skipped, so a nightly
re-sync only pays for embeddings of documents that actually changed.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from ..embeddings import Embedder
from ..storage.base import Chunk, Document, Store, context_label
from ..telemetry import tracer
from .chunking import chunk_text

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".rst"}


def embedding_text(title: str, heading: str, text: str) -> str:
    """Prefix each chunk with its document title and section path before embedding.

    A chunk like "Set it to 30 seconds." means little alone; with
    "Gateway config > Timeouts" in front it retrieves for timeout questions.
    """
    context = context_label(title, heading)
    return f"{context}\n{text}" if context else text


@dataclass
class IngestResult:
    document_id: str
    title: str
    status: str  # "created" | "updated" | "unchanged"
    chunks: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class Ingestor:
    def __init__(self, store: Store, embedder: Embedder, chunk_size: int = 800, overlap: int = 150):
        self.store = store
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.overlap = overlap

    @staticmethod
    def document_id_for(source: str) -> str:
        return "doc_" + hashlib.sha256(source.encode()).hexdigest()[:16]

    def ingest(self, text: str, title: str, source: str, document_id: str | None = None,
               metadata: dict | None = None) -> IngestResult:
        if not text.strip():
            raise ValueError(f"document '{title}' is empty")
        doc_id = document_id or self.document_id_for(source)
        content_hash = hashlib.sha256(f"{title}\n{text}".encode()).hexdigest()

        existing = self.store.get_document(doc_id)
        if existing and existing.content_hash == content_hash:
            return IngestResult(doc_id, title, "unchanged", existing.chunk_count)

        with tracer.start_as_current_span("rag.ingest") as span:
            pieces = chunk_text(text, self.chunk_size, self.overlap)
            span.set_attribute("nexus.rag.chunks", len(pieces))
            vectors = self.embedder.embed([embedding_text(title, p.heading, p.text) for p in pieces], task="document")
            chunks = [
                Chunk(
                    id=f"{doc_id}:{p.ordinal}",
                    document_id=doc_id,
                    ordinal=p.ordinal,
                    text=p.text,
                    heading=p.heading,
                    title=title,
                    source=source,
                    embedding=vec,
                )
                for p, vec in zip(pieces, vectors, strict=True)
            ]
            document = Document(doc_id, title, source, content_hash, len(chunks), metadata or {})
            self.store.upsert_document(document, chunks)
        return IngestResult(doc_id, title, "updated" if existing else "created", len(chunks))

    def ingest_path(self, path: str | Path, root: str | Path | None = None) -> IngestResult:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        source = str(path.relative_to(root)) if root else str(path)
        first_heading = next((line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("# ")), None)
        return self.ingest(text, first_heading or path.stem.replace("_", " ").replace("-", " "), source)

    def ingest_directory(self, directory: str | Path) -> list[IngestResult]:
        directory = Path(directory)
        files = sorted(p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
        return [self.ingest_path(p, root=directory) for p in files]
