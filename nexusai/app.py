"""Flask application factory and HTTP API."""
from __future__ import annotations

import functools
import hmac
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from flask import Flask, jsonify, redirect, request, send_from_directory

from . import __version__
from .cache import ExactCache, build_kv
from .config import Settings
from .embeddings import Embedder, EmbeddingError, build_embedder
from .gateway import Gateway, GatewayError, route_summary
from .pricing import CATALOG
from .providers import Provider, build_providers
from .providers.base import ChatRequest, Message
from .rag import Ingestor, RAGPipeline, Retriever
from .routing import DEFAULT_RULES, RuleValidationError
from .storage import Store, build_store
from .telemetry import setup_telemetry

log = logging.getLogger("nexusai")
STATIC_DIR = Path(__file__).parent / "static"
SAMPLE_DOCS_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_docs"
VALID_ROLES = {"system", "user", "assistant"}


@dataclass
class Services:
    settings: Settings
    store: Store
    gateway: Gateway
    ingestor: Ingestor
    retriever: Retriever
    rag: RAGPipeline
    kv: object


def build_services(settings: Settings | None = None, *, providers: dict[str, Provider] | None = None,
                   store: Store | None = None, embedder: Embedder | None = None, kv=None,
                   sleep=None) -> Services:
    settings = settings or Settings()
    store = store or build_store(settings)
    embedder = embedder or build_embedder(settings)
    kv = kv if kv is not None else build_kv(settings)
    providers = providers or build_providers(settings)
    kwargs = {"sleep": sleep} if sleep else {}
    gateway = Gateway(settings, providers, store, ExactCache(kv, settings.exact_cache_ttl_s), embedder, **kwargs)
    retriever = Retriever(store, embedder, min_score=settings.rag_min_score)
    return Services(
        settings=settings,
        store=store,
        gateway=gateway,
        ingestor=Ingestor(store, embedder, settings.rag_chunk_size, settings.rag_chunk_overlap),
        retriever=retriever,
        rag=RAGPipeline(gateway, retriever, store, settings.rag_top_k),
        kv=kv,
    )


def _bad_request(message: str, status: int = 400):
    return jsonify({"error": message}), status


def parse_chat_request(body: dict) -> ChatRequest:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("'messages' must be a non-empty list")
    parsed = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") not in VALID_ROLES or not isinstance(m.get("content"), str):
            raise ValueError(f"messages[{i}] needs a role ({'/'.join(sorted(VALID_ROLES))}) and string content")
        parsed.append(Message(m["role"], m["content"]))
    if parsed[-1].role != "user":
        raise ValueError("the last message must have role 'user'")
    temperature = float(body.get("temperature", 0.0))
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("'temperature' must be between 0 and 2")
    max_tokens = int(body.get("max_tokens", 2048))
    if not 1 <= max_tokens <= 8192:
        raise ValueError("'max_tokens' must be between 1 and 8192")
    model = body.get("model", "auto") or "auto"
    if model != "auto" and model not in CATALOG:
        raise ValueError(f"unknown model '{model}'")
    return ChatRequest(
        messages=parsed,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        task=str(body.get("task", "chat")),
        tier=body.get("tier"),
        use_cache=bool(body.get("cache", True)),
    )


def create_app(services: Services | None = None) -> Flask:
    setup_telemetry()
    services = services or build_services()
    settings = services.settings

    # Load routing rules saved from the admin UI; seed defaults on first run.
    saved = services.store.get_rules()
    if saved:
        services.gateway.router.set_rules(saved)
    else:
        services.store.save_rules(services.gateway.router.rules)

    if settings.ingest_on_start:
        try:
            results = services.ingestor.ingest_directory(settings.ingest_on_start)
            log.info("startup ingest: %d documents from %s", len(results), settings.ingest_on_start)
        except Exception as exc:  # noqa: BLE001 - a failed demo ingest must not stop the gateway
            log.warning("startup ingest of %s failed: %s", settings.ingest_on_start, exc)

    app = Flask(__name__, static_folder=None)
    app.config["SERVICES"] = services

    def require_key(admin: bool = False):
        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                # Admin routes need the admin key; API routes accept the API key or the admin key.
                accepted = [settings.admin_api_key] if admin else [settings.api_key, settings.admin_api_key]
                required = settings.admin_api_key if admin else settings.api_key
                if required:
                    supplied = request.headers.get("X-API-Key", "")
                    auth = request.headers.get("Authorization", "")
                    if auth.startswith("Bearer "):
                        supplied = auth[7:]
                    if not any(k and hmac.compare_digest(supplied, k) for k in accepted):
                        return _bad_request("invalid or missing API key", 401)
                return fn(*args, **kwargs)
            return wrapper
        return decorator

    def json_body() -> dict:
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
        return body

    @app.errorhandler(ValueError)
    @app.errorhandler(KeyError)
    def handle_value_error(exc):
        return _bad_request(str(exc).strip("'\""))

    @app.errorhandler(EmbeddingError)
    def handle_embedding_error(exc: EmbeddingError):
        return jsonify({"error": f"embedding service unavailable: {exc}"}), 503

    @app.errorhandler(GatewayError)
    def handle_gateway_error(exc: GatewayError):
        status = exc.status if 400 <= exc.status < 600 else 502
        return jsonify({"error": str(exc), "attempts": exc.attempts, "route": route_summary(exc.attempts)}), status

    # health ---------------------------------------------------------------
    @app.get("/health")
    def health():
        try:
            kv_ok = bool(services.kv.ping())
        except Exception:  # noqa: BLE001 - health checks report, never raise
            kv_ok = False
        return jsonify({
            "status": "ok" if kv_ok else "degraded",
            "version": __version__,
            "commit": os.getenv("NEXUS_BUILD_SHA", "dev"),
            "mode": settings.provider_mode,
            "providers": services.gateway.provider_health(),
            "store": type(services.store).__name__,
            "cache_backend": type(services.kv).__name__,
            "cache_ok": kv_ok,
            "embeddings": settings.embedding_backend,
            "available_models": [
                {"name": m.name, "provider": m.provider, "tier": m.tier}
                for m in CATALOG.values() if m.provider in services.gateway.providers
            ],
        })

    # gateway --------------------------------------------------------------
    @app.post("/v1/chat")
    @require_key()
    def chat():
        response = services.gateway.chat(parse_chat_request(json_body()))
        return jsonify(response.to_dict())

    @app.get("/v1/models")
    @require_key()
    def models():
        return jsonify([m.__dict__ for m in CATALOG.values()])

    # RAG ------------------------------------------------------------------
    @app.post("/v1/rag/ingest")
    @require_key()
    def rag_ingest():
        docs = json_body().get("documents")
        if not isinstance(docs, list) or not docs:
            raise ValueError("'documents' must be a non-empty list")
        results = []
        for i, doc in enumerate(docs):
            if not isinstance(doc, dict) or not doc.get("text") or not doc.get("title"):
                raise ValueError(f"documents[{i}] needs 'title' and 'text'")
            results.append(services.ingestor.ingest(
                doc["text"], doc["title"], doc.get("source") or doc["title"], doc.get("id"), doc.get("metadata"),
            ).to_dict())
        return jsonify({"ingested": results})

    @app.post("/v1/rag/query")
    @require_key()
    def rag_query():
        body = json_body()
        top_k = int(body.get("top_k", settings.rag_top_k))
        if not 1 <= top_k <= 20:
            raise ValueError("'top_k' must be between 1 and 20")
        result = services.rag.answer(str(body.get("question", "")), top_k=top_k, tier=body.get("tier"))
        return jsonify(result.to_dict())

    @app.post("/v1/rag/search")
    @require_key()
    def rag_search():
        body = json_body()
        query = str(body.get("query", "")).strip()
        if not query:
            raise ValueError("'query' is required")
        k = int(body.get("top_k", settings.rag_top_k))
        return jsonify({"results": [r.public() for r in services.retriever.retrieve(query, k)]})

    @app.get("/v1/rag/documents")
    @require_key()
    def rag_documents():
        return jsonify([
            {"id": d.id, "title": d.title, "source": d.source, "chunks": d.chunk_count}
            for d in services.store.list_documents()
        ])

    @app.delete("/v1/rag/documents/<document_id>")
    @require_key(admin=True)
    def rag_delete(document_id: str):
        if not services.store.delete_document(document_id):
            return _bad_request("document not found", 404)
        return jsonify({"deleted": document_id})

    # admin ----------------------------------------------------------------
    @app.get("/")
    def root():
        return redirect("/admin")

    @app.get("/admin/")
    def admin_trailing_slash():
        return redirect("/admin")

    @app.get("/admin")
    def admin_page():
        return send_from_directory(STATIC_DIR, "admin.html")

    @app.post("/admin/api/sample-docs")
    @require_key(admin=True)
    def admin_sample_docs():
        if not SAMPLE_DOCS_DIR.is_dir():
            return _bad_request("sample documents are not bundled with this deployment", 404)
        results = services.ingestor.ingest_directory(SAMPLE_DOCS_DIR)
        return jsonify({"ingested": [r.to_dict() for r in results]})

    @app.get("/admin/static/<path:filename>")
    def admin_static(filename: str):
        return send_from_directory(STATIC_DIR, filename)

    @app.get("/admin/api/stats")
    @require_key(admin=True)
    def admin_stats():
        return jsonify({**services.store.stats(), "baseline_model": settings.baseline_model})

    @app.post("/admin/api/stats/reset")
    @require_key(admin=True)
    def admin_stats_reset():
        services.store.reset_stats()
        return jsonify({"reset": True})

    @app.get("/admin/api/requests")
    @require_key(admin=True)
    def admin_requests():
        limit = min(int(request.args.get("limit", 50)), 500)
        return jsonify(services.store.recent_requests(limit))

    @app.get("/admin/api/providers")
    @require_key(admin=True)
    def admin_providers():
        return jsonify(services.gateway.provider_health())

    @app.get("/admin/api/rules")
    @require_key(admin=True)
    def admin_get_rules():
        return jsonify({"rules": services.gateway.router.rules, "models": [m.__dict__ for m in CATALOG.values()]})

    @app.put("/admin/api/rules")
    @require_key(admin=True)
    def admin_put_rules():
        try:
            services.gateway.router.set_rules(json_body().get("rules"))
        except RuleValidationError as exc:
            return _bad_request(str(exc), 422)
        services.store.save_rules(services.gateway.router.rules)
        return jsonify({"rules": services.gateway.router.rules})

    @app.post("/admin/api/rules/reset")
    @require_key(admin=True)
    def admin_reset_rules():
        services.gateway.router.set_rules(DEFAULT_RULES)
        services.store.save_rules(services.gateway.router.rules)
        return jsonify({"rules": services.gateway.router.rules})

    return app


def main() -> None:  # pragma: no cover - entry point
    logging.basicConfig(level=logging.INFO)
    create_app().run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), threaded=True)


if __name__ == "__main__":  # pragma: no cover
    main()
