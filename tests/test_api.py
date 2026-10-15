from nexusai.app import build_services, create_app
from nexusai.cache import MemoryKV
from nexusai.embeddings import HashEmbedder
from nexusai.storage import MemoryStore


def test_health(client):
    body = client.get("/health").get_json()
    assert body["status"] == "ok" and body["mode"] == "mock"
    assert set(body["providers"]) == {"openai", "anthropic"}


def test_chat_roundtrip_and_cache(client):
    payload = {"messages": [{"role": "user", "content": "What is Redis?"}]}
    first = client.post("/v1/chat", json=payload).get_json()
    second = client.post("/v1/chat", json=payload).get_json()
    assert first["cache"] == "miss" and second["cache"] == "exact"
    assert first["model"] == "gpt-4o-mini"


def test_chat_validation_errors(client):
    assert client.post("/v1/chat", json={}).status_code == 400
    bad_role = {"messages": [{"role": "robot", "content": "hi"}]}
    assert client.post("/v1/chat", json=bad_role).status_code == 400
    last_not_user = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
    assert client.post("/v1/chat", json=last_not_user).status_code == 400
    unknown_model = {"messages": [{"role": "user", "content": "hi"}], "model": "gpt-99"}
    assert client.post("/v1/chat", json=unknown_model).status_code == 400


def test_all_providers_down_returns_502_with_attempts(client, providers):
    for p in providers.values():
        p.down = True
    resp = client.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 502 and resp.get_json()["attempts"]


def test_rag_endpoints(client, docs_dir):
    docs = [{"title": p.stem, "text": p.read_text(), "source": p.name} for p in sorted(docs_dir.glob("*.md"))]
    ingested = client.post("/v1/rag/ingest", json={"documents": docs}).get_json()["ingested"]
    assert all(r["status"] == "created" for r in ingested)
    assert len(client.get("/v1/rag/documents").get_json()) == len(docs)

    answer = client.post("/v1/rag/query", json={"question": "Which HTTP status codes does the gateway retry?"}).get_json()
    assert answer["grounded"] and answer["citations"][0]["source"] == "failover.md"

    search = client.post("/v1/rag/search", json={"query": "circuit breaker half-open"}).get_json()
    assert search["results"][0]["source"] == "failover.md"

    doc_id = ingested[0]["document_id"]
    assert client.delete(f"/v1/rag/documents/{doc_id}").status_code == 200
    assert client.delete(f"/v1/rag/documents/{doc_id}").status_code == 404


def test_rules_admin_api(client):
    rules = client.get("/admin/api/rules").get_json()["rules"]
    rules[-1]["tier"] = "standard"  # route the catch-all to standard
    assert client.put("/admin/api/rules", json={"rules": rules}).status_code == 200
    reply = client.post("/v1/chat", json={"messages": [{"role": "user", "content": "hello"}]}).get_json()
    assert reply["tier"] == "standard"

    broken = [{"name": "only", "priority": 1, "match": {"task": "chat"}, "tier": "economy"}]
    assert client.put("/admin/api/rules", json={"rules": broken}).status_code == 422
    assert client.post("/admin/api/rules/reset").get_json()["rules"][-1]["tier"] == "economy"


def test_admin_page_and_assets(client):
    assert b"NexusAI" in client.get("/admin").data
    assert client.get("/admin/static/admin.js").status_code == 200


def test_api_keys(settings, providers):
    settings.api_key, settings.admin_api_key = "user-key", "admin-key"
    services = build_services(settings, providers=providers, store=MemoryStore(),
                              embedder=HashEmbedder(settings.embedding_dim), kv=MemoryKV(), sleep=lambda s: None)
    client = create_app(services).test_client()
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert client.post("/v1/chat", json=body).status_code == 401
    assert client.post("/v1/chat", json=body, headers={"X-API-Key": "user-key"}).status_code == 200
    assert client.post("/v1/chat", json=body, headers={"Authorization": "Bearer admin-key"}).status_code == 200
    assert client.get("/admin/api/stats", headers={"X-API-Key": "user-key"}).status_code == 401
    assert client.get("/admin/api/stats", headers={"X-API-Key": "admin-key"}).status_code == 200


def test_health_reports_version_and_commit(client, monkeypatch):
    from nexusai import __version__

    monkeypatch.setenv("NEXUS_BUILD_SHA", "abc1234")
    body = client.get("/health").get_json()
    assert body["version"] == __version__ and body["commit"] == "abc1234"


def test_ingest_on_start_loads_documents(services, docs_dir):
    services.settings.ingest_on_start = str(docs_dir)
    client = create_app(services).test_client()
    assert len(client.get("/v1/rag/documents").get_json()) == len(list(docs_dir.glob("*.md")))


def test_failed_ingest_on_start_does_not_stop_the_app(services):
    services.settings.ingest_on_start = "/does/not/exist"
    client = create_app(services).test_client()
    assert client.get("/health").status_code == 200
