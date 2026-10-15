import pytest

from nexusai.rag import chunk_text, evaluate, load_eval_set
from nexusai.rag.chunking import split_sections
from nexusai.rag.evaluation import retrieval_metrics
from nexusai.rag.pipeline import NO_CONTEXT_ANSWER, extract_citations


def test_sections_follow_heading_path():
    text = "# Guide\nintro\n## Setup\nsetup text\n### Docker\ndocker text\n## Usage\nusage text"
    headings = [h for h, _ in split_sections(text)]
    assert headings == ["Guide", "Guide > Setup", "Guide > Setup > Docker", "Guide > Usage"]


def test_headings_inside_code_blocks_are_ignored():
    text = "# Real\n```\n# not a heading\n```\nbody"
    assert [h for h, _ in split_sections(text)] == ["Real"]


def test_chunks_respect_size_and_overlap():
    paragraph = " ".join(f"Sentence number {i} talks about caching." for i in range(60))
    chunks = chunk_text(f"# Doc\n{paragraph}", chunk_size=300, overlap=60)
    assert len(chunks) > 3
    assert all(len(c.text) <= 300 for c in chunks)
    # each chunk after the first starts with text from the end of the previous one
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        assert cur.text.split("\n\n")[0] in prev.text


def test_chunk_validation():
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=100, overlap=100)


def test_ingest_is_idempotent(services):
    first = services.ingestor.ingest("# A\nsome text about caching", "A", "a.md")
    again = services.ingestor.ingest("# A\nsome text about caching", "A", "a.md")
    changed = services.ingestor.ingest("# A\nnew text about routing", "A", "a.md")
    assert (first.status, again.status, changed.status) == ("created", "unchanged", "updated")
    assert len(services.store.list_documents()) == 1


def test_retrieval_finds_the_right_document(services, docs_dir):
    services.ingestor.ingest_directory(docs_dir)
    results = services.retriever.retrieve("How many consecutive failures open the circuit breaker?", k=3)
    assert results[0].chunk.source == "failover.md"
    assert results[0].score >= results[-1].score


def test_max_chunks_per_document(services, docs_dir):
    services.ingestor.ingest_directory(docs_dir)
    services.retriever.max_per_document = 1
    results = services.retriever.retrieve("semantic cache threshold namespace TTL", k=5)
    sources = [r.chunk.source for r in results]
    assert len(sources) == len(set(sources))


def test_answer_has_valid_citations_and_logs_through_gateway(services, docs_dir):
    services.ingestor.ingest_directory(docs_dir)
    result = services.rag.answer("What is the default similarity threshold for the semantic cache?")
    assert result.grounded and result.citations
    assert result.citations[0]["source"] == "caching.md"
    assert result.model == "gpt-4.1"  # rag-answers rule -> standard tier, cheapest first
    assert services.store.stats()["requests"] == 1


def test_no_context_answer_skips_the_llm(services, providers, docs_dir):
    services.ingestor.ingest_directory(docs_dir)
    result = services.rag.answer("zzqx flurble wombat")
    assert result.answer == NO_CONTEXT_ANSWER and not result.grounded
    assert providers["openai"].calls == providers["anthropic"].calls == 0
    assert services.store.stats()["rag_answered_without_llm"] == 1


def test_extract_citations_ignores_out_of_range_and_duplicates():
    assert extract_citations("A [2] b [1][2] c [9]", n_sources=3) == [2, 1]


def test_retrieval_metrics():
    m = retrieval_metrics(["b.md", "b.md", "a.md", "c.md"], {"a.md", "z.md"}, k=2)
    assert m == {"hit": 1.0, "recall": 0.5, "mrr": 0.5}


def test_eval_on_sample_corpus(services, docs_dir):
    services.ingestor.ingest_directory(docs_dir)
    cases = load_eval_set(docs_dir.parent / "eval_questions.jsonl")
    report = evaluate(cases, services.retriever, k=5, pipeline=services.rag)
    assert report["hit@5"] >= 0.9
    assert report["abstention_accuracy"] is not None


def test_context_label_does_not_repeat_the_title():
    from nexusai.rag.ingest import context_label

    assert context_label("Caching", "Caching > Exact cache") == "Caching > Exact cache"
    assert context_label("Caching", "") == "Caching"
    assert context_label("Caching", "Setup > Docker") == "Caching > Setup > Docker"


# ---- live-eval resilience (free-tier quotas) ---------------------------------

def test_rag_answer_survives_flash_quota_by_degrading_to_flash_lite(settings, docs_dir):
    """Reproduces a live failure: both Flash models rate-limited during a RAG answer."""
    from nexusai.app import build_services
    from nexusai.providers.base import ProviderError

    settings.mock_providers = "gemini"
    services = build_services(settings, sleep=lambda s: None)
    services.ingestor.ingest_directory(docs_dir)
    gemini = services.gateway.providers["gemini"]
    original = gemini.chat

    def flash_quota_exhausted(req, model):
        if model in ("gemini-3.6-flash", "gemini-3.8-flash"):
            raise ProviderError("quota exceeded", provider="gemini", status=429)
        return original(req, model)

    gemini.chat = flash_quota_exhausted
    result = services.rag.answer("How many consecutive failures open a provider's circuit breaker?")
    assert result.model == "gemini-3.1-flash-lite" and result.grounded


def test_eval_records_failures_and_retries_after_rate_limit(services, docs_dir):
    from nexusai.gateway import GatewayError

    services.ingestor.ingest_directory(docs_dir)
    cases = load_eval_set(docs_dir.parent / "eval_questions.jsonl")[:3]
    calls, waits, seen = {"n": 0}, [], []
    original = services.rag.answer

    def flaky(question, top_k=None, tier=None):
        calls["n"] += 1
        if calls["n"] == 1:  # first question: quota hit once, then fine
            raise GatewayError("Rate limited by every available model", status=429,
                               attempts=[{"model": "m", "error": "quota exceeded", "status": 429}])
        if question == cases[1].question:  # second question: hard failure, must not abort the run
            raise GatewayError("All providers failed for this request", status=502)
        return original(question, top_k=top_k)

    services.rag.answer = flaky
    report = evaluate(cases, services.retriever, k=5, pipeline=services.rag, rate_limit_wait_s=60,
                      progress=lambda i, n, q, status: seen.append(status), sleep=waits.append)
    assert waits == [60]
    assert report["answered"] == 2 and len(report["answer_errors"]) == 1
    assert "All providers failed" in report["answer_errors"][0]["error"]
    assert len(seen) == 3 and seen[1].startswith("ERROR")


def test_dotenv_loader(tmp_path, monkeypatch):
    from nexusai.config import load_dotenv

    monkeypatch.delenv("NEXUS_SKIP_DOTENV", raising=False)
    monkeypatch.setenv("KEEP_ME", "from-shell")
    for var in ("A_KEY", "B_KEY", "C_KEY"):
        monkeypatch.delenv(var, raising=False)
    env = tmp_path / ".env"
    env.write_text('# comment\nA_KEY=plain\nexport B_KEY="quoted value"\nC_KEY=0.62  # note\nKEEP_ME=from-file\n')
    loaded = load_dotenv(env)
    import os
    assert os.environ["A_KEY"] == "plain" and os.environ["B_KEY"] == "quoted value"
    assert os.environ["C_KEY"] == "0.62" and os.environ["KEEP_ME"] == "from-shell"
    assert set(loaded) == {"A_KEY", "B_KEY", "C_KEY"}
