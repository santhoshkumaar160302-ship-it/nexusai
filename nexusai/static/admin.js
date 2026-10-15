// NexusAI admin UI: stats, provider health, routing rules editor, RAG playground.
(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const KEY_STORAGE = "nexusai.adminKey";

  function readKey() {
    try { return localStorage.getItem(KEY_STORAGE) || ""; } catch { return ""; }
  }
  function writeKey(value) {
    try { localStorage.setItem(KEY_STORAGE, value); } catch { /* storage unavailable: keep in memory */ }
  }
  let adminKey = readKey();

  async function api(path, options = {}) {
    const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
    if (adminKey) headers["X-API-Key"] = adminKey;
    const res = await fetch(path, { ...options, headers });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);
    return body;
  }

  const el = (tag, attrs = {}, ...children) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = v; else node.setAttribute(k, v);
    }
    for (const child of children.flat()) {
      if (child !== null && child !== undefined) node.append(child instanceof Node ? child : String(child));
    }
    return node;
  };

  const pct = (x) => `${(x * 100).toFixed(1)}%`;
  const usd = (x) => `$${Number(x).toFixed(x < 1 ? 4 : 2)}`;

  function setMsg(target, text, kind) {
    target.textContent = text;
    target.className = `msg ${kind || ""}`;
  }

  // ---- overview -------------------------------------------------------------
  async function loadStats() {
    const [stats, providers] = await Promise.all([api("/admin/api/stats"), api("/admin/api/providers")]);
    const cards = [
      ["Requests", stats.requests.toLocaleString(), `${stats.failed} failed`],
      ["Success rate", pct(stats.success_rate), `${stats.failovers} failovers`],
      ["Cache hit rate", pct(stats.cache_hit_rate), `${stats.cache_hits_exact} exact, ${stats.cache_hits_semantic} semantic`],
      ["API call reduction", pct(stats.api_call_reduction), `${stats.upstream_requests} upstream requests`],
      ["Cost savings", pct(stats.cost_savings),
        `${pct(stats.savings_from_caching)} caching, ${pct(stats.savings_from_routing)} routing`],
      ["Spend", usd(stats.cost_usd), `${usd(stats.baseline_cost_usd)} on baseline model`],
      ["Latency p95", `${stats.latency_p95_ms} ms`, `p50 ${stats.latency_p50_ms} ms`],
    ];
    $("#stats").replaceChildren(...cards.map(([label, value, sub]) =>
      el("div", { class: "card" }, el("div", { class: "label" }, label), el("div", { class: "value" }, value),
        el("div", { class: "sub" }, sub))));

    $("#providers").replaceChildren(...Object.entries(providers).map(([name, p]) => {
      const kind = p.state === "closed" ? "ok" : p.state === "half_open" ? "warn" : "bad";
      return el("div", { class: "row" }, el("span", {}, name),
        el("span", {}, el("span", { class: `pill ${kind}` }, p.state.replace("_", " ")),
          ` ${p.consecutive_failures} recent failures`));
    }));

    const models = Object.entries(stats.requests_by_model);
    $("#by-model").replaceChildren(...(models.length
      ? models.map(([m, n]) => el("div", { class: "row" }, el("span", {}, m), el("span", {}, n.toLocaleString())))
      : [el("p", { class: "hint" }, "No requests yet.")]));
  }

  async function loadRequests() {
    const rows = await api("/admin/api/requests?limit=25");
    const head = el("tr", {}, ["Time", "Task", "Rule", "Model", "Cache", "Status", "Attempts", "Cost", "Latency"]
      .map((h) => el("th", {}, h)));
    const body = rows.map((r) => el("tr", {},
      el("td", {}, r.created_at ? new Date(r.created_at).toLocaleTimeString() : "just now"),
      el("td", {}, r.task), el("td", {}, r.rule || ""), el("td", {}, r.model || "—"),
      el("td", {}, r.cache),
      el("td", {}, el("span", { class: `pill ${r.status === "ok" ? "ok" : "bad"}` }, r.status)),
      el("td", { class: "num" }, r.attempts), el("td", { class: "num" }, usd(r.cost_usd)),
      el("td", { class: "num" }, `${Math.round(r.latency_ms)} ms`)));
    $("#requests-table").replaceChildren(el("thead", {}, head), el("tbody", {}, body));
  }

  // ---- routing rules --------------------------------------------------------
  function renderRules(rules) {
    const head = el("tr", {}, ["Priority", "Name", "Match", "Routes to", "Description"].map((h) => el("th", {}, h)));
    const body = rules.map((r) => el("tr", {},
      el("td", { class: "num" }, r.priority), el("td", {}, r.name),
      el("td", {}, el("code", {}, Object.keys(r.match || {}).length ? JSON.stringify(r.match) : "everything")),
      el("td", {}, r.tier ? `${r.tier} tier` : `model ${r.model}`), el("td", {}, r.description || "")));
    $("#rules-table").replaceChildren(el("thead", {}, head), el("tbody", {}, body));
    $("#rules-json").value = JSON.stringify(rules, null, 2);
  }

  async function loadRules() {
    renderRules((await api("/admin/api/rules")).rules);
  }

  async function saveRules() {
    const msg = $("#rules-msg");
    let rules;
    try { rules = JSON.parse($("#rules-json").value); } catch (e) { return setMsg(msg, `Invalid JSON: ${e.message}`, "error"); }
    try {
      renderRules((await api("/admin/api/rules", { method: "PUT", body: JSON.stringify({ rules }) })).rules);
      setMsg(msg, "Rules saved and active.", "ok");
    } catch (e) { setMsg(msg, e.message, "error"); }
  }

  // ---- RAG ------------------------------------------------------------------
  async function loadDocuments() {
    const docs = await api("/v1/rag/documents");
    $("#documents").replaceChildren(...(docs.length ? docs.map((d) => {
      const del = el("button", { class: "secondary danger", type: "button" }, "Delete");
      del.addEventListener("click", async () => {
        try { await api(`/v1/rag/documents/${encodeURIComponent(d.id)}`, { method: "DELETE" }); loadDocuments(); }
        catch (e) { setMsg($("#ingest-msg"), e.message, "error"); }
      });
      return el("li", {}, el("div", {}, el("div", {}, `${d.title} (${d.chunks} chunks)`), el("div", { class: "src" }, d.source)), del);
    }) : [el("li", { class: "hint" }, "No documents yet.")]));
  }

  async function ingest(event) {
    event.preventDefault();
    const form = event.target;
    const data = Object.fromEntries(new FormData(form));
    try {
      const res = await api("/v1/rag/ingest", { method: "POST", body: JSON.stringify({ documents: [data] }) });
      const r = res.ingested[0];
      setMsg($("#ingest-msg"), `${r.title}: ${r.status}, ${r.chunks} chunks.`, "ok");
      form.reset();
      loadDocuments();
    } catch (e) { setMsg($("#ingest-msg"), e.message, "error"); }
  }

  async function ask(event) {
    event.preventDefault();
    const data = Object.fromEntries(new FormData(event.target));
    const out = $("#answer");
    out.replaceChildren(el("p", { class: "hint" }, "Thinking…"));
    try {
      const r = await api("/v1/rag/query", {
        method: "POST", body: JSON.stringify({ question: data.question, top_k: Number(data.top_k) }),
      });
      const cited = new Set(r.citations.map((c) => c.chunk_id));
      const meta = r.model
        ? `${r.model} via ${r.provider} · cache ${r.cache} · ${usd(r.cost_usd)} · retrieval ${r.retrieval_ms} ms · total ${r.total_ms} ms`
        : `No relevant sources found, so no LLM call was made · ${r.total_ms} ms`;
      out.replaceChildren(
        el("div", { class: "text" }, r.answer),
        el("div", { class: "meta" }, meta),
        ...r.sources.map((s, i) => el("div", { class: `source ${cited.has(s.chunk_id) ? "cited" : ""}` },
          el("div", { class: "head" }, `[${i + 1}] ${s.label.replaceAll(" > ", " › ")} · ${s.source} · score ${s.score}`),
          el("div", { class: "body" }, s.text))));
      loadAll();
    } catch (e) { out.replaceChildren(el("p", { class: "msg error" }, e.message)); }
  }

  // ---- wiring ---------------------------------------------------------------
  async function loadAll() {
    try {
      await Promise.all([loadStats(), loadRequests()]);
    } catch (e) {
      $("#stats").replaceChildren(el("p", { class: "msg error" }, `Could not load stats: ${e.message}`));
    }
  }

  $("#admin-key").value = adminKey;
  $("#key-form").addEventListener("submit", (e) => {
    e.preventDefault();
    adminKey = $("#admin-key").value.trim();
    writeKey(adminKey);
    init();
  });
  $("#refresh").addEventListener("click", loadAll);
  $("#reset-stats").addEventListener("click", async () => {
    if (!confirm("Clear the request log and all stats?")) return;
    await api("/admin/api/stats/reset", { method: "POST" });
    loadAll();
  });
  $("#save-rules").addEventListener("click", saveRules);
  $("#reset-rules").addEventListener("click", async () => {
    renderRules((await api("/admin/api/rules/reset", { method: "POST" })).rules);
    setMsg($("#rules-msg"), "Default rules restored.", "ok");
  });
  $("#ingest-form").addEventListener("submit", ingest);
  $("#ask-form").addEventListener("submit", ask);

  function init() {
    loadAll();
    loadRules().catch((e) => setMsg($("#rules-msg"), e.message, "error"));
    loadDocuments().catch(() => {});
  }
  init();
  setInterval(loadAll, 10000);
})();
