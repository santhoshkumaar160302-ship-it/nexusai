// NexusAI console ("Inspector" layout): conversation in the middle, request inspector on the right.
(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const KEY_STORAGE = "nexusai.adminKey";
  const TIER_NAMES = { economy: "fast", standard: "balanced", premium: "most capable" };
  const PROVIDER_NAMES = { openai: "OpenAI", anthropic: "Anthropic", gemini: "Gemini" };
  const VIEWS = {
    ask: ["Ask your documents", "Answers come only from your documents, with sources. Ask follow-ups, or say \"continue\". Select an answer to inspect it."],
    chat: ["Chat", "Remembers this conversation until you start a new chat. Select a reply to inspect it."],
    documents: ["Documents", "What the assistant can answer from. Each document is split into sections and indexed by meaning."],
    monitor: ["Monitor", ""],
    settings: ["Settings", ""],
  };
  const SAMPLE_QUESTIONS = [
    "How many failures open a provider's circuit breaker?",
    "What does the semantic cache do?",
    "Which HTTP status codes does the gateway retry?",
  ];
  // How many earlier exchanges are sent with each chat message. Older ones are left
  // out so long conversations don't grow the prompt (and the cost) without limit.
  const HISTORY_TURNS = 10;
  const CHAT_SUGGESTIONS = [
    "What is a Bloom filter?",
    "Explain step by step how a TCP handshake works.",
    "Give me three tips for clear commit messages.",
  ];

  const state = {
    key: "", health: null, rules: [], docs: [], tab: "ask",
    items: { ask: [], chat: [] }, selected: { ask: null, chat: null }, nextId: 1,
  };

  // ---------- helpers ----------
  const storage = {
    get() { try { return localStorage.getItem(KEY_STORAGE) || ""; } catch { return ""; } },
    set(v) { try { localStorage.setItem(KEY_STORAGE, v); } catch { /* storage unavailable */ } },
  };

  const el = (tag, attrs = {}, ...children) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === undefined || v === null || v === false) continue;
      if (k === "class") node.className = v;
      else if (k === "style") node.setAttribute("style", v);
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    }
    append(node, children);
    return node;
  };
  function append(node, children) {
    for (const c of children.flat(Infinity)) {
      if (c === null || c === undefined || c === false) continue;
      node.append(c instanceof Node ? c : String(c));
    }
    return node;
  }
  const fill = (node, ...children) => { node.replaceChildren(); return append(node, children); };

  const usd = (x) => { const n = Number(x) || 0; return n === 0 ? "$0" : n < 0.01 ? `$${n.toFixed(5)}` : `$${n.toFixed(2)}`; };
  const pct = (x) => `${Math.round((Number(x) || 0) * 1000) / 10}%`;
  const ms = (x) => { const n = Number(x) || 0; return n >= 1000 ? `${(n / 1000).toFixed(2)} s` : n >= 10 ? `${Math.round(n)} ms` : `${n.toFixed(1)} ms`; };
  const providerName = (p) => PROVIDER_NAMES[p] || p;
  const mono = (t) => el("span", { class: "mono" }, t);

  class ApiError extends Error {
    constructor(message, status, body) { super(message); this.status = status; this.body = body || {}; }
  }

  async function api(path, options = {}) {
    const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
    if (state.key) headers["X-API-Key"] = state.key;
    let res;
    try { res = await fetch(path, { ...options, headers }); } catch {
      banner("Can't reach the gateway. Check that the server is still running in your terminal.");
      throw new ApiError("Network error", 0);
    }
    const body = await res.json().catch(() => ({}));
    if (res.status === 401) banner("This server requires the admin key. Add it in Settings.", "#settings");
    if (!res.ok) throw new ApiError(body.error || `Request failed (HTTP ${res.status})`, res.status, body);
    $("#banner").hidden = true;
    return body;
  }

  function banner(text, link) {
    const b = $("#banner");
    fill(b, text, " ", link ? el("a", { href: link }, "Open Settings") : null);
    b.hidden = false;
  }
  function setMsg(target, text, kind = "") { target.textContent = text; target.className = `msg ${kind}`; }
  async function busy(button, fn) {
    const label = button.textContent; button.disabled = true;
    try { return await fn(); } finally { button.disabled = false; button.textContent = label; }
  }
  function describeError(e) {
    if (e.status === 429) return "Every available model is rate limited right now. On a free tier this usually clears within a minute.";
    if (e.status === 503 && /embedding/i.test(e.message)) return "Document search is unavailable right now (the embedding service didn't respond). Try again shortly.";
    if (e.status === 503) return "No AI provider is available right now. Check provider health in Monitor.";
    if (e.status === 401) return "This server requires the admin key. Add it in Settings.";
    if (e.status === 0) return "Can't reach the gateway. Check that the server is running.";
    return e.message;
  }

  // ---------- navigation ----------
  function showTab(name) {
    state.tab = VIEWS[name] ? name : "ask";
    document.body.dataset.tab = state.tab;
    for (const id of Object.keys(VIEWS)) $(`#view-${id}`).hidden = id !== state.tab;
    document.querySelectorAll(".rail a[data-tab]").forEach((a) => {
      if (a.dataset.tab === state.tab) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
    const [title, subtitle] = VIEWS[state.tab];
    $("#view-title").textContent = title;
    $("#view-subtitle").textContent = subtitle;
    updateNewChat();
    if (state.tab === "ask" || state.tab === "chat") { renderConversation(state.tab); renderInspector(); }
    if (state.tab === "documents") loadDocs();
    if (state.tab === "monitor") loadMonitor();
    if (state.tab === "settings") { loadRules(); renderAbout(); }
  }
  window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

  // ---------- status ----------
  const BREAKER = { closed: ["", "healthy"], half_open: ["warn", "testing recovery"], open: ["bad", "paused after failures"] };

  async function loadHealth() {
    try { state.health = await api("/health"); } catch { return; }
    const h = state.health;
    const live = h.mode === "live";
    const pill = $("#mode-pill");
    pill.className = live ? "pill" : "pill sim";
    pill.textContent = live ? `Live on ${Object.keys(h.providers).map(providerName).join(" and ")}` : "Simulated: no real AI calls";
    fill($("#providers"), Object.entries(h.providers).map(([name, p]) => {
      const [cls, text] = BREAKER[p.state] || ["warn", p.state];
      return el("span", { title: `${providerName(name)}: ${text}` }, el("i", { class: `dot ${cls}` }), providerName(name));
    }));
    const select = $("#chat-model");
    if (select.options.length === 1) {
      for (const m of h.available_models) select.append(el("option", { value: m.name }, `${m.name} (${TIER_NAMES[m.tier] || m.tier})`));
    }
    if (state.tab === "settings") renderAbout();
  }

  // ---------- conversation ----------
  function addItem(kind, question) {
    const item = { id: state.nextId++, kind, question, pending: true, result: null, error: null };
    state.items[kind].push(item);
    state.selected[kind] = item.id;
    return item;
  }
  const currentItem = () => state.items[state.tab]?.find((i) => i.id === state.selected[state.tab]);

  function select(kind, id, sourceN) {
    state.selected[kind] = id;
    renderConversation(kind);
    renderInspector(sourceN);
  }

  function citedText(text, item) {
    const parts = String(text).split(/(\[\d+\])/g);
    const n = item.result?.sources?.length || 0;
    return parts.map((part) => {
      const m = part.match(/^\[(\d+)\]$/);
      const k = m ? Number(m[1]) : 0;
      if (!k || k > n) return part;
      return el("button", {
        type: "button", class: "cite", title: `Show source ${k}`,
        onclick: (e) => { e.stopPropagation(); select(item.kind, item.id, k); },
      }, String(k));
    });
  }

  // ---------- markdown (safe: builds DOM nodes, never uses innerHTML) ----------
  // Models answer in Markdown. This renders the common parts: headings, lists,
  // bold, italics, inline code and code blocks. Anything else stays as text.
  const INLINE_RE = /(`[^`\n]+`)|(\*\*(?!\s)[^*]+?\*\*|__(?!\s)[^_]+?__)|(\*(?![\s*])[^*\n]+?\*|(?<![\w])_(?![\s_])[^_\n]+?_(?![\w]))/g;

  function inline(text, cite) {
    const out = [];
    let last = 0;
    for (const m of text.matchAll(INLINE_RE)) {
      if (m.index > last) out.push(cite(text.slice(last, m.index)));
      const t = m[0];
      if (m[1]) out.push(el("code", {}, t.slice(1, -1)));
      else if (m[2]) out.push(el("strong", {}, inline(t.slice(2, -2), cite)));
      else out.push(el("em", {}, inline(t.slice(1, -1), cite)));
      last = m.index + t.length;
    }
    if (last < text.length) out.push(cite(text.slice(last)));
    return out;
  }

  function markdown(text, cite = (s) => s) {
    const lines = String(text).replace(/\r\n/g, "\n").split("\n");
    const blocks = [];
    let para = [];
    let list = null;
    const flushPara = () => { if (para.length) { blocks.push(el("p", {}, inline(para.join("\n"), cite))); para = []; } };
    const flushList = () => { if (list) { blocks.push(list); list = null; } };
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      let m;
      if (/^\s*```/.test(line)) {
        flushPara(); flushList();
        const code = [];
        while (++i < lines.length && !/^\s*```/.test(lines[i])) code.push(lines[i]);
        blocks.push(el("pre", {}, el("code", {}, code.join("\n"))));
      } else if ((m = line.match(/^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/))) {
        flushPara(); flushList();
        blocks.push(el("p", { class: `md-h md-h${m[1].length}` }, inline(m[2], cite)));
      } else if (/^\s{0,3}([-*_])(\s*\1){2,}\s*$/.test(line)) {
        flushPara(); flushList();
        blocks.push(el("hr"));
      } else if ((m = line.match(/^\s*([*+-]|\d+[.)])\s+(.*)$/))) {
        flushPara();
        const tag = /\d/.test(m[1]) ? "ol" : "ul";
        if (!list || list.tagName.toLowerCase() !== tag) {
          flushList();
          list = el(tag, tag === "ol" && parseInt(m[1], 10) > 1 ? { start: parseInt(m[1], 10) } : {});
        }
        list.append(el("li", {}, inline(m[2], cite)));
      } else if (!line.trim()) {
        flushPara(); flushList();
      } else if (list && /^\s{2,}\S/.test(line)) {
        append(list.lastElementChild, [" ", inline(line.trim(), cite)]);  // wrapped list item
      } else {
        flushList();
        para.push(line.trim());
      }
    }
    flushPara(); flushList();
    return el("div", { class: "md" }, blocks);
  }

  function footFor(item) {
    if (item.pending) return null;
    if (item.error) return el("span", { class: "bubble-foot" }, el("span", { class: "tag bad" }, "failed"));
    const r = item.result;
    const tags = [];
    if (r.cache === "exact" || r.cache === "semantic") tags.push(el("span", { class: "tag mem" }, "from memory"));
    else if (r.cache === "no_context") tags.push(el("span", { class: "tag" }, "no matching sources"));
    else tags.push(el("span", { class: "tag acc" }, r.model));
    if (r.failover) tags.push(el("span", { class: "tag bad" }, "backup used"));
    if (r.truncated) tags.push(el("span", { class: "tag bad", title: "The answer reached the length limit and stopped early" }, "cut off"));
    tags.push(el("span", {}, ms(r.total_ms ?? r.latency_ms)));
    if (r.cost_usd) tags.push(el("span", {}, usd(r.cost_usd)));
    return el("span", { class: "bubble-foot" }, tags);
  }

  function renderConversation(kind) {
    const log = $(`#${kind}-log`);
    const items = state.items[kind];
    if (!items.length) { renderEmpty(kind, log); return; }
    fill(log, items.map((item) => {
      const answerText = item.pending ? (kind === "ask" ? "Searching your documents…" : "Routing your message…")
        : item.error ? describeError(item.error)
        : kind === "ask" ? item.result.answer : item.result.content;
      const classes = ["bubble", item.pending ? "pending" : "", item.error ? "failed" : "", item.id === state.selected[kind] ? "selected" : ""].join(" ");
      const body = item.pending || item.error ? answerText
        : markdown(answerText, kind === "ask" ? (s) => citedText(s, item) : undefined);
      const cutoff = item.result?.truncated
        ? el("p", { class: "cutoff" }, kind === "chat"
          ? "This answer hit the length limit and stopped early. Reply \"continue\" to get the rest."
          : "This answer hit the length limit and stopped early. Reply \"continue\" to get the rest.")
        : null;
      const open = () => select(kind, item.id);
      return [
        el("div", { class: "msg-row me" }, el("div", { class: "bubble" }, item.question)),
        el("div", { class: "msg-row" },
          el("div", {
            class: classes, role: "button", tabindex: "0", "aria-pressed": String(item.id === state.selected[kind]),
            onclick: open,
            onkeydown: (e) => { if ((e.key === "Enter" || e.key === " ") && e.target === e.currentTarget) { e.preventDefault(); open(); } },
          }, body, cutoff, footFor(item))),
      ];
    }));
    log.lastElementChild?.scrollIntoView({ block: "nearest" });
  }

  function renderEmpty(kind, log) {
    if (kind === "chat") {
      fill(log, el("div", { class: "empty" },
        el("h2", {}, "Send a message"),
        el("p", {}, "It remembers the conversation, so you can ask follow-ups. Simple questions go to the fast tier and reasoning-heavy ones to the most capable tier. Use New chat to start fresh."),
        el("div", { class: "chips" }, CHAT_SUGGESTIONS.map((q) => el("button", { type: "button", onclick: () => chat(q) }, q)))));
      return;
    }
    if (!state.docs.length) {
      fill(log, el("div", { class: "empty" },
        el("h2", {}, "Add documents to get started"),
        el("p", {}, "There's nothing to search yet. Load the sample documents, or add your own in Documents."),
        el("div", { class: "row" },
          el("button", { type: "button", class: "primary", onclick: (e) => loadSamples(e.currentTarget) }, "Load sample documents"),
          el("a", { href: "#documents" }, "Add your own"))));
      return;
    }
    const hasSamples = state.docs.some((d) => d.source === "failover.md");
    fill(log, el("div", { class: "empty" },
      el("h2", {}, `Ready to answer from ${state.docs.length} document${state.docs.length === 1 ? "" : "s"}`),
      el("p", {}, hasSamples ? "Try one of these, or ask your own question below." : "Ask a question below."),
      hasSamples ? el("div", { class: "chips" }, SAMPLE_QUESTIONS.map((q) => el("button", { type: "button", onclick: () => ask(q) }, q))) : null));
  }

  async function send(kind, text, path, payload, input, button) {
    text = text.trim();
    if (!text) return;
    const item = addItem(kind, text);
    input.value = ""; autosize(input);
    renderConversation(kind); renderInspector(); updateNewChat();
    await busy(button, async () => {
      try { item.result = await api(path, { method: "POST", body: JSON.stringify(payload(text, item)) }); }
      catch (e) { item.error = e; }
      item.pending = false;
    });
    if (state.tab === kind) { renderConversation(kind); renderInspector(); updateNewChat(); }
    loadHealth();
  }

  // Earlier exchanges in this chat, oldest first, as the messages the API expects.
  // Failed and still-pending replies are skipped; so is the message being sent now.
  function chatHistory() {
    const done = state.items.chat.filter((i) => !i.pending && !i.error && i.result);
    return done.slice(-HISTORY_TURNS).flatMap((i) => [
      { role: "user", content: i.question },
      { role: "assistant", content: i.result.content },
    ]);
  }

  // The same for Ask, but shorter (the server keeps the last three exchanges). Each earlier
  // question is sent as it was understood ("and how long?" becomes the full question), so a
  // later "continue" searches for the right thing.
  function askHistory() {
    const done = state.items.ask.filter((i) => !i.pending && !i.error && i.result && i.result.cache !== "no_context");
    return done.slice(-3).flatMap((i) => [
      { role: "user", content: i.result.mode && i.result.mode !== "new" ? i.result.search_query : i.question },
      { role: "assistant", content: i.result.answer },
    ]);
  }

  function updateNewChat() {
    const button = $("#new-chat");
    const convo = state.tab === "ask" || state.tab === "chat";
    button.hidden = !convo;
    if (!convo) return;
    const items = state.items[state.tab];
    button.lastElementChild.textContent = state.tab === "chat" ? "New chat" : "New conversation";
    button.disabled = !items.length || items.some((i) => i.pending);
  }

  function newConversation() {
    const kind = state.tab;
    if (!state.items[kind] || state.items[kind].some((i) => i.pending)) return;
    state.items[kind] = [];
    state.selected[kind] = null;
    renderConversation(kind); renderInspector(); updateNewChat();
    $(`#${kind}-input`).focus();
  }
  $("#new-chat").addEventListener("click", newConversation);

  const ask = (q) => send("ask", q, "/v1/rag/query",
    (question, item) => {
      const history = askHistory();
      item.historyCount = history.length;
      return { question, top_k: Number($("#ask-k").value), history };
    },
    $("#ask-input"), $("#ask-form .primary"));
  const chat = (m) => send("chat", m, "/v1/chat",
    (message, item) => {
      const history = chatHistory();
      item.historyCount = history.length;
      return { messages: [...history, { role: "user", content: message }], model: $("#chat-model").value, cache: $("#chat-cache").checked };
    },
    $("#chat-input"), $("#chat-form .primary"));

  // ---------- inspector ----------
  function timelineRows(item) {
    const r = item.result || {};
    const rows = [];
    if (item.kind === "ask") rows.push({ label: "find sources", ms: r.retrieval_ms, cls: "" });
    if (r.cache !== "no_context") {
      const hit = r.cache === "exact" || r.cache === "semantic";
      rows.push({ label: hit ? "memory (hit)" : "memory lookup", ms: r.cache_ms, cls: hit ? "memory" : "" });
    }
    const route = item.error ? (item.error.body?.route || []) : (r.route || []);
    for (const hop of route) {
      // Red bars mean failed or rate limited; "What happened" below spells out why.
      const tries = hop.count > 1 ? ` ×${hop.count}` : "";
      rows.push({ label: `${hop.model}${tries}`, ms: hop.ms, cls: hop.outcome === "answered" ? "answered" : "problem" });
    }
    return rows;
  }

  function renderWaterfall(item) {
    const rows = timelineRows(item);
    if (!rows.length) return null;
    const sum = rows.reduce((a, row) => a + (Number(row.ms) || 0), 0);
    const total = Math.max(sum, Number(item.result?.total_ms ?? item.result?.latency_ms) || 0, 0.001);
    let offset = 0;
    return el("div", { class: "insp-section" },
      el("h3", {}, "Timeline"),
      el("div", { class: "waterfall", role: "list" }, rows.map((row) => {
        const width = ((Number(row.ms) || 0) / total) * 100;
        const left = (offset / total) * 100;
        offset += Number(row.ms) || 0;
        return el("div", { class: "wf-row", role: "listitem" },
          el("span", { class: "wf-label", title: row.label }, row.label),
          el("span", { class: "wf-track" }, el("span", { class: `wf-bar ${row.cls}`, style: `left:${left}%;width:${width}%` })),
          el("span", { class: "wf-ms" }, ms(row.ms)));
      })),
      el("div", { class: "legend" },
        el("span", { style: "--k:var(--check)" }, "checks"),
        el("span", { style: "--k:var(--mem)" }, "memory"),
        el("span", { style: "--k:var(--bad)" }, "failed or rate limited"),
        el("span", { style: "--k:var(--acc)" }, "answered")));
  }

  function routeSteps(item) {
    const r = item.result || {};
    const steps = [];
    if (r.cache === "no_context") {
      steps.push(["memory", "Searched your documents: nothing relevant enough"]);
      steps.push(["", "Answered \"I don't know\" without calling a model"]);
      return steps;
    }
    if (r.cache === "exact") return [["memory", "Found the same question in memory"], ["answered", ["Reused the answer from ", mono(r.model)]]];
    if (r.cache === "semantic") {
      return [["memory", `Found a similar question in memory (${Math.round((r.similarity || 0) * 100)}% match)`],
        ["answered", ["Reused the answer from ", mono(r.model)]]];
    }
    if (item.kind === "ask" && r.sources) steps.push(["", `Found ${r.sources.length} relevant section${r.sources.length === 1 ? "" : "s"} in your documents`]);
    steps.push(["", "Checked memory: not seen before"]);
    const rule = state.rules.find((x) => x.name === r.rule);
    if (r.rule) {
      steps.push(["", r.rule === "caller-model" ? ["You picked ", mono(r.model)]
        : [`Rule `, mono(r.rule), ` chose the ${TIER_NAMES[rule?.tier || r.tier] || r.tier} tier`]]);
    }
    const route = item.error ? (item.error.body?.route || []) : (r.route || []);
    for (const hop of route) {
      const tries = hop.count > 1 ? ` (${hop.count} tries)` : "";
      if (hop.outcome === "answered") steps.push(["answered", [mono(hop.model), " answered"]]);
      else if (hop.outcome === "rate_limited") steps.push(["problem", [mono(hop.model), ` was rate limited${tries}`]]);
      else if (hop.outcome === "skipped") steps.push(["problem", [mono(hop.model), " was skipped: provider paused after failures"]]);
      else steps.push(["problem", [mono(hop.model), ` failed: ${hop.detail}${tries}`]]);
    }
    if (item.error) steps.push(["problem", describeError(item.error)]);
    return steps;
  }

  function renderSources(item, focusN) {
    const r = item.result;
    if (!r?.sources?.length) return null;
    const cited = new Set(r.citations.map((c) => c.chunk_id));
    return el("div", { class: "insp-section" },
      el("h3", {}, `Sources (${r.sources.length})`),
      r.sources.map((s, i) => {
        const n = i + 1;
        const open = n === focusN;
        return el("button", {
          type: "button", class: `src ${cited.has(s.chunk_id) ? "cited" : ""} ${open ? "flash" : ""}`,
          "aria-expanded": String(open), "data-n": n,
          onclick: (e) => { const b = e.currentTarget; b.setAttribute("aria-expanded", String(b.getAttribute("aria-expanded") !== "true")); },
        },
          el("span", { class: "src-head" }, el("span", {}, `[${n}] ${s.label || s.title}`),
            el("span", { class: "muted" }, `${Math.round(s.score * 100)}%${cited.has(s.chunk_id) ? ", cited" : ""}`)),
          el("span", { class: "src-body" }, `${s.source}\n\n${s.text}`));
      }));
  }

  function renderInspector(focusN) {
    const body = $("#inspector-body");
    if (state.tab !== "ask" && state.tab !== "chat") return;
    const item = currentItem();
    if (!item) {
      fill(body, el("p", { class: "insp-empty" }, state.tab === "ask"
        ? "Ask a question and the inspector shows how it was answered: the timeline, which models were tried, the cost, and the sources used."
        : "Send a message and the inspector shows how the gateway handled it: memory, routing, every model tried, timing and cost."));
      return;
    }
    if (item.pending) {
      fill(body, el("p", { class: "insp-question" }, item.question), el("p", { class: "insp-empty" }, "Waiting for the gateway…"));
      return;
    }
    const r = item.result || {};
    const fromMemory = r.cache === "exact" || r.cache === "semantic";
    const kv = [
      ["Answered by", item.error ? "nobody" : r.cache === "no_context" ? "no model needed" : r.model, !item.error && r.model],
      ["Total time", ms(r.total_ms ?? r.latency_ms)],
      ["Cost", item.error ? "$0" : fromMemory ? "$0, from memory" : usd(r.cost_usd)],
      ...(item.kind === "chat" ? [["Conversation sent", item.historyCount ? `${item.historyCount / 2} earlier exchange${item.historyCount === 2 ? "" : "s"}` : "just this message"]] : []),
      ...(item.kind === "ask" && r.mode && r.mode !== "new" ? [["Searched for", r.mode === "continue"
        ? `the previous question again, to continue the answer` : `"${r.search_query}"`]] : []),
      ["Tokens in / out", fromMemory || r.cache === "no_context" || item.error ? "none" : `${(r.prompt_tokens || 0).toLocaleString()} / ${(r.completion_tokens || 0).toLocaleString()}${r.truncated ? " (hit the limit)" : ""}`],
    ];
    fill(body,
      el("p", { class: "insp-question" }, item.question),
      el("dl", { class: "kv" }, kv.map(([k, v, isModel]) => el("div", {}, el("dt", {}, k), el("dd", { class: isModel ? "mono" : "" }, v)))),
      item.error && !item.error.body?.route?.length ? null : renderWaterfall(item),
      el("div", { class: "insp-section" }, el("h3", {}, "What happened"),
        el("ol", { class: "steps" }, routeSteps(item).map(([cls, text]) => el("li", { class: cls }, text)))),
      renderSources(item, focusN));
    if (focusN) body.querySelector(`[data-n="${focusN}"]`)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  // ---------- documents ----------
  async function loadDocs() {
    try { state.docs = await api("/v1/rag/documents"); } catch { return; }
    $("#doc-count").textContent = state.docs.length ? String(state.docs.length) : "";
    const sections = state.docs.reduce((a, d) => a + d.chunks, 0);
    $("#doc-summary").textContent = state.docs.length ? `${state.docs.length} documents, ${sections} sections` : "";
    const head = el("thead", {}, el("tr", {}, el("th", {}, "Title"), el("th", {}, "Source"), el("th", { class: "num" }, "Sections"), el("th", {}, el("span", { class: "sr-only" }, "Actions"))));
    const rows = state.docs.map((d) => el("tr", {},
      el("td", { class: "wrap" }, el("strong", {}, d.title)),
      el("td", { class: "wrap muted" }, d.source),
      el("td", { class: "num" }, d.chunks),
      el("td", { class: "num" }, el("button", {
        type: "button", class: "danger", "aria-label": `Delete ${d.title}`,
        onclick: async (e) => {
          if (!confirm(`Delete "${d.title}"? Questions will no longer use it.`)) return;
          await busy(e.currentTarget, async () => {
            try { await api(`/v1/rag/documents/${encodeURIComponent(d.id)}`, { method: "DELETE" }); setMsg($("#docs-msg"), `Deleted "${d.title}".`, "ok"); }
            catch (err) { setMsg($("#docs-msg"), describeError(err), "error"); }
          });
          loadDocs();
        },
      }, "Delete"))));
    fill($("#doc-table"), head, el("tbody", {}, rows.length ? rows
      : el("tr", {}, el("td", { colspan: 4, class: "muted" }, "No documents yet. Upload a file, paste text, or load the samples."))));
    if (state.tab === "ask" && !state.items.ask.length) renderConversation("ask");
  }

  function summarizeIngest(results) {
    const count = (s) => results.filter((r) => r.status === s).length;
    const parts = [];
    if (count("created")) parts.push(`${count("created")} added`);
    if (count("updated")) parts.push(`${count("updated")} updated`);
    if (count("unchanged")) parts.push(`${count("unchanged")} already up to date`);
    return parts.join(", ") || "nothing to add";
  }

  async function loadSamples(button) {
    await busy(button, async () => {
      button.textContent = "Loading…";
      try { const r = await api("/admin/api/sample-docs", { method: "POST" }); setMsg($("#docs-msg"), `Sample documents: ${summarizeIngest(r.ingested)}.`, "ok"); }
      catch (e) { setMsg($("#docs-msg"), describeError(e), "error"); }
    });
    await loadDocs();
  }

  const ingest = async (documents) => (await api("/v1/rag/ingest", { method: "POST", body: JSON.stringify({ documents }) })).ingested;

  async function uploadFiles(files) {
    const msg = $("#docs-msg");
    const accepted = [...files].filter((f) => /\.(md|markdown|txt)$/i.test(f.name));
    if (!accepted.length) { setMsg(msg, "Choose .md or .txt files.", "error"); return; }
    setMsg(msg, `Reading ${accepted.length} file${accepted.length === 1 ? "" : "s"}…`);
    try {
      const documents = await Promise.all(accepted.map(async (f) => {
        const text = await f.text();
        const heading = text.split("\n").find((line) => line.startsWith("# "));
        return { title: heading ? heading.slice(2).trim() : f.name.replace(/\.[^.]+$/, "").replace(/[-_]/g, " "), text, source: f.name };
      }));
      const empty = documents.filter((d) => !d.text.trim());
      if (empty.length) throw new Error(`${empty.map((d) => d.source).join(", ")} is empty.`);
      setMsg(msg, "Indexing…");
      setMsg(msg, `Uploaded: ${summarizeIngest(await ingest(documents))}.`, "ok");
    } catch (e) { setMsg(msg, e instanceof ApiError ? describeError(e) : e.message, "error"); }
    loadDocs();
  }

  // ---------- monitor ----------
  async function loadMonitor() {
    let stats, requests;
    try { [stats, requests] = await Promise.all([api("/admin/api/stats"), api("/admin/api/requests?limit=30")]); } catch { return; }
    state.baseline = stats.baseline_model;
    const metric = (cls, label, value, explain) => el("div", { class: `metric ${cls}` },
      el("dt", {}, label), el("dd", {}, el("div", { class: "value" }, value), el("div", { class: "explain" }, explain)));
    fill($("#stats"),
      metric("", "Requests", stats.requests.toLocaleString(), stats.failed ? `${stats.failed} could not be answered` : "All answered"),
      metric("", "Success rate", pct(stats.success_rate), stats.failovers ? `${stats.failovers} needed a backup model` : "No backup model needed"),
      metric("mem", "Answered from memory", pct(stats.cache_hit_rate), `${stats.cache_hits_exact} repeated, ${stats.cache_hits_semantic} reworded`),
      metric("", "Cost saved", pct(stats.cost_savings), `vs. all on ${stats.baseline_model}: ${pct(stats.savings_from_caching)} memory, ${pct(stats.savings_from_routing)} routing`),
      metric("", "Spent", usd(stats.cost_usd), `${usd(stats.baseline_cost_usd)} on ${stats.baseline_model}`),
      metric("", "Median response", ms(stats.latency_p50_ms), `95th percentile ${ms(stats.latency_p95_ms)}`));

    const models = Object.entries(stats.requests_by_model);
    const max = Math.max(1, ...models.map(([, n]) => n));
    fill($("#by-model"), models.length ? models.map(([m, n]) => el("div", { class: "bar-row" },
      mono(m), el("span", {}, n.toLocaleString()),
      el("div", { class: "bar", "aria-hidden": "true" }, el("span", { style: `width:${(n / max) * 100}%` }))))
      : el("p", { class: "muted" }, "No answers yet. Try Ask or Chat."));

    fill($("#health-detail"), Object.entries(state.health?.providers || {}).map(([name, p]) => {
      const [cls, text] = BREAKER[p.state] || ["warn", p.state];
      return el("div", { class: "health-row" }, el("i", { class: `dot ${cls}` }),
        el("div", {}, el("strong", {}, `${providerName(name)}: ${text}`),
          el("div", { class: "muted" }, p.consecutive_failures ? `${p.consecutive_failures} failure${p.consecutive_failures === 1 ? "" : "s"} in a row` : "No recent failures")));
    }));

    const memTag = (c) => ({
      exact: el("span", { class: "tag mem" }, "repeated"), semantic: el("span", { class: "tag mem" }, "reworded"),
      no_context: el("span", { class: "tag" }, "no match"),
    }[c] || el("span", { class: "tag" }, "new"));
    const head = el("thead", {}, el("tr", {}, ["When", "Type", "Answered by", "Memory", "Result"].map((h) => el("th", {}, h)),
      el("th", { class: "num" }, "Cost"), el("th", { class: "num" }, "Time")));
    const rows = requests.map((r) => el("tr", {},
      el("td", {}, r.created_at ? new Date(r.created_at).toLocaleTimeString() : "just now"),
      el("td", {}, r.task === "rag" ? "Document question" : "Chat"),
      el("td", {}, r.model ? mono(r.model) : el("span", { class: "muted" }, "none")),
      el("td", {}, memTag(r.cache)),
      el("td", {}, r.status === "ok" ? el("span", { class: "tag ok" }, r.failover ? "ok, backup used" : "ok") : el("span", { class: "tag bad", title: r.error || "" }, "failed")),
      el("td", { class: "num" }, usd(r.cost_usd)),
      el("td", { class: "num" }, ms(r.latency_ms))));
    fill($("#requests"), head, el("tbody", {}, rows.length ? rows : el("tr", {}, el("td", { colspan: 7, class: "muted" }, "No requests yet."))));
  }

  // ---------- settings ----------
  function describeMatch(match) {
    const parts = [];
    if (match.task === "rag") parts.push("it's a question about your documents");
    else if (match.task) parts.push(`the task is "${match.task}"`);
    if (match.keywords_any?.length) parts.push(`the message mentions ${match.keywords_any.map((k) => `"${k}"`).join(", ")}`);
    if (match.min_prompt_tokens) parts.push(`the message is over about ${match.min_prompt_tokens.toLocaleString()} tokens`);
    if (match.max_prompt_tokens) parts.push(`the message is under about ${match.max_prompt_tokens.toLocaleString()} tokens`);
    return parts.length ? `When ${parts.join(" and ")}` : "Everything else";
  }

  function renderRules(rules) {
    state.rules = rules;
    const available = state.health?.available_models || [];
    fill($("#rules"), rules.map((r) => {
      const models = r.tier ? available.filter((m) => m.tier === r.tier).map((m) => m.name) : [];
      return el("li", {},
        el("div", {}, `${describeMatch(r.match || {})}: `,
          el("strong", {}, r.model ? ["always use ", mono(r.model)] : `use the ${TIER_NAMES[r.tier] || r.tier} tier`),
          models.length ? el("span", { class: "muted" }, " (", models.map((m, i) => [i ? ", " : "", mono(m)]), ")") : null),
        el("div", { class: "rule-meta" }, mono(r.name), r.description ? `: ${r.description}` : ""));
    }));
    $("#rules-json").value = JSON.stringify(rules, null, 2);
  }
  async function loadRules() { try { renderRules((await api("/admin/api/rules")).rules); } catch { /* banner explains */ } }

  function renderAbout() {
    const h = state.health;
    if (!h) return;
    const rows = [
      ["Version", h.version], ["Build", h.commit === "dev" ? "local development" : h.commit.slice(0, 12)],
      ["Mode", h.mode === "live" ? "live (real AI calls)" : "simulated"],
      ["Providers", Object.keys(h.providers).map(providerName).join(", ")],
      ["Embeddings", h.embeddings], ["Storage", `${h.store}, cache in ${h.cache_backend}`],
      ["Models", h.available_models.map((m) => m.name).join(", ")],
    ];
    fill($("#about"), rows.map(([k, v]) => [el("dt", {}, k), el("dd", {}, v)]));
  }

  // ---------- wiring ----------
  function autosize(t) { t.style.height = "auto"; t.style.height = `${Math.min(t.scrollHeight, 160)}px`; }
  for (const [form, input, fn] of [["#ask-form", "#ask-input", () => ask($("#ask-input").value)], ["#chat-form", "#chat-input", () => chat($("#chat-input").value)]]) {
    $(form).addEventListener("submit", (e) => { e.preventDefault(); fn(); });
    $(input).addEventListener("input", (e) => autosize(e.target));
    $(input).addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $(form).requestSubmit(); }
    });
  }

  $("#load-samples").addEventListener("click", (e) => loadSamples(e.currentTarget));
  $("#file-input").addEventListener("change", (e) => { uploadFiles(e.target.files); e.target.value = ""; });
  const dz = $("#dropzone");
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("over"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("over"));
  dz.addEventListener("drop", (e) => { e.preventDefault(); dz.classList.remove("over"); uploadFiles(e.dataTransfer.files); });

  $("#paste-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target;
    const data = Object.fromEntries(new FormData(form));
    await busy(form.querySelector(".primary"), async () => {
      try {
        const [r] = await ingest([{ title: data.title, text: data.text, source: data.source || data.title }]);
        setMsg($("#docs-msg"), r.status === "unchanged" ? `"${r.title}" is already up to date.` : `Added "${r.title}" in ${r.chunks} section${r.chunks === 1 ? "" : "s"}. Ask about it in Ask.`, "ok");
        form.reset();
      } catch (err) { setMsg($("#docs-msg"), describeError(err), "error"); }
    });
    loadDocs();
  });

  $("#refresh").addEventListener("click", () => { loadHealth(); loadMonitor(); });
  $("#reset-stats").addEventListener("click", async () => {
    if (!confirm("Clear the request history and all numbers in Monitor?")) return;
    try { await api("/admin/api/stats/reset", { method: "POST" }); } catch { return; }
    loadMonitor();
  });

  $("#key-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    state.key = $("#admin-key").value.trim();
    storage.set(state.key);
    try {
      await api("/admin/api/stats");
      setMsg($("#key-msg"), state.key ? "Key saved. The console can use the admin API." : "Key cleared.", "ok");
      init();
    } catch (err) {
      setMsg($("#key-msg"), err.status === 401 ? "That key was rejected. Check NEXUS_ADMIN_API_KEY on the server." : describeError(err), "error");
    }
  });

  $("#save-rules").addEventListener("click", async () => {
    const msg = $("#rules-msg");
    let rules;
    try { rules = JSON.parse($("#rules-json").value); } catch (err) { setMsg(msg, `That isn't valid JSON: ${err.message}`, "error"); return; }
    try { renderRules((await api("/admin/api/rules", { method: "PUT", body: JSON.stringify({ rules }) })).rules); setMsg(msg, "Rules saved and in use.", "ok"); }
    catch (err) { setMsg(msg, err.message, "error"); }
  });
  $("#reset-rules").addEventListener("click", async () => {
    if (!confirm("Replace your rules with the defaults?")) return;
    try { renderRules((await api("/admin/api/rules/reset", { method: "POST" })).rules); setMsg($("#rules-msg"), "Default rules restored.", "ok"); }
    catch (err) { setMsg($("#rules-msg"), err.message, "error"); }
  });

  async function init() {
    await loadHealth();
    await Promise.all([loadDocs(), loadRules()]);
    showTab(location.hash.slice(1) || "ask");
  }

  state.key = storage.get();
  $("#admin-key").value = state.key;
  init();
  setInterval(() => {
    if (document.hidden) return;
    loadHealth();
    if (state.tab === "monitor") loadMonitor();
  }, 10000);
})();