/* Customer Intelligence FE — vanilla JS (no build step). */

/* --------------------------------------------------------------------------
 * render_hint payload renderer (mirrors apps/agent/render.py + FE contract)
 * ------------------------------------------------------------------------ */
const KNOWN_KINDS = ["table", "chart", "cards", "qa", "markdown"];

function normalizeHint(hint) {
  if (!hint || typeof hint !== "object") return { kind: "markdown", data: null, spec: {}, text: "" };
  const payload = hint.payload && typeof hint.payload === "object" ? hint.payload : hint;
  let kind = payload.kind || hint.kind || "markdown";
  if (!KNOWN_KINDS.includes(kind)) kind = "markdown";
  return { kind, data: payload.data ?? null, spec: payload.spec || {}, text: payload.text || "" };
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function fmt(v) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return v.toLocaleString(undefined, { maximumFractionDigits: 2 });
  return String(v);
}

function renderTable(rows) {
  rows = rows || [];
  if (!rows.length) return '<p class="muted">No data.</p>';
  const cols = Object.keys(rows[0]);
  const head = cols.map(c => `<th>${escapeHtml(c)}</th>`).join("");
  const body = rows.map(r => `<tr>${cols.map(c => `<td>${escapeHtml(fmt(r[c]))}</td>`).join("")}</tr>`).join("");
  return `<div class="table-wrap"><table class="data"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function renderCards(kpis) {
  kpis = kpis || [];
  if (!kpis.length) return "";
  return kpis.map(k => `
    <div class="kpi">
      <div class="label">${escapeHtml(k.label || "")}</div>
      <div class="value">${escapeHtml(fmt(k.value))}</div>
      ${k.delta ? `<div class="muted">${escapeHtml(k.delta)}</div>` : ""}
    </div>`).join("");
}

function renderChart(el, payload) {
  const data = payload.data || [];
  const spec = payload.spec || {};
  const type = spec.type || "bar";
  if (!data.length) { el.innerHTML = '<p class="muted">No chart data.</p>'; return; }
  const xKey = spec.x || Object.keys(data[0])[0];
  const yKey = spec.y || Object.keys(data[0])[1] || Object.keys(data[0])[0];
  const x = data.map(r => r[xKey]);
  const y = data.map(r => r[yKey]);
  let trace;
  if (type === "pie") {
    trace = { type: "pie", labels: x, values: y };
  } else if (type === "line") {
    trace = { type: "scatter", mode: "lines+markers", x, y, name: yKey };
  } else {
    trace = { type: "bar", x, y };
  }
  Plotly.newPlot(el, [trace], { title: spec.title || "", margin: { t: 40, b: 30, l: 50, r: 10 }, height: 360 }, { responsive: true });
}

function renderHintEl(el, payload) {
  const h = normalizeHint(payload);
  if (h.kind === "table") { el.innerHTML = renderTable(h.data); }
  else if (h.kind === "cards") { el.innerHTML = `<div class="kpi-row">${renderCards(h.data)}</div>`; }
  else if (h.kind === "chart") { renderChart(el, h); }
  else { el.innerHTML = escapeHtml(h.text || ""); }
}

/* --------------------------------------------------------------------------
 * Answer rendering (summary + sections + render_hint)
 * ------------------------------------------------------------------------ */
function renderAnswer(answer) {
  if (!answer) return "";
  const conf = answer.confidence ? `<div class="conf">Confidence: ${escapeHtml(answer.confidence)}</div>` : "";
  const summary = answer.summary ? `<div><strong>${escapeHtml(answer.summary)}</strong></div>` : "";

  let visual = "";
  if (answer.render_hint) {
    const h = normalizeHint(answer.render_hint);
    if (h.kind === "table") visual = `<div class="section"><h4>Table</h4>${renderTable(h.data)}</div>`;
    else if (h.kind === "cards") visual = `<div class="kpi-row">${renderCards(h.data)}</div>`;
    else if (h.kind === "chart") visual = `<div class="section chart" data-chart='${JSON.stringify(h).replace(/'/g, "&#39;")}'></div>`;
  }

  const sections = [];
  for (const name of ["facts", "interpretation", "recommendation", "other_sections"]) {
    for (const sec of answer[name] || []) {
      const cites = (sec.citations || []).length
        ? `<div class="cites">📎 ${sec.citations.map(escapeHtml).join(", ")}</div>` : "";
      sections.push(`<div class="section"><h4>${escapeHtml(sec.heading || name)}</h4><p>${escapeHtml(sec.content)}</p>${cites}</div>`);
    }
  }
  return `${conf}${summary}${visual}${sections.join("")}`;
}

function afterAnswerRendered(container) {
  container.querySelectorAll(".chart").forEach(el => {
    try {
      const h = JSON.parse(el.dataset.chart);
      renderChart(el, h);
    } catch (e) { /* ignore malformed */ }
  });
}

/* --------------------------------------------------------------------------
 * Chat (SSE)
 * ------------------------------------------------------------------------ */
function sseChat(question, onAnswer, onError) {
  const body = {
    question,
    rerank_enabled: document.getElementById("rerank-toggle").checked,
    moderation_enabled: document.getElementById("moderation-toggle").checked,
    conversation: window.__chatHistory || [],
  };
  fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then(resp => {
    if (!resp.ok || !resp.body) { onError(`HTTP ${resp.status}`); return null; }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    const pump = () => reader.read().then(({ done, value }) => {
      if (done) return;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) >= 0) {
        const chunk = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const line = chunk.split("\n").find(l => l.startsWith("data: "));
        if (!line) continue;
        try {
          const msg = JSON.parse(line.slice(6));
          if (msg.error) onError(msg.message);
          else if (msg.answer) onAnswer(msg);
        } catch (e) { /* skip */ }
      }
      return pump();
    });
    return pump();
  }).catch(e => onError(String(e)));
}

/* --------------------------------------------------------------------------
 * Chat page — Agent Workspace (3-panel layout)
 * ------------------------------------------------------------------------ */
function appendChatMessage(role, html) {
  const el = document.createElement("div");
  el.className = `msg-row ${role}`;
  const isUser = role === "user";
  const icon = isUser ? "person" : "smart_toy";
  const name = isUser ? "User" : "CustIntel Agent";
  const now = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  el.innerHTML = `
    <div class="avatar"><span class="material-symbols-outlined">${icon}</span></div>
    <div class="msg-body">
      <div class="msg-meta">
        <span>${name}</span>
        <span class="time">${now}</span>
      </div>
      <div class="msg-content">${html}</div>
    </div>`;
  const messages = document.getElementById("messages");
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
  return el.querySelector(".msg-content");
}

function appendThinking() {
  const el = document.createElement("div");
  el.className = "msg-row assistant";
  el.innerHTML = `
    <div class="avatar"><span class="material-symbols-outlined">smart_toy</span></div>
    <div class="msg-body">
      <div class="msg-meta">
        <span>CustIntel Agent</span>
        <span class="time">${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
      </div>
      <div class="msg-content"><div class="thinking-dots"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div></div>
    </div>`;
  const messages = document.getElementById("messages");
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
  return el.querySelector(".msg-content");
}

/* --------------------------------------------------------------------------
 * Trace panel rendering (persistent timeline)
 * ------------------------------------------------------------------------ */
const TRACE_ICONS = {
  node: "account_tree",
  tool: "build",
  llm: "psychology",
  retrieval: "search",
  guardrail: "shield",
  memory: "memory",
  error: "error",
  internal: "settings",
};

function renderTrace(trace) {
  const body = document.getElementById("bts-content");
  const stats = document.getElementById("trace-stats");
  const status = document.getElementById("trace-status");

  if (!trace || !trace.spans || !trace.spans.length) {
    body.innerHTML = '<p class="muted">No trace data.</p>';
    stats.innerHTML = "";
    status.textContent = "Idle";
    status.classList.remove("running");
    return;
  }

  status.textContent = "Complete";
  status.classList.remove("running");

  const spans = trace.spans.map((s, i) => {
    const icon = TRACE_ICONS[s.kind] || "settings";
    const isLast = i === trace.spans.length - 1;
    const meta = s.metadata || {};
    const bits = [];
    if (meta.intent) bits.push(`intent: ${escapeHtml(meta.intent)}`);
    if (meta.routed_node) bits.push(`→ ${escapeHtml(meta.routed_node)}`);
    if (meta.tools) bits.push(`${meta.tools.length} tool(s)`);
    if (meta.used_llm) bits.push(`${meta.input_tokens ?? "?"} in / ${meta.output_tokens ?? "?"} out tokens`);

    const expandHtml = s.result
      ? `<div class="trace-expand"><div class="label">result (JSON)</div><pre>${escapeHtml(JSON.stringify(s.result, null, 2).slice(0, 2000))}</pre></div>`
      : "";

    const badgesHtml = bits.length
      ? `<div class="trace-badges">${bits.map(b => `<span class="trace-badge meta">${b}</span>`).join("")}</div>`
      : "";

    return `
      <div class="trace-item">
        <div class="trace-icon-col">
          <div class="trace-icon"><span class="material-symbols-outlined">${icon}</span></div>
          ${!isLast ? '<div class="trace-line"></div>' : ''}
        </div>
        <div class="trace-detail">
          <div class="trace-detail-head">
            <span class="trace-label">${escapeHtml(s.name)}</span>
            <span class="trace-time">${s.latency_ms != null ? s.latency_ms + 'ms' : ''}</span>
          </div>
          <div class="trace-sub">${escapeHtml(s.kind || '')}${s.status !== 'ok' ? ' · ' + escapeHtml(s.status) : ''}</div>
          ${badgesHtml}
          ${expandHtml}
        </div>
      </div>`;
  }).join("");

  const eventsHtml = (trace.events || []).length
    ? `<div style="margin-top:8px;"><div class="trace-sub" style="margin-bottom:6px;">Events</div>${trace.events.map(e => `<div class="trace-sub">• ${escapeHtml(e.event)}</div>`).join("")}</div>`
    : "";

  body.innerHTML = spans + eventsHtml;

  // Footer stats
  const totalMs = trace.spans.reduce((acc, s) => acc + (s.latency_ms || 0), 0);
  stats.innerHTML = `
    <span class="trace-stat">Total: ${Math.round(totalMs)}ms</span>
    <span class="trace-stat">Spans: ${trace.spans.length}</span>
  `;
}

function setTraceRunning() {
  const status = document.getElementById("trace-status");
  status.textContent = "Running";
  status.classList.add("running");
  document.getElementById("bts-content").innerHTML = "";
  document.getElementById("trace-stats").innerHTML = "";
}

/* --------------------------------------------------------------------------
 * Chat submission
 * ------------------------------------------------------------------------ */
function submitQuestion(question) {
  if (!question.trim()) return;
  const welcome = document.querySelector(".welcome");
  if (welcome) welcome.remove();
  appendChatMessage("user", escapeHtml(question));
  const assistant = appendThinking();
  setTraceRunning();
  sseChat(question, msg => {
    assistant.innerHTML = renderAnswer(msg.answer);
    afterAnswerRendered(assistant);
    renderTrace(msg.trace);
    const flags = (msg.guardrails || []).filter(g => g.severity === "block" || g.severity === "flag");
    if (flags.length) {
      const flagEl = document.createElement("div");
      flagEl.className = "flags";
      flagEl.innerHTML = flags.map(f => `<span>[${escapeHtml(f.rule)}] ${escapeHtml(f.message)}</span>`).join("");
      assistant.appendChild(flagEl);
    }
    window.__chatHistory = (window.__chatHistory || []).concat(
      [{ role: "user", content: question }, { role: "assistant", content: msg.answer ? msg.answer.summary : "" }]
    ).slice(-20);
    saveSession(question);
  }, err => {
    assistant.innerHTML = `<span class="muted">⚠ ${escapeHtml(err)}</span>`;
    renderTrace(null);
  });
}

/* --------------------------------------------------------------------------
 * Session management (localStorage)
 * ------------------------------------------------------------------------ */
function loadSessions() {
  try {
    return JSON.parse(localStorage.getItem("ci_sessions") || "[]");
  } catch { return []; }
}

function saveSession(question) {
  const sessions = loadSessions();
  const label = question.length > 40 ? question.slice(0, 40) + "…" : question;
  sessions.unshift({ label, ts: Date.now() });
  if (sessions.length > 20) sessions.length = 20;
  localStorage.setItem("ci_sessions", JSON.stringify(sessions));
  renderSessions(sessions);
}

function renderSessions(sessions) {
  const el = document.getElementById("sessions-list");
  if (!el) return;
  if (!sessions.length) {
    el.innerHTML = '<p class="muted" style="padding:8px 12px;font-size:12px;">No recent sessions.</p>';
    return;
  }
  el.innerHTML = sessions.map(s => `
    <button class="session-item" title="${escapeHtml(s.label)}">
      <span class="material-symbols-outlined">chat_bubble</span>
      <span class="session-label">${escapeHtml(s.label)}</span>
    </button>
  `).join("");
}

/* --------------------------------------------------------------------------
 * Chat page init
 * ------------------------------------------------------------------------ */
function initChat() {
  const form = document.getElementById("composer");
  const input = document.getElementById("question-input");

  // Auto-resize textarea
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 200) + "px";
  });

  form.addEventListener("submit", e => {
    e.preventDefault();
    submitQuestion(input.value);
    input.value = "";
    input.style.height = "auto";
  });

  document.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => submitQuestion(chip.dataset.q));
  });

  // New session button
  document.getElementById("new-session").addEventListener("click", () => {
    window.__chatHistory = [];
    const messages = document.getElementById("messages");
    messages.innerHTML = `
      <div class="welcome">
        <div class="context-chip"><span class="dot"></span> Customer Intelligence Analyst Ready</div>
        <div class="welcome-title">How can I help?</div>
        <div class="welcome-sub">Ask about feedback themes, customer risk, revenue at risk, or any customer.</div>
        <div id="suggestions" class="suggestions">
          <button class="chip" data-q="What are the top feedback themes?">Top feedback themes</button>
          <button class="chip" data-q="Which customers need attention?">Which customers need attention?</button>
          <button class="chip" data-q="How much revenue is at risk?">Revenue at risk</button>
          <button class="chip" data-q="What is happening with customer CUST-0001?">CUST-0001</button>
        </div>
      </div>`;
    // Re-bind chip clicks
    document.querySelectorAll(".chip").forEach(chip => {
      chip.addEventListener("click", () => submitQuestion(chip.dataset.q));
    });
    renderTrace(null);
  });

  // Load sessions
  renderSessions(loadSessions());
}

/* --------------------------------------------------------------------------
 * Dashboard / customer / admin pages
 * ------------------------------------------------------------------------ */
function renderKpis(container, kpis) {
  container.innerHTML = renderCards(kpis);
}

function initDashboard() {
  fetch("/api/dashboard").then(r => r.json()).then(d => {
    const kpi = d.kpis && d.kpis[0];
    if (kpi) renderKpis(document.getElementById("kpis"), [
      { label: "MRR", value: kpi.mrr },
      { label: "Customers", value: kpi.customers },
      { label: "Churned", value: kpi.churned },
      { label: "Open tickets", value: kpi.open_tickets },
    ]);
    if (d.themes) renderChart(document.getElementById("chart-themes"),
      { kind: "chart", data: d.themes, spec: { type: "bar", x: "theme", y: "count", title: "Themes" } });
    if (d.tickets_by_category) renderChart(document.getElementById("chart-tickets"),
      { kind: "chart", data: d.tickets_by_category, spec: { type: "bar", x: "category", y: "count", title: "Tickets" } });
    if (d.risk) document.getElementById("table-risk").innerHTML = renderTable(d.risk);
  });
}

function initCustomer() {
  const load = id => {
    fetch(`/api/customer/${encodeURIComponent(id)}`).then(r => r.json()).then(d => {
      const p = d.profile;
      document.getElementById("customer-profile").innerHTML = p
        ? renderCards([
            { label: "Status", value: p.account_status },
            { label: "Plan", value: p.subscription_plan },
            { label: "Revenue", value: p.monthly_revenue },
            { label: "Segment", value: p.customer_segment },
          ])
        : '<p class="muted">Customer not found.</p>';
      document.getElementById("customer-tickets").innerHTML = renderTable(d.tickets);
      document.getElementById("customer-feedback").innerHTML = renderTable(
        d.feedback.map(f => ({ feedback_id: f.feedback_id, created_at: String(f.created_at || "").slice(0, 10), rating: f.rating, feedback_text: f.feedback_text }))
      );
      if (d.usage_trend && d.usage_trend.length) renderChart(document.getElementById("customer-usage"),
        { kind: "chart", data: d.usage_trend, spec: { type: "line", x: "date", y: "sessions", title: "Sessions" } });
      else document.getElementById("customer-usage").innerHTML = '<p class="muted">No usage data.</p>';
    });
  };
  document.getElementById("customer-form").addEventListener("submit", e => {
    e.preventDefault();
    load(document.getElementById("customer-id").value.trim());
  });
  load(document.getElementById("customer-id").value);
}

function initAdmin() {
  fetch("/api/admin").then(r => r.json()).then(d => {
    document.getElementById("admin-quality").innerHTML = renderTable(d.quality_report.map(q => ({
      rule: q.rule, table: q.table_name, count: q.count,
    })));
    document.getElementById("admin-embedding").innerHTML = renderTable(d.embedding_meta) || '<p class="muted">No embeddings yet.</p>';
    document.getElementById("admin-agent").innerHTML =
      `<p>Memory entries: <strong>${d.memory_count}</strong></p><p>Traces: <strong>${d.trace_count}</strong></p>`;
    document.getElementById("admin-tables").innerHTML = renderTable(
      d.tables.map(t => ({ schema: t.table_schema, table: t.table_name }))
    );
  });
}

/* boot */
document.addEventListener("DOMContentLoaded", () => {
  if (window.PAGE === "dashboard") initDashboard();
  else if (window.PAGE === "customer") initCustomer();
  else if (window.PAGE === "admin") initAdmin();
  else initChat();
});