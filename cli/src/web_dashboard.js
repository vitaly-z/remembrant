// -----------------------------------------------------------------------
// State
// -----------------------------------------------------------------------
let allSessions = [];
let currentProject = "";
let sortField = "started_at";
let sortAsc = false;
let searchMode = "text";
let activeAgents = new Set(["claude_code", "codex", "gemini"]);
let factsActiveOnly = true;
let charts = {};
const AGENT_COLORS = {
  claude: "#58a6ff",
  claude_code: "#58a6ff",
  codex: "#00ff9c",
  gemini: "#bc8cff",
};

// -----------------------------------------------------------------------
// API helpers
// -----------------------------------------------------------------------
async function api(path, opts) {
  const res = await fetch("/api/" + path, opts);
  if (!res.ok) throw new Error("API " + res.status);
  return res.json();
}
async function apiPut(path, body) {
  return api(path, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
async function apiPost(path, body) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
async function apiDel(path) {
  return api(path, { method: "DELETE" });
}
function esc(value) {
  return value
    ? String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
    : "";
}
function jsArg(value) {
  return String(value ?? "")
    .replace(/\\/g, "\\\\")
    .replace(/'/g, "\\'")
    .replace(/"/g, "&quot;")
    .replace(/\r/g, "\\r")
    .replace(/\n/g, "\\n");
}
function parseUtcDate(value) {
  const raw = String(value ?? "");
  const iso = raw.includes(" ") ? raw.replace(" ", "T") : raw;
  const hasZone = /(?:Z|[+-]\d\d:?\d\d)$/i.test(iso);
  return new Date(hasZone ? iso : `${iso}Z`);
}
function formatIsoDay(value) {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(value ?? ""));
  if (!match) return String(value ?? "");
  const months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
  ];
  const month = months[Number(match[2]) - 1] ?? match[2];
  return `${month} ${Number(match[3])}`;
}
function fmtDate(s) {
  try {
    return esc(
      parseUtcDate(s).toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
      }),
    );
  } catch {
    return esc(s || "");
  }
}
function fmtDateTime(s) {
  try {
    return esc(parseUtcDate(s).toLocaleString());
  } catch {
    return esc(s || "");
  }
}
function fmtNum(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1000) return (n / 1000).toFixed(1) + "k";
  return String(n);
}
function confClass(c) {
  return c >= 0.8 ? "conf-high" : c >= 0.5 ? "conf-mid" : "conf-low";
}
function confBar(c) {
  const pct = Math.round(c * 100);
  return (
    '<span class="conf-bar"><span class="conf-track"><span class="conf-fill ' +
    confClass(c) +
    '" style="width:' +
    pct +
    '%"></span></span><span class="dim" style="font-size:10px">' +
    pct +
    "%</span></span>"
  );
}
function agentTag(a) {
  return (
    '<span class="agent-tag agent-' +
    safeToken(a || "unknown") +
    '">' +
    esc(a || "unknown") +
    "</span>"
  );
}
function safeToken(value) {
  return String(value || "unknown")
    .toLowerCase()
    .replace(/[^a-z0-9_.-]/g, "-");
}
function memoryTags(m) {
  return Array.isArray(m.tags) && m.tags.length
    ? '<div class="card-footer">tags: ' + m.tags.map(esc).join(", ") + "</div>"
    : "";
}

// Day.js relative time
if (window.dayjs && dayjs.extend) {
  dayjs.extend(window.dayjs_plugin_relativeTime);
}
function relTime(s) {
  try {
    return dayjs(parseUtcDate(s)).fromNow();
  } catch {
    return fmtDate(s);
  }
}

function renderSparkline(data, color) {
  if (!data || !data.length) return "";
  const w = 60,
    h = 18,
    pad = 1;
  const vals = data.map((d) => (typeof d === "number" ? d : d.value || 0));
  const max = Math.max(...vals, 1);
  const min = Math.min(...vals, 0);
  const range = max - min || 1;
  const pts = vals
    .map((v, i) => {
      const x = pad + (i / (vals.length - 1 || 1)) * (w - 2 * pad);
      const y = h - pad - ((v - min) / range) * (h - 2 * pad);
      return x.toFixed(1) + "," + y.toFixed(1);
    })
    .join(" ");
  return (
    '<svg width="' +
    w +
    '" height="' +
    h +
    '" viewBox="0 0 ' +
    w +
    " " +
    h +
    '"><polyline points="' +
    pts +
    '" fill="none" stroke="' +
    (color || "var(--accent)") +
    '" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'
  );
}

function trendIndicator(current, previous) {
  if (previous === 0 && current === 0)
    return '<span class="trend trend-flat">&rarr; steady</span>';
  if (previous === 0) return '<span class="trend trend-up">&uarr; new</span>';
  const pct = Math.round(((current - previous) / previous) * 100);
  if (pct > 0) return '<span class="trend trend-up">&uarr; ' + pct + "%</span>";
  if (pct < 0)
    return (
      '<span class="trend trend-down">&darr; ' + Math.abs(pct) + "%</span>"
    );
  return '<span class="trend trend-flat">&rarr; steady</span>';
}

async function loadBriefing() {
  try {
    const b = await api("briefing");
    document.getElementById("briefingHeadline").innerHTML = esc(
      b.headline || "",
    ).replace(/(\d+)/g, "<strong>$1</strong>");
    // Metrics row
    const mEl = document.getElementById("briefingMetrics");
    const metrics = b.metrics || {};
    let mHtml = "";
    for (const [key, m] of Object.entries(metrics)) {
      if (!m) continue;
      mHtml +=
        '<div class="briefing-metric"><div class="briefing-metric-value">' +
        fmtNum(m.current || 0) +
        " " +
        trendIndicator(m.current || 0, m.previous || 0) +
        '</div><div class="briefing-metric-label">' +
        esc(key) +
        " today</div></div>";
    }
    if (b.active_agents && b.active_agents.length) {
      mHtml +=
        '<div class="briefing-metric"><div class="briefing-metric-value" style="font-size:13px">' +
        b.active_agents.map((a) => agentTag(a)).join(" ") +
        '</div><div class="briefing-metric-label">active agents</div></div>';
    }
    mEl.innerHTML = mHtml;

    // Sparklines on stat cards
    const sparks = b.sparklines || {};
    if (sparks.sessions)
      document.getElementById("sparkSessions").innerHTML = renderSparkline(
        sparks.sessions,
        "var(--accent)",
      );
    if (sparks.memories)
      document.getElementById("sparkMemories").innerHTML = renderSparkline(
        sparks.memories,
        "var(--cyan)",
      );
    if (sparks.decisions)
      document.getElementById("sparkDecisions").innerHTML = renderSparkline(
        sparks.decisions,
        "var(--purple)",
      );

    // Trend indicators on stat cards
    if (metrics.sessions)
      document.getElementById("trendSessions").innerHTML = trendIndicator(
        metrics.sessions.current || 0,
        metrics.sessions.previous || 0,
      );
    if (metrics.memories)
      document.getElementById("trendMemories").innerHTML = trendIndicator(
        metrics.memories.current || 0,
        metrics.memories.previous || 0,
      );
    if (metrics.decisions)
      document.getElementById("trendDecisions").innerHTML = trendIndicator(
        metrics.decisions.current || 0,
        metrics.decisions.previous || 0,
      );

    // Digest grid (merged from loadFullBriefing)
    const digest = document.getElementById("briefingDigest");
    let dHtml = "";
    if (b.project_breakdown && b.project_breakdown.length) {
      dHtml +=
        '<div class="digest-card"><div class="digest-card-title">projects today</div><ul class="digest-list">';
      b.project_breakdown.forEach((p) => {
        let name = (p.project || "unknown").split("/").pop();
        dHtml +=
          "<li><strong>" +
          esc(name) +
          "</strong> &mdash; " +
          (p.sessions || 0) +
          " sessions, " +
          fmtNum(p.tokens || 0) +
          " tokens</li>";
      });
      dHtml += "</ul></div>";
    }
    if (b.decisions_today && b.decisions_today.length) {
      dHtml +=
        '<div class="digest-card"><div class="digest-card-title">decisions today</div><ul class="digest-list">';
      b.decisions_today.slice(0, 5).forEach((d) => {
        dHtml +=
          "<li>" +
          esc(d.what || "") +
          (d.why
            ? ' <span class="dim">&mdash; ' + esc(d.why) + "</span>"
            : "") +
          "</li>";
      });
      dHtml += "</ul></div>";
    }
    if (b.new_facts && b.new_facts.length) {
      dHtml +=
        '<div class="digest-card"><div class="digest-card-title">new facts learned</div><ul class="digest-list">';
      b.new_facts.slice(0, 5).forEach((f) => {
        dHtml +=
          "<li><strong>" +
          esc(f.subject || "") +
          "</strong> " +
          esc(f.predicate || "") +
          " " +
          esc(f.object || "") +
          "</li>";
      });
      dHtml += "</ul></div>";
    }
    if (b.top_files && b.top_files.length) {
      dHtml +=
        '<div class="digest-card"><div class="digest-card-title">hottest files</div><ul class="digest-list">';
      b.top_files.slice(0, 5).forEach((f) => {
        let path = f.file_path || f.path || "";
        let name = path.split("/").pop() || path;
        dHtml +=
          '<li><span style="color:var(--cyan)">' +
          esc(name) +
          "</span> &mdash; " +
          (f.change_frequency || f.changes || 0) +
          " changes</li>";
      });
      dHtml += "</ul></div>";
    }
    if (dHtml) {
      digest.innerHTML = dHtml;
      digest.style.display = "";
    }
  } catch (e) {
    // Show welcome message on error
    document.getElementById("briefingHeadline").innerHTML =
      "<strong>Welcome to Remembrant</strong>";
    document.getElementById("briefingMetrics").innerHTML =
      '<div style="font-size:12px;color:var(--text-secondary)">Your AI agent activity will appear here once you ingest session data.</div>';
    console.log("Briefing not available:", e.message);
  }
}

async function loadAttention() {
  let typeLabels = {
    conflict: "Cross-agent conflict",
    stale_memory: "Stale memory",
    high_churn: "Frequently changed file",
    contradictory_fact: "Contradicting facts",
  };
  try {
    const data = await api("attention");
    const items = data.items || [];
    const bar = document.getElementById("attentionBar");
    bar.style.display = "";
    if (!items.length) {
      bar.innerHTML =
        '<div style="font-size:11px;color:var(--text-dim);padding:8px 0">&#x2713; No issues detected</div>';
      return;
    }
    bar.innerHTML = items
      .map((item) => {
        const sev = item.severity || "medium";
        const type = item.type || "unknown";
        const label = typeLabels[type] || type.replace(/_/g, " ");
        return (
          '<div class="attention-item severity-' +
          sev +
          '">' +
          '<div class="attention-type type-' +
          type +
          '">' +
          esc(label) +
          "</div>" +
          '<div class="attention-title">' +
          esc(item.title || "") +
          "</div>" +
          '<div class="attention-detail">' +
          esc(item.detail || "") +
          "</div></div>"
        );
      })
      .join("");
  } catch (e) {
    console.log("Attention not available:", e.message);
  }
}

// -----------------------------------------------------------------------
// Tab navigation
// -----------------------------------------------------------------------
function switchTab(tab) {
  document
    .querySelectorAll(".nav-item")
    .forEach((n) => n.classList.remove("active"));
  document
    .querySelectorAll(".tab-content")
    .forEach((p) => p.classList.remove("active"));
  const navEl = document.querySelector('.nav-item[data-tab="' + tab + '"]');
  if (navEl) navEl.classList.add("active");
  document.getElementById(tab + "Panel").classList.add("active");
  if (tab === "home") {
    loadBriefing();
    loadAttention();
    loadHomeActivity();
  }
  if (tab === "sessions") loadSessions(currentProject);
  if (tab === "memories") loadMemories();
  if (tab === "decisions") loadDecisions();
  if (tab === "facts") loadFacts();
  if (tab === "analytics") loadAnalytics();
}

// -----------------------------------------------------------------------
// Stats
// -----------------------------------------------------------------------
async function loadStats() {
  try {
    const d = await api("stats");
    let today = d.today || null;
    function statHtml(todayVal, totalVal) {
      if (today && typeof todayVal === "number") {
        return (
          fmtNum(todayVal) +
          '</div><div style="font-size:10px;color:var(--text-dim)">' +
          fmtNum(totalVal) +
          " total"
        );
      }
      return fmtNum(totalVal);
    }
    document.getElementById("statSessions").innerHTML = statHtml(
      today ? today.sessions : null,
      d.sessions || 0,
    );
    document.getElementById("statMemories").innerHTML = statHtml(
      today ? today.memories : null,
      d.memories || 0,
    );
    document.getElementById("statDecisions").innerHTML = statHtml(
      today ? today.decisions : null,
      d.decisions || 0,
    );
    document.getElementById("statToolCalls").innerHTML = statHtml(
      today ? today.tool_calls : null,
      d.tool_calls || 0,
    );
    document.getElementById("statProjects").textContent = d.projects || 0;
    document.getElementById("navSessions").textContent = fmtNum(
      d.sessions || 0,
    );
    document.getElementById("navMemories").textContent = fmtNum(
      d.memories || 0,
    );
    document.getElementById("navDecisions").textContent = fmtNum(
      d.decisions || 0,
    );
    document.getElementById("sbProjects").textContent =
      (d.projects || 0) + " projects";
    const factCount = d.active_facts ?? d.facts ?? 0;
    document.getElementById("statFacts").textContent = fmtNum(factCount);
    document.getElementById("navFacts").textContent = fmtNum(factCount);
    document.getElementById("sbRefresh").textContent =
      "updated " + new Date().toLocaleTimeString();
  } catch (e) {
    document.getElementById("statSessions").textContent = "ERR";
  }
}

async function loadProjects() {
  try {
    const projects = await api("projects");
    const sel = document.getElementById("projectFilter");
    projects.forEach((p) => {
      const o = document.createElement("option");
      o.value = p;
      o.textContent = p;
      sel.appendChild(o);
    });
    document.getElementById("sbAgents").textContent =
      projects.length + " projects tracked";
  } catch (e) {
    console.error("Projects:", e);
  }
}

// -----------------------------------------------------------------------
// Sessions
// -----------------------------------------------------------------------
async function loadSessions(project) {
  try {
    let url = "sessions?limit=200";
    if (project) url += "&project=" + encodeURIComponent(project);
    const sessions = await api(url);
    allSessions = sessions;
    document.getElementById("sessionsCount").textContent =
      "(" + sessions.length + ")";
    renderSessions(sessions);
  } catch (e) {
    document.getElementById("sessionsContent").innerHTML =
      '<div class="empty-state"><div class="empty-icon">&#x2297;</div><div class="empty-text">Error: ' +
      esc(String(e)) +
      "</div></div>";
  }
}

function getDayLabel(dateStr) {
  try {
    let d = parseUtcDate(dateStr);
    let now = new Date();
    let today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    let yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    let sessionDay = new Date(d.getFullYear(), d.getMonth(), d.getDate());
    if (sessionDay.getTime() === today.getTime()) return "Today";
    if (sessionDay.getTime() === yesterday.getTime()) return "Yesterday";
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  } catch {
    return "";
  }
}

function renderSessionItem(s) {
  let agent = safeToken(s.agent);
  let timeStr = s.started_at ? relTime(s.started_at) : "-";
  let sum = s.summary || "\u2014";
  let project = s.project_id ? s.project_id.split("/").pop() : "";
  return (
    '<div class="feed-item" onclick="showDetail(\'' +
    jsArg(s.id) +
    "')\">" +
    '<div class="feed-icon feed-icon-' +
    agent +
    '">' +
    (agent === "claude" || agent === "claude_code"
      ? "&#x2318;"
      : agent === "codex"
        ? "&#x25C8;"
        : "&#x25C6;") +
    "</div>" +
    '<div class="feed-body">' +
    '<div class="feed-header"><span class="feed-agent" style="color:' +
    (AGENT_COLORS[agent] || "var(--text-primary)") +
    '">' +
    esc(s.agent || "unknown") +
    (project
      ? ' <span class="dim" style="font-weight:400;font-size:10px">' +
        esc(project) +
        "</span>"
      : "") +
    "</span>" +
    '<span class="feed-time">' +
    timeStr +
    "</span></div>" +
    '<div class="feed-summary">' +
    esc(sum) +
    "</div>" +
    '<div class="feed-meta">' +
    (s.message_count
      ? "<span>&#x2709; " + s.message_count + " msgs</span>"
      : "") +
    (s.tool_call_count
      ? "<span>&#x2699; " + s.tool_call_count + " tools</span>"
      : "") +
    (s.total_tokens
      ? "<span>&#x26A1; " + fmtNum(s.total_tokens) + " tokens</span>"
      : "") +
    (s.duration_minutes
      ? "<span>&#x23F1; " + s.duration_minutes + "min</span>"
      : "") +
    "</div></div></div>"
  );
}

function renderSessions(sessions) {
  const filtered = sessions.filter((s) => activeAgents.has(safeToken(s.agent)));
  const el = document.getElementById("sessionsContent");
  if (!filtered.length) {
    el.innerHTML =
      '<div class="empty-state">' +
      '<div class="empty-icon">&#x229E;</div>' +
      '<div class="empty-title">No sessions yet</div>' +
      '<div class="empty-desc">Sessions track your AI coding agent conversations. Start by ingesting agent data:</div>' +
      '<div class="empty-commands"><code>rem ingest</code> <span class="cmd-comment"># Import all detected agent sessions</span></div></div>';
    return;
  }
  let html = '<div class="activity-feed">';
  let lastDay = "";
  filtered.forEach((s) => {
    let day = s.started_at ? getDayLabel(s.started_at) : "";
    if (day && day !== lastDay) {
      html += '<div class="feed-day-header">' + esc(day) + "</div>";
      lastDay = day;
    }
    html += renderSessionItem(s);
  });
  html += "</div>";
  el.innerHTML = html;
}

async function loadHomeActivity() {
  try {
    let url = "sessions?limit=10";
    if (currentProject) url += "&project=" + encodeURIComponent(currentProject);
    let sessions = await api(url);
    let el = document.getElementById("homeActivityContent");
    let filtered = sessions.filter((s) => activeAgents.has(safeToken(s.agent)));
    if (!filtered.length) {
      el.innerHTML =
        '<div class="empty-state"><div class="empty-icon">&#x229E;</div><div class="empty-title">No recent activity</div>' +
        '<div class="empty-desc">Ingest agent data to see activity here.</div></div>';
      return;
    }
    let html = '<div class="activity-feed">';
    let lastDay = "";
    filtered.forEach((s) => {
      let day = s.started_at ? getDayLabel(s.started_at) : "";
      if (day && day !== lastDay) {
        html += '<div class="feed-day-header">' + esc(day) + "</div>";
        lastDay = day;
      }
      html += renderSessionItem(s);
    });
    html += "</div>";
    el.innerHTML = html;
  } catch (e) {
    document.getElementById("homeActivityContent").innerHTML =
      '<div class="empty-state"><div class="empty-icon">&#x2297;</div><div class="empty-text">Error loading activity</div></div>';
  }
  // Also load recent decisions for home
  try {
    let dUrl = "decisions";
    if (currentProject)
      dUrl += "?project=" + encodeURIComponent(currentProject);
    let decisions = await api(dUrl);
    let dEl = document.getElementById("homeDecisionsContent");
    let recent = decisions.slice(0, 5);
    if (!recent.length) {
      dEl.innerHTML =
        '<div class="empty-state" style="padding:20px"><div class="empty-title" style="font-size:12px">No decisions yet</div></div>';
      return;
    }
    dEl.innerHTML = recent
      .map((d) => {
        let dt = d.created_at ? fmtDate(d.created_at) : "";
        return (
          '<div class="decision-card">' +
          '<div class="card-meta"><span class="card-type">' +
          esc(d.decision_type || "decision") +
          '</span><span class="card-date">' +
          dt +
          "</span></div>" +
          '<div class="card-body" style="color:var(--text-primary);font-weight:500">' +
          esc(d.what) +
          "</div>" +
          (d.why
            ? '<div class="card-body" style="margin-top:4px">\u21B3 ' +
              esc(d.why) +
              "</div>"
            : "") +
          "</div>"
        );
      })
      .join("");
  } catch (e) {
    document.getElementById("homeDecisionsContent").innerHTML =
      '<div class="empty-state" style="padding:20px"><div class="empty-text">Could not load decisions</div></div>';
  }
}

function sortBy(field) {
  if (sortField === field) sortAsc = !sortAsc;
  else {
    sortField = field;
    sortAsc = true;
  }
  allSessions.sort((a, b) => {
    const av = a[field] || "",
      bv = b[field] || "";
    return sortAsc ? (av > bv ? 1 : -1) : av < bv ? 1 : -1;
  });
  renderSessions(allSessions);
}

async function showDetail(id) {
  try {
    const data = await api("sessions/" + encodeURIComponent(id));
    const s = data.session;
    const tcs = data.tool_calls || [];
    let html =
      '<div class="detail-grid">' +
      '<div class="detail-item"><label>id</label><div class="val dim" style="font-size:11px;word-break:break-all">' +
      esc(s.id) +
      "</div></div>" +
      '<div class="detail-item"><label>agent</label><div class="val">' +
      agentTag(s.agent) +
      "</div></div>" +
      '<div class="detail-item"><label>project</label><div class="val">' +
      esc(s.project_id || "-") +
      "</div></div>" +
      '<div class="detail-item"><label>duration</label><div class="val">' +
      (s.duration_minutes || "-") +
      " min</div></div>" +
      '<div class="detail-item"><label>started</label><div class="val mono">' +
      fmtDateTime(s.started_at) +
      "</div></div>" +
      '<div class="detail-item"><label>ended</label><div class="val mono">' +
      fmtDateTime(s.ended_at) +
      "</div></div>" +
      '<div class="detail-item"><label>messages</label><div class="val mono">' +
      (s.message_count || "-") +
      "</div></div>" +
      '<div class="detail-item"><label>tokens</label><div class="val mono">' +
      (s.total_tokens ? fmtNum(s.total_tokens) : "-") +
      "</div></div></div>";
    if (s.summary)
      html +=
        '<div style="margin-bottom:14px"><label style="font-size:10px;color:var(--text-dim);text-transform:uppercase;letter-spacing:1px">summary</label><div style="margin-top:4px;color:var(--text-secondary);font-size:12px;line-height:1.6">' +
        esc(s.summary) +
        "</div></div>";
    if (s.files_changed && s.files_changed.length)
      html +=
        '<div style="margin-bottom:14px"><label style="font-size:10px;color:var(--text-dim);text-transform:uppercase;letter-spacing:1px">files changed (' +
        s.files_changed.length +
        ')</label><ul class="files-list">' +
        s.files_changed.map((f) => "<li>" + esc(f) + "</li>").join("") +
        "</ul></div>";
    if (tcs.length) {
      html +=
        '<div><label style="font-size:10px;color:var(--text-dim);text-transform:uppercase;letter-spacing:1px">tool calls (' +
        tcs.length +
        ")</label>";
      tcs.slice(0, 50).forEach((tc) => {
        const cls = tc.success === false ? " failed" : "";
        const ts = tc.timestamp
          ? parseUtcDate(tc.timestamp).toLocaleTimeString()
          : "";
        html +=
          '<div class="tool-call-item' +
          cls +
          '"><div class="tc-header"><span>' +
          esc(tc.tool_name || "unknown") +
          '</span><span class="dim">' +
          ts +
          "</span></div>" +
          (tc.command
            ? '<div class="tc-cmd">' +
              esc(tc.command.substring(0, 150)) +
              (tc.command.length > 150 ? "\u2026" : "") +
              "</div>"
            : "") +
          (tc.error_message
            ? '<div class="tc-err">' + esc(tc.error_message) + "</div>"
            : "") +
          "</div>";
      });
      if (tcs.length > 50)
        html +=
          '<div class="dim" style="padding:8px;font-size:11px">...and ' +
          (tcs.length - 50) +
          " more</div>";
      html += "</div>";
    }
    document.getElementById("modalTitle").textContent =
      "Session \u2014 " + (s.agent || "");
    document.getElementById("detailContent").innerHTML = html;
    document.getElementById("detailModal").style.display = "block";
  } catch (e) {
    console.error("Detail:", e);
  }
}

// -----------------------------------------------------------------------
// Memories
// -----------------------------------------------------------------------
async function loadMemories() {
  const el = document.getElementById("memoriesContent");
  try {
    let url = "memories?limit=200";
    if (currentProject) url += "&project=" + encodeURIComponent(currentProject);
    const memories = await api(url);
    if (!memories.length) {
      el.innerHTML =
        '<div class="empty-state">' +
        '<div class="empty-icon">&#x25C8;</div>' +
        '<div class="empty-title">No memories extracted</div>' +
        '<div class="empty-desc">Memories are automatically created when agents are ingested. They capture key learnings, patterns, and context.</div>' +
        '<div class="empty-commands"><code>rem ingest</code> <span class="cmd-comment"># Memories are extracted automatically</span><br>' +
        '<code>rem note &quot;My custom note&quot;</code> <span class="cmd-comment"># Or add a manual memory</span></div></div>';
      return;
    }
    el.innerHTML = memories
      .map((m) => {
        const t = m.memory_type || "general";
        const d = m.created_at ? fmtDate(m.created_at) : "";
        const conf = typeof m.confidence === "number" ? m.confidence : 1.0;
        return (
          '<div class="memory-card" id="mem-' +
          esc(m.id) +
          '" data-tags="' +
          esc((m.tags || []).join(", ")) +
          '">' +
          '<div class="card-meta"><span class="card-type">' +
          esc(t) +
          "</span>" +
          '<div style="display:flex;align-items:center;gap:8px">' +
          confBar(conf) +
          '<span class="card-date">' +
          d +
          "</span>" +
          '<div class="card-actions"><button onclick="editMemory(\'' +
          jsArg(m.id) +
          '\',event)" title="edit">&#x270E;</button><button class="del" onclick="deleteMemory(\'' +
          jsArg(m.id) +
          '\',event)" title="delete">&#x2715;</button></div>' +
          "</div></div>" +
          '<div class="card-body">' +
          esc(m.content) +
          "</div>" +
          (m.project_id
            ? '<div class="card-footer">project: ' +
              esc(m.project_id) +
              "</div>"
            : "") +
          memoryTags(m) +
          "</div>"
        );
      })
      .join("");
  } catch (e) {
    el.innerHTML =
      '<div class="empty-state"><div class="empty-icon">&#x2297;</div><div class="empty-text">failed to load</div></div>';
  }
}

async function editMemory(id, ev) {
  ev.stopPropagation();
  const card = document.getElementById("mem-" + id);
  if (card.querySelector(".edit-form")) return;
  const body = card.querySelector(".card-body");
  const current = body.textContent;
  const form = document.createElement("div");
  form.className = "edit-form";
  form.innerHTML =
    "<textarea>" +
    esc(current) +
    "</textarea>" +
    '<input type="text" class="tag-input" value="' +
    esc(card.dataset.tags || "") +
    '" placeholder="tags (comma-separated)" style="width:calc(100% - 22px);margin-top:6px;background:var(--bg-primary);border:1px solid var(--border);color:var(--text-primary);padding:7px;border-radius:4px;font-family:inherit;font-size:11px" />' +
    '<div class="edit-row"><span style="font-size:10px;color:var(--text-dim)">confidence:</span><input type="range" min="0" max="100" value="80" id="confSlider-' +
    id +
    '"><span id="confVal-' +
    id +
    '" style="font-size:10px;color:var(--text-dim);width:30px">80%</span></div>' +
    '<div style="display:flex;gap:6px;justify-content:flex-end;margin-top:8px"><button class="btn btn-sm btn-ghost" onclick="this.closest(\'.edit-form\').remove()">cancel</button><button class="btn btn-sm" onclick="saveMemory(\'' +
    esc(id) +
    "',this)\">save</button></div>";
  card.appendChild(form);
  const slider = form.querySelector("input[type=range]");
  const valEl = form.querySelector("[id^=confVal]");
  slider.oninput = () => {
    valEl.textContent = slider.value + "%";
  };
}

async function saveMemory(id, btn) {
  const form = btn.closest(".edit-form");
  const content = form.querySelector("textarea").value;
  const conf = parseInt(form.querySelector("input[type=range]").value) / 100;
  const tags = (form.querySelector(".tag-input").value || "")
    .split(",")
    .map((tag) => tag.trim())
    .filter(Boolean);
  try {
    await apiPut("memories/" + encodeURIComponent(id), {
      content,
      confidence: conf,
      tags,
    });
    loadMemories();
  } catch (e) {
    alert("Failed: " + e.message);
  }
}

async function deleteMemory(id, ev) {
  ev.stopPropagation();
  if (!confirm("Delete this memory?")) return;
  try {
    await apiDel("memories/" + encodeURIComponent(id));
    loadMemories();
  } catch (e) {
    alert("Failed: " + e.message);
  }
}

// -----------------------------------------------------------------------
// Facts
// -----------------------------------------------------------------------
async function loadFacts() {
  const el = document.getElementById("factsContent");
  el.innerHTML = '<div class="loading-state">loading</div>';
  try {
    const url = "facts?limit=200&active_only=" + factsActiveOnly;
    if (currentProject) url += "&project=" + encodeURIComponent(currentProject);
    const facts = await api(url);
    document.getElementById("factsCount").textContent =
      "(" + facts.length + ")";
    if (!facts.length) {
      el.innerHTML =
        '<div class="empty-state">' +
        '<div class="empty-icon">&#x25C6;</div>' +
        '<div class="empty-title">Knowledge base is empty</div>' +
        '<div class="empty-desc">Facts are structured knowledge triples (subject-predicate-object) that agents learn and share across sessions.</div>' +
        '<div class="empty-commands"><span class="cmd-comment"># Facts are auto-extracted during ingestion, or add manually:</span><br>' +
        "<code>rem ingest</code> to distill facts from agent sessions.</div></div>";
      return;
    }
    el.innerHTML =
      "<table><thead><tr><th>subject</th><th>predicate</th><th>object</th><th>confidence</th><th>agent</th><th>status</th></tr></thead><tbody>" +
      facts
        .map((f) => {
          const active = !f.invalid_at;
          const cls = active ? "fact-active" : "fact-inactive";
          return (
            '<tr class="' +
            cls +
            '" onclick="showFactHistory(\'' +
            jsArg(f.id) +
            '\')" style="cursor:pointer">' +
            "<td" +
            (active ? "" : ' class="fact-superseded"') +
            ">" +
            esc(f.subject) +
            "</td>" +
            '<td class="dim">' +
            esc(f.predicate) +
            "</td>" +
            "<td" +
            (active ? "" : ' class="fact-superseded"') +
            ">" +
            esc(f.object) +
            "</td>" +
            "<td>" +
            confBar(f.confidence || 1.0) +
            "</td>" +
            '<td class="dim">' +
            esc(f.source_agent || "-") +
            "</td>" +
            '<td><span class="status-badge">' +
            (active ? "active" : "superseded") +
            "</span></td></tr>"
          );
        })
        .join("") +
      "</tbody></table>";
  } catch (e) {
    el.innerHTML =
      '<div class="empty-state"><div class="empty-icon">&#x2297;</div><div class="empty-text">failed to load facts</div></div>';
  }
}

function toggleFactsFilter() {
  factsActiveOnly = !factsActiveOnly;
  document.getElementById("factsToggle").textContent = factsActiveOnly
    ? "active only"
    : "show all";
  loadFacts();
}

async function showFactHistory(id) {
  try {
    const history = await api("facts/" + encodeURIComponent(id) + "/history");
    if (!Array.isArray(history) || !history.length) {
      alert("No history found");
      return;
    }
    const subject = history[0].subject;
    let html =
      '<h3 style="font-size:13px;margin-bottom:12px;color:var(--text-secondary)">Evolution of: <span style="color:var(--accent)">' +
      esc(subject) +
      "</span></h3>";
    html += '<div class="timeline">';
    history.forEach((f, i) => {
      const active = !f.invalid_at;
      const prev = i > 0 ? history[i - 1] : null;
      html +=
        '<div class="timeline-item' +
        (active ? "" : " superseded") +
        '">' +
        '<div class="timeline-meta">' +
        fmtDateTime(f.valid_at || f.created_at) +
        "<br>" +
        '<span style="color:' +
        (active ? "var(--accent)" : "var(--text-dim)") +
        '">' +
        (active ? "ACTIVE" : "superseded") +
        "</span>" +
        (f.source_agent ? "<br>" + esc(f.source_agent) : "") +
        "</div>" +
        '<div class="timeline-content">' +
        (prev && prev.object !== f.object
          ? '<div class="timeline-diff-del">' +
            esc(prev.predicate) +
            " " +
            esc(prev.object) +
            "</div>"
          : "") +
        "<div" +
        (prev && prev.object !== f.object ? ' class="timeline-diff-add"' : "") +
        ">" +
        esc(f.predicate) +
        " <strong>" +
        esc(f.object) +
        "</strong></div>" +
        '<div style="margin-top:4px">' +
        confBar(f.confidence || 1.0) +
        "</div>" +
        "</div></div>";
    });
    html += "</div>";
    document.getElementById("factModalTitle").textContent =
      "Fact History \u2014 " + subject;
    document.getElementById("factModalContent").innerHTML = html;
    document.getElementById("factModal").style.display = "block";
  } catch (e) {
    console.error("Fact history:", e);
  }
}

// -----------------------------------------------------------------------
// Decisions
// -----------------------------------------------------------------------
async function loadDecisions() {
  const el = document.getElementById("decisionsContent");
  try {
    let url = "decisions";
    if (currentProject) url += "?project=" + encodeURIComponent(currentProject);
    const decisions = await api(url);
    if (!decisions.length) {
      el.innerHTML =
        '<div class="empty-state">' +
        '<div class="empty-icon">&#x2298;</div>' +
        '<div class="empty-title">No decisions recorded</div>' +
        '<div class="empty-desc">Decisions capture the &quot;what&quot; and &quot;why&quot; of choices made by AI agents during coding sessions.</div>' +
        '<div class="empty-commands">Use <strong>+ record</strong> to add the first decision.</div></div>';
      return;
    }
    el.innerHTML = decisions
      .map((d) => {
        const dt = d.created_at ? fmtDate(d.created_at) : "";
        return (
          '<div class="decision-card">' +
          '<div class="card-meta"><span class="card-type">' +
          esc(d.decision_type || "decision") +
          '</span><span class="card-date">' +
          dt +
          "</span></div>" +
          '<div class="card-body" style="color:var(--text-primary);font-weight:500">' +
          esc(d.what) +
          "</div>" +
          (d.why
            ? '<div class="card-body" style="margin-top:4px">\u21B3 ' +
              esc(d.why) +
              "</div>"
            : "") +
          (d.alternatives && d.alternatives.length
            ? '<div class="card-footer">alternatives: ' +
              d.alternatives.map(esc).join(" \u00B7 ") +
              "</div>"
            : "") +
          (d.project_id
            ? '<div class="card-footer">project: ' +
              esc(d.project_id) +
              "</div>"
            : "") +
          "</div>"
        );
      })
      .join("");
  } catch (e) {
    el.innerHTML =
      '<div class="empty-state"><div class="empty-icon">&#x2297;</div><div class="empty-text">failed to load</div></div>';
  }
}

// -----------------------------------------------------------------------
// Analytics
// -----------------------------------------------------------------------
const chartDefaults = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      labels: {
        color: "#8b949e",
        font: { family: "JetBrains Mono", size: 10 },
      },
    },
  },
  scales: {
    x: {
      ticks: { color: "#484f58", font: { family: "JetBrains Mono", size: 9 } },
      grid: { color: "#21262d" },
    },
    y: {
      ticks: { color: "#484f58", font: { family: "JetBrains Mono", size: 9 } },
      grid: { color: "#21262d" },
    },
  },
};

async function loadAnalytics() {
  await Promise.all([
    loadAgentStats(),
    loadTimeline(),
    loadToolStats(),
    loadHotfiles(),
  ]);
}

async function loadAgentStats() {
  try {
    const agents = await api("stats/agents");
    if (!Array.isArray(agents)) return;
    // Metric cards
    const el = document.getElementById("agentMetrics");
    el.innerHTML = agents
      .map((a) => {
        const color =
          AGENT_COLORS[a.agent?.toLowerCase()] || "var(--text-primary)";
        return (
          '<div class="metric-card"><div class="metric-agent" style="color:' +
          color +
          '">' +
          esc(a.agent) +
          "</div>" +
          '<div class="metric-val">' +
          fmtNum(a.sessions || 0) +
          '</div><div class="metric-label">sessions</div>' +
          '<div style="margin-top:8px;font-size:12px;color:var(--text-secondary)">' +
          fmtNum(a.total_tokens || 0) +
          " tokens</div>" +
          '<div class="metric-label">' +
          (a.avg_duration ? Math.round(a.avg_duration) + " min avg" : "-") +
          "</div></div>"
        );
      })
      .join("");
    // Doughnut chart
    if (charts.agent) charts.agent.destroy();
    charts.agent = new Chart(document.getElementById("agentChart"), {
      type: "doughnut",
      data: {
        labels: agents.map((a) => a.agent),
        datasets: [
          {
            data: agents.map((a) => a.sessions || 0),
            backgroundColor: agents.map(
              (a) => AGENT_COLORS[a.agent?.toLowerCase()] || "#8b949e",
            ),
            borderWidth: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "right",
            labels: {
              color: "#8b949e",
              font: { family: "JetBrains Mono", size: 10 },
              padding: 12,
            },
          },
        },
      },
    });
    // Token bar chart
    if (charts.token) charts.token.destroy();
    charts.token = new Chart(document.getElementById("tokenChart"), {
      type: "bar",
      data: {
        labels: agents.map((a) => a.agent),
        datasets: [
          {
            label: "Tokens",
            data: agents.map((a) => a.total_tokens || 0),
            backgroundColor: agents.map(
              (a) => AGENT_COLORS[a.agent?.toLowerCase()] || "#8b949e",
            ),
            borderRadius: 4,
          },
        ],
      },
      options: {
        ...chartDefaults,
        indexAxis: "y",
        plugins: { ...chartDefaults.plugins, legend: { display: false } },
      },
    });
  } catch (e) {
    console.error("Agent stats:", e);
  }
}

async function loadTimeline() {
  try {
    const data = await api("stats/timeline?days=30");
    if (!Array.isArray(data) || !data.length) return;
    const dates = [...new Set(data.map((d) => d.date))].sort();
    const agentNames = [...new Set(data.map((d) => d.agent))];
    const datasets = agentNames.map((agent) => ({
      label: agent,
      data: dates.map((date) => {
        const m = data.find((d) => d.date === date && d.agent === agent);
        return m ? m.count : 0;
      }),
      borderColor: AGENT_COLORS[agent?.toLowerCase()] || "#8b949e",
      backgroundColor: (AGENT_COLORS[agent?.toLowerCase()] || "#8b949e") + "33",
      fill: true,
      tension: 0.3,
      borderWidth: 2,
      pointRadius: 1,
    }));
    if (charts.timeline) charts.timeline.destroy();
    charts.timeline = new Chart(document.getElementById("timelineChart"), {
      type: "line",
      data: {
        labels: dates.map((d) => {
          return formatIsoDay(d);
        }),
        datasets,
      },
      options: {
        ...chartDefaults,
        plugins: {
          ...chartDefaults.plugins,
          legend: {
            labels: {
              color: "#8b949e",
              font: { family: "JetBrains Mono", size: 10 },
            },
          },
        },
      },
    });
  } catch (e) {
    console.error("Timeline:", e);
  }
}

async function loadToolStats() {
  try {
    const tools = await api("stats/tools");
    if (!Array.isArray(tools) || !tools.length) return;
    const top10 = tools.slice(0, 10);
    const colors = [
      "#00ff9c",
      "#58a6ff",
      "#bc8cff",
      "#ffa657",
      "#ff7b72",
      "#e3b341",
      "#79c0ff",
      "#7ee787",
      "#d2a8ff",
      "#ffa198",
    ];
    if (charts.tool) charts.tool.destroy();
    charts.tool = new Chart(document.getElementById("toolChart"), {
      type: "doughnut",
      data: {
        labels: top10.map((t) => t.tool_name || "unknown"),
        datasets: [
          {
            data: top10.map((t) => t.count || 0),
            backgroundColor: colors,
            borderWidth: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "right",
            labels: {
              color: "#8b949e",
              font: { family: "JetBrains Mono", size: 9 },
              padding: 8,
            },
          },
        },
      },
    });
    // Table
    const el = document.getElementById("toolStatsContent");
    el.innerHTML =
      "<table><thead><tr><th>tool</th><th>calls</th><th>success rate</th><th>avg duration</th></tr></thead><tbody>" +
      tools
        .map((t) => {
          const rate =
            t.count > 0 ? Math.round((t.success_count / t.count) * 100) : 0;
          const rateColor =
            rate >= 90
              ? "var(--accent)"
              : rate >= 70
                ? "var(--yellow)"
                : "var(--red)";
          return (
            "<tr><td>" +
            esc(t.tool_name || "unknown") +
            '</td><td class="mono">' +
            fmtNum(t.count) +
            "</td>" +
            '<td class="mono" style="color:' +
            rateColor +
            '">' +
            rate +
            "%</td>" +
            '<td class="mono dim">' +
            (t.avg_duration_ms ? Math.round(t.avg_duration_ms) + "ms" : "-") +
            "</td></tr>"
          );
        })
        .join("") +
      "</tbody></table>";
  } catch (e) {
    console.error("Tool stats:", e);
  }
}

async function loadHotfiles() {
  try {
    const files = await api("hotfiles?limit=15");
    if (!Array.isArray(files) || !files.length) return;
    if (charts.hotfile) charts.hotfile.destroy();
    const labels = files.map((f) => {
      const p = f.file_path || f[0] || "";
      return p.length > 40 ? "..." + p.slice(-37) : p;
    });
    const data = files.map((f) => f.change_frequency || f[1] || 0);
    charts.hotfile = new Chart(document.getElementById("hotfileChart"), {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Changes",
            data,
            backgroundColor: "#00ff9c44",
            borderColor: "#00ff9c",
            borderWidth: 1,
            borderRadius: 3,
          },
        ],
      },
      options: {
        ...chartDefaults,
        indexAxis: "y",
        plugins: { ...chartDefaults.plugins, legend: { display: false } },
        scales: {
          ...chartDefaults.scales,
          y: {
            ...chartDefaults.scales.y,
            ticks: {
              ...chartDefaults.scales.y.ticks,
              font: { family: "JetBrains Mono", size: 8 },
            },
          },
        },
      },
    });
  } catch (e) {
    console.error("Hotfiles:", e);
  }
}

// -----------------------------------------------------------------------
// Search
// -----------------------------------------------------------------------
function setSearchMode(mode) {
  searchMode = mode;
  document
    .querySelectorAll(".mode-btn")
    .forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  const input = document.getElementById("searchInput");
  if (mode === "xpath") input.placeholder = '//Session[node~"auth"]/Decision';
  else if (mode === "facts")
    input.placeholder = "search facts by subject or object...";
  else input.placeholder = "search sessions, memories, facts...";
}

function toggleAgent(el, agent) {
  el.classList.toggle("active");
  if (activeAgents.has(agent)) activeAgents.delete(agent);
  else activeAgents.add(agent);
  renderSessions(allSessions);
}

async function performSearch() {
  const q = document.getElementById("searchInput").value.trim();
  if (!q) return;
  const el = document.getElementById("searchContent");
  el.innerHTML = '<div class="loading-state">searching</div>';
  document.getElementById("navSearchItem").style.display = "";
  switchTab("search");

  try {
    const eq = encodeURIComponent(q);
    let html = "";
    if (searchMode === "xpath") {
      const response = await api("search/xpath?q=" + eq + "&limit=20");
      const results = Array.isArray(response?.results) ? response.results : [];
      if (results.length) {
        html +=
          '<div class="section-label">xpath results (' +
          results.length +
          ")</div>";
        results.forEach((r) => {
          html +=
            '<div class="memory-card"><div class="card-meta"><span class="card-type">' +
            esc(r.node_type || r.result_type || "node") +
            "</span>" +
            '<span class="card-date">score: ' +
            (r.weight || r.score || 0).toFixed(2) +
            "</span></div>" +
            '<div class="card-body">' +
            esc(r.name || r.content || "") +
            "</div>" +
            (r.path
              ? '<div class="card-footer">path: ' +
                r.path.map(esc).join(" \u2192 ") +
                "</div>"
              : "") +
            "</div>";
        });
      } else
        html =
          '<div class="empty-state"><div class="empty-icon">&#x2295;</div><div class="empty-text">no XPath results for "' +
          esc(q) +
          '"</div></div>';
    } else if (searchMode === "facts") {
      const facts = await api("search/facts?q=" + eq);
      if (Array.isArray(facts) && facts.length) {
        html += '<div class="section-label">facts (' + facts.length + ")</div>";
        facts.forEach((f) => {
          const active = !f.invalid_at;
          html +=
            '<div class="memory-card" style="border-left-color:var(--orange)"><div class="card-meta"><span class="card-type" style="color:var(--orange)">fact</span>' +
            '<span class="status-badge">' +
            (active ? "active" : "superseded") +
            "</span></div>" +
            '<div class="card-body"><strong>' +
            esc(f.subject) +
            "</strong> " +
            esc(f.predicate) +
            " <strong>" +
            esc(f.object) +
            "</strong></div>" +
            '<div class="card-footer">' +
            confBar(f.confidence || 1.0) +
            "</div></div>";
        });
      } else
        html =
          '<div class="empty-state"><div class="empty-icon">&#x2295;</div><div class="empty-text">no facts matching "' +
          esc(q) +
          '"</div></div>';
    } else {
      const [sessions, memories] = await Promise.all([
        api("search/sessions?q=" + eq),
        api("search/memories?q=" + eq),
      ]);
      if (sessions.length) {
        html +=
          '<div class="section-label">sessions (' + sessions.length + ")</div>";
        sessions.forEach((s) => {
          const dt = s.started_at ? fmtDate(s.started_at) : "";
          html +=
            '<div class="memory-card" style="cursor:pointer;border-left-color:var(--blue)" onclick="showDetail(\'' +
            jsArg(s.id) +
            "')\">" +
            '<div class="card-meta"><div style="display:flex;gap:6px;align-items:center">' +
            agentTag(s.agent) +
            '<span class="dim">' +
            esc(s.project_id || "") +
            '</span></div><span class="card-date">' +
            dt +
            "</span></div>" +
            '<div class="card-body">' +
            esc(s.summary || "\u2014") +
            "</div></div>";
        });
      }
      if (memories.length) {
        html +=
          '<div class="section-label">memories (' + memories.length + ")</div>";
        memories.forEach((m) => {
          html +=
            '<div class="memory-card"><div class="card-meta"><span class="card-type">' +
            esc(m.memory_type || "general") +
            '</span></div><div class="card-body">' +
            esc(m.content) +
            "</div>" +
            memoryTags(m) +
            "</div>";
        });
      }
      if (!sessions.length && !memories.length)
        html =
          '<div class="empty-state"><div class="empty-icon">&#x2295;</div><div class="empty-text">no results for "' +
          esc(q) +
          '"</div></div>';
    }
    el.innerHTML = html;
  } catch (e) {
    el.innerHTML =
      '<div class="empty-state"><div class="empty-icon">&#x2297;</div><div class="empty-text">search failed: ' +
      esc(e.message) +
      "</div></div>";
  }
}

function clearSearch() {
  document.getElementById("searchInput").value = "";
  document.getElementById("navSearchItem").style.display = "none";
  switchTab("sessions");
}

// -----------------------------------------------------------------------
// Add Note / Decision
// -----------------------------------------------------------------------
function showAddNote() {
  document.getElementById("noteModal").style.display = "block";
  document.getElementById("noteText").focus();
}
function showAddDecision() {
  document.getElementById("decisionModal").style.display = "block";
  document.getElementById("decWhat").focus();
}

async function submitNote() {
  const text = document.getElementById("noteText").value.trim();
  if (!text) return;
  const tags = document
    .getElementById("noteTags")
    .value.split(",")
    .map((tag) => tag.trim())
    .filter(Boolean);
  try {
    await apiPost("notes", {
      text,
      tags,
      project: currentProject || undefined,
    });
    document.getElementById("noteText").value = "";
    document.getElementById("noteTags").value = "";
    closeModal("noteModal");
    loadMemories();
  } catch (e) {
    alert("Failed: " + e.message);
  }
}

async function submitDecision() {
  const what = document.getElementById("decWhat").value.trim();
  const why = document.getElementById("decWhy").value.trim();
  if (!what) return;
  try {
    await apiPost("decisions", {
      what,
      why: why || undefined,
      project: currentProject || undefined,
    });
    document.getElementById("decWhat").value = "";
    document.getElementById("decWhy").value = "";
    closeModal("decisionModal");
    loadDecisions();
  } catch (e) {
    alert("Failed: " + e.message);
  }
}

// -----------------------------------------------------------------------
// Modals & Command Palette
// -----------------------------------------------------------------------
function closeModal(id) {
  document.getElementById(id).style.display = "none";
}

function openCmdPalette() {
  document.getElementById("cmdPalette").style.display = "block";
  const input = document.getElementById("cmdInput");
  input.value = "";
  input.focus();
}
function closeCmdPalette() {
  document.getElementById("cmdPalette").style.display = "none";
}

function cmdGo(tab) {
  closeCmdPalette();
  switchTab(tab);
}

// -----------------------------------------------------------------------
// Keyboard shortcuts
// -----------------------------------------------------------------------
document.addEventListener("keydown", (e) => {
  // Cmd+K or Ctrl+K → command palette
  if ((e.metaKey || e.ctrlKey) && e.key === "k") {
    e.preventDefault();
    openCmdPalette();
    return;
  }
  // Escape → close modals
  if (e.key === "Escape") {
    if (document.getElementById("cmdPalette").style.display === "block") {
      closeCmdPalette();
      return;
    }
    document
      .querySelectorAll(".modal-overlay")
      .forEach((m) => (m.style.display = "none"));
    return;
  }
  // Don't handle shortcuts when typing in inputs
  if (
    e.target.tagName === "INPUT" ||
    e.target.tagName === "TEXTAREA" ||
    e.target.tagName === "SELECT"
  )
    return;
  // / → focus search
  if (e.key === "/") {
    e.preventDefault();
    document.getElementById("searchInput").focus();
  }
});

document.getElementById("searchInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") performSearch();
});

document.getElementById("cmdInput").addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeCmdPalette();
  if (e.key === "Enter") {
    const val = e.target.value.trim().toLowerCase();
    const tabs = [
      "home",
      "sessions",
      "memories",
      "facts",
      "decisions",
      "analytics",
    ];
    const match = tabs.find((t) => t.startsWith(val));
    if (match) cmdGo(match);
    else {
      closeCmdPalette();
      document.getElementById("searchInput").value = e.target.value;
      performSearch();
    }
  }
});

document.getElementById("cmdPalette").addEventListener("click", (e) => {
  if (e.target === document.getElementById("cmdPalette")) closeCmdPalette();
});
document.querySelectorAll(".modal-overlay").forEach((m) =>
  m.addEventListener("click", (e) => {
    if (e.target === m) m.style.display = "none";
  }),
);

document.getElementById("projectFilter").addEventListener("change", (e) => {
  currentProject = e.target.value;
  loadSessions(currentProject);
});

// -----------------------------------------------------------------------
// Init
// -----------------------------------------------------------------------
switchTab("home");
loadStats();
loadProjects();
