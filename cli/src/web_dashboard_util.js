// Remembrant dashboard — pure helper functions.
//
// These helpers have no DOM or network dependencies so they can be unit
// tested with `node --test` and reused from web_dashboard.js. The file is a
// UMD module: in the browser it attaches to `window.RemembrantUtil`, and in
// Node it is importable via `require()`.
(function (root, factory) {
  if (typeof module === "object" && typeof module.exports === "object") {
    module.exports = factory();
  } else {
    root.RemembrantUtil = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // Escape a value for safe interpolation into HTML. Falsy values become "".
  function esc(value) {
    return value
      ? String(value)
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;")
      : "";
  }

  // Escape a value for safe interpolation into a single-quoted JS string
  // literal inside an HTML attribute.
  function jsArg(value) {
    return String(value ?? "")
      .replace(/\\/g, "\\\\")
      .replace(/'/g, "\\'")
      .replace(/"/g, "&quot;")
      .replace(/\r/g, "\\r")
      .replace(/\n/g, "\\n");
  }

  // Parse a stored (naive-UTC or zoned) timestamp into a Date. Never throws.
  function parseUtcDate(value) {
    try {
      const raw = String(value ?? "");
      if (!raw) return new Date(NaN);
      const iso = raw.includes(" ") ? raw.replace(" ", "T") : raw;
      const hasZone = /(?:Z|[+-]\d\d:?\d\d)$/i.test(iso);
      return new Date(hasZone ? iso : iso + "Z");
    } catch {
      return new Date(NaN);
    }
  }

  // Format an ISO day ("2026-08-22") as "Aug 22". Falls back to input.
  function formatIsoDay(value) {
    const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(value ?? ""));
    if (!match) return String(value ?? "");
    const months = [
      "Jan", "Feb", "Mar", "Apr", "May", "Jun",
      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    const month = months[Number(match[2]) - 1] ?? match[2];
    return month + " " + Number(match[3]);
  }

  // Compact number formatting (1200 -> "1.2k").
  function fmtNum(n) {
    if (typeof n !== "number" || !isFinite(n)) return String(n);
    if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
    if (n >= 1000) return (n / 1000).toFixed(1) + "k";
    return String(n);
  }

  // CSS class bucket for a confidence value.
  function confClass(c) {
    return c >= 0.8 ? "conf-high" : c >= 0.5 ? "conf-mid" : "conf-low";
  }

  // Inline confidence bar markup.
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

  // Normalize an agent name into a safe token for CSS/keys.
  function safeToken(value) {
    return String(value || "unknown")
      .toLowerCase()
      .replace(/[^a-z0-9_.-]/g, "-");
  }

  // Trend arrow markup comparing current vs previous.
  function trendIndicator(current, previous) {
    if (previous === 0 && current === 0)
      return '<span class="trend trend-flat">&rarr; steady</span>';
    if (previous === 0)
      return '<span class="trend trend-up">&uarr; new</span>';
    const pct = Math.round(((current - previous) / previous) * 100);
    if (pct > 0)
      return '<span class="trend trend-up">&uarr; ' + pct + "%</span>";
    if (pct < 0)
      return '<span class="trend trend-down">&darr; ' + Math.abs(pct) + "%</span>";
    return '<span class="trend trend-flat">&rarr; steady</span>';
  }

  // Inline SVG sparkline for a series of numbers.
  function renderSparkline(data, color) {
    if (!data || !data.length) return "";
    const w = 60,
      h = 18,
      pad = 1;
    const vals = data.map((d) => (typeof d === "number" ? d : d.value || 0));
    const max = Math.max.apply(null, vals.concat([1]));
    const min = Math.min.apply(null, vals.concat([0]));
    const range = max - min || 1;
    const pts = vals
      .map(function (v, i) {
        const x = pad + (i / (vals.length - 1 || 1)) * (w - 2 * pad);
        const y = h - pad - ((v - min) / range) * (h - 2 * pad);
        return x.toFixed(1) + "," + y.toFixed(1);
      })
      .join(" ");
    return (
      '<svg width="' + w + '" height="' + h + '" viewBox="0 0 ' + w + " " + h +
      '"><polyline points="' + pts + '" fill="none" stroke="' +
      (color || "var(--accent)") +
      '" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    );
  }

  // Build a human-readable API error that preserves the server-provided body.
  // Fixes issue #35: previously the response body was discarded and users saw
  // a bare "API 500" with no actionable detail.
  function describeApiError(status, bodyText) {
    const detail = String(bodyText ?? "").trim();
    if (detail) {
      const short = detail.length > 300 ? detail.slice(0, 300) + "…" : detail;
      return "API " + status + ": " + short;
    }
    return "API " + status;
  }

  // Safely format a timestamp as a local time string. Returns "" on any
  // parse/format failure instead of throwing (issue #35: a malformed
  // tool-call timestamp used to blank the whole session modal).
  function safeTime(value) {
    try {
      const d = parseUtcDate(value);
      if (isNaN(d.getTime())) return "";
      return d.toLocaleTimeString();
    } catch {
      return "";
    }
  }

  // --- Memory-edit confidence helpers (issue #27) -------------------------

  // Initial slider position (0..100) for a memory's current confidence.
  // Falls back to 80 when the stored confidence is missing or invalid.
  function initialSliderValue(currentConfidence) {
    if (typeof currentConfidence === "number" && isFinite(currentConfidence)) {
      const clamped = Math.min(1, Math.max(0, currentConfidence));
      return Math.round(clamped * 100);
    }
    return 80;
  }

  // Decide which confidence to persist when saving an edited memory.
  //
  // If the slider is still at its initialized position we preserve the
  // memory's exact stored confidence (so editing text/tags alone never
  // changes confidence). Otherwise we adopt the user's slider value.
  function confidencePayload(currentConfidence, initialValue, sliderValue) {
    const unchanged = sliderValue === initialValue;
    if (
      unchanged &&
      typeof currentConfidence === "number" &&
      isFinite(currentConfidence)
    ) {
      return currentConfidence;
    }
    return sliderValue / 100;
  }

  // Build the comma-separated `agent=` query value for list endpoints from
  // the set of currently active agent chips. Returns "" when all (or none)
  // of the known agents are active, i.e. no filtering is needed (issue #28).
  function agentQueryParams(activeAgents, knownAgents) {
    const known = knownAgents || ["claude_code", "codex", "gemini"];
    const selected = known.filter(function (a) {
      return activeAgents.has(a);
    });
    if (!selected.length || selected.length === known.length) return "";
    return "agent=" + encodeURIComponent(selected.join(","));
  }

  // Append an extra query string to a URL, choosing "?" vs "&" correctly.
  function appendQuery(url, extra) {
    if (!extra) return url;
    return url + (url.indexOf("?") === -1 ? "?" : "&") + extra;
  }

  return {
    esc: esc,
    jsArg: jsArg,
    parseUtcDate: parseUtcDate,
    formatIsoDay: formatIsoDay,
    fmtNum: fmtNum,
    confClass: confClass,
    confBar: confBar,
    safeToken: safeToken,
    trendIndicator: trendIndicator,
    renderSparkline: renderSparkline,
    describeApiError: describeApiError,
    safeTime: safeTime,
    initialSliderValue: initialSliderValue,
    confidencePayload: confidencePayload,
    agentQueryParams: agentQueryParams,
    appendQuery: appendQuery,
  };
});
