// Unit tests for cli/src/web_dashboard_util.js — run with `node --test cli/js-tests`.
// These cover the dashboard's pure logic: HTML escaping, timestamp parsing,
// confidence-preservation on memory edits (issue #27), agent-filter query
// building (issue #28), and API error formatting (issue #35).
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const util = require("../src/web_dashboard_util.js");

test("esc escapes HTML special characters", () => {
  assert.equal(util.esc('<script>alert("x")&</script>'),
    "&lt;script&gt;alert(&quot;x&quot;)&amp;&lt;/script&gt;");
});

test("esc returns empty string for falsy input", () => {
  assert.equal(util.esc(""), "");
  assert.equal(util.esc(null), "");
  assert.equal(util.esc(undefined), "");
  assert.equal(util.esc(0), "");
});

test("jsArg escapes single quotes, backslashes and newlines", () => {
  assert.equal(util.jsArg("a'b\\c\"d\ne\rf"),
    "a\\'b\\\\c&quot;d\\ne\\rf");
});

test("parseUtcDate treats naive timestamps as UTC", () => {
  const d = util.parseUtcDate("2026-08-22 15:30:00");
  assert.equal(d.toISOString(), "2026-08-22T15:30:00.000Z");
});

test("parseUtcDate keeps explicit zones intact", () => {
  const d = util.parseUtcDate("2026-08-22T15:30:00+02:00");
  assert.equal(d.toISOString(), "2026-08-22T13:30:00.000Z");
});

test("parseUtcDate never throws on garbage", () => {
  const d = util.parseUtcDate("not a date");
  assert.ok(Number.isNaN(d.getTime()));
  assert.ok(Number.isNaN(util.parseUtcDate(null).getTime()));
});

test("formatIsoDay renders short month names", () => {
  assert.equal(util.formatIsoDay("2026-08-22"), "Aug 22");
  assert.equal(util.formatIsoDay("2026-01-05"), "Jan 5");
  assert.equal(util.formatIsoDay("garbage"), "garbage");
});

test("fmtNum compacts thousands and millions", () => {
  assert.equal(util.fmtNum(999), "999");
  assert.equal(util.fmtNum(1200), "1.2k");
  assert.equal(util.fmtNum(2500000), "2.5M");
});

test("confClass buckets confidence values", () => {
  assert.equal(util.confClass(0.9), "conf-high");
  assert.equal(util.confClass(0.8), "conf-high");
  assert.equal(util.confClass(0.6), "conf-mid");
  assert.equal(util.confClass(0.5), "conf-mid");
  assert.equal(util.confClass(0.2), "conf-low");
});

test("confBar embeds the percentage width", () => {
  const html = util.confBar(0.87);
  assert.ok(html.includes("width:87%"));
  assert.ok(html.includes("87%"));
});

test("safeToken normalizes agent names", () => {
  assert.equal(util.safeToken("Claude Code"), "claude-code");
  assert.equal(util.safeToken("codex"), "codex");
  assert.equal(util.safeToken(null), "unknown");
});

test("trendIndicator handles zero baselines without dividing by zero", () => {
  assert.ok(util.trendIndicator(0, 0).includes("steady"));
  assert.ok(util.trendIndicator(5, 0).includes("new"));
  assert.ok(util.trendIndicator(15, 10).includes("50%"));
  assert.ok(util.trendIndicator(5, 10).includes("50%"));
  assert.ok(util.trendIndicator(10, 10).includes("steady"));
});

test("renderSparkline returns empty for no data and svg otherwise", () => {
  assert.equal(util.renderSparkline([]), "");
  assert.equal(util.renderSparkline(null), "");
  const svg = util.renderSparkline([1, 5, 3]);
  assert.ok(svg.startsWith("<svg"));
  assert.ok(svg.includes("polyline"));
  // single point must not divide by zero
  assert.ok(util.renderSparkline([7]).includes("polyline"));
});

test("describeApiError preserves the server body (issue #35)", () => {
  assert.equal(util.describeApiError(500, "store unavailable"),
    "API 500: store unavailable");
  assert.equal(util.describeApiError(500, ""), "API 500");
  assert.equal(util.describeApiError(404, null), "API 404");
  const long = "x".repeat(500);
  assert.ok(util.describeApiError(500, long).length < 400);
});

test("safeTime degrades to empty string on malformed input (issue #35)", () => {
  assert.equal(util.safeTime("garbage"), "");
  assert.equal(util.safeTime(null), "");
  assert.equal(typeof util.safeTime("2026-08-22T15:30:00Z"), "string");
  assert.ok(util.safeTime("2026-08-22T15:30:00Z").length > 0);
});

test("initialSliderValue reflects current confidence (issue #27)", () => {
  assert.equal(util.initialSliderValue(0.95), 95);
  assert.equal(util.initialSliderValue(0.3), 30);
  // clamps out-of-range values
  assert.equal(util.initialSliderValue(1.7), 100);
  assert.equal(util.initialSliderValue(-0.4), 0);
  // falls back for missing/invalid data
  assert.equal(util.initialSliderValue(undefined), 80);
  assert.equal(util.initialSliderValue(NaN), 80);
});

test("confidencePayload preserves stored confidence when slider untouched (issue #27)", () => {
  // Slider initialized at 95 from stored 0.95, user saves without touching:
  assert.equal(util.confidencePayload(0.95, 95, 95), 0.95);
  // Precision is preserved exactly (no 0.8 overwrite):
  assert.equal(util.confidencePayload(0.83, 83, 83), 0.83);
  // User moved the slider: adopt the slider value.
  assert.equal(util.confidencePayload(0.95, 95, 40), 0.4);
  // No stored confidence: use the slider either way.
  assert.equal(util.confidencePayload(undefined, 80, 80), 0.8);
  assert.equal(util.confidencePayload(NaN, 80, 55), 0.55);
});

test("agentQueryParams builds repeatable agent params (issue #28)", () => {
  const all = new Set(["claude_code", "codex", "gemini"]);
  assert.equal(util.agentQueryParams(all), "");
  assert.equal(util.agentQueryParams(new Set()), "");
  const subset = new Set(["codex"]);
  assert.equal(util.agentQueryParams(subset), "agent=codex");
  const two = new Set(["claude_code", "gemini"]);
  assert.equal(util.agentQueryParams(two),
    "agent=claude_code%2Cgemini");
});

test("appendQuery chooses ? vs & correctly (issue #28)", () => {
  assert.equal(util.appendQuery("memories?limit=200", "agent=codex"),
    "memories?limit=200&agent=codex");
  assert.equal(util.appendQuery("decisions", "agent=codex"),
    "decisions?agent=codex");
  assert.equal(util.appendQuery("decisions", ""), "decisions");
});
