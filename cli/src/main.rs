use std::collections::HashMap;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::{
    Json as AxumJson, Router,
    extract::{Path as AxumPath, Query, State},
    http::StatusCode,
    routing::{delete, get, post},
};
use chrono::NaiveDateTime;
use clap::{Parser, Subcommand};
mod mcp_server;

use remembrant_engine::AppConfig;
use remembrant_engine::distill::Distiller;
use remembrant_engine::embed_pipeline::EmbedPipeline;
use remembrant_engine::embedding::{EmbedProvider, LmStudioEmbedder};
use remembrant_engine::graph_builder::{self, GraphBackend, GraphBuilder};
use remembrant_engine::repo_embed::RepoEmbedder;
use remembrant_engine::store::{DuckStore, LanceStore};

#[derive(Parser)]
#[command(
    name = "rem",
    about = "Remembrant: shared persistent memory for coding agents"
)]
#[command(version, propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize remembrant, create config, scan agents
    Init,

    /// Start file watcher daemon
    Watch,

    /// Stop daemon
    Stop,

    /// Semantic search across sessions
    Search {
        /// Search query
        query: String,

        /// Filter by project
        #[arg(long)]
        project: Option<String>,

        /// Filter by agent
        #[arg(long)]
        agent: Option<String>,

        /// Filter by date (e.g. "2d", "1w", ISO date)
        #[arg(long)]
        since: Option<String>,

        /// Filter by content type
        #[arg(long = "type", alias = "content-type")]
        content_type: Option<String>,

        /// Use exact matching instead of semantic
        #[arg(long)]
        exact: bool,

        /// Output as JSON (for agent consumption)
        #[arg(long)]
        json: bool,
    },

    /// Exact text match
    Find {
        /// Text to find
        query: String,
    },

    /// Recent sessions
    Recent {
        /// Maximum number of results
        #[arg(long, default_value_t = 20)]
        limit: usize,

        /// Filter by agent
        #[arg(long)]
        agent: Option<String>,

        /// Filter by project
        #[arg(long)]
        project: Option<String>,
    },

    /// Daily context briefing
    Brief {
        /// Filter by project
        #[arg(long)]
        project: Option<String>,

        /// Only show today's activity
        #[arg(long)]
        today: bool,

        /// Output as JSON (for agent consumption)
        #[arg(long)]
        json: bool,

        /// LLM-optimized compact context block
        #[arg(long)]
        for_agent: bool,

        /// Max tokens for agent context (default: 1000)
        #[arg(long, default_value_t = 1000)]
        max_tokens: usize,
    },

    /// Topic-specific context assembly for agents
    Context {
        /// Topic to assemble context for
        topic: String,

        /// Filter by project
        #[arg(long)]
        project: Option<String>,

        /// Output as JSON instead of prompt block
        #[arg(long)]
        json: bool,

        /// Max tokens (default: 800)
        #[arg(long, default_value_t = 800)]
        max_tokens: usize,
    },

    /// Memory consolidation: decay scores, duplicate detection, TTL expiry
    Consolidate {
        /// Filter by project
        #[arg(long)]
        project: Option<String>,

        /// Similarity threshold for merge candidates (0.0-1.0, default: 0.6)
        #[arg(long, default_value_t = 0.6)]
        threshold: f64,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Cross-project patterns
    Patterns {
        /// Optional topic to focus on
        topic: Option<String>,
    },

    /// Decision journal
    Decisions {
        /// Filter by project
        #[arg(long)]
        project: Option<String>,

        /// Show all decisions (not just recent)
        #[arg(long)]
        all: bool,
    },

    /// Find related content for a file
    Related {
        /// File path to find related content for
        path: String,
    },

    /// Dependency graph for a file
    Graph {
        /// File path to graph
        path: String,
    },

    /// Chronological view of a topic
    Timeline {
        /// Topic to view
        topic: String,

        /// Filter by date
        #[arg(long)]
        since: Option<String>,
    },

    /// Quick manual note
    Note {
        /// Note text
        text: String,

        /// Associate with project
        #[arg(long)]
        project: Option<String>,

        /// Add a tag
        #[arg(long)]
        tag: Option<Vec<String>>,
    },

    /// Remove a session
    Forget {
        /// Session ID to remove
        #[arg(long)]
        session: String,
    },

    /// Generate agent memory files
    Export {
        /// Filter by project
        #[arg(long)]
        project: Option<String>,

        /// Output format (e.g. markdown, json)
        #[arg(long, default_value = "markdown")]
        format: String,

        /// Output path
        #[arg(long)]
        output: Option<String>,
    },

    /// Embed a repository
    Embed {
        /// Repository path
        path: String,

        /// Update existing embeddings
        #[arg(long)]
        update: bool,
    },

    /// Full ingest pipeline: parse agents → DuckDB → embed → distill → graph
    Ingest {
        /// Skip embedding step (no LM Studio needed)
        #[arg(long)]
        skip_embed: bool,

        /// Skip LLM distillation step
        #[arg(long)]
        skip_distill: bool,
    },

    /// Daemon and DB status
    Status,

    /// Analytics
    Stats,

    /// Garbage collect old/orphaned data
    Gc,

    /// Launch web dashboard on localhost
    Web {
        /// Port to listen on
        #[arg(long, default_value_t = 3000)]
        port: u16,
    },

    /// Start MCP (Model Context Protocol) server for Claude Code/Cursor integration
    Mcp,

    /// Search using Semantic XPath query
    #[command(name = "xpath")]
    XPath {
        /// The XPath query (e.g., '//Session[node~"auth"]/Decision')
        query: String,

        /// Maximum tree depth to load (default: 4)
        #[arg(long, default_value = "4")]
        depth: usize,

        /// Maximum results to show
        #[arg(long, short, default_value = "20")]
        limit: usize,

        /// Show tree structure of results
        #[arg(long)]
        tree: bool,

        /// Output as JSON (for agent consumption)
        #[arg(long)]
        json: bool,
    },
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Expand a leading `~/` to the actual home directory.
fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(rest);
    }
    PathBuf::from(path)
}

/// Open DuckDB at the configured (tilde-expanded) path.
fn open_store(config: &AppConfig) -> Result<DuckStore> {
    let db_path = expand_tilde(&config.storage.duckdb_path);
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }
    let store = DuckStore::open(&db_path)?;

    // Try to initialize FTS indexes for BM25 search. Non-fatal if the
    // FTS extension is unavailable (falls back to ILIKE).
    if let Err(e) = store.init_fts() {
        tracing::debug!("FTS extension not available, falling back to ILIKE: {e}");
    }

    Ok(store)
}

/// Return the path to the PID file.
fn pid_file_path() -> Result<PathBuf> {
    let dir = AppConfig::config_dir()?;
    Ok(dir.join("daemon.pid"))
}

/// Return whether a Unix process is currently alive.
#[cfg(unix)]
fn process_is_running(pid: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// Process probing is not currently supported on non-Unix targets.
#[cfg(not(unix))]
fn process_is_running(_pid: u32) -> bool {
    false
}

/// Reject an active watcher PID file and remove stale markers.
fn ensure_no_active_pid_file(pid_path: &std::path::Path) -> Result<()> {
    if !pid_path.exists() {
        return Ok(());
    }

    let existing = std::fs::read_to_string(pid_path)
        .with_context(|| format!("failed to read {}", pid_path.display()))?;
    let existing = existing
        .trim()
        .parse::<u32>()
        .with_context(|| format!("invalid PID in {}: {existing}", pid_path.display()))?;
    if process_is_running(existing) {
        anyhow::bail!("watcher PID {existing} is already running; run `rem stop` first");
    }

    std::fs::remove_file(pid_path)
        .with_context(|| format!("failed to remove stale {}", pid_path.display()))
}

/// Ingest only sessions whose normalized values changed.
///
/// Replacing a changed session atomically removes stale transcript-derived
/// rows while preserving manual notes and decisions without a session source.
fn ingest_watch_snapshot(
    store: &DuckStore,
    registry: &remembrant_engine::ingest::AdapterRegistry,
    adapter_ids: Option<&HashSet<String>>,
) -> Result<usize> {
    let mut known: HashMap<String, _> = store
        .get_recent_sessions(100_000)?
        .into_iter()
        .map(|session| (session.id.clone(), session))
        .collect();
    let mut ingested = 0;

    for adapter in registry.detected() {
        let meta = adapter.meta();
        if adapter_ids.is_some_and(|ids| !ids.contains(&meta.id)) {
            continue;
        }

        match adapter.ingest() {
            Ok(result) => {
                for error in result.errors {
                    eprintln!("[watch] {} parse error: {error}", meta.display_name);
                }

                let mut changed_ids = HashSet::new();
                for session in &result.sessions {
                    let changed = known.get(&session.id).is_none_or(|previous| {
                        previous.started_at != session.started_at
                            || previous.ended_at != session.ended_at
                            || previous.summary != session.summary
                            || previous.message_count != session.message_count
                            || previous.tool_call_count != session.tool_call_count
                            || previous.total_tokens != session.total_tokens
                            || previous.files_changed != session.files_changed
                    });
                    if !changed {
                        continue;
                    }

                    store
                        .insert_or_replace_session(session)
                        .with_context(|| format!("failed to persist session {}", session.id))?;
                    known.insert(session.id.clone(), session.clone());
                    changed_ids.insert(session.id.clone());
                }

                if changed_ids.is_empty() {
                    continue;
                }

                for tool_call in &result.tool_calls {
                    if tool_call
                        .session_id
                        .as_ref()
                        .is_some_and(|session_id| changed_ids.contains(session_id))
                        && let Err(error) = store.insert_tool_call(tool_call)
                    {
                        eprintln!(
                            "[watch] {} tool call persistence error: {error}",
                            meta.display_name
                        );
                    }
                }
                for memory in &result.memories {
                    if memory
                        .source_session_id
                        .as_ref()
                        .is_some_and(|session_id| changed_ids.contains(session_id))
                        && let Err(error) = store.insert_memory(memory)
                    {
                        eprintln!(
                            "[watch] {} memory persistence error: {error}",
                            meta.display_name
                        );
                    }
                }

                println!(
                    "[watch] {}: ingested {} changed session(s)",
                    meta.display_name,
                    changed_ids.len()
                );
                ingested += changed_ids.len();
            }
            Err(error) => {
                eprintln!("[watch] {} ingestion error: {error}", meta.display_name);
            }
        }
    }

    Ok(ingested)
}

/// Wait for the daemon's graceful shutdown signals.
async fn wait_for_shutdown_signal() -> Result<()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};

        let mut interrupt = signal(SignalKind::interrupt())?;
        let mut terminate = signal(SignalKind::terminate())?;
        tokio::select! {
            result = interrupt.recv() => result.context("interrupt channel closed")?,
            result = terminate.recv() => result.context("terminate channel closed")?,
        }
    }

    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c().await?;
    }

    Ok(())
}

/// Format a file size in human-readable form.
fn human_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// Parse relative date strings like "2d", "1w", or ISO dates.
fn parse_since(s: &str) -> Option<NaiveDateTime> {
    let now = chrono::Utc::now().naive_utc();
    if let Some(days) = s.strip_suffix('d') {
        let n: i64 = days.parse().ok()?;
        return now.checked_sub_signed(chrono::Duration::try_days(n)?);
    }
    if let Some(weeks) = s.strip_suffix('w') {
        let n: i64 = weeks.parse().ok()?;
        return now.checked_sub_signed(chrono::Duration::try_weeks(n)?);
    }
    if let Some(hours) = s.strip_suffix('h') {
        let n: i64 = hours.parse().ok()?;
        return now.checked_sub_signed(chrono::Duration::try_hours(n)?);
    }
    // RFC 3339 values are normalized to UTC; naive ISO values use the UTC
    // convention used by Remembrant's DuckDB timestamps.
    chrono::DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|timestamp| timestamp.naive_utc())
        .or_else(|| {
            chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f")
                .ok()
                .or_else(|| {
                    chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d")
                        .ok()
                        .map(|date| date.and_hms_opt(0, 0, 0).unwrap())
                })
        })
}

fn parse_since_required(since: Option<&str>) -> Result<Option<NaiveDateTime>> {
    since
        .map(|value| parse_since(value).with_context(|| format!("invalid --since value: {value}")))
        .transpose()
}

fn parse_metadata_timestamp(value: &str) -> Option<NaiveDateTime> {
    NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S%.f")
        .ok()
        .or_else(|| {
            chrono::DateTime::parse_from_rfc3339(value)
                .ok()
                .map(|timestamp| timestamp.naive_utc())
        })
}

fn filter_search_results(
    results: Vec<remembrant_engine::HybridResult>,
    project: Option<&str>,
    agent: Option<&str>,
    content_type: Option<&str>,
    since: Option<NaiveDateTime>,
) -> Vec<remembrant_engine::HybridResult> {
    results
        .into_iter()
        .filter(|result| {
            if let Some(project_filter) = project {
                let result_project = result
                    .metadata
                    .get("project")
                    .map(|value| value.as_str())
                    .unwrap_or_default();
                if !result_project
                    .to_lowercase()
                    .contains(&project_filter.to_lowercase())
                {
                    return false;
                }
            }

            if let Some(agent_filter) = agent {
                let result_agent = result
                    .metadata
                    .get("agent")
                    .map(|value| value.as_str())
                    .unwrap_or_default();
                if !result_agent.eq_ignore_ascii_case(agent_filter) {
                    return false;
                }
            }

            if let Some(type_filter) = content_type {
                let result_type = result.result_type.to_string().to_lowercase();
                let metadata_type = result
                    .metadata
                    .get("type")
                    .map(|value| value.as_str())
                    .unwrap_or_default()
                    .to_lowercase();
                if !result_type.contains(&type_filter.to_lowercase())
                    && !metadata_type.contains(&type_filter.to_lowercase())
                {
                    return false;
                }
            }

            if let Some(since) = since {
                let timestamp = result
                    .metadata
                    .get("started_at")
                    .or_else(|| result.metadata.get("created_at"))
                    .or_else(|| result.metadata.get("valid_at"))
                    .map(|value| value.as_str())
                    .and_then(parse_metadata_timestamp);
                if timestamp.is_none_or(|timestamp| timestamp < since) {
                    return false;
                }
            }

            true
        })
        .collect()
}

/// Open LanceDB at the configured (tilde-expanded) path.
async fn open_lance_store(config: &AppConfig) -> Result<LanceStore> {
    let lance_path = expand_tilde(&config.storage.lancedb_path);
    std::fs::create_dir_all(&lance_path)
        .with_context(|| format!("failed to create directory {}", lance_path.display()))?;
    LanceStore::open_with_dim(&lance_path, config.embedding.dimensions as i32).await
}

/// Truncate a string to a maximum character width, appending "..." if truncated.
fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars.saturating_sub(3)).collect();
        format!("{truncated}...")
    }
}

/// Return the start of the UTC calendar day containing `now`.
fn utc_day_start(now: NaiveDateTime) -> NaiveDateTime {
    now.date().and_hms_opt(0, 0, 0).unwrap_or(now)
}

// ---------------------------------------------------------------------------
// Command implementations
// ---------------------------------------------------------------------------

fn cmd_init() -> Result<()> {
    println!("Remembrant: initializing...\n");

    // 1. Load or create config
    let config = AppConfig::load()?;
    let config_dir = AppConfig::config_dir()?;
    std::fs::create_dir_all(&config_dir)
        .with_context(|| format!("failed to create {}", config_dir.display()))?;
    println!("[config] Configuration at {}", config_dir.display());

    // 2. Detect agents
    println!("\n--- Agent Detection ---");
    let registry = remembrant_engine::ingest::build_registry_from_config(&config)?;
    let detected: Vec<_> = registry
        .adapters()
        .iter()
        .filter(|adapter| adapter.detect())
        .map(|adapter| adapter.meta().clone())
        .collect();

    for meta in &detected {
        println!("  [+] {} -- {}", meta.display_name, meta.default_path);
    }
    let agents_found = detected.len();
    if agents_found == 0 {
        println!("  (no agents detected)");
    }

    // 3. Open DuckDB and init schema
    let db_path = expand_tilde(&config.storage.duckdb_path);
    println!("\n[storage] DuckDB at {}", db_path.display());
    let store = open_store(&config)?;
    println!("[storage] Schema initialized");

    // 4. Initial ingestion
    println!("\n--- Initial Ingestion ---");
    let mut total_sessions = 0usize;
    let mut total_memories = 0usize;
    let mut total_tool_calls = 0usize;
    let mut parse_errors = 0usize;

    for adapter in registry.detected() {
        let meta = adapter.meta();
        let result = adapter
            .ingest()
            .with_context(|| format!("failed to ingest {}", meta.display_name))?;

        for session in &result.sessions {
            store
                .insert_or_replace_session(session)
                .with_context(|| format!("failed to persist {} session {}", meta.id, session.id))?;
        }
        for memory in &result.memories {
            store
                .insert_memory(memory)
                .with_context(|| format!("failed to persist {} memory", meta.id))?;
        }
        for tool_call in &result.tool_calls {
            store.insert_tool_call(tool_call).with_context(|| {
                format!("failed to persist {} tool call {}", meta.id, tool_call.id)
            })?;
        }

        total_sessions += result.sessions.len();
        total_memories += result.memories.len();
        total_tool_calls += result.tool_calls.len();
        println!(
            "  {}: {} sessions, {} tool calls, {} memories",
            meta.display_name,
            result.sessions.len(),
            result.tool_calls.len(),
            result.memories.len()
        );
        for error in &result.errors {
            eprintln!("  [!] {} parse error: {error}", meta.id);
        }
        parse_errors += result.errors.len();
    }

    if parse_errors > 0 {
        anyhow::bail!(
            "{parse_errors} agent artifact parse error(s) occurred during initialization"
        );
    }

    // Populate projects and file_stats from ingested sessions
    let session_count = store.count_sessions()?;
    let all_sessions = store.get_recent_sessions(session_count.max(1))?;
    let mut projects_seen = std::collections::HashSet::new();
    let mut files_tracked = 0usize;
    for s in &all_sessions {
        if let Some(ref pid) = s.project_id {
            if projects_seen.insert(pid.clone()) {
                let name = pid.rsplit('/').next().unwrap_or(pid);
                store
                    .upsert_project(pid, name, pid)
                    .with_context(|| format!("failed to persist project {pid}"))?;
            }
            for f in &s.files_changed {
                store
                    .upsert_file_stat(f, pid)
                    .with_context(|| format!("failed to persist file stat {f}"))?;
                files_tracked += 1;
            }
        }
    }

    println!("\n--- Summary ---");
    println!("  Agents detected:   {agents_found}");
    println!("  Sessions ingested: {total_sessions}");
    println!("  Tool calls found:  {total_tool_calls}");
    println!("  Memories ingested: {total_memories}");
    println!("  Projects tracked:  {}", projects_seen.len());
    println!("  File changes:      {files_tracked}");
    println!("\nInitialization complete. Run `rem status` to verify.");

    Ok(())
}

fn cmd_status() -> Result<()> {
    let config = AppConfig::load()?;
    let config_dir = AppConfig::config_dir()?;

    println!("Remembrant Status\n");

    // 1. Daemon status
    let pid_path = pid_file_path()?;
    if pid_path.exists() {
        let pid_text = std::fs::read_to_string(&pid_path)
            .with_context(|| format!("failed to read {}", pid_path.display()))?;
        let pid = pid_text
            .trim()
            .parse::<u32>()
            .with_context(|| format!("invalid PID in {}: {pid_text}", pid_path.display()))?;
        let running = process_is_running(pid);

        if running {
            println!("[daemon] Running (PID {pid})");
        } else {
            println!("[daemon] Stale PID file (process {pid} not running)");
        }
    } else {
        println!("[daemon] Not running");
    }

    // 2. Detected agents
    println!("\n--- Agents ---");
    let registry = remembrant_engine::ingest::build_registry_from_config(&config)?;
    let detected_agents = registry.detected();
    for adapter in &detected_agents {
        let meta = adapter.meta();
        println!("  {}: {}", meta.display_name, meta.default_path);
    }
    if detected_agents.is_empty() {
        println!("  (no agents detected)");
    }

    // 3. Storage
    println!("\n--- Storage ---");
    println!("  Config dir:  {}", config_dir.display());

    let db_path = expand_tilde(&config.storage.duckdb_path);
    if db_path.exists() {
        let size = std::fs::metadata(&db_path)
            .with_context(|| format!("failed to read metadata for {}", db_path.display()))?
            .len();
        println!(
            "  DuckDB:      {} ({})",
            db_path.display(),
            human_size(size)
        );

        let store = open_store(&config)?;
        println!("  Sessions:    {}", store.count_sessions()?);
        println!("  Memories:    {}", store.count_memories()?);
    } else {
        println!("  DuckDB:      {} (not created yet)", db_path.display());
        println!("  Run `rem init` first.");
    }

    let lance_path = expand_tilde(&config.storage.lancedb_path);
    if lance_path.exists() {
        println!("  LanceDB:     {}", lance_path.display());
    }

    Ok(())
}

async fn cmd_watch() -> Result<()> {
    let config = AppConfig::load()?;
    let pid_path = pid_file_path()?;
    ensure_no_active_pid_file(&pid_path)?;
    let registry = Arc::new(remembrant_engine::ingest::build_registry_from_config(
        &config,
    )?);
    let store = Arc::new(open_store(&config)?);
    let watcher = remembrant_engine::FileWatcher::from_config(&config);

    // Native JSONL/JSON/markdown adapters are event-driven. Configured SQLite
    // adapters can be a single database file, so they retain a low-frequency
    // polling fallback instead of forcing a filesystem watch onto a file that
    // may be replaced atomically.
    let native_adapter_ids: HashSet<String> = [
        config.agents.claude_code.enabled.then_some("claude_code"),
        config.agents.codex.enabled.then_some("codex"),
        config.agents.gemini.enabled.then_some("gemini"),
    ]
    .into_iter()
    .flatten()
    .map(str::to_string)
    .chain(
        config
            .agents
            .dynamic
            .iter()
            .filter(|agent| agent.enabled && agent.adapter_type == "jsonl")
            .map(|agent| agent.id.clone()),
    )
    .collect();
    let dynamic_adapter_ids: HashSet<String> = config
        .agents
        .dynamic
        .iter()
        .filter(|agent| agent.enabled && agent.adapter_type == "sqlite")
        .map(|agent| agent.id.clone())
        .collect();
    let active_native_paths = watcher
        .watch_paths()
        .iter()
        .filter(|path| path.is_dir())
        .count();
    let poll_interval = std::time::Duration::from_millis(config.watch.debounce_ms.max(1_000));
    if registry.detected().is_empty() {
        anyhow::bail!(
            "no enabled agent artifacts were detected; run `rem status` to inspect configured paths"
        );
    }

    // Write the PID file after the configuration and database are known to be
    // valid so a failed startup never leaves a stale daemon marker.
    let pid = std::process::id();
    if let Some(parent) = pid_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::write(&pid_path, pid.to_string())
        .with_context(|| format!("failed to write {}", pid_path.display()))?;
    println!("[watch] PID {pid} written to {}", pid_path.display());
    println!(
        "[watch] Event watching {active_native_paths} native artifact director(y/ies); \
        polling {} dynamic adapter(s) every {} ms. Press Ctrl+C to stop.",
        dynamic_adapter_ids.len(),
        poll_interval.as_millis()
    );

    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    let watcher_debounce = config.watch.debounce_ms;
    std::thread::spawn(move || {
        let file_watcher =
            remembrant_engine::FileWatcher::new(watcher.watch_paths().to_vec(), watcher_debounce);
        let sender = event_tx;
        if let Err(error) = file_watcher.run(move |events| {
            // An event is only a hint: ingestion performs a normalized
            // diff, so duplicate or partial-write events are safe.
            println!("[watch] received {} artifact change(s)", events.len());
            let _ = sender.send(());
        }) {
            eprintln!("[watch] filesystem watcher error: {error}");
        }
        tracing::debug!("native watcher thread exited");
    });

    let event_registry = Arc::clone(&registry);
    let event_store = Arc::clone(&store);
    let native_ids = native_adapter_ids.clone();
    tokio::spawn(async move {
        while event_rx.recv().await.is_some() {
            match ingest_watch_snapshot(&event_store, &event_registry, Some(&native_ids)) {
                Ok(changed_sessions) if changed_sessions > 0 => match build_graph(&event_store) {
                    Ok(graph) => println!(
                        "[watch] rebuilt graph with {} nodes and {} edges",
                        graph.node_count(),
                        graph.edge_count()
                    ),
                    Err(error) => {
                        eprintln!("[watch] graph rebuild error: {error:#}");
                    }
                },
                Ok(_) => {}
                Err(error) => {
                    eprintln!("[watch] persistence error: {error:#}");
                }
            }
        }
    });

    let poll_registry = Arc::clone(&registry);
    let poll_store = Arc::clone(&store);
    let poll_loop = async move {
        if dynamic_adapter_ids.is_empty() {
            std::future::pending::<()>().await;
        }
        loop {
            tokio::time::sleep(poll_interval).await;
            match ingest_watch_snapshot(&poll_store, &poll_registry, Some(&dynamic_adapter_ids)) {
                Ok(changed_sessions) if changed_sessions > 0 => match build_graph(&poll_store) {
                    Ok(graph) => println!(
                        "[watch] rebuilt graph with {} nodes and {} edges",
                        graph.node_count(),
                        graph.edge_count()
                    ),
                    Err(error) => {
                        eprintln!("[watch] graph rebuild error: {error:#}");
                    }
                },
                Ok(_) => {}
                Err(error) => eprintln!("[watch] persistence error: {error:#}"),
            }
        }
    };

    wait_for_shutdown_signal().await?;
    println!("\n[watch] Shutting down...");
    drop(poll_loop);

    if pid_path.exists() {
        std::fs::remove_file(&pid_path)
            .with_context(|| format!("failed to remove {}", pid_path.display()))?;
        println!("[watch] PID file removed");
    }

    Ok(())
}

fn cmd_stop() -> Result<()> {
    let pid_path = pid_file_path()?;

    if !pid_path.exists() {
        println!("[stop] No daemon PID file found. Is the watcher running?");
        return Ok(());
    }

    let pid_text = std::fs::read_to_string(&pid_path).context("failed to read PID file")?;
    let pid = pid_text
        .trim()
        .parse::<u32>()
        .with_context(|| format!("invalid PID in {}: {pid_text}", pid_path.display()))?;

    println!("[stop] Sending SIGTERM to PID {pid}...");

    let status = std::process::Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status()
        .context("failed to send SIGTERM")?;

    if status.success() {
        println!("[stop] Signal sent successfully");
    } else {
        if process_is_running(pid) {
            anyhow::bail!("failed to stop watcher PID {pid}");
        }
        println!("[stop] PID {pid} was already stopped; removing stale PID file");
    }

    // Give the daemon a moment to clean up its own marker, then finish the job
    // if it exited before flushing.
    for _ in 0..100 {
        if !pid_path.exists() {
            println!("[stop] PID file removed");
            println!("[stop] Daemon stopped.");
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    if process_is_running(pid) {
        anyhow::bail!("watcher PID {pid} did not stop within five seconds");
    }
    std::fs::remove_file(&pid_path)
        .with_context(|| format!("failed to remove stale PID file {}", pid_path.display()))?;
    println!("[stop] PID file removed");
    println!("[stop] Daemon stopped.");
    Ok(())
}

fn cmd_recent(limit: usize, agent: Option<&str>, project: Option<&str>) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;
    // Fetch more than needed so post-filtering still returns enough results
    let fetch_limit = if agent.is_some() || project.is_some() {
        limit * 5
    } else {
        limit
    };
    let sessions: Vec<_> = store
        .get_recent_sessions(fetch_limit)?
        .into_iter()
        .filter(|s| {
            if let Some(a) = agent
                && !s.agent.eq_ignore_ascii_case(a)
            {
                return false;
            }
            if let Some(p) = project
                && s.project_id.as_deref() != Some(p)
            {
                return false;
            }
            true
        })
        .take(limit)
        .collect();

    if sessions.is_empty() {
        println!("No sessions found. Run `rem init` to ingest agent data.");
        return Ok(());
    }

    println!(
        "{:<36}  {:<12}  {:<20}  {:>5}  {:>5}  SUMMARY",
        "SESSION ID", "AGENT", "STARTED", "MSGS", "TOOLS"
    );
    println!("{}", "-".repeat(110));

    for s in &sessions {
        let started = s
            .started_at
            .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_else(|| "-".to_string());
        let msgs = s
            .message_count
            .map(|n| n.to_string())
            .unwrap_or_else(|| "-".to_string());
        let tools = s
            .tool_call_count
            .map(|n| n.to_string())
            .unwrap_or_else(|| "-".to_string());
        let summary = s
            .summary
            .as_deref()
            .unwrap_or("-")
            .chars()
            .take(40)
            .collect::<String>();

        let id_display = if s.id.len() > 36 {
            truncate(&s.id, 36)
        } else {
            s.id.clone()
        };

        println!(
            "{:<36}  {:<12}  {:<20}  {:>5}  {:>5}  {}",
            id_display, s.agent, started, msgs, tools, summary
        );
    }

    println!("\n{} session(s) shown", sessions.len());
    Ok(())
}

async fn cmd_search(
    query: &str,
    project: Option<&str>,
    agent: Option<&str>,
    since: Option<&str>,
    content_type: Option<&str>,
    exact: bool,
    json_output: bool,
) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    let since_dt = parse_since_required(since)?;

    // JSON output mode: uses HybridSearch text-only for fast agent-friendly results
    if json_output {
        let search = remembrant_engine::HybridSearch::new(&store);

        let results = if remembrant_engine::is_xpath_query(query) {
            search.search_xpath(query, 20)?
        } else {
            search.search_text_only(query, 20)?
        };

        let results = filter_search_results(results, project, agent, content_type, since_dt);

        let json_results: Vec<serde_json::Value> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "id": r.id,
                    "type": r.result_type.to_string(),
                    "content": r.content,
                    "score": r.score,
                    "sources": r.sources,
                    "metadata": r.metadata,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "query": query,
                "count": json_results.len(),
                "results": json_results,
            }))?
        );
        return Ok(());
    }

    // Unified hybrid search for all non-JSON modes.
    // Tries full hybrid (text + vector + graph + xpath + recency), falls back to text-only.
    let search = remembrant_engine::HybridSearch::new(&store);

    let results = if exact || remembrant_engine::is_xpath_query(query) {
        // Exact/XPath: text-only (no embeddings needed)
        if remembrant_engine::is_xpath_query(query) {
            search.search_xpath(query, 20)?
        } else {
            search.search_text_only(query, 40)?
        }
    } else {
        // Full hybrid: try vector search, fall back to text-only
        let embedder = LmStudioEmbedder::from_config(&config.embedding);
        match open_lance_store(&config).await {
            Ok(lance) => {
                search
                    .search(query, 40, Some(&lance), Some(&store), Some(&embedder))
                    .await?
            }
            Err(e) => {
                eprintln!("Warning: LanceDB unavailable ({e:#}). Using text-only search.");
                search.search_text_only(query, 40)?
            }
        }
    };

    // Post-filter by project, agent, content_type, since
    let results = filter_search_results(results, project, agent, content_type, since_dt)
        .into_iter()
        .take(20)
        .collect::<Vec<_>>();

    if results.is_empty() {
        println!("No results found for: {query}");
        return Ok(());
    }

    let mode_label = if exact { "Exact" } else { "Hybrid" };
    println!("{mode_label} search results for \"{query}\":\n");
    for r in &results {
        let sources = r.sources.join("+");
        println!(
            "[{:.3}] {} ({}) -- {}",
            r.score,
            r.result_type,
            sources,
            r.metadata
                .get("project")
                .or_else(|| r.metadata.get("file_path"))
                .map(|s| s.as_str())
                .unwrap_or("-")
        );
        println!("  {}", truncate(&r.content, 80));
        println!();
    }
    println!("{} result(s).", results.len());

    Ok(())
}

fn cmd_find(query: &str) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    let mut memories = store.search_memories(query)?;
    let mut memory_ids: std::collections::HashSet<String> =
        memories.iter().map(|memory| memory.id.clone()).collect();
    for memory in store.search_memories_by_tag(query, 100)? {
        if memory_ids.insert(memory.id.clone()) {
            memories.push(memory);
        }
    }
    let sessions = store.search_sessions_by_summary(query)?;

    let total = memories.len() + sessions.len();
    if total == 0 {
        println!("No results found for: {query}");
        return Ok(());
    }

    println!("Find results for \"{query}\":\n");

    if !memories.is_empty() {
        println!("--- Memories ({}) ---", memories.len());
        for m in &memories {
            let mtype = m.memory_type.as_deref().unwrap_or("unknown");
            let proj = m.project_id.as_deref().unwrap_or("-");
            println!("  [{mtype}] {proj}");
            println!("    {}", truncate(&m.content, 80));
            println!();
        }
    }

    if !sessions.is_empty() {
        println!("--- Sessions ({}) ---", sessions.len());
        for s in &sessions {
            let started = s
                .started_at
                .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                .unwrap_or_else(|| "-".to_string());
            let summary = s.summary.as_deref().unwrap_or("-");
            let id_short = if s.id.len() > 12 {
                truncate(&s.id, 12)
            } else {
                s.id.clone()
            };
            println!(
                "  [{id_short}] {}: {} -- {}",
                s.agent,
                started,
                truncate(summary, 60)
            );
        }
        println!();
    }

    println!("{total} result(s) total.");
    Ok(())
}

fn cmd_brief(project: Option<&str>, today: bool, json_output: bool) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    let now = chrono::Utc::now().naive_utc();
    let today_date = now.date();

    // Determine time window
    let since = if today {
        utc_day_start(now)
    } else {
        now - chrono::Duration::days(3)
    };

    let sessions = store.search_sessions(None, project, Some(since), 1000)?;
    let memories = store.get_memories(project, 10)?;
    let facts = store.get_active_facts(project, 20)?;
    let decisions = store.get_decisions(project, 10)?;

    if json_output {
        let json_sessions: Vec<serde_json::Value> = sessions
            .iter()
            .map(|s| {
                serde_json::json!({
                    "id": s.id,
                    "agent": s.agent,
                    "project": s.project_id,
                    "summary": s.summary,
                    "files_changed": s.files_changed,
                    "messages": s.message_count,
                    "tools": s.tool_call_count,
                    "duration_min": s.duration_minutes,
                })
            })
            .collect();
        let json_facts: Vec<serde_json::Value> = facts
            .iter()
            .map(|f| {
                serde_json::json!({
                    "subject": f.subject,
                    "predicate": f.predicate,
                    "object": f.object,
                    "confidence": f.confidence,
                })
            })
            .collect();
        let json_memories: Vec<serde_json::Value> = memories
            .iter()
            .map(|m| {
                serde_json::json!({
                    "id": m.id,
                    "type": m.memory_type,
                    "content": m.content,
                    "confidence": m.confidence,
                })
            })
            .collect();
        let json_decisions: Vec<serde_json::Value> = decisions
            .iter()
            .map(|d| {
                serde_json::json!({
                    "what": d.what,
                    "why": d.why,
                    "alternatives": d.alternatives,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "date": today_date.format("%Y-%m-%d").to_string(),
                "project": project,
                "sessions": json_sessions,
                "facts": json_facts,
                "memories": json_memories,
                "decisions": json_decisions,
            }))?
        );
        return Ok(());
    }

    let window_label = if today { "today" } else { "3 days" };

    println!("=== Daily Brief ({}) ===\n", today_date.format("%Y-%m-%d"));

    // Recent sessions
    println!("Recent Sessions (last {window_label}):");
    if sessions.is_empty() {
        println!("  (no sessions)");
    } else {
        for s in &sessions {
            let summary = s.summary.as_deref().unwrap_or("(no summary)");
            let duration = s
                .duration_minutes
                .map(|d| format!("{d} min"))
                .unwrap_or_else(|| "-".to_string());
            let msgs = s
                .message_count
                .map(|n| format!("{n} msgs"))
                .unwrap_or_else(|| "-".to_string());
            let proj = s.project_id.as_deref().unwrap_or("-");
            println!(
                "  - [{}] {}: {} ({}, {})",
                s.agent,
                proj,
                truncate(summary, 50),
                duration,
                msgs
            );
        }
    }
    println!();

    // Recent memories
    println!("Recent Memories:");
    if memories.is_empty() {
        println!("  (no memories)");
    } else {
        for m in &memories {
            let mtype = m.memory_type.as_deref().unwrap_or("unknown");
            println!("  - [{mtype}] {}", truncate(&m.content, 60));
        }
    }
    println!();

    // Active projects
    let mut project_counts: HashMap<String, (usize, i32, i32)> = HashMap::new();
    let mut total_msgs = 0i32;
    let mut total_tools = 0i32;

    for s in &sessions {
        let proj = s.project_id.as_deref().unwrap_or("(unknown)").to_string();
        let entry = project_counts.entry(proj).or_insert((0, 0, 0));
        entry.0 += 1;
        entry.1 += s.message_count.unwrap_or(0);
        entry.2 += s.tool_call_count.unwrap_or(0);
        total_msgs += s.message_count.unwrap_or(0);
        total_tools += s.tool_call_count.unwrap_or(0);
    }

    println!("Active Projects:");
    if project_counts.is_empty() {
        println!("  (none)");
    } else {
        let mut sorted: Vec<_> = project_counts.iter().collect();
        sorted.sort_by_key(|item| std::cmp::Reverse(item.1.0));
        for (proj, (count, _, _)) in &sorted {
            let label = if *count == 1 { "session" } else { "sessions" };
            println!("  - {proj} ({count} {label})");
        }
    }
    println!();

    println!(
        "Summary: {} sessions, {} messages, {} tool calls across {} projects.",
        sessions.len(),
        total_msgs,
        total_tools,
        project_counts.len()
    );

    Ok(())
}

fn cmd_context_brief(project: Option<&str>, max_tokens: usize, json: bool) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;
    let assembler = remembrant_engine::ContextAssembler::new(&store).with_max_tokens(max_tokens);
    let ctx = assembler.project_context(project)?;
    if json {
        println!("{}", ctx.to_json()?);
    } else {
        print!("{}", ctx.to_prompt_block());
    }
    Ok(())
}

fn cmd_context_topic(
    topic: &str,
    project: Option<&str>,
    max_tokens: usize,
    json: bool,
) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;
    let assembler = remembrant_engine::ContextAssembler::new(&store).with_max_tokens(max_tokens);
    let ctx = assembler.topic_context(topic, project)?;
    if json {
        println!("{}", ctx.to_json()?);
    } else {
        print!("{}", ctx.to_prompt_block());
    }
    Ok(())
}

fn cmd_consolidate(project: Option<&str>, threshold: f64, json: bool) -> Result<()> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        anyhow::bail!("--threshold must be between 0.0 and 1.0");
    }

    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    let (stats, scores, candidates) = remembrant_engine::consolidate(&store, project, threshold)?;

    if json {
        let json_scores: Vec<serde_json::Value> = scores
            .iter()
            .take(20)
            .map(|s| {
                serde_json::json!({
                    "memory_id": s.memory_id,
                    "score": (s.score * 1000.0).round() / 1000.0,
                    "confidence": s.components.confidence,
                    "access_freq": (s.components.access_frequency * 100.0).round() / 100.0,
                    "recency": (s.components.recency * 100.0).round() / 100.0,
                })
            })
            .collect();
        let json_candidates: Vec<serde_json::Value> = candidates
            .iter()
            .take(10)
            .map(|c| {
                serde_json::json!({
                    "memory_a": c.memory_a,
                    "memory_b": c.memory_b,
                    "similarity": (c.similarity * 100.0).round() / 100.0,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "expired": stats.expired_count,
                "merge_candidates": json_candidates,
                "decay_scores": json_scores,
            }))?
        );
        return Ok(());
    }

    println!("=== Memory Consolidation ===\n");
    println!(
        "Expired {} stale memories (past valid_until).",
        stats.expired_count
    );
    println!("Scored {} memories.", stats.scored_count);
    println!();

    if !candidates.is_empty() {
        println!(
            "Merge Candidates ({} found, threshold: {:.0}%):",
            candidates.len(),
            threshold * 100.0
        );
        for c in candidates.iter().take(10) {
            println!(
                "  {:.0}% similar: {} <-> {}",
                c.similarity * 100.0,
                c.memory_a,
                c.memory_b
            );
        }
        println!();
    }

    println!("Top Memories by Decay Score:");
    for (i, s) in scores.iter().take(15).enumerate() {
        println!(
            "  {}. [{:.3}] {} (conf={:.0}% access={:.2} recency={:.2})",
            i + 1,
            s.score,
            s.memory_id,
            s.components.confidence * 100.0,
            s.components.access_frequency,
            s.components.recency,
        );
    }

    if scores.len() > 15 {
        println!("  ... and {} more", scores.len() - 15);
    }

    Ok(())
}

fn cmd_related(path: &str) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    let builder = build_graph(&store)?;

    let node_id = match builder.find_node_id(path)? {
        Some(id) => id,
        None => {
            println!("No node found matching: {path}");
            println!("Tip: use a file path (e.g. src/main.rs), session ID, or project ID.");
            return Ok(());
        }
    };

    let neighbors = builder.find_related(&node_id, 2)?;
    let output = graph_builder::format_related(path, &neighbors);
    print!("{output}");
    Ok(())
}

fn cmd_graph(path: &str) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    let builder = build_graph(&store)?;

    let node_id = match builder.find_node_id(path)? {
        Some(id) => id,
        None => {
            println!("No node found matching: {path}");
            println!("Tip: use a file path (e.g. src/main.rs), session ID, or project ID.");
            return Ok(());
        }
    };

    let node_name = GraphBackend::get_node(builder.backend(), &node_id)?
        .map(|(_, _, name, _)| name)
        .unwrap_or_else(|| path.to_string());

    let neighbors = builder.find_related(&node_id, 2)?;
    let output = graph_builder::format_graph_tree(path, &node_name, &neighbors);
    print!("{output}");
    Ok(())
}

fn cmd_timeline(topic: &str, since: Option<&str>) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    let sessions = store.search_sessions_by_summary(topic)?;
    let memories = store.search_memories(topic)?;

    // Apply optional since filter
    let (sessions, memories) = if let Some(since_str) = since {
        let since_dt = parse_since_required(Some(since_str))?
            .context("a --since value must produce a timestamp")?;
        let filtered_sessions: Vec<_> = sessions
            .into_iter()
            .filter(|s| s.started_at.is_some_and(|dt| dt >= since_dt))
            .collect();
        let filtered_memories: Vec<_> = memories
            .into_iter()
            .filter(|m| m.created_at.is_some_and(|dt| dt >= since_dt))
            .collect();
        (filtered_sessions, filtered_memories)
    } else {
        (sessions, memories)
    };

    let output = graph_builder::format_timeline(topic, &sessions, &memories);
    print!("{output}");
    Ok(())
}

fn cmd_patterns(topic: Option<&str>) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    // Load pattern memories
    let all_memories = store.get_memories(None, 10_000)?;
    let pattern_memories: Vec<_> = all_memories
        .iter()
        .filter(|m| {
            m.memory_type
                .as_deref()
                .map(|t| t.contains("pattern"))
                .unwrap_or(false)
        })
        .filter(|m| {
            if let Some(t) = topic {
                m.content.to_lowercase().contains(&t.to_lowercase())
            } else {
                true
            }
        })
        .collect();

    if pattern_memories.is_empty() {
        println!("No patterns found.");
        if topic.is_some() {
            println!("Try without a topic filter, or run distillation to extract patterns.");
        }
        return Ok(());
    }

    // Group patterns by content and collect projects
    let mut pattern_groups: HashMap<String, (Vec<String>, Option<NaiveDateTime>)> = HashMap::new();
    for m in &pattern_memories {
        let content = m.content.clone();
        let entry = pattern_groups
            .entry(content)
            .or_insert_with(|| (Vec::new(), None));
        if let Some(ref pid) = m.project_id
            && !entry.0.contains(pid)
        {
            entry.0.push(pid.clone());
        }
        if let Some(created) = m.created_at {
            match entry.1 {
                None => entry.1 = Some(created),
                Some(existing) if created < existing => entry.1 = Some(created),
                _ => {}
            }
        }
    }

    // Sort by number of projects (descending)
    let mut sorted: Vec<_> = pattern_groups.into_iter().collect();
    sorted.sort_by_key(|item| std::cmp::Reverse(item.1.0.len()));

    println!("Cross-Project Patterns:\n");
    for (content, (projects, first_seen)) in &sorted {
        let count = projects.len();
        let projects_str = if projects.is_empty() {
            "(no project)".to_string()
        } else {
            projects.join(", ")
        };
        let first_str = first_seen
            .map(|dt| dt.format("%Y-%m-%d").to_string())
            .unwrap_or_else(|| "unknown".to_string());
        println!("[{count}x] {}", truncate(content, 80));
        println!("  Projects: {projects_str}");
        println!("  First seen: {first_str}");
        println!();
    }

    let project_set: std::collections::HashSet<&str> = sorted
        .iter()
        .flat_map(|(_, (projs, _))| projs.iter().map(|s| s.as_str()))
        .collect();
    println!(
        "Found {} patterns across {} projects.",
        sorted.len(),
        project_set.len()
    );
    Ok(())
}

fn cmd_decisions(project: Option<&str>, all: bool) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    let limit = if all { 10_000 } else { 20 };
    let decisions = store.get_decisions(project, limit)?;

    if decisions.is_empty() {
        println!("No decisions found.");
        return Ok(());
    }

    let total = store.count_decisions()?;

    println!("Decision Journal:\n");
    for d in &decisions {
        let date = d
            .created_at
            .map(|dt| dt.format("%Y-%m-%d").to_string())
            .unwrap_or_else(|| "????-??-??".to_string());
        let proj = d.project_id.as_deref().unwrap_or("?");
        println!("{}  [{}] {}", date, proj, truncate(&d.what, 70));
        if let Some(ref why) = d.why {
            println!("  Why: {}", truncate(why, 70));
        }
        if !d.alternatives.is_empty() {
            println!("  Alternatives: {}", d.alternatives.join(", "));
        }
        println!();
    }

    if !all && total > decisions.len() {
        println!(
            "Showing {} of {} decisions. Use --all to see all.",
            decisions.len(),
            total
        );
    }

    Ok(())
}

fn cmd_note(text: &str, project: Option<&str>, tags: Option<&[String]>) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    let id = store.insert_note_with_tags(text, project, tags.unwrap_or_default())?;

    let tag_str = tags.map(|t| t.join(", ")).unwrap_or_default();

    println!("Note saved: {id}");
    if let Some(p) = project {
        println!("  Project: {p}");
    }
    if !tag_str.is_empty() {
        println!("  Tags: {tag_str}");
    }
    Ok(())
}

fn cmd_forget(session_id: &str) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    if store.delete_session(session_id)? {
        println!("Session {session_id} deleted.");
    } else {
        anyhow::bail!("Session {session_id} not found");
    }
    Ok(())
}

fn cmd_export(project: Option<&str>, format: &str, output: Option<&str>) -> Result<()> {
    if !matches!(format, "markdown" | "json") {
        anyhow::bail!("unsupported export format '{format}'; use markdown or json");
    }

    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    let memories = store.get_memories(project, 10_000)?;
    let decisions = store.get_decisions(project, 10_000)?;
    let sessions = if let Some(proj) = project {
        store.get_project_sessions(proj)?
    } else {
        let session_count = store.count_sessions()?;
        store.get_recent_sessions(session_count.max(1))?
    };

    let content = match format {
        "json" => {
            let data = serde_json::json!({
                "project": project.unwrap_or("all"),
                "generated_at": chrono::Utc::now().naive_utc().format("%Y-%m-%d").to_string(),
                "memories": memories,
                "decisions": decisions,
                "sessions": sessions,
            });
            serde_json::to_string_pretty(&data).context("failed to serialize JSON")?
        }
        _ => {
            // Markdown: CLAUDE.md-style document
            let mut md = String::new();
            let proj_label = project.unwrap_or("All Projects");
            md.push_str(&format!("# Project Memory: {proj_label}\n\n"));

            // Key Decisions
            md.push_str("## Key Decisions\n");
            if decisions.is_empty() {
                md.push_str("- (none recorded)\n");
            } else {
                for d in &decisions {
                    let date = d
                        .created_at
                        .map(|dt| dt.format("%Y-%m-%d").to_string())
                        .unwrap_or_default();
                    md.push_str(&format!("- {} ({})\n", d.what, date));
                }
            }
            md.push('\n');

            // Patterns
            let patterns: Vec<_> = memories
                .iter()
                .filter(|m| {
                    m.memory_type
                        .as_deref()
                        .map(|t| t.contains("pattern"))
                        .unwrap_or(false)
                })
                .collect();
            md.push_str("## Patterns\n");
            if patterns.is_empty() {
                md.push_str("- (none recorded)\n");
            } else {
                for m in &patterns {
                    md.push_str(&format!("- {}\n", m.content));
                }
            }
            md.push('\n');

            // Insights
            let insights: Vec<_> = memories
                .iter()
                .filter(|m| {
                    m.memory_type
                        .as_deref()
                        .map(|t| t == "insight" || t == "note")
                        .unwrap_or(false)
                })
                .collect();
            md.push_str("## Insights\n");
            if insights.is_empty() {
                md.push_str("- (none recorded)\n");
            } else {
                for m in &insights {
                    md.push_str(&format!("- {}\n", m.content));
                }
            }
            md.push('\n');

            // Recent Sessions
            md.push_str("## Recent Sessions\n");
            let session_limit = sessions.len().min(20);
            if sessions.is_empty() {
                md.push_str("- (none)\n");
            } else {
                for s in &sessions[..session_limit] {
                    let summary = s.summary.as_deref().unwrap_or("(no summary)");
                    let date = s
                        .started_at
                        .map(|dt| dt.format("%Y-%m-%d").to_string())
                        .unwrap_or_default();
                    let id_short = truncate(&s.id, 8);
                    md.push_str(&format!("- {id_short}: {summary} ({date}, {})\n", s.agent));
                }
            }
            md.push('\n');

            let today = chrono::Utc::now()
                .naive_utc()
                .format("%Y-%m-%d")
                .to_string();
            md.push_str(&format!("Generated by Remembrant on {today}\n"));

            md
        }
    };

    match output {
        Some(path) => {
            let out_path = expand_tilde(path);
            std::fs::write(&out_path, &content)
                .with_context(|| format!("failed to write to {}", out_path.display()))?;
            println!("Exported to {}", out_path.display());
        }
        None => {
            print!("{content}");
        }
    }

    Ok(())
}

fn cmd_stats() -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    let session_count = store.count_sessions()?;
    let memory_count = store.count_memories()?;
    let decision_count = store.count_decisions()?;
    let tool_call_count = store.count_tool_calls()?;
    let fact_count = store.count_facts()?;
    let active_fact_count = store.count_active_facts()?;

    // Per-agent breakdown
    let agent_counts = store.get_agent_session_counts()?;
    let agent_str = if agent_counts.is_empty() {
        String::new()
    } else {
        let parts: Vec<String> = agent_counts
            .iter()
            .map(|(agent, count)| format!("{agent}: {count}"))
            .collect();
        format!(" ({})", parts.join(", "))
    };

    println!("Remembrant Statistics:\n");
    println!("Sessions:    {session_count}{agent_str}");
    println!("Memories:    {memory_count}");
    println!("Decisions:   {decision_count}");
    println!("Facts:       {active_fact_count} active / {fact_count} total");
    println!("Tool calls:  {tool_call_count}");

    // Per-project
    let project_counts = store.get_project_session_counts()?;
    if !project_counts.is_empty() {
        println!("\nTop Projects:");
        for (project, count) in &project_counts {
            let label = if *count == 1 { "session" } else { "sessions" };
            println!("  {:<20} {count} {label}", project);
        }
    }

    // Storage sizes
    let db_path = expand_tilde(&config.storage.duckdb_path);
    let lance_path = expand_tilde(&config.storage.lancedb_path);

    let duck_size = if db_path.exists() {
        std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0)
    } else {
        0
    };

    let lance_size = if lance_path.exists() {
        dir_size(&lance_path)
    } else {
        0
    };

    println!(
        "\nStorage: DuckDB {}, LanceDB {}",
        human_size(duck_size),
        human_size(lance_size)
    );

    Ok(())
}

/// Compute total size of a directory recursively.
fn dir_size(path: &PathBuf) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let meta = entry.metadata();
            if let Ok(meta) = meta {
                if meta.is_file() {
                    total += meta.len();
                } else if meta.is_dir() {
                    total += dir_size(&entry.path());
                }
            }
        }
    }
    total
}

fn cmd_gc() -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    let retention_days = config.retention.raw_transcripts_days;
    let retention_duration = chrono::Duration::try_days(retention_days as i64)
        .context("retention duration is out of range")?;
    let cutoff = chrono::Utc::now()
        .naive_utc()
        .checked_sub_signed(retention_duration)
        .context("retention cutoff is out of range")?;

    // Get DB size before
    let db_path = expand_tilde(&config.storage.duckdb_path);
    let size_before = if db_path.exists() {
        std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0)
    } else {
        0
    };

    let deleted = store.gc_sessions_before(cutoff)?;
    let orphaned_tool_calls = store.delete_orphaned_tool_calls()?;

    let size_after = if db_path.exists() {
        std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0)
    } else {
        0
    };

    let freed = size_before.saturating_sub(size_after);

    println!("Garbage collection:");
    println!("  Deleted {deleted} sessions older than {retention_days} days");
    println!("  Deleted {orphaned_tool_calls} orphaned tool calls");
    println!("  Freed ~{}", human_size(freed));

    Ok(())
}

async fn cmd_ingest(skip_embed: bool, skip_distill: bool) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    println!("╔══════════════════════════════════════════╗");
    println!("║  REMEMBRANT — Full Ingest Pipeline       ║");
    println!("╚══════════════════════════════════════════╝\n");

    // ── Step 1: Detect & parse agent artifacts ──────────────────────
    println!("▸ Step 1/4: Parsing agent artifacts...");

    let mut all_sessions = Vec::new();
    let mut all_memories = Vec::new();
    let mut all_tool_calls = Vec::new();
    let mut parse_errors = 0usize;

    let registry = remembrant_engine::ingest::build_registry_from_config(&config)?;
    for adapter in registry.detected() {
        let meta = adapter.meta();
        match adapter.ingest() {
            Ok(result) => {
                println!(
                    "  ✓ {}: {} sessions, {} tool calls, {} memories",
                    meta.display_name,
                    result.sessions.len(),
                    result.tool_calls.len(),
                    result.memories.len()
                );

                for session in &result.sessions {
                    store.insert_or_replace_session(session).with_context(|| {
                        format!("failed to persist {} session {}", meta.id, session.id)
                    })?;
                }
                for memory in &result.memories {
                    store
                        .insert_memory(memory)
                        .with_context(|| format!("failed to persist {} memory", meta.id))?;
                }
                for tool_call in &result.tool_calls {
                    store.insert_tool_call(tool_call).with_context(|| {
                        format!("failed to persist {} tool call {}", meta.id, tool_call.id)
                    })?;
                }

                all_sessions.extend(result.sessions);
                all_memories.extend(result.memories);
                all_tool_calls.extend(result.tool_calls);
                for error in &result.errors {
                    eprintln!("  ! {} parse error: {error}", meta.id);
                }
                parse_errors += result.errors.len();
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("failed to ingest {}", meta.display_name));
            }
        }
    }
    let total_s = all_sessions.len();
    let total_m = all_memories.len();
    let total_tc = all_tool_calls.len();
    println!("\n  Total: {total_s} sessions, {total_tc} tool calls, {total_m} memories → DuckDB ✓");
    if parse_errors > 0 {
        anyhow::bail!("{parse_errors} agent artifact parse error(s) occurred during ingestion");
    }

    // ── Step 2: LLM Distillation ────────────────────────────────────
    if skip_distill {
        println!("\n▸ Step 2/4: Distillation — skipped (--skip-distill)");
    } else {
        println!("\n▸ Step 2/4: Distilling with LLM...");

        let distiller = Distiller::new(&config.distillation);
        let has_llm = !config.distillation.llm_model.is_empty();

        if has_llm {
            println!("  Using LLM: {}", config.distillation.llm_model);
        } else {
            println!("  No LLM configured — using keyword extraction fallback");
            println!("  Tip: Set distillation.llm_model in ~/.remembrant/config.yaml");
        }

        let mut decisions_count = 0usize;
        let mut patterns_count = 0usize;
        let mut problems_count = 0usize;
        let mut facts_count = 0usize;

        for session in &all_sessions {
            let summary = session.summary.as_deref().unwrap_or("");
            let files = session.files_changed.join(", ");
            let text = format!(
                "Session: {}\nAgent: {}\nSummary: {}\nFiles: {}",
                session.id, session.agent, summary, files
            );

            match distiller.distill_session(session, &text).await {
                Ok(distilled) => {
                    for d in distiller.to_decisions(&distilled) {
                        store
                            .insert_decision(&d)
                            .context("failed to persist distilled decision")?;
                        decisions_count += 1;
                    }
                    for m in distiller.to_memories(&distilled) {
                        store
                            .insert_memory(&m)
                            .context("failed to persist distilled memory")?;
                        patterns_count += 1;
                    }
                    for f in distiller.to_facts(&distilled, Some(&session.agent)) {
                        store
                            .upsert_fact(&f)
                            .context("failed to persist distilled fact")?;
                        facts_count += 1;
                    }
                    problems_count += distilled.problems.len();
                }
                Err(e) => {
                    eprintln!("  ✗ Distill session {}: {e}", session.id);
                }
            }
        }

        println!(
            "  Extracted: {decisions_count} decisions, {patterns_count} patterns, {problems_count} problems, {facts_count} facts"
        );
    }

    // ── Step 3: Embeddings ──────────────────────────────────────────
    if skip_embed {
        println!("\n▸ Step 3/4: Embedding — skipped (--skip-embed)");
    } else {
        println!("\n▸ Step 3/4: Embedding with LM Studio...");
        println!("  Model: {}", config.embedding.model);
        println!("  Dimensions: {}", config.embedding.dimensions);

        let embedder = LmStudioEmbedder::from_config(&config.embedding);

        // Test connection first
        match embedder.embed_texts(&["test"]).await {
            Ok(_) => println!("  ✓ LM Studio connection OK"),
            Err(e) => {
                eprintln!("  ✗ LM Studio not reachable: {e}");
                eprintln!("  Start LM Studio and load an embedding model, then retry.");
                eprintln!("  Skipping embedding step.\n");
                println!("▸ Step 4/4: Building graph...");
                let graph = build_graph(&store)?;
                let node_count = graph.node_count();
                let edge_count = graph.edge_count();
                println!("  Graph: {node_count} nodes, {edge_count} edges");
                print_summary(total_s, total_m, total_tc);
                return Ok(());
            }
        }

        let lance = open_lance_store(&config).await?;
        let pipeline = EmbedPipeline::new(lance, config.embedding.batch_size);

        let stats = pipeline
            .run(&all_sessions, &all_memories, &all_tool_calls, &embedder)
            .await?;

        println!(
            "  Embedded: {} chunks ({} stored, {} errors)",
            stats.chunks_embedded, stats.chunks_stored, stats.errors
        );
        if stats.errors > 0 {
            anyhow::bail!("embedding pipeline reported {} errors", stats.errors);
        }
    }

    // ── Step 4: Graph ───────────────────────────────────────────────
    println!("\n▸ Step 4/4: Building relationship graph...");
    let graph = build_graph(&store)?;
    let node_count = graph.node_count();
    let edge_count = graph.edge_count();
    println!("  Graph: {node_count} nodes, {edge_count} edges");

    print_summary(total_s, total_m, total_tc);
    Ok(())
}

fn print_summary(sessions: usize, memories: usize, tool_calls: usize) {
    println!("\n╔══════════════════════════════════════════╗");
    println!("║  Pipeline complete                       ║");
    println!("╠══════════════════════════════════════════╣");
    println!("║  Sessions:   {:>6}                      ║", sessions);
    println!("║  Memories:   {:>6}                      ║", memories);
    println!("║  Tool calls: {:>6}                      ║", tool_calls);
    println!("╚══════════════════════════════════════════╝");
    println!("\nRun `rem stats` for full analytics.");
    println!("Run `rem recent` to browse sessions.");
    println!("Run `rem search <query>` to search.");
}

/// Build an in-memory graph from all DuckDB data.
fn build_graph(store: &DuckStore) -> Result<GraphBuilder<&DuckStore>> {
    let session_count = store.count_sessions()?;
    let memory_count = store.count_memories()?;
    let decision_count = store.count_decisions()?;
    let sessions = store.get_recent_sessions(session_count.max(1))?;
    let memories = store.get_memories(None, memory_count.max(1))?;
    let decisions = store.get_decisions(None, decision_count.max(1))?;
    let tool_calls = Vec::new();

    let builder = GraphBuilder::with_backend(store);
    builder.build_from_data(&sessions, &memories, &decisions, &tool_calls)?;
    Ok(builder)
}

async fn cmd_embed(path: &str, update: bool) -> Result<()> {
    let config = AppConfig::load()?;
    let abs_path =
        std::fs::canonicalize(path).with_context(|| format!("failed to resolve path: {path}"))?;

    println!("Embedding repository: {}", abs_path.display());
    if update {
        println!("Mode: replace existing embeddings for this project");
    }

    let embedder = RepoEmbedder::new(&abs_path);

    // Discover and chunk first so we can report counts before embedding.
    let (chunks, file_count) = embedder.chunk_all()?;
    println!("Found {} chunks from {} files", chunks.len(), file_count);

    if chunks.is_empty() {
        println!("No embeddable files found. Nothing to do.");
        return Ok(());
    }

    // Create LM Studio embedder.
    let embed_provider = LmStudioEmbedder::from_config(&config.embedding);

    // Open LanceDB.
    let lance_store = open_lance_store(&config).await?;

    let result = embedder
        .embed_and_store_with_update(
            &embed_provider,
            &lance_store,
            config.embedding.batch_size,
            update,
        )
        .await;

    match result {
        Ok(result) => {
            println!("\nEmbed complete:");
            println!("  Files:  {}", result.files_found);
            println!(
                "  Chunks: {} created, {} embedded",
                result.chunks_created, result.chunks_embedded
            );
            if result.errors > 0 {
                println!("  Errors: {}", result.errors);
                anyhow::bail!("{} repository chunk(s) failed to embed", result.errors);
            }
        }
        Err(e) => {
            let err_msg = format!("{e:#}");
            if err_msg.contains("connect") || err_msg.contains("LM Studio") {
                eprintln!(
                    "Error: Could not connect to LM Studio.\n\n\
                     To use `rem embed`, you need LM Studio running locally:\n\
                     1. Download LM Studio from https://lmstudio.ai\n\
                     2. Start it and load an embedding model (e.g. nomic-embed-text)\n\
                     3. Start the local server (it listens on http://localhost:1234)\n\
                     4. Run `rem embed` again\n\n\
                     Underlying error: {e:#}"
                );
                return Err(e).context("embed failed");
            } else {
                return Err(e).context("embed failed");
            }
        }
    }

    Ok(())
}

fn cmd_xpath(
    query: &str,
    depth: usize,
    limit: usize,
    show_tree: bool,
    json_output: bool,
) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    // Parse the query
    let parsed = remembrant_engine::xpath_query::parse(query)
        .map_err(|e| anyhow::anyhow!("Parse error at position {}: {}", e.position, e.message))?;

    // Build the tree
    let builder = remembrant_engine::TreeBuilder::new(&store);
    let root = builder.build_tree(depth)?;

    // Evaluate with keyword scorer (no embeddings needed)
    let scorer = remembrant_engine::semantic_scorer::keyword_scorer;
    let results = remembrant_engine::xpath_query::evaluate(&parsed, &root, &scorer);

    if json_output {
        // Agent-friendly JSON output
        let json_results: Vec<serde_json::Value> = results
            .iter()
            .take(limit)
            .map(|r| {
                serde_json::json!({
                    "node_id": r.node_id,
                    "node_type": r.node_type,
                    "name": r.name,
                    "weight": r.weight,
                    "path": r.path,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "query": query,
                "count": results.len(),
                "results": json_results,
            }))?
        );
        return Ok(());
    }

    // Display results
    if results.is_empty() {
        println!("No results found for: {query}");
        return Ok(());
    }

    let display_count = results.len().min(limit);
    println!("Found {} results for: {query}\n", results.len());

    for (i, result) in results.iter().take(display_count).enumerate() {
        println!(
            "{}. [{}] {} (score: {:.3})",
            i + 1,
            result.node_type,
            result.name,
            result.weight,
        );
        if show_tree {
            let path_str = result.path.join(" -> ");
            println!("   Path: {path_str}");
        }
    }

    if results.len() > display_count {
        println!(
            "\n... and {} more results (use --limit to show more)",
            results.len() - display_count
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Web server
// ---------------------------------------------------------------------------

struct WebState {
    config: AppConfig,
}

impl WebState {
    fn store(&self) -> Result<DuckStore, StatusCode> {
        open_store(&self.config).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
    }
}

/// Keep local API page sizes bounded without rejecting client typos such as
/// zero or an accidentally enormous value.
fn api_limit(requested: Option<usize>, default: usize, maximum: usize) -> usize {
    requested.unwrap_or(default).clamp(1, maximum)
}

/// Parse a comma-separated `agent=` filter value (e.g. `codex,gemini`)
/// into a clean list, dropping empty segments and surrounding whitespace.
fn parse_agent_filter(raw: Option<String>) -> Vec<String> {
    raw.map(|value| {
        value
            .split(',')
            .map(str::trim)
            .filter(|agent| !agent.is_empty())
            .map(str::to_string)
            .collect()
    })
    .unwrap_or_default()
}

async fn web_index() -> impl axum::response::IntoResponse {
    (
        [
            (axum::http::header::CONTENT_TYPE, "text/html; charset=utf-8"),
            (axum::http::header::CACHE_CONTROL, "no-store"),
        ],
        include_str!("web_dashboard.html"),
    )
}

async fn web_dashboard_css() -> impl axum::response::IntoResponse {
    (
        [
            (axum::http::header::CONTENT_TYPE, "text/css; charset=utf-8"),
            (axum::http::header::CACHE_CONTROL, "no-cache"),
        ],
        include_str!("web_dashboard.css"),
    )
}

async fn web_dashboard_util_js() -> impl axum::response::IntoResponse {
    (
        [
            (
                axum::http::header::CONTENT_TYPE,
                "application/javascript; charset=utf-8",
            ),
            (axum::http::header::CACHE_CONTROL, "no-cache"),
        ],
        include_str!("web_dashboard_util.js"),
    )
}

async fn web_dashboard_js() -> impl axum::response::IntoResponse {
    (
        [
            (
                axum::http::header::CONTENT_TYPE,
                "application/javascript; charset=utf-8",
            ),
            (axum::http::header::CACHE_CONTROL, "no-cache"),
        ],
        include_str!("web_dashboard.js"),
    )
}

async fn web_stats(
    State(state): State<Arc<WebState>>,
) -> Result<axum::Json<serde_json::Value>, StatusCode> {
    let store = state.store()?;
    let map_store_error = |_| StatusCode::INTERNAL_SERVER_ERROR;
    let sessions = store.count_sessions().map_err(map_store_error)?;
    let memories = store.count_memories().map_err(map_store_error)?;
    let decisions = store.count_decisions().map_err(map_store_error)?;
    let tool_calls = store.count_tool_calls().map_err(map_store_error)?;
    let facts = store.count_facts().map_err(map_store_error)?;
    let active_facts = store.count_active_facts().map_err(map_store_error)?;
    let projects = store.get_project_ids().map_err(map_store_error)?;

    let now = chrono::Utc::now().naive_utc();
    let today_start = now.date().and_hms_opt(0, 0, 0).unwrap_or(now);
    let week_start = (now - chrono::Duration::days(7))
        .date()
        .and_hms_opt(0, 0, 0)
        .unwrap_or(now);

    let today_sessions = store
        .count_sessions_since(today_start)
        .map_err(map_store_error)?;
    let today_memories = store
        .count_memories_since(today_start)
        .map_err(map_store_error)?;
    let today_decisions = store
        .count_decisions_since(today_start)
        .map_err(map_store_error)?;
    let today_tool_calls = store
        .count_tool_calls_since(today_start)
        .map_err(map_store_error)?;

    let week_sessions = store
        .count_sessions_since(week_start)
        .map_err(map_store_error)?;
    let week_memories = store
        .count_memories_since(week_start)
        .map_err(map_store_error)?;
    let week_decisions = store
        .count_decisions_since(week_start)
        .map_err(map_store_error)?;
    let week_tool_calls = store
        .count_tool_calls_since(week_start)
        .map_err(map_store_error)?;

    Ok(axum::Json(serde_json::json!({
        "sessions": sessions,
        "memories": memories,
        "decisions": decisions,
        "tool_calls": tool_calls,
        "facts": facts,
        "active_facts": active_facts,
        "projects": projects.len(),
        "today": {
            "sessions": today_sessions,
            "memories": today_memories,
            "decisions": today_decisions,
            "tool_calls": today_tool_calls,
        },
        "week": {
            "sessions": week_sessions,
            "memories": week_memories,
            "decisions": week_decisions,
            "tool_calls": week_tool_calls,
        },
    })))
}

async fn web_projects(
    State(state): State<Arc<WebState>>,
) -> Result<axum::Json<serde_json::Value>, StatusCode> {
    let store = state.store()?;
    let projects = store
        .get_project_ids()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(axum::Json(
        serde_json::to_value(&projects).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
    ))
}

#[derive(serde::Deserialize)]
struct SessionsQuery {
    limit: Option<usize>,
    project: Option<String>,
    agent: Option<String>,
}

async fn web_sessions(
    State(state): State<Arc<WebState>>,
    Query(q): Query<SessionsQuery>,
) -> Result<axum::Json<serde_json::Value>, StatusCode> {
    let store = state.store()?;
    let limit = api_limit(q.limit, 200, 1_000);
    let sessions = store
        .search_sessions(q.agent.as_deref(), q.project.as_deref(), None, limit)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(axum::Json(
        serde_json::to_value(&sessions).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
    ))
}

async fn web_session_detail(
    State(state): State<Arc<WebState>>,
    AxumPath(id): AxumPath<String>,
) -> Result<axum::Json<serde_json::Value>, StatusCode> {
    let store = state.store()?;
    let session = store
        .get_session(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let session = session.ok_or(StatusCode::NOT_FOUND)?;
    let tool_calls = store
        .get_tool_calls_for_session(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(axum::Json(serde_json::json!({
        "session": session,
        "tool_calls": tool_calls,
    })))
}

#[derive(serde::Deserialize)]
struct MemoriesQuery {
    project: Option<String>,
    tag: Option<String>,
    limit: Option<usize>,
    /// Comma-separated: `?agent=codex,gemini`.
    agent: Option<String>,
}

async fn web_memories(
    State(state): State<Arc<WebState>>,
    Query(q): Query<MemoriesQuery>,
) -> Result<axum::Json<serde_json::Value>, StatusCode> {
    let store = state.store()?;
    let limit = api_limit(q.limit, 200, 1_000);
    let agents = parse_agent_filter(q.agent);
    let memories = if let Some(tag) = q.tag.as_deref() {
        store
            .search_memories_by_tag(tag, limit)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            .into_iter()
            .filter(|memory| {
                q.project
                    .as_deref()
                    .is_none_or(|project| memory.project_id.as_deref() == Some(project))
            })
            .collect::<Vec<_>>()
    } else {
        store
            .get_memories_filtered(q.project.as_deref(), &agents, limit)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    };
    let mut values = serde_json::to_value(&memories)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .as_array_mut()
        .map(std::mem::take)
        .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;
    for (memory, value) in memories.iter().zip(&mut values) {
        let tags = store
            .get_memory_tags(&memory.id)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        value["tags"] =
            serde_json::to_value(tags).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    }
    Ok(axum::Json(serde_json::Value::Array(values)))
}

#[derive(serde::Deserialize)]
struct DecisionsQuery {
    project: Option<String>,
    /// Comma-separated: `?agent=codex,gemini`.
    agent: Option<String>,
}

async fn web_decisions(
    State(state): State<Arc<WebState>>,
    Query(q): Query<DecisionsQuery>,
) -> Result<axum::Json<serde_json::Value>, StatusCode> {
    let store = state.store()?;
    let agents = parse_agent_filter(q.agent);
    let decisions = store
        .get_decisions_filtered(q.project.as_deref(), &agents, 100)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(axum::Json(
        serde_json::to_value(&decisions).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
    ))
}

#[derive(serde::Deserialize)]
struct SearchQuery {
    q: String,
}

async fn web_search_sessions(
    State(state): State<Arc<WebState>>,
    Query(sq): Query<SearchQuery>,
) -> Result<axum::Json<serde_json::Value>, StatusCode> {
    let store = state.store()?;
    let sessions = store
        .search_sessions_by_summary(&sq.q)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(axum::Json(
        serde_json::to_value(&sessions).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
    ))
}

async fn web_search_memories(
    State(state): State<Arc<WebState>>,
    Query(sq): Query<SearchQuery>,
) -> Result<axum::Json<serde_json::Value>, StatusCode> {
    let store = state.store()?;
    let memories = store
        .search_memories(&sq.q)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(axum::Json(
        serde_json::to_value(&memories).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
    ))
}

// ---------------------------------------------------------------------------
// New API endpoint handlers
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct FactsQuery {
    project: Option<String>,
    active_only: Option<bool>,
    limit: Option<usize>,
    /// Comma-separated: `?agent=codex,gemini`.
    agent: Option<String>,
}

async fn web_facts(
    State(state): State<Arc<WebState>>,
    Query(q): Query<FactsQuery>,
) -> Result<axum::Json<serde_json::Value>, StatusCode> {
    let store = state.store()?;
    let limit = api_limit(q.limit, 100, 10_000);
    let active_only = q.active_only.unwrap_or(true);
    let agents = parse_agent_filter(q.agent);
    let facts = if active_only {
        store
            .get_active_facts_filtered(q.project.as_deref(), &agents, limit)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    } else {
        store
            .get_all_facts_filtered(q.project.as_deref(), &agents, limit)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    };
    Ok(axum::Json(
        serde_json::to_value(&facts).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
    ))
}

async fn web_fact_history(
    State(state): State<Arc<WebState>>,
    AxumPath(id): AxumPath<String>,
) -> Result<axum::Json<serde_json::Value>, StatusCode> {
    let store = state.store()?;
    let fact = store
        .get_fact(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;
    let history = store
        .get_fact_history(&fact.subject)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(axum::Json(
        serde_json::to_value(&history).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
    ))
}

async fn web_stats_agents(
    State(state): State<Arc<WebState>>,
) -> Result<axum::Json<serde_json::Value>, StatusCode> {
    let store = state.store()?;
    let agent_stats = store
        .get_agent_stats()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let result: Vec<serde_json::Value> = agent_stats
        .into_iter()
        .map(|(agent, sessions, total_tokens, avg_duration)| {
            serde_json::json!({
                "agent": agent,
                "sessions": sessions,
                "total_tokens": total_tokens,
                "avg_duration": avg_duration,
            })
        })
        .collect();
    Ok(axum::Json(serde_json::json!(result)))
}

async fn web_stats_tools(
    State(state): State<Arc<WebState>>,
) -> Result<axum::Json<serde_json::Value>, StatusCode> {
    let store = state.store()?;
    let tool_stats = store
        .get_tool_call_stats()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let result: Vec<serde_json::Value> = tool_stats
        .into_iter()
        .map(|(tool_name, count, success_count, avg_duration_ms)| {
            serde_json::json!({
                "tool_name": tool_name,
                "count": count,
                "success_count": success_count,
                "avg_duration_ms": avg_duration_ms,
            })
        })
        .collect();
    Ok(axum::Json(serde_json::json!(result)))
}

#[derive(serde::Deserialize)]
struct TimelineQuery {
    days: Option<i64>,
    agent: Option<String>,
}

async fn web_stats_timeline(
    State(state): State<Arc<WebState>>,
    Query(q): Query<TimelineQuery>,
) -> Result<axum::Json<serde_json::Value>, StatusCode> {
    let store = state.store()?;
    let days = q.days.unwrap_or(30).clamp(1, 365);
    let timeline = store
        .get_session_timeline(days, q.agent.as_deref())
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let result: Vec<serde_json::Value> = timeline
        .into_iter()
        .map(|(date, agent, count)| {
            serde_json::json!({
                "date": date,
                "agent": agent,
                "count": count,
            })
        })
        .collect();
    Ok(axum::Json(serde_json::json!(result)))
}

#[derive(serde::Deserialize)]
struct HotFilesQuery {
    project: Option<String>,
    limit: Option<usize>,
}

async fn web_hotfiles(
    State(state): State<Arc<WebState>>,
    Query(q): Query<HotFilesQuery>,
) -> Result<axum::Json<serde_json::Value>, StatusCode> {
    let store = state.store()?;
    let limit = api_limit(q.limit, 20, 100);
    let files = store
        .get_hot_files(q.project.as_deref(), limit)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let result: Vec<serde_json::Value> = files
        .into_iter()
        .map(|(path, freq)| {
            serde_json::json!({
                "file_path": path,
                "change_frequency": freq,
            })
        })
        .collect();
    Ok(axum::Json(serde_json::json!(result)))
}

async fn web_attention(
    State(state): State<Arc<WebState>>,
) -> Result<axum::Json<serde_json::Value>, StatusCode> {
    let store = state.store()?;
    let items = store
        .get_attention_items()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let total = items.len();
    Ok(axum::Json(serde_json::json!({
        "items": items,
        "total": total,
    })))
}

async fn web_search_facts(
    State(state): State<Arc<WebState>>,
    Query(sq): Query<SearchQuery>,
) -> Result<axum::Json<serde_json::Value>, StatusCode> {
    let store = state.store()?;
    let facts = store
        .search_facts(&sq.q)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(axum::Json(
        serde_json::to_value(&facts).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
    ))
}

#[derive(serde::Deserialize)]
struct XPathQuery {
    q: String,
    limit: Option<usize>,
}

async fn web_search_xpath(
    State(state): State<Arc<WebState>>,
    Query(xq): Query<XPathQuery>,
) -> Result<axum::Json<serde_json::Value>, StatusCode> {
    let store = state.store()?;
    let limit = api_limit(xq.limit, 20, 500);

    let parsed =
        remembrant_engine::xpath_query::parse(&xq.q).map_err(|_| StatusCode::BAD_REQUEST)?;

    let builder = remembrant_engine::TreeBuilder::new(&store);
    let root = builder
        .build_tree(3)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let scorer = remembrant_engine::semantic_scorer::keyword_scorer;
    let results = remembrant_engine::xpath_query::evaluate(&parsed, &root, &scorer);

    let json_results: Vec<serde_json::Value> = results
        .iter()
        .take(limit)
        .map(|r| {
            serde_json::json!({
                "node_id": r.node_id,
                "node_type": r.node_type,
                "name": r.name,
                "weight": r.weight,
                "path": r.path,
            })
        })
        .collect();

    Ok(axum::Json(serde_json::json!({
        "query": xq.q,
        "count": results.len(),
        "results": json_results,
    })))
}

#[derive(serde::Deserialize)]
struct SymbolsQuery {
    project: Option<String>,
    file: Option<String>,
    limit: Option<usize>,
}

async fn web_symbols(
    State(state): State<Arc<WebState>>,
    Query(q): Query<SymbolsQuery>,
) -> Result<axum::Json<serde_json::Value>, StatusCode> {
    let store = state.store()?;
    let limit = api_limit(q.limit, 100, 1_000);
    let symbols = if let (Some(file), Some(project)) = (&q.file, &q.project) {
        store
            .get_symbols_in_file(file, project)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    } else if let Some(project) = &q.project {
        store
            .get_symbols_for_project(project, limit)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    } else {
        // project is required for symbol queries
        return Err(StatusCode::BAD_REQUEST);
    };
    Ok(axum::Json(
        serde_json::to_value(&symbols).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
    ))
}

async fn web_graph_neighbors(
    State(state): State<Arc<WebState>>,
    AxumPath(id): AxumPath<String>,
) -> Result<axum::Json<serde_json::Value>, StatusCode> {
    let store = state.store()?;
    let neighbors = store
        .query_graph_neighbors(&id, None)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(axum::Json(
        serde_json::to_value(&neighbors).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
    ))
}

// --- Mutation endpoints ---

#[derive(serde::Deserialize)]
struct UpdateMemoryBody {
    content: Option<String>,
    confidence: Option<f32>,
    tags: Option<Vec<String>>,
}

async fn web_update_memory(
    State(state): State<Arc<WebState>>,
    AxumPath(id): AxumPath<String>,
    AxumJson(body): AxumJson<UpdateMemoryBody>,
) -> Result<axum::Json<serde_json::Value>, StatusCode> {
    let store = state.store()?;
    if body
        .content
        .as_deref()
        .is_some_and(|content| content.trim().is_empty())
    {
        return Err(StatusCode::BAD_REQUEST);
    }
    let updated = store
        .update_memory(&id, body.content.as_deref(), body.confidence)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    if !updated {
        return Err(StatusCode::NOT_FOUND);
    }

    if let Some(tags) = &body.tags {
        store
            .set_memory_tags(&id, tags)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    }
    Ok(axum::Json(serde_json::json!({"ok": true})))
}

async fn web_get_memory(
    State(state): State<Arc<WebState>>,
    AxumPath(id): AxumPath<String>,
) -> Result<axum::Json<serde_json::Value>, StatusCode> {
    let store = state.store()?;
    let memory = store
        .get_memory(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;
    let tags = store
        .get_memory_tags(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let mut value = serde_json::to_value(memory).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    value["tags"] = serde_json::to_value(tags).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(axum::Json(value))
}

async fn web_delete_memory(
    State(state): State<Arc<WebState>>,
    AxumPath(id): AxumPath<String>,
) -> Result<axum::Json<serde_json::Value>, StatusCode> {
    let store = state.store()?;
    let deleted = store
        .delete_memory(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    if deleted {
        Ok(axum::Json(serde_json::json!({"ok": true})))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

async fn web_delete_fact(
    State(state): State<Arc<WebState>>,
    AxumPath(id): AxumPath<String>,
) -> Result<axum::Json<serde_json::Value>, StatusCode> {
    let store = state.store()?;
    let deleted = store
        .delete_fact(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    if deleted {
        Ok(axum::Json(serde_json::json!({"ok": true})))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

#[derive(serde::Deserialize)]
struct CreateNoteBody {
    text: String,
    project: Option<String>,
    tags: Option<Vec<String>>,
}

async fn web_create_note(
    State(state): State<Arc<WebState>>,
    AxumJson(body): AxumJson<CreateNoteBody>,
) -> Result<(StatusCode, axum::Json<serde_json::Value>), StatusCode> {
    let store = state.store()?;
    if body.text.trim().is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }
    let tags = body.tags.clone().unwrap_or_default();
    let id = store
        .insert_note_with_tags(&body.text, body.project.as_deref(), &tags)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let tags = store
        .get_memory_tags(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok((
        StatusCode::CREATED,
        axum::Json(serde_json::json!({"id": id, "tags": tags})),
    ))
}

#[derive(serde::Deserialize)]
struct CreateDecisionBody {
    what: String,
    why: Option<String>,
    alternatives: Option<Vec<String>>,
    project: Option<String>,
}

async fn web_create_decision(
    State(state): State<Arc<WebState>>,
    AxumJson(body): AxumJson<CreateDecisionBody>,
) -> Result<(StatusCode, axum::Json<serde_json::Value>), StatusCode> {
    use remembrant_engine::store::Decision;

    if body.what.trim().is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    let id = uuid::Uuid::new_v4().to_string();
    let now = chrono::Utc::now().naive_utc();
    let decision = Decision {
        id: id.clone(),
        session_id: None,
        project_id: body.project,
        decision_type: None,
        what: body.what,
        why: body.why,
        alternatives: body.alternatives.unwrap_or_default(),
        outcome: None,
        created_at: Some(now),
        valid_until: None,
    };
    let store = state.store()?;
    store
        .insert_decision(&decision)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok((
        StatusCode::CREATED,
        axum::Json(serde_json::json!({"id": id})),
    ))
}

async fn web_briefing(
    State(state): State<Arc<WebState>>,
) -> Result<axum::Json<serde_json::Value>, StatusCode> {
    let store = state.store()?;
    let map_store_error = |_| StatusCode::INTERNAL_SERVER_ERROR;
    let now = chrono::Utc::now();
    // "Today" and "yesterday" are the user's LOCAL calendar days, expressed
    // as naive-UTC windows (issue #25). Computing them from UTC midnight
    // misattributes activity for every user not at UTC+0.
    let day = remembrant_engine::timeutil::local_day_window(now);
    let today_start = day.start;
    let yesterday_start = day.shifted_back(1).start;
    let tz_offset = chrono::Local::now().format("%:z").to_string();

    // Current (today) counts
    let cur_sessions = store
        .count_sessions_since(today_start)
        .map_err(map_store_error)?;
    let cur_memories = store
        .count_memories_since(today_start)
        .map_err(map_store_error)?;
    let cur_decisions = store
        .count_decisions_since(today_start)
        .map_err(map_store_error)?;
    let recent_sessions = store
        .get_recent_sessions_for_briefing(yesterday_start)
        .map_err(map_store_error)?;
    let cur_tokens: i64 = recent_sessions
        .iter()
        .filter(|s| s.started_at.map(|t| t >= today_start).unwrap_or(false))
        .filter_map(|s| s.total_tokens)
        .map(|t| t as i64)
        .sum();

    // Previous (yesterday) counts
    let prev_sessions = store
        .count_sessions_since(yesterday_start)
        .map_err(map_store_error)?
        .saturating_sub(cur_sessions);
    let prev_memories = store
        .count_memories_since(yesterday_start)
        .map_err(map_store_error)?
        .saturating_sub(cur_memories);
    let prev_decisions = store
        .count_decisions_since(yesterday_start)
        .map_err(map_store_error)?
        .saturating_sub(cur_decisions);
    let prev_tokens: i64 = recent_sessions
        .iter()
        .filter(|s| {
            s.started_at
                .map(|t| t >= yesterday_start && t < today_start)
                .unwrap_or(false)
        })
        .filter_map(|s| s.total_tokens)
        .map(|t| t as i64)
        .sum();

    let trend = |current: f64, previous: f64| -> f64 {
        if previous == 0.0 {
            if current > 0.0 { 100.0 } else { 0.0 }
        } else {
            ((current - previous) / previous * 100.0 * 10.0).round() / 10.0
        }
    };

    // Active agents & projects today
    let active_agents = store
        .get_active_agents_since(today_start)
        .map_err(map_store_error)?;
    let projects_today: Vec<String> = recent_sessions
        .iter()
        .filter(|s| s.started_at.map(|t| t >= today_start).unwrap_or(false))
        .filter_map(|s| s.project_id.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    // Headline
    let sess_trend_pct = trend(cur_sessions as f64, prev_sessions as f64);
    let trend_word = if sess_trend_pct > 0.0 {
        format!("up {sess_trend_pct:.0}%")
    } else if sess_trend_pct < 0.0 {
        format!("down {:.0}%", sess_trend_pct.abs())
    } else {
        "unchanged".to_string()
    };
    let headline = format!(
        "{} made {} session(s) across {} project(s) today, {} from yesterday",
        if active_agents.is_empty() {
            "Agents".to_string()
        } else if active_agents.len() == 1 {
            active_agents[0].clone()
        } else {
            format!("{} agents", active_agents.join(", "))
        },
        cur_sessions,
        projects_today.len(),
        trend_word,
    );

    // Sparklines (last 7 local days)
    let spark_sessions = store
        .get_daily_session_counts(6, now, &chrono::Local)
        .map_err(map_store_error)?;
    let spark_memories = store
        .get_daily_memory_counts(6, now, &chrono::Local)
        .map_err(map_store_error)?;
    let spark_decisions = store
        .get_daily_decision_counts(6, now, &chrono::Local)
        .map_err(map_store_error)?;

    // Project breakdown
    let project_breakdown: Vec<serde_json::Value> = store
        .get_project_breakdown_today(today_start)
        .map_err(map_store_error)?;

    // Decisions today
    let decisions_today = store
        .get_decisions_today(today_start)
        .map_err(map_store_error)?;

    // New facts
    let new_facts = store
        .get_new_facts_today(today_start)
        .map_err(map_store_error)?;

    // Top files
    let top_files: Vec<serde_json::Value> = store
        .get_top_files_today(today_start, 10)
        .map_err(map_store_error)?
        .into_iter()
        .map(|(path, changes)| serde_json::json!({"file_path": path, "change_frequency": changes}))
        .collect();

    // Session summaries
    let session_summaries = store
        .get_session_summaries_today(today_start)
        .map_err(map_store_error)?;

    let date_str = now
        .with_timezone(&chrono::Local)
        .format("%Y-%m-%d")
        .to_string();

    // Top projects (limited to 5)
    let top_projects = &project_breakdown[..std::cmp::min(5, project_breakdown.len())];

    Ok(axum::Json(serde_json::json!({
        "headline": headline,
        "date": date_str,
        "period": "today",
        "tz_offset": tz_offset,
        "metrics": {
            "sessions": {
                "current": cur_sessions,
                "previous": prev_sessions,
                "trend": trend(cur_sessions as f64, prev_sessions as f64),
            },
            "memories": {
                "current": cur_memories,
                "previous": prev_memories,
                "trend": trend(cur_memories as f64, prev_memories as f64),
            },
            "decisions": {
                "current": cur_decisions,
                "previous": prev_decisions,
                "trend": trend(cur_decisions as f64, prev_decisions as f64),
            },
            "tokens": {
                "current": cur_tokens,
                "previous": prev_tokens,
                "trend": trend(cur_tokens as f64, prev_tokens as f64),
            },
        },
        "sparklines": {
            "sessions": spark_sessions,
            "memories": spark_memories,
            "decisions": spark_decisions,
        },
        "active_agents": active_agents,
        "top_projects": top_projects,
        "recent_decisions": &decisions_today,
        "project_breakdown": &project_breakdown,
        "decisions_today": &decisions_today,
        "new_facts": new_facts,
        "top_files": top_files,
        "session_summaries": session_summaries,
    })))
}

fn cmd_mcp() -> Result<()> {
    let config = AppConfig::load()?;
    let db_path = expand_tilde(&config.storage.duckdb_path);

    if !db_path.exists() {
        anyhow::bail!(
            "Database not found at {}. Run 'rem ingest' first.",
            db_path.display()
        );
    }

    let store = DuckStore::open(&db_path)?;
    let server = mcp_server::McpServer::new(store);

    eprintln!("Remembrant MCP server started (stdio)");
    server.run()?;
    Ok(())
}

async fn cmd_web(port: u16) -> Result<()> {
    let config = AppConfig::load()?;

    // Verify DB exists
    let db_path = expand_tilde(&config.storage.duckdb_path);
    if !db_path.exists() {
        anyhow::bail!(
            "DuckDB not found at {}. Run `rem init` first.",
            db_path.display()
        );
    }

    let state = Arc::new(WebState { config });

    let app = Router::new()
        .route("/", get(web_index))
        .route("/assets/dashboard.css", get(web_dashboard_css))
        .route("/assets/dashboard.util.js", get(web_dashboard_util_js))
        .route("/assets/dashboard.js", get(web_dashboard_js))
        .route("/api/stats", get(web_stats))
        .route("/api/projects", get(web_projects))
        .route("/api/sessions", get(web_sessions))
        .route("/api/sessions/{id}", get(web_session_detail))
        .route("/api/memories", get(web_memories))
        .route(
            "/api/decisions",
            get(web_decisions).post(web_create_decision),
        )
        .route("/api/search/sessions", get(web_search_sessions))
        .route("/api/search/memories", get(web_search_memories))
        // New GET endpoints
        .route("/api/facts", get(web_facts))
        .route("/api/facts/{id}/history", get(web_fact_history))
        .route("/api/stats/agents", get(web_stats_agents))
        .route("/api/stats/tools", get(web_stats_tools))
        .route("/api/stats/timeline", get(web_stats_timeline))
        .route("/api/hotfiles", get(web_hotfiles))
        .route("/api/attention", get(web_attention))
        .route("/api/search/facts", get(web_search_facts))
        .route("/api/search/xpath", get(web_search_xpath))
        .route("/api/symbols", get(web_symbols))
        .route("/api/graph/neighbors/{id}", get(web_graph_neighbors))
        // Mutation endpoints
        .route(
            "/api/memories/{id}",
            get(web_get_memory)
                .put(web_update_memory)
                .delete(web_delete_memory),
        )
        .route("/api/facts/{id}", delete(web_delete_fact))
        .route("/api/notes", post(web_create_note))
        .route("/api/briefing", get(web_briefing))
        .with_state(state);

    let addr = format!("127.0.0.1:{port}");
    println!("Remembrant web dashboard: http://{addr}");
    println!("Press Ctrl+C to stop.\n");

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("failed to bind to {addr}"))?;
    axum::serve(listener, app).await.context("server error")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Init => {
            cmd_init()?;
        }
        Commands::Watch => {
            cmd_watch().await?;
        }
        Commands::Stop => {
            cmd_stop()?;
        }
        Commands::Search {
            query,
            project,
            agent,
            since,
            content_type,
            exact,
            json,
        } => {
            cmd_search(
                &query,
                project.as_deref(),
                agent.as_deref(),
                since.as_deref(),
                content_type.as_deref(),
                exact,
                json,
            )
            .await?;
        }
        Commands::Find { query } => {
            cmd_find(&query)?;
        }
        Commands::Recent {
            limit,
            agent,
            project,
        } => {
            cmd_recent(limit, agent.as_deref(), project.as_deref())?;
        }
        Commands::Brief {
            project,
            today,
            json,
            for_agent,
            max_tokens,
        } => {
            if for_agent {
                cmd_context_brief(project.as_deref(), max_tokens, json)?;
            } else {
                cmd_brief(project.as_deref(), today, json)?;
            }
        }
        Commands::Context {
            topic,
            project,
            json,
            max_tokens,
        } => {
            cmd_context_topic(&topic, project.as_deref(), max_tokens, json)?;
        }
        Commands::Consolidate {
            project,
            threshold,
            json,
        } => {
            cmd_consolidate(project.as_deref(), threshold, json)?;
        }
        Commands::Patterns { topic } => {
            cmd_patterns(topic.as_deref())?;
        }
        Commands::Decisions { project, all } => {
            cmd_decisions(project.as_deref(), all)?;
        }
        Commands::Related { path } => {
            cmd_related(&path)?;
        }
        Commands::Graph { path } => {
            cmd_graph(&path)?;
        }
        Commands::Timeline { topic, since } => {
            cmd_timeline(&topic, since.as_deref())?;
        }
        Commands::Note { text, project, tag } => {
            cmd_note(&text, project.as_deref(), tag.as_deref())?;
        }
        Commands::Forget { session } => {
            cmd_forget(&session)?;
        }
        Commands::Export {
            project,
            format,
            output,
        } => {
            cmd_export(project.as_deref(), &format, output.as_deref())?;
        }
        Commands::Embed { path, update } => {
            cmd_embed(&path, update).await?;
        }
        Commands::Ingest {
            skip_embed,
            skip_distill,
        } => {
            cmd_ingest(skip_embed, skip_distill).await?;
        }
        Commands::Status => {
            cmd_status()?;
        }
        Commands::Stats => {
            cmd_stats()?;
        }
        Commands::Gc => {
            cmd_gc()?;
        }
        Commands::Web { port } => {
            cmd_web(port).await?;
        }
        Commands::Mcp => {
            cmd_mcp()?;
        }
        Commands::XPath {
            query,
            depth,
            limit,
            tree,
            json,
        } => {
            cmd_xpath(&query, depth, limit, tree, json)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_handles_unicode_and_small_limits() {
        assert_eq!(truncate("🔐🔐🔐🔐🔐", 4), "🔐...");
        assert_eq!(truncate("🔐", 1), "🔐");
        assert_eq!(truncate("unchanged", 20), "unchanged");
    }

    #[test]
    fn utc_day_start_uses_midnight_not_rolling_hours() {
        let timestamp =
            NaiveDateTime::parse_from_str("2026-08-22 00:00:01", "%Y-%m-%d %H:%M:%S").unwrap();
        assert_eq!(
            utc_day_start(timestamp),
            NaiveDateTime::parse_from_str("2026-08-22 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        );
    }

    #[test]
    fn parse_since_accepts_rfc3339_offsets() {
        let parsed = parse_since("2026-08-21T20:00:00-04:00").unwrap();
        assert_eq!(
            parsed,
            NaiveDateTime::parse_from_str("2026-08-22 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        );
    }

    #[test]
    fn parse_since_rejects_out_of_range_relative_values() {
        assert!(parse_since("999999999999d").is_none());
    }

    #[test]
    fn api_limits_are_bounded() {
        assert_eq!(api_limit(None, 20, 100), 20);
        assert_eq!(api_limit(Some(0), 20, 100), 1);
        assert_eq!(api_limit(Some(999), 20, 100), 100);
    }

    #[test]
    fn agent_filter_parses_comma_separated_values() {
        assert!(parse_agent_filter(None).is_empty());
        assert!(parse_agent_filter(Some(String::new())).is_empty());
        assert!(parse_agent_filter(Some(" , ".to_string())).is_empty());
        assert_eq!(
            parse_agent_filter(Some("codex".to_string())),
            vec!["codex".to_string()]
        );
        assert_eq!(
            parse_agent_filter(Some(" codex , gemini ,".to_string())),
            vec!["codex".to_string(), "gemini".to_string()]
        );
    }
}
