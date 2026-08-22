use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDateTime, Utc};
use duckdb::{Connection, params};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Mutex;
use tracing::warn;

// ---------------------------------------------------------------------------
// Domain structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub project_id: Option<String>,
    pub agent: String,
    pub started_at: Option<NaiveDateTime>,
    pub ended_at: Option<NaiveDateTime>,
    pub duration_minutes: Option<i32>,
    pub message_count: Option<i32>,
    pub tool_call_count: Option<i32>,
    pub total_tokens: Option<i32>,
    pub files_changed: Vec<String>,
    pub summary: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    pub id: String,
    pub session_id: Option<String>,
    pub project_id: Option<String>,
    pub decision_type: Option<String>,
    pub what: String,
    pub why: Option<String>,
    pub alternatives: Vec<String>,
    pub outcome: Option<String>,
    pub created_at: Option<NaiveDateTime>,
    pub valid_until: Option<NaiveDateTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub project_id: Option<String>,
    pub content: String,
    pub memory_type: Option<String>,
    pub source_session_id: Option<String>,
    pub confidence: f32,
    pub access_count: i32,
    pub created_at: Option<NaiveDateTime>,
    pub updated_at: Option<NaiveDateTime>,
    pub valid_until: Option<NaiveDateTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub session_id: Option<String>,
    pub tool_name: Option<String>,
    pub command: Option<String>,
    pub success: Option<bool>,
    pub error_message: Option<String>,
    pub duration_ms: Option<i32>,
    pub timestamp: Option<NaiveDateTime>,
}

/// A temporal fact extracted from a coding session.
/// Facts have validity windows: they are true from `valid_at` until `invalid_at`.
/// When a contradicting fact is found, the old fact's `invalid_at` is set and a new
/// fact is created, preserving the full history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub id: String,
    pub project_id: Option<String>,
    pub subject: String,   // entity the fact is about (e.g., "auth module")
    pub predicate: String, // relationship (e.g., "uses", "depends_on", "is_located_at")
    pub object: String,    // value (e.g., "JWT tokens", "src/auth.rs")
    pub confidence: f32,
    pub source_session_id: Option<String>,
    pub source_agent: Option<String>,
    pub valid_at: Option<NaiveDateTime>,
    pub invalid_at: Option<NaiveDateTime>, // None = still valid
    pub superseded_by: Option<String>,     // ID of the fact that replaced this one
    pub created_at: Option<NaiveDateTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileStat {
    pub file_path: String,
    pub project_id: String,
    pub language: Option<String>,
    pub lines_of_code: Option<i32>,
    pub token_count: Option<i32>,
    pub complexity: Option<f64>,
    pub change_frequency: i32,
    pub last_modified: Option<NaiveDateTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeSymbol {
    pub id: String, // "project:file:symbol:line"
    pub project_id: String,
    pub file_path: String,
    pub symbol_name: String,
    pub symbol_kind: String, // function, class, struct, method, etc.
    pub signature: Option<String>,
    pub docstring: Option<String>,
    pub start_line: i32,
    pub end_line: i32,
    pub visibility: Option<String>, // public, private, protected
    pub parent_symbol: Option<String>,
    pub pagerank_score: f64,
    pub reference_count: i32,
    pub language: Option<String>,
    pub content_hash: Option<String>,
    pub indexed_at: Option<NaiveDateTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeDependency {
    pub id: String,
    pub project_id: String,
    pub from_symbol: String,
    pub to_symbol: String,
    pub relationship: String, // calls, imports, inherits, implements, references
    pub from_file: String,
    pub to_file: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisRun {
    pub project_id: String,
    pub commit_hash: Option<String>,
    pub files_analyzed: i32,
    pub symbols_extracted: i32,
    pub dependencies_found: i32,
    pub chunks_generated: i32,
    pub duration_ms: i32,
    pub analyzed_at: Option<NaiveDateTime>,
}

// ---------------------------------------------------------------------------
// Helpers — store Vec<String> as JSON text in DuckDB
// ---------------------------------------------------------------------------

fn vec_to_json(v: &[String]) -> String {
    serde_json::to_string(v).unwrap_or_else(|_| "[]".to_string())
}

/// Count timestamp rows per local calendar day in `tz`.
///
/// `sql` must select a single `TIMESTAMP` column filtered by `>= ?`.
fn count_by_local_day<Tz>(
    conn: &Connection,
    sql: &str,
    cutoff: NaiveDateTime,
    tz: &Tz,
) -> Result<std::collections::HashMap<chrono::NaiveDate, i64>>
where
    Tz: chrono::TimeZone,
    Tz::Offset: std::fmt::Display,
{
    let mut stmt = conn
        .prepare(sql)
        .context("failed to prepare daily count query")?;
    let rows = stmt
        .query_map(params![cutoff], |row| row.get::<_, NaiveDateTime>(0))
        .context("failed to query daily count timestamps")?;
    let mut counts: std::collections::HashMap<chrono::NaiveDate, i64> =
        std::collections::HashMap::new();
    for row in rows {
        let ts = row.context("failed to read daily count row")?;
        let local_day = ts.and_utc().with_timezone(tz).date_naive();
        *counts.entry(local_day).or_insert(0) += 1;
    }
    Ok(counts)
}

/// Lower-cased identifier-like tokens of at least 3 characters, with common
/// English stop words removed. Used to decide whether two memories are
/// topically related (see `get_attention_items`).
fn significant_tokens(text: &str) -> std::collections::HashSet<String> {
    const STOP_WORDS: &[&str] = &[
        "the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "will", "have",
        "has", "had", "not", "but", "its", "our", "out", "use", "used", "using", "than", "then",
        "them", "they", "into", "over", "after", "before", "about", "all", "any", "can", "should",
        "would", "could", "when", "where", "which", "who",
    ];
    let stop: std::collections::HashSet<&str> = STOP_WORDS.iter().copied().collect();
    text.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|tok| tok.len() >= 3)
        .map(|tok| tok.to_lowercase())
        .filter(|tok| !stop.contains(tok.as_str()))
        .collect()
}

fn json_to_vec(s: &str) -> Vec<String> {
    serde_json::from_str(s).unwrap_or_default()
}

/// Delete a graph node and incident edges before a relational transaction.
///
/// DuckDB's foreign-key implementation can reject a parent-node deletion in
/// the same transaction that removes its child edges. Graph cleanup is therefore
/// committed first; callers rebuild derived graph nodes after relational writes.
fn delete_graph_node_with_edges(conn: &mut duckdb::Connection, node_id: &str) -> Result<()> {
    conn.execute(
        "DELETE FROM graph_edges WHERE from_id = ? OR to_id = ?",
        params![node_id, node_id],
    )
    .context("failed to delete graph edges")?;
    conn.execute("DELETE FROM graph_nodes WHERE id = ?", params![node_id])
        .context("failed to delete graph node")?;
    Ok(())
}

fn normalize_tags(tags: &[String]) -> Result<Vec<String>> {
    let mut normalized = Vec::new();
    for tag in tags {
        let tag = tag.trim();
        if tag.is_empty() {
            anyhow::bail!("tags cannot be empty");
        }
        if tag.len() > 80 {
            anyhow::bail!("tag exceeds 80 characters: {tag}");
        }
        if !normalized
            .iter()
            .any(|existing: &String| existing.eq_ignore_ascii_case(tag))
        {
            normalized.push(tag.to_string());
        }
    }
    Ok(normalized)
}

// ---------------------------------------------------------------------------
// Graph row structs (for DuckPGQ-backed graph storage)
// ---------------------------------------------------------------------------

/// A row from the `graph_nodes` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNodeRow {
    pub id: String,
    pub kind: String,
    pub name: String,
    pub properties: String, // JSON
}

/// A neighbor row returned from a graph adjacency query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNeighborRow {
    pub node: GraphNodeRow,
    pub edge_kind: String,
    pub direction: String, // "outgoing" or "incoming"
}

// ---------------------------------------------------------------------------
// DuckStore
// ---------------------------------------------------------------------------

/// Persistent store backed by an embedded DuckDB database.
pub struct DuckStore {
    conn: Mutex<Connection>,
}

impl DuckStore {
    /// Access the underlying connection mutex (for modules that need custom queries).
    pub fn connection(&self) -> &Mutex<Connection> {
        &self.conn
    }

    /// Open (or create) a DuckDB database at `path` and initialise the schema.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path.as_ref())
            .with_context(|| format!("failed to open DuckDB at {}", path.as_ref().display()))?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.init_schema()?;
        Ok(store)
    }

    /// Create an in-memory DuckDB instance (useful for tests).
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().context("failed to open in-memory DuckDB")?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.init_schema()?;
        Ok(store)
    }

    // -----------------------------------------------------------------------
    // Schema
    // -----------------------------------------------------------------------

    /// Create all tables if they do not already exist.
    pub fn init_schema(&self) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");

        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                visibility TEXT DEFAULT 'public',
                tags TEXT,
                created_at TIMESTAMP DEFAULT current_timestamp,
                updated_at TIMESTAMP DEFAULT current_timestamp
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                agent TEXT NOT NULL,
                started_at TIMESTAMP,
                ended_at TIMESTAMP,
                duration_minutes INTEGER,
                message_count INTEGER,
                tool_call_count INTEGER,
                total_tokens INTEGER,
                files_changed TEXT,
                summary TEXT
            );

            CREATE TABLE IF NOT EXISTS decisions (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                project_id TEXT,
                decision_type TEXT,
                what TEXT NOT NULL,
                why TEXT,
                alternatives TEXT,
                outcome TEXT,
                created_at TIMESTAMP DEFAULT current_timestamp,
                valid_until TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                content TEXT NOT NULL,
                memory_type TEXT,
                source_session_id TEXT,
                confidence REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT current_timestamp,
                updated_at TIMESTAMP DEFAULT current_timestamp,
                valid_until TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS memory_tags (
                memory_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT current_timestamp,
                PRIMARY KEY (memory_id, tag),
                FOREIGN KEY (memory_id) REFERENCES memories(id)
            );

            CREATE TABLE IF NOT EXISTS tool_calls (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                tool_name TEXT,
                command TEXT,
                success BOOLEAN,
                error_message TEXT,
                duration_ms INTEGER,
                timestamp TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source_session_id TEXT,
                source_agent TEXT,
                valid_at TIMESTAMP DEFAULT current_timestamp,
                invalid_at TIMESTAMP,
                superseded_by TEXT,
                created_at TIMESTAMP DEFAULT current_timestamp
            );

            CREATE TABLE IF NOT EXISTS file_stats (
                file_path TEXT,
                project_id TEXT,
                language TEXT,
                lines_of_code INTEGER,
                token_count INTEGER,
                complexity REAL,
                change_frequency INTEGER DEFAULT 0,
                last_modified TIMESTAMP,
                PRIMARY KEY (file_path, project_id)
            );

            CREATE TABLE IF NOT EXISTS code_symbols (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                symbol_name TEXT NOT NULL,
                symbol_kind TEXT NOT NULL,
                signature TEXT,
                docstring TEXT,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                visibility TEXT,
                parent_symbol TEXT,
                pagerank_score REAL DEFAULT 0.0,
                reference_count INTEGER DEFAULT 0,
                language TEXT,
                content_hash TEXT,
                indexed_at TIMESTAMP DEFAULT current_timestamp
            );

            CREATE TABLE IF NOT EXISTS code_dependencies (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                from_symbol TEXT NOT NULL,
                to_symbol TEXT NOT NULL,
                relationship TEXT NOT NULL,
                from_file TEXT NOT NULL,
                to_file TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS code_analysis_runs (
                project_id TEXT PRIMARY KEY,
                commit_hash TEXT,
                files_analyzed INTEGER DEFAULT 0,
                symbols_extracted INTEGER DEFAULT 0,
                dependencies_found INTEGER DEFAULT 0,
                chunks_generated INTEGER DEFAULT 0,
                duration_ms INTEGER DEFAULT 0,
                analyzed_at TIMESTAMP DEFAULT current_timestamp
            );

            CREATE TABLE IF NOT EXISTS graph_nodes (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                name TEXT NOT NULL,
                properties TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS graph_edges (
                id TEXT PRIMARY KEY,
                from_id TEXT NOT NULL,
                to_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                properties TEXT DEFAULT '{}',
                FOREIGN KEY (from_id) REFERENCES graph_nodes(id),
                FOREIGN KEY (to_id) REFERENCES graph_nodes(id)
            );
            ",
        )
        .context("failed to initialise DuckDB schema")?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Full-Text Search (BM25 via DuckDB FTS extension)
    // -----------------------------------------------------------------------

    /// Install and load the DuckDB FTS extension, then create full-text
    /// indexes on key tables. Call this once after `init_schema` (or on
    /// demand before the first FTS query).
    ///
    /// FTS indexes use BM25 scoring which is far superior to ILIKE for
    /// identifier and keyword search. The GrepRAG paper shows simple lexical
    /// retrieval matches complex methods for code search.
    ///
    /// This is idempotent — safe to call multiple times. If the FTS extension
    /// is unavailable the method returns an error but doesn't crash.
    pub fn init_fts(&self) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");

        conn.execute_batch("INSTALL fts; LOAD fts;")
            .context("failed to load DuckDB FTS extension")?;

        // Drop existing FTS indexes before recreating (idempotent).
        // DuckDB FTS uses PRAGMA which doesn't support IF NOT EXISTS,
        // so we drop first to avoid "already exists" errors.
        let _ = conn.execute_batch("PRAGMA drop_fts_index('memories');");
        let _ = conn.execute_batch("PRAGMA drop_fts_index('facts');");
        let _ = conn.execute_batch("PRAGMA drop_fts_index('sessions');");
        let _ = conn.execute_batch("PRAGMA drop_fts_index('code_symbols');");
        let _ = conn.execute_batch("PRAGMA drop_fts_index('decisions');");

        // Create FTS indexes on searchable columns.
        // stemmer='none' preserves identifiers exactly (important for code).
        conn.execute_batch(
            "PRAGMA create_fts_index('memories', 'id', 'content', stemmer='porter', overwrite=1);
             PRAGMA create_fts_index('facts', 'id', 'subject', 'predicate', 'object', stemmer='porter', overwrite=1);
             PRAGMA create_fts_index('sessions', 'id', 'summary', stemmer='porter', overwrite=1);
             PRAGMA create_fts_index('code_symbols', 'id', 'symbol_name', 'file_path', 'signature', 'docstring', stemmer='none', overwrite=1);
             PRAGMA create_fts_index('decisions', 'id', 'what', 'why', 'alternatives', stemmer='porter', overwrite=1);",
        )
        .context("failed to create FTS indexes")?;

        tracing::info!("FTS indexes created on memories, facts, sessions, code_symbols, decisions");
        Ok(())
    }

    /// BM25 full-text search over memories. Returns results ranked by relevance.
    /// Falls back to ILIKE if FTS indexes haven't been created.
    pub fn search_memories_fts(&self, query: &str) -> Result<Vec<(Memory, f64)>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let mut stmt = conn
            .prepare(
                "SELECT m.id, m.project_id, m.content, m.memory_type, m.source_session_id,
                    m.confidence, m.access_count, m.created_at, m.updated_at, m.valid_until,
                    fts.score
             FROM memories m
             JOIN (SELECT id, fts_main_memories.match_bm25(id, ?) AS score
                   FROM memories) fts ON m.id = fts.id
             WHERE fts.score IS NOT NULL
             ORDER BY fts.score DESC",
            )
            .context("failed to prepare FTS memory search")?;

        let rows = stmt
            .query_map(params![query], |row| {
                let memory = Memory {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    content: row.get(2)?,
                    memory_type: row.get(3)?,
                    source_session_id: row.get(4)?,
                    confidence: row.get::<_, f32>(5).unwrap_or(1.0),
                    access_count: row.get::<_, i32>(6).unwrap_or(0),
                    created_at: row.get(7)?,
                    updated_at: row.get(8)?,
                    valid_until: row.get(9)?,
                };
                let score: f64 = row.get::<_, f64>(10).unwrap_or(0.0);
                Ok((memory, score))
            })
            .context("failed to query FTS memories")?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.context("failed to read FTS memory row")?);
        }
        Ok(results)
    }

    /// BM25 full-text search over facts.
    pub fn search_facts_fts(&self, query: &str) -> Result<Vec<(Fact, f64)>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let mut stmt = conn
            .prepare(
                "SELECT f.id, f.project_id, f.subject, f.predicate, f.object,
                    f.confidence, f.source_session_id, f.source_agent,
                    f.valid_at, f.invalid_at, f.superseded_by, f.created_at,
                    fts.score
             FROM facts f
             JOIN (SELECT id, fts_main_facts.match_bm25(id, ?) AS score
                   FROM facts) fts ON f.id = fts.id
             WHERE fts.score IS NOT NULL AND f.invalid_at IS NULL
             ORDER BY fts.score DESC",
            )
            .context("failed to prepare FTS fact search")?;

        let rows = stmt
            .query_map(params![query], |row| {
                let fact = Fact {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    subject: row.get(2)?,
                    predicate: row.get(3)?,
                    object: row.get(4)?,
                    confidence: row.get::<_, f32>(5).unwrap_or(1.0),
                    source_session_id: row.get(6)?,
                    source_agent: row.get(7)?,
                    valid_at: row.get(8)?,
                    invalid_at: row.get(9)?,
                    superseded_by: row.get(10)?,
                    created_at: row.get(11)?,
                };
                let score: f64 = row.get::<_, f64>(12).unwrap_or(0.0);
                Ok((fact, score))
            })
            .context("failed to query FTS facts")?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.context("failed to read FTS fact row")?);
        }
        Ok(results)
    }

    /// BM25 full-text search over sessions (by summary).
    pub fn search_sessions_fts(&self, query: &str) -> Result<Vec<(Session, f64)>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let mut stmt = conn
            .prepare(
                "SELECT s.id, s.project_id, s.agent, s.started_at, s.ended_at,
                    s.duration_minutes, s.message_count, s.tool_call_count,
                    s.total_tokens, s.files_changed, s.summary,
                    fts.score
             FROM sessions s
             JOIN (SELECT id, fts_main_sessions.match_bm25(id, ?) AS score
                   FROM sessions) fts ON s.id = fts.id
             WHERE fts.score IS NOT NULL
             ORDER BY fts.score DESC",
            )
            .context("failed to prepare FTS session search")?;

        let rows = stmt
            .query_map(params![query], |row| {
                let files_str: String = row.get::<_, String>(9).unwrap_or_default();
                let session = Session {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    agent: row.get(2)?,
                    started_at: row.get(3)?,
                    ended_at: row.get(4)?,
                    duration_minutes: row.get(5)?,
                    message_count: row.get(6)?,
                    tool_call_count: row.get(7)?,
                    total_tokens: row.get(8)?,
                    files_changed: json_to_vec(&files_str),
                    summary: row.get(10)?,
                };
                let score: f64 = row.get::<_, f64>(11).unwrap_or(0.0);
                Ok((session, score))
            })
            .context("failed to query FTS sessions")?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.context("failed to read FTS session row")?);
        }
        Ok(results)
    }

    /// BM25 full-text search over code symbols.
    /// Uses stemmer='none' for exact identifier matching.
    pub fn search_code_symbols_fts(&self, query: &str) -> Result<Vec<(CodeSymbol, f64)>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let mut stmt = conn
            .prepare(
                "SELECT cs.id, cs.project_id, cs.file_path, cs.symbol_name, cs.symbol_kind,
                    cs.signature, cs.docstring, cs.start_line, cs.end_line,
                    cs.visibility, cs.parent_symbol, cs.pagerank_score,
                    cs.reference_count, cs.language, cs.content_hash, cs.indexed_at,
                    fts.score
             FROM code_symbols cs
             JOIN (SELECT id, fts_main_code_symbols.match_bm25(id, ?) AS score
                   FROM code_symbols) fts ON cs.id = fts.id
             WHERE fts.score IS NOT NULL
             ORDER BY fts.score DESC",
            )
            .context("failed to prepare FTS code_symbols search")?;

        let rows = stmt
            .query_map(params![query], |row| {
                let symbol = CodeSymbol {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    file_path: row.get(2)?,
                    symbol_name: row.get(3)?,
                    symbol_kind: row.get(4)?,
                    signature: row.get(5)?,
                    docstring: row.get(6)?,
                    start_line: row.get(7)?,
                    end_line: row.get(8)?,
                    visibility: row.get(9)?,
                    parent_symbol: row.get(10)?,
                    pagerank_score: row.get::<_, f64>(11).unwrap_or(0.0),
                    reference_count: row.get::<_, i32>(12).unwrap_or(0),
                    language: row.get(13)?,
                    content_hash: row.get(14)?,
                    indexed_at: row.get(15)?,
                };
                let score: f64 = row.get::<_, f64>(16).unwrap_or(0.0);
                Ok((symbol, score))
            })
            .context("failed to query FTS code_symbols")?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.context("failed to read FTS code_symbol row")?);
        }
        Ok(results)
    }

    /// BM25 full-text search over decisions.
    pub fn search_decisions_fts(&self, query: &str) -> Result<Vec<(Decision, f64)>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let mut stmt = conn
            .prepare(
                "SELECT d.id, d.session_id, d.project_id, d.decision_type, d.what,
                    d.why, d.alternatives, d.outcome, d.created_at, d.valid_until,
                    fts.score
             FROM decisions d
             JOIN (SELECT id, fts_main_decisions.match_bm25(id, ?) AS score
                   FROM decisions) fts ON d.id = fts.id
             WHERE fts.score IS NOT NULL
             ORDER BY fts.score DESC",
            )
            .context("failed to prepare FTS decision search")?;

        let rows = stmt
            .query_map(params![query], |row| {
                let alts_str: String = row.get::<_, String>(6).unwrap_or_default();
                let decision = Decision {
                    id: row.get(0)?,
                    session_id: row.get(1)?,
                    project_id: row.get(2)?,
                    decision_type: row.get(3)?,
                    what: row.get(4)?,
                    why: row.get(5)?,
                    alternatives: json_to_vec(&alts_str),
                    outcome: row.get(7)?,
                    created_at: row.get(8)?,
                    valid_until: row.get(9)?,
                };
                let score: f64 = row.get::<_, f64>(10).unwrap_or(0.0);
                Ok((decision, score))
            })
            .context("failed to query FTS decisions")?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.context("failed to read FTS decision row")?);
        }
        Ok(results)
    }

    /// Check whether FTS indexes have been created.
    pub fn has_fts(&self) -> bool {
        let conn = self.conn.lock().expect("lock poisoned");
        // If the FTS macro table exists, indexes are active.
        conn.prepare("SELECT * FROM fts_main_memories.docs LIMIT 0")
            .is_ok()
    }

    // -----------------------------------------------------------------------
    // Inserts
    // -----------------------------------------------------------------------

    /// Insert a session record.
    pub fn insert_session(&self, session: &Session) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        let files_json = vec_to_json(&session.files_changed);
        conn.execute(
            "INSERT INTO sessions (
                id, project_id, agent, started_at, ended_at,
                duration_minutes, message_count, tool_call_count,
                total_tokens, files_changed, summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                session.id,
                session.project_id,
                session.agent,
                session.started_at,
                session.ended_at,
                session.duration_minutes,
                session.message_count,
                session.tool_call_count,
                session.total_tokens,
                files_json,
                session.summary,
            ],
        )
        .context("failed to insert session")?;
        Ok(())
    }

    /// Insert a decision record.
    pub fn insert_decision(&self, decision: &Decision) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        let alts_json = vec_to_json(&decision.alternatives);
        conn.execute(
            "INSERT INTO decisions (
                id, session_id, project_id, decision_type, what,
                why, alternatives, outcome, created_at, valid_until
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                decision.id,
                decision.session_id,
                decision.project_id,
                decision.decision_type,
                decision.what,
                decision.why,
                alts_json,
                decision.outcome,
                decision
                    .created_at
                    .unwrap_or_else(|| Utc::now().naive_utc()),
                decision.valid_until,
            ],
        )
        .context("failed to insert decision")?;
        Ok(())
    }

    /// Insert a tool call record.
    pub fn insert_tool_call(&self, tool_call: &ToolCall) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        conn.execute(
            "INSERT INTO tool_calls (
                id, session_id, tool_name, command, success,
                error_message, duration_ms, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                tool_call.id,
                tool_call.session_id,
                tool_call.tool_name,
                tool_call.command,
                tool_call.success,
                tool_call.error_message,
                tool_call.duration_ms,
                tool_call.timestamp,
            ],
        )
        .context("failed to insert tool_call")?;
        Ok(())
    }

    /// Replace a session and its transcript-derived relational rows.
    ///
    /// Re-ingestion is idempotent. DuckDB cannot reliably delete graph parent
    /// nodes and child edges in one transaction, so stale derived graph nodes
    /// are removed first; callers rebuild the graph after successful writes.
    pub fn insert_or_replace_session(&self, session: &Session) -> Result<()> {
        let mut conn = self.conn.lock().expect("lock poisoned");
        let files_json = vec_to_json(&session.files_changed);

        let stale_graph_ids: Vec<String> = {
            let mut statement = conn
                .prepare(
                    "SELECT 'session:' || ? AS graph_id
                     UNION ALL
                     SELECT 'memory:' || id FROM memories WHERE source_session_id = ?
                     UNION ALL
                     SELECT 'decision:' || id FROM decisions WHERE session_id = ?",
                )
                .context("failed to prepare stale graph lookup")?;
            let rows = statement
                .query_map(params![session.id, session.id, session.id], |row| {
                    row.get::<_, String>(0)
                })
                .context("failed to query stale graph IDs")?;
            let mut ids = Vec::new();
            for row in rows {
                ids.push(row.context("failed to read stale graph ID")?);
            }
            ids
        };
        for graph_id in &stale_graph_ids {
            delete_graph_node_with_edges(&mut conn, graph_id)?;
        }
        conn.execute(
            "DELETE FROM memory_tags WHERE memory_id IN (
                SELECT id FROM memories WHERE source_session_id = ?
            )",
            params![session.id],
        )
        .context("failed to delete old session memory tags")?;

        let transaction = conn
            .transaction()
            .context("failed to begin session replacement transaction")?;
        transaction
            .execute(
                "DELETE FROM tool_calls WHERE session_id = ?",
                params![session.id],
            )
            .context("failed to delete old session tool calls")?;
        transaction
            .execute(
                "DELETE FROM memories WHERE source_session_id = ?",
                params![session.id],
            )
            .context("failed to delete old session memories")?;
        transaction
            .execute(
                "DELETE FROM decisions WHERE session_id = ?",
                params![session.id],
            )
            .context("failed to delete old session decisions")?;
        transaction
            .execute(
                "DELETE FROM facts WHERE source_session_id = ?",
                params![session.id],
            )
            .context("failed to delete old session facts")?;
        transaction
            .execute(
                "INSERT OR REPLACE INTO sessions (
                    id, project_id, agent, started_at, ended_at,
                    duration_minutes, message_count, tool_call_count,
                    total_tokens, files_changed, summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                params![
                    session.id,
                    session.project_id,
                    session.agent,
                    session.started_at,
                    session.ended_at,
                    session.duration_minutes,
                    session.message_count,
                    session.tool_call_count,
                    session.total_tokens,
                    files_json,
                    session.summary,
                ],
            )
            .context("failed to insert_or_replace session")?;
        transaction
            .commit()
            .context("failed to commit session replacement transaction")?;
        Ok(())
    }
    /// Insert a memory record.
    pub fn insert_memory(&self, memory: &Memory) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        conn.execute(
            "INSERT INTO memories (
                id, project_id, content, memory_type, source_session_id,
                confidence, access_count, created_at, updated_at, valid_until
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                memory.id,
                memory.project_id,
                memory.content,
                memory.memory_type,
                memory.source_session_id,
                memory.confidence,
                memory.access_count,
                memory.created_at.unwrap_or_else(|| Utc::now().naive_utc()),
                memory.updated_at.unwrap_or_else(|| Utc::now().naive_utc()),
                memory.valid_until,
            ],
        )
        .context("failed to insert memory")?;
        Ok(())
    }

    /// Increment access_count for a memory (called on retrieval).
    pub fn touch_memory(&self, memory_id: &str) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        conn.execute(
            "UPDATE memories SET access_count = access_count + 1, updated_at = ? WHERE id = ?",
            params![Utc::now().naive_utc(), memory_id],
        )
        .context("failed to touch memory")?;
        Ok(())
    }

    /// Update a memory's content and/or confidence.
    pub fn update_memory(
        &self,
        memory_id: &str,
        new_content: Option<&str>,
        new_confidence: Option<f32>,
    ) -> Result<bool> {
        let conn = self.conn.lock().expect("lock poisoned");
        let now = Utc::now().naive_utc();

        let (set_clause, mut values): (String, Vec<Box<dyn duckdb::ToSql>>) =
            match (new_content, new_confidence) {
                (Some(c), Some(conf)) => (
                    "content = ?, confidence = ?, updated_at = ?".into(),
                    vec![
                        Box::new(c.to_string()) as Box<dyn duckdb::ToSql>,
                        Box::new(conf),
                        Box::new(now),
                    ],
                ),
                (Some(c), None) => (
                    "content = ?, updated_at = ?".into(),
                    vec![
                        Box::new(c.to_string()) as Box<dyn duckdb::ToSql>,
                        Box::new(now),
                    ],
                ),
                (None, Some(conf)) => (
                    "confidence = ?, updated_at = ?".into(),
                    vec![Box::new(conf) as Box<dyn duckdb::ToSql>, Box::new(now)],
                ),
                (None, None) => return Ok(false),
            };

        values.push(Box::new(memory_id.to_string()));
        let sql = format!("UPDATE memories SET {set_clause} WHERE id = ?");
        let params_ref: Vec<&dyn duckdb::ToSql> = values.iter().map(|b| b.as_ref()).collect();
        let affected = conn
            .execute(&sql, params_ref.as_slice())
            .context("failed to update memory")?;
        Ok(affected > 0)
    }

    /// Delete a memory by ID.
    pub fn delete_memory(&self, memory_id: &str) -> Result<bool> {
        let mut conn = self.conn.lock().expect("lock poisoned");
        delete_graph_node_with_edges(&mut conn, &format!("memory:{memory_id}"))?;
        conn.execute(
            "DELETE FROM memory_tags WHERE memory_id = ?",
            params![memory_id],
        )
        .context("failed to delete memory tags")?;
        let transaction = conn
            .transaction()
            .context("failed to begin memory deletion transaction")?;
        let affected = transaction
            .execute("DELETE FROM memories WHERE id = ?", params![memory_id])
            .context("failed to delete memory")?;
        transaction
            .commit()
            .context("failed to commit memory deletion transaction")?;
        Ok(affected > 0)
    }

    /// Delete a fact by ID (hard delete, not invalidation).
    pub fn delete_fact(&self, fact_id: &str) -> Result<bool> {
        let conn = self.conn.lock().expect("lock poisoned");
        let affected = conn
            .execute("DELETE FROM facts WHERE id = ?", params![fact_id])
            .context("failed to delete fact")?;
        Ok(affected > 0)
    }

    /// Get a single memory by ID.
    pub fn get_memory(&self, memory_id: &str) -> Result<Option<Memory>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, project_id, content, memory_type, source_session_id,
                    confidence, access_count, created_at, updated_at, valid_until
             FROM memories WHERE id = ?",
        )?;
        let mut rows = stmt.query(params![memory_id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(Memory {
                id: row.get(0)?,
                project_id: row.get(1)?,
                content: row.get(2)?,
                memory_type: row.get(3)?,
                source_session_id: row.get(4)?,
                confidence: row.get(5)?,
                access_count: row.get(6)?,
                created_at: row.get(7)?,
                updated_at: row.get(8)?,
                valid_until: row.get(9)?,
            }))
        } else {
            Ok(None)
        }
    }

    /// Get a single fact by ID.
    pub fn get_fact(&self, fact_id: &str) -> Result<Option<Fact>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, project_id, subject, predicate, object, confidence,
                    source_session_id, source_agent, valid_at, invalid_at,
                    superseded_by, created_at
             FROM facts WHERE id = ?",
        )?;
        let mut rows = stmt.query(params![fact_id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(Fact {
                id: row.get(0)?,
                project_id: row.get(1)?,
                subject: row.get(2)?,
                predicate: row.get(3)?,
                object: row.get(4)?,
                confidence: row.get(5)?,
                source_session_id: row.get(6)?,
                source_agent: row.get(7)?,
                valid_at: row.get(8)?,
                invalid_at: row.get(9)?,
                superseded_by: row.get(10)?,
                created_at: row.get(11)?,
            }))
        } else {
            Ok(None)
        }
    }

    /// Upsert a project record.
    pub fn upsert_project(&self, id: &str, name: &str, path: &str) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        conn.execute(
            "INSERT INTO projects (id, name, path, updated_at)
             VALUES (?, ?, ?, ?)
             ON CONFLICT (id) DO UPDATE SET
                name = excluded.name,
                path = excluded.path,
                updated_at = excluded.updated_at",
            params![id, name, path, Utc::now().naive_utc()],
        )
        .context("failed to upsert project")?;
        Ok(())
    }

    /// Upsert file stats (change_frequency increments on each call).
    pub fn upsert_file_stat(&self, file_path: &str, project_id: &str) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        conn.execute(
            "INSERT INTO file_stats (file_path, project_id, change_frequency, last_modified)
             VALUES (?, ?, 1, ?)
             ON CONFLICT (file_path, project_id) DO UPDATE SET
                change_frequency = file_stats.change_frequency + 1,
                last_modified = excluded.last_modified",
            params![file_path, project_id, Utc::now().naive_utc()],
        )
        .context("failed to upsert file_stat")?;
        Ok(())
    }

    /// Get hot files (most frequently changed) for a project.
    pub fn get_hot_files(&self, project: Option<&str>, limit: usize) -> Result<Vec<(String, i32)>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut result = Vec::new();

        if let Some(proj) = project {
            let mut stmt = conn.prepare(
                "SELECT file_path, change_frequency FROM file_stats WHERE project_id = ? ORDER BY change_frequency DESC LIMIT ?"
            )?;
            let rows = stmt.query_map(params![proj, limit as i64], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i32>(1)?))
            })?;
            for row in rows {
                result.push(row?);
            }
        } else {
            let mut stmt = conn.prepare(
                "SELECT file_path, change_frequency FROM file_stats ORDER BY change_frequency DESC LIMIT ?"
            )?;
            let rows = stmt.query_map(params![limit as i64], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i32>(1)?))
            })?;
            for row in rows {
                result.push(row?);
            }
        }

        Ok(result)
    }

    /// Insert a fact record.
    pub fn insert_fact(&self, fact: &Fact) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        conn.execute(
            "INSERT INTO facts (
                id, project_id, subject, predicate, object, confidence,
                source_session_id, source_agent, valid_at, invalid_at,
                superseded_by, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                fact.id,
                fact.project_id,
                fact.subject,
                fact.predicate,
                fact.object,
                fact.confidence,
                fact.source_session_id,
                fact.source_agent,
                fact.valid_at.unwrap_or_else(|| Utc::now().naive_utc()),
                fact.invalid_at,
                fact.superseded_by,
                fact.created_at.unwrap_or_else(|| Utc::now().naive_utc()),
            ],
        )
        .context("failed to insert fact")?;
        Ok(())
    }

    /// Invalidate a fact by setting `invalid_at` and optionally linking to successor.
    pub fn invalidate_fact(&self, fact_id: &str, superseded_by: Option<&str>) -> Result<bool> {
        let conn = self.conn.lock().expect("lock poisoned");
        let now = Utc::now().naive_utc();
        let affected = conn
            .execute(
                "UPDATE facts SET invalid_at = ?, superseded_by = ?
                 WHERE id = ? AND invalid_at IS NULL",
                params![now, superseded_by, fact_id],
            )
            .context("failed to invalidate fact")?;
        Ok(affected > 0)
    }

    /// Insert a new fact, automatically invalidating any contradicting facts
    /// (same subject + predicate + project, still valid).
    pub fn upsert_fact(&self, fact: &Fact) -> Result<()> {
        // Find existing valid facts with the same subject+predicate
        let existing = self.get_active_facts_for_subject(
            &fact.subject,
            &fact.predicate,
            fact.project_id.as_deref(),
        )?;

        // Invalidate contradicting facts (different object value)
        for old in &existing {
            if old.object != fact.object {
                self.invalidate_fact(&old.id, Some(&fact.id))?;
            }
        }

        // If an identical fact already exists and is valid, skip insertion
        if existing.iter().any(|f| f.object == fact.object) {
            return Ok(());
        }

        self.insert_fact(fact)
    }

    /// Get all currently valid facts (invalid_at IS NULL).
    pub fn get_active_facts(&self, project: Option<&str>, limit: usize) -> Result<Vec<Fact>> {
        self.get_active_facts_filtered(project, &[], limit)
    }

    /// Get currently valid facts with optional project and source-agent
    /// filters.
    pub fn get_active_facts_filtered(
        &self,
        project: Option<&str>,
        agents: &[String],
        limit: usize,
    ) -> Result<Vec<Fact>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let mut conditions: Vec<String> = vec!["invalid_at IS NULL".to_string()];
        let mut param_values: Vec<Box<dyn duckdb::ToSql>> = Vec::new();
        if let Some(project) = project {
            conditions.push("project_id ILIKE ?".to_string());
            param_values.push(Box::new(format!("%{project}%")));
        }
        if !agents.is_empty() {
            conditions.push(format!(
                "source_agent IN ({})",
                vec!["?"; agents.len()].join(", ")
            ));
            for agent in agents {
                param_values.push(Box::new(agent.clone()));
            }
        }
        let sql = format!(
            "SELECT id, project_id, subject, predicate, object, confidence,
                    source_session_id, source_agent, valid_at, invalid_at,
                    superseded_by, created_at
             FROM facts
             WHERE {}
             ORDER BY valid_at DESC NULLS LAST
             LIMIT ?",
            conditions.join(" AND ")
        );
        param_values.push(Box::new(limit as i64));

        let params_ref: Vec<&dyn duckdb::ToSql> =
            param_values.iter().map(|pv| pv.as_ref()).collect();
        let mut stmt = conn
            .prepare(&sql)
            .context("failed to prepare get_active_facts_filtered")?;

        let map_row = |row: &duckdb::Row| -> duckdb::Result<Fact> {
            Ok(Fact {
                id: row.get(0)?,
                project_id: row.get(1)?,
                subject: row.get(2)?,
                predicate: row.get(3)?,
                object: row.get(4)?,
                confidence: row.get::<_, f32>(5).unwrap_or(1.0),
                source_session_id: row.get(6)?,
                source_agent: row.get(7)?,
                valid_at: row.get(8)?,
                invalid_at: row.get(9)?,
                superseded_by: row.get(10)?,
                created_at: row.get(11)?,
            })
        };

        let rows = stmt
            .query_map(params_ref.as_slice(), map_row)
            .context("failed to query active facts")?;

        let mut facts = Vec::new();
        for row in rows {
            facts.push(row.context("failed to read fact row")?);
        }
        Ok(facts)
    }

    /// Get active facts for a specific subject and predicate.
    fn get_active_facts_for_subject(
        &self,
        subject: &str,
        predicate: &str,
        project: Option<&str>,
    ) -> Result<Vec<Fact>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let (sql, params_vec): (&str, Vec<Box<dyn duckdb::ToSql>>) = if let Some(proj) = project {
            (
                "SELECT id, project_id, subject, predicate, object, confidence,
                        source_session_id, source_agent, valid_at, invalid_at,
                        superseded_by, created_at
                 FROM facts
                 WHERE invalid_at IS NULL AND subject = ? AND predicate = ? AND project_id = ?",
                vec![
                    Box::new(subject.to_string()),
                    Box::new(predicate.to_string()),
                    Box::new(proj.to_string()),
                ],
            )
        } else {
            (
                "SELECT id, project_id, subject, predicate, object, confidence,
                        source_session_id, source_agent, valid_at, invalid_at,
                        superseded_by, created_at
                 FROM facts
                 WHERE invalid_at IS NULL AND subject = ? AND predicate = ?",
                vec![
                    Box::new(subject.to_string()),
                    Box::new(predicate.to_string()),
                ],
            )
        };

        let params_ref: Vec<&dyn duckdb::ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();
        let mut stmt = conn
            .prepare(sql)
            .context("failed to prepare subject facts query")?;

        let rows = stmt
            .query_map(params_ref.as_slice(), |row| {
                Ok(Fact {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    subject: row.get(2)?,
                    predicate: row.get(3)?,
                    object: row.get(4)?,
                    confidence: row.get::<_, f32>(5).unwrap_or(1.0),
                    source_session_id: row.get(6)?,
                    source_agent: row.get(7)?,
                    valid_at: row.get(8)?,
                    invalid_at: row.get(9)?,
                    superseded_by: row.get(10)?,
                    created_at: row.get(11)?,
                })
            })
            .context("failed to query subject facts")?;

        let mut facts = Vec::new();
        for row in rows {
            facts.push(row.context("failed to read fact row")?);
        }
        Ok(facts)
    }

    /// Search facts by subject or object text (ILIKE).
    pub fn search_facts(&self, query: &str) -> Result<Vec<Fact>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let pattern = format!("%{query}%");
        let mut stmt = conn
            .prepare(
                "SELECT id, project_id, subject, predicate, object, confidence,
                        source_session_id, source_agent, valid_at, invalid_at,
                        superseded_by, created_at
                 FROM facts
                 WHERE (subject ILIKE ? OR object ILIKE ?)
                 ORDER BY invalid_at IS NULL DESC, valid_at DESC NULLS LAST",
            )
            .context("failed to prepare search_facts")?;

        let rows = stmt
            .query_map(params![pattern, pattern], |row| {
                Ok(Fact {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    subject: row.get(2)?,
                    predicate: row.get(3)?,
                    object: row.get(4)?,
                    confidence: row.get::<_, f32>(5).unwrap_or(1.0),
                    source_session_id: row.get(6)?,
                    source_agent: row.get(7)?,
                    valid_at: row.get(8)?,
                    invalid_at: row.get(9)?,
                    superseded_by: row.get(10)?,
                    created_at: row.get(11)?,
                })
            })
            .context("failed to query facts")?;

        let mut facts = Vec::new();
        for row in rows {
            facts.push(row.context("failed to read fact row")?);
        }
        Ok(facts)
    }

    /// Get the full temporal history for a subject (all facts, including invalidated).
    pub fn get_fact_history(&self, subject: &str) -> Result<Vec<Fact>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT id, project_id, subject, predicate, object, confidence,
                        source_session_id, source_agent, valid_at, invalid_at,
                        superseded_by, created_at
                 FROM facts
                 WHERE subject = ?
                 ORDER BY valid_at ASC NULLS LAST",
            )
            .context("failed to prepare get_fact_history")?;

        let rows = stmt
            .query_map(params![subject], |row| {
                Ok(Fact {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    subject: row.get(2)?,
                    predicate: row.get(3)?,
                    object: row.get(4)?,
                    confidence: row.get::<_, f32>(5).unwrap_or(1.0),
                    source_session_id: row.get(6)?,
                    source_agent: row.get(7)?,
                    valid_at: row.get(8)?,
                    invalid_at: row.get(9)?,
                    superseded_by: row.get(10)?,
                    created_at: row.get(11)?,
                })
            })
            .context("failed to query fact history")?;

        let mut facts = Vec::new();
        for row in rows {
            facts.push(row.context("failed to read fact row")?);
        }
        Ok(facts)
    }

    /// Count facts in the database.
    pub fn count_facts(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM facts", [], |row| row.get(0))
            .context("failed to count facts")?;
        Ok(count as usize)
    }

    /// Count currently valid facts.
    pub fn count_active_facts(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM facts WHERE invalid_at IS NULL",
                [],
                |row| row.get(0),
            )
            .context("failed to count active facts")?;
        Ok(count as usize)
    }

    /// Insert a code symbol record.
    pub fn insert_code_symbol(&self, symbol: &CodeSymbol) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        conn.execute(
            "INSERT INTO code_symbols (
                id, project_id, file_path, symbol_name, symbol_kind,
                signature, docstring, start_line, end_line, visibility,
                parent_symbol, pagerank_score, reference_count, language,
                content_hash, indexed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                symbol.id,
                symbol.project_id,
                symbol.file_path,
                symbol.symbol_name,
                symbol.symbol_kind,
                symbol.signature,
                symbol.docstring,
                symbol.start_line,
                symbol.end_line,
                symbol.visibility,
                symbol.parent_symbol,
                symbol.pagerank_score,
                symbol.reference_count,
                symbol.language,
                symbol.content_hash,
                symbol.indexed_at.unwrap_or_else(|| Utc::now().naive_utc()),
            ],
        )
        .context("failed to insert code_symbol")?;
        Ok(())
    }

    /// Insert a code dependency record.
    pub fn insert_code_dependency(&self, dep: &CodeDependency) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        conn.execute(
            "INSERT INTO code_dependencies (
                id, project_id, from_symbol, to_symbol, relationship,
                from_file, to_file
            ) VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![
                dep.id,
                dep.project_id,
                dep.from_symbol,
                dep.to_symbol,
                dep.relationship,
                dep.from_file,
                dep.to_file,
            ],
        )
        .context("failed to insert code_dependency")?;
        Ok(())
    }

    /// Insert an analysis run record.
    pub fn insert_analysis_run(&self, run: &AnalysisRun) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        conn.execute(
            "INSERT OR REPLACE INTO code_analysis_runs (
                project_id, commit_hash, files_analyzed, symbols_extracted,
                dependencies_found, chunks_generated, duration_ms, analyzed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                run.project_id,
                run.commit_hash,
                run.files_analyzed,
                run.symbols_extracted,
                run.dependencies_found,
                run.chunks_generated,
                run.duration_ms,
                run.analyzed_at.unwrap_or_else(|| Utc::now().naive_utc()),
            ],
        )
        .context("failed to insert analysis_run")?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Return the most recent sessions ordered by `started_at` descending.
    pub fn get_recent_sessions(&self, limit: usize) -> Result<Vec<Session>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT id, project_id, agent, started_at, ended_at,
                        duration_minutes, message_count, tool_call_count,
                        total_tokens, files_changed, summary
                 FROM sessions
                 ORDER BY started_at DESC NULLS LAST
                 LIMIT ?",
            )
            .context("failed to prepare get_recent_sessions")?;

        let rows = stmt
            .query_map(params![limit as i64], |row| {
                let files_str: String = row.get::<_, String>(9).unwrap_or_default();
                Ok(Session {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    agent: row.get(2)?,
                    started_at: row.get(3)?,
                    ended_at: row.get(4)?,
                    duration_minutes: row.get(5)?,
                    message_count: row.get(6)?,
                    tool_call_count: row.get(7)?,
                    total_tokens: row.get(8)?,
                    files_changed: json_to_vec(&files_str),
                    summary: row.get(10)?,
                })
            })
            .context("failed to query recent sessions")?;

        let mut sessions = Vec::new();
        for row in rows {
            sessions.push(row.context("failed to read session row")?);
        }
        Ok(sessions)
    }

    /// Return a session by its exact identifier.
    pub fn get_session(&self, session_id: &str) -> Result<Option<Session>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT id, project_id, agent, started_at, ended_at,
                        duration_minutes, message_count, tool_call_count,
                        total_tokens, files_changed, summary
                 FROM sessions
                 WHERE id = ?",
            )
            .context("failed to prepare get_session")?;

        let mut rows = stmt
            .query_map(params![session_id], |row| {
                let files_str: String = row.get::<_, String>(9).unwrap_or_default();
                Ok(Session {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    agent: row.get(2)?,
                    started_at: row.get(3)?,
                    ended_at: row.get(4)?,
                    duration_minutes: row.get(5)?,
                    message_count: row.get(6)?,
                    tool_call_count: row.get(7)?,
                    total_tokens: row.get(8)?,
                    files_changed: json_to_vec(&files_str),
                    summary: row.get(10)?,
                })
            })
            .context("failed to query session by id")?;

        rows.next()
            .transpose()
            .context("failed to read session by id")
    }

    /// Search sessions by agent, project, or since date.
    pub fn search_sessions(
        &self,
        agent: Option<&str>,
        project: Option<&str>,
        since: Option<NaiveDateTime>,
        limit: usize,
    ) -> Result<Vec<Session>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let mut conditions = Vec::new();
        let mut param_values: Vec<Box<dyn duckdb::ToSql>> = Vec::new();

        if let Some(agent) = agent {
            conditions.push("agent ILIKE ?".to_string());
            param_values.push(Box::new(format!("%{agent}%")));
        }
        if let Some(project) = project {
            conditions.push("project_id ILIKE ?".to_string());
            param_values.push(Box::new(format!("%{project}%")));
        }
        if let Some(since) = since {
            conditions.push("started_at >= ?".to_string());
            param_values.push(Box::new(since));
        }

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };

        let sql = format!(
            "SELECT id, project_id, agent, started_at, ended_at,
                    duration_minutes, message_count, tool_call_count,
                    total_tokens, files_changed, summary
             FROM sessions
             {where_clause}
             ORDER BY started_at DESC NULLS LAST
             LIMIT ?"
        );

        param_values.push(Box::new(limit as i64));

        let params_ref: Vec<&dyn duckdb::ToSql> = param_values.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn
            .prepare(&sql)
            .context("failed to prepare search_sessions")?;

        let rows = stmt
            .query_map(params_ref.as_slice(), |row| {
                let files_str: String = row.get::<_, String>(9).unwrap_or_default();
                Ok(Session {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    agent: row.get(2)?,
                    started_at: row.get(3)?,
                    ended_at: row.get(4)?,
                    duration_minutes: row.get(5)?,
                    message_count: row.get(6)?,
                    tool_call_count: row.get(7)?,
                    total_tokens: row.get(8)?,
                    files_changed: json_to_vec(&files_str),
                    summary: row.get(10)?,
                })
            })
            .context("failed to query sessions")?;

        let mut sessions = Vec::new();
        for row in rows {
            sessions.push(row.context("failed to read session row")?);
        }
        Ok(sessions)
    }

    /// Get all sessions from the current UTC calendar day.
    pub fn get_todays_sessions(&self, since: NaiveDateTime) -> Result<Vec<Session>> {
        self.search_sessions(None, None, Some(since), i32::MAX as usize)
    }

    /// Get tool calls for a session.
    pub fn get_tool_calls_for_session(&self, session_id: &str) -> Result<Vec<ToolCall>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, tool_name, command, success,
                        error_message, duration_ms, timestamp
                 FROM tool_calls
                 WHERE session_id = ?
                 ORDER BY timestamp ASC NULLS LAST",
            )
            .context("failed to prepare get_tool_calls_for_session")?;

        let rows = stmt
            .query_map(params![session_id], |row| {
                Ok(ToolCall {
                    id: row.get(0)?,
                    session_id: row.get(1)?,
                    tool_name: row.get(2)?,
                    command: row.get(3)?,
                    success: row.get(4)?,
                    error_message: row.get(5)?,
                    duration_ms: row.get(6)?,
                    timestamp: row.get(7)?,
                })
            })
            .context("failed to query tool calls")?;

        let mut tool_calls = Vec::new();
        for row in rows {
            tool_calls.push(row.context("failed to read tool_call row")?);
        }
        Ok(tool_calls)
    }

    /// Get all memories, optionally filtered by project.
    pub fn get_memories(&self, project: Option<&str>, limit: usize) -> Result<Vec<Memory>> {
        self.get_memories_filtered(project, &[], limit)
    }

    /// List memories with optional project and source-agent filters.
    ///
    /// When `agents` is non-empty, only memories whose source session was
    /// produced by one of the listed agents are returned; manual notes
    /// (no source session) are excluded in that case.
    pub fn get_memories_filtered(
        &self,
        project: Option<&str>,
        agents: &[String],
        limit: usize,
    ) -> Result<Vec<Memory>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let mut sql = String::from(
            "SELECT m.id, m.project_id, m.content, m.memory_type, m.source_session_id,
                    m.confidence, m.access_count, m.created_at, m.updated_at, m.valid_until
             FROM memories m",
        );
        if !agents.is_empty() {
            sql.push_str(" JOIN sessions s ON m.source_session_id = s.id");
        }
        let mut conditions: Vec<String> = Vec::new();
        let mut param_values: Vec<Box<dyn duckdb::ToSql>> = Vec::new();
        if let Some(project) = project {
            conditions.push("m.project_id ILIKE ?".to_string());
            param_values.push(Box::new(format!("%{project}%")));
        }
        if !agents.is_empty() {
            conditions.push(format!(
                "s.agent IN ({})",
                vec!["?"; agents.len()].join(", ")
            ));
            for agent in agents {
                param_values.push(Box::new(agent.clone()));
            }
        }
        if !conditions.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&conditions.join(" AND "));
        }
        sql.push_str(" ORDER BY m.updated_at DESC NULLS LAST LIMIT ?");
        param_values.push(Box::new(limit as i64));

        let params_ref: Vec<&dyn duckdb::ToSql> =
            param_values.iter().map(|pv| pv.as_ref()).collect();
        let mut stmt = conn
            .prepare(&sql)
            .context("failed to prepare get_memories_filtered")?;

        let map_row = |row: &duckdb::Row| -> duckdb::Result<Memory> {
            Ok(Memory {
                id: row.get(0)?,
                project_id: row.get(1)?,
                content: row.get(2)?,
                memory_type: row.get(3)?,
                source_session_id: row.get(4)?,
                confidence: row.get::<_, f32>(5).unwrap_or(1.0),
                access_count: row.get::<_, i32>(6).unwrap_or(0),
                created_at: row.get(7)?,
                updated_at: row.get(8)?,
                valid_until: row.get(9)?,
            })
        };

        let rows = stmt
            .query_map(params_ref.as_slice(), map_row)
            .context("failed to query memories")?;

        let mut memories = Vec::new();
        for row in rows {
            memories.push(row.context("failed to read memory row")?);
        }
        Ok(memories)
    }

    /// Count sessions in the database.
    pub fn count_sessions(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM sessions", [], |row| row.get(0))
            .context("failed to count sessions")?;
        Ok(count as usize)
    }

    /// Count memories in the database.
    pub fn count_memories(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
            .context("failed to count memories")?;
        Ok(count as usize)
    }

    /// Count tool calls in the database.
    pub fn count_tool_calls(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM tool_calls", [], |row| row.get(0))
            .context("failed to count tool_calls")?;
        Ok(count as usize)
    }

    /// Search sessions by summary ILIKE.
    pub fn search_sessions_by_summary(&self, query: &str) -> Result<Vec<Session>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let pattern = format!("%{query}%");
        let mut stmt = conn
            .prepare(
                "SELECT id, project_id, agent, started_at, ended_at,
                        duration_minutes, message_count, tool_call_count,
                        total_tokens, files_changed, summary
                 FROM sessions
                 WHERE summary ILIKE ?
                 ORDER BY started_at DESC NULLS LAST",
            )
            .context("failed to prepare search_sessions_by_summary")?;

        let rows = stmt
            .query_map(params![pattern], |row| {
                let files_str: String = row.get::<_, String>(9).unwrap_or_default();
                Ok(Session {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    agent: row.get(2)?,
                    started_at: row.get(3)?,
                    ended_at: row.get(4)?,
                    duration_minutes: row.get(5)?,
                    message_count: row.get(6)?,
                    tool_call_count: row.get(7)?,
                    total_tokens: row.get(8)?,
                    files_changed: json_to_vec(&files_str),
                    summary: row.get(10)?,
                })
            })
            .context("failed to query sessions by summary")?;

        let mut sessions = Vec::new();
        for row in rows {
            sessions.push(row.context("failed to read session row")?);
        }
        Ok(sessions)
    }

    /// Count decisions in the database.
    pub fn count_decisions(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM decisions", [], |row| row.get(0))
            .context("failed to count decisions")?;
        Ok(count as usize)
    }

    /// Count sessions since a given timestamp.
    pub fn count_sessions_since(&self, since: NaiveDateTime) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sessions WHERE started_at >= ?",
                params![since],
                |row| row.get(0),
            )
            .context("failed to count sessions since")?;
        Ok(count as usize)
    }

    /// Count memories since a given timestamp.
    pub fn count_memories_since(&self, since: NaiveDateTime) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE created_at >= ?",
                params![since],
                |row| row.get(0),
            )
            .context("failed to count memories since")?;
        Ok(count as usize)
    }

    /// Count decisions since a given timestamp.
    pub fn count_decisions_since(&self, since: NaiveDateTime) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM decisions WHERE created_at >= ?",
                params![since],
                |row| row.get(0),
            )
            .context("failed to count decisions since")?;
        Ok(count as usize)
    }

    /// Count tool calls since a given timestamp.
    pub fn count_tool_calls_since(&self, since: NaiveDateTime) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM tool_calls WHERE timestamp >= ?",
                params![since],
                |row| row.get(0),
            )
            .context("failed to count tool calls since")?;
        Ok(count as usize)
    }

    /// Get distinct agent names active since a timestamp.
    pub fn get_active_agents_since(&self, since: NaiveDateTime) -> Result<Vec<String>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare("SELECT DISTINCT agent FROM sessions WHERE started_at >= ? ORDER BY agent")
            .context("failed to prepare active agents since")?;
        let rows = stmt
            .query_map(params![since], |row| row.get::<_, String>(0))
            .context("failed to query active agents since")?;
        let mut agents = Vec::new();
        for row in rows {
            agents.push(row.context("failed to read agent row")?);
        }
        Ok(agents)
    }

    /// Get per-day session counts for the last N local days (sparklines).
    pub fn get_daily_session_counts<Tz>(
        &self,
        days: i64,
        now: DateTime<Utc>,
        tz: &Tz,
    ) -> Result<Vec<i64>>
    where
        Tz: chrono::TimeZone,
        Tz::Offset: std::fmt::Display,
    {
        Ok(self
            .get_daily_counts(days, now, tz)?
            .into_iter()
            .map(|(_, sessions, _, _)| sessions)
            .collect())
    }

    /// Get per-day memory counts for the last N local days (sparklines).
    pub fn get_daily_memory_counts<Tz>(
        &self,
        days: i64,
        now: DateTime<Utc>,
        tz: &Tz,
    ) -> Result<Vec<i64>>
    where
        Tz: chrono::TimeZone,
        Tz::Offset: std::fmt::Display,
    {
        Ok(self
            .get_daily_counts(days, now, tz)?
            .into_iter()
            .map(|(_, _, memories, _)| memories)
            .collect())
    }

    /// Get per-day decision counts for the last N local days (sparklines).
    pub fn get_daily_decision_counts<Tz>(
        &self,
        days: i64,
        now: DateTime<Utc>,
        tz: &Tz,
    ) -> Result<Vec<i64>>
    where
        Tz: chrono::TimeZone,
        Tz::Offset: std::fmt::Display,
    {
        Ok(self
            .get_daily_counts(days, now, tz)?
            .into_iter()
            .map(|(_, _, _, decisions)| decisions)
            .collect())
    }

    /// Returns (date_str, sessions_count, memories_count, decisions_count)
    /// for each of the last `days` calendar days in timezone `tz`, ending
    /// with the local day containing `now`. Oldest day first.
    ///
    /// Buckets are computed on local calendar dates, not UTC dates, so
    /// sparklines match what the user considers "today".
    pub fn get_daily_counts<Tz>(
        &self,
        days: i64,
        now: DateTime<Utc>,
        tz: &Tz,
    ) -> Result<Vec<(String, i64, i64, i64)>>
    where
        Tz: chrono::TimeZone,
        Tz::Offset: std::fmt::Display,
    {
        let days = days.max(0);
        let window = crate::timeutil::day_window_in_tz(now, tz);
        let cutoff = window.cutoff_days_back(days);
        let conn = self.conn.lock().expect("lock poisoned");

        // Query each table separately and merge in Rust, bucketing by the
        // local calendar date of each timestamp.
        let sess_map = count_by_local_day(
            &conn,
            "SELECT started_at FROM sessions WHERE started_at >= ?",
            cutoff,
            tz,
        )
        .context("daily session counts")?;
        let mem_map = count_by_local_day(
            &conn,
            "SELECT created_at FROM memories WHERE created_at >= ?",
            cutoff,
            tz,
        )
        .context("daily memory counts")?;
        let dec_map = count_by_local_day(
            &conn,
            "SELECT created_at FROM decisions WHERE created_at >= ?",
            cutoff,
            tz,
        )
        .context("daily decision counts")?;

        // Build ordered result for each local day in the range.
        let today_local = now.with_timezone(tz).date_naive();
        let mut result = Vec::new();
        for i in (0..=days).rev() {
            let date = today_local - chrono::Duration::days(i);
            let date_str = date.format("%Y-%m-%d").to_string();
            let sess = sess_map.get(&date).copied().unwrap_or(0);
            let mem = mem_map.get(&date).copied().unwrap_or(0);
            let dec = dec_map.get(&date).copied().unwrap_or(0);
            result.push((date_str, sess, mem, dec));
        }

        Ok(result)
    }

    /// Returns sessions since `since` (start of the local yesterday window)
    /// for briefing comparison.
    pub fn get_recent_sessions_for_briefing(&self, since: NaiveDateTime) -> Result<Vec<Session>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let cutoff = since;

        let mut stmt = conn
            .prepare(
                "SELECT id, project_id, agent, started_at, ended_at,
                        duration_minutes, message_count, tool_call_count,
                        total_tokens, files_changed, summary
                 FROM sessions
                 WHERE started_at >= ?
                 ORDER BY started_at DESC",
            )
            .context("failed to prepare recent sessions for briefing")?;

        let rows = stmt
            .query_map(params![cutoff], |row| {
                let files_str: String = row.get::<_, String>(9).unwrap_or_default();
                Ok(Session {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    agent: row.get(2)?,
                    started_at: row.get(3)?,
                    ended_at: row.get(4)?,
                    duration_minutes: row.get(5)?,
                    message_count: row.get(6)?,
                    tool_call_count: row.get(7)?,
                    total_tokens: row.get(8)?,
                    files_changed: json_to_vec(&files_str),
                    summary: row.get(10)?,
                })
            })
            .context("failed to query recent sessions for briefing")?;

        let mut sessions = Vec::new();
        for row in rows {
            sessions.push(row.context("failed to read briefing session row")?);
        }
        Ok(sessions)
    }

    /// Get all decisions, optionally filtered by project.
    pub fn get_decisions(&self, project: Option<&str>, limit: usize) -> Result<Vec<Decision>> {
        self.get_decisions_filtered(project, &[], limit)
    }

    /// List decisions with optional project and source-agent filters.
    ///
    /// When `agents` is non-empty, only decisions recorded in sessions of
    /// the listed agents are returned.
    pub fn get_decisions_filtered(
        &self,
        project: Option<&str>,
        agents: &[String],
        limit: usize,
    ) -> Result<Vec<Decision>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let mut sql = String::from(
            "SELECT d.id, d.session_id, d.project_id, d.decision_type, d.what,
                    d.why, d.alternatives, d.outcome, d.created_at, d.valid_until
             FROM decisions d",
        );
        if !agents.is_empty() {
            sql.push_str(" JOIN sessions s ON d.session_id = s.id");
        }
        let mut conditions: Vec<String> = Vec::new();
        let mut param_values: Vec<Box<dyn duckdb::ToSql>> = Vec::new();
        if let Some(project) = project {
            conditions.push("d.project_id ILIKE ?".to_string());
            param_values.push(Box::new(format!("%{project}%")));
        }
        if !agents.is_empty() {
            conditions.push(format!(
                "s.agent IN ({})",
                vec!["?"; agents.len()].join(", ")
            ));
            for agent in agents {
                param_values.push(Box::new(agent.clone()));
            }
        }
        if !conditions.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&conditions.join(" AND "));
        }
        sql.push_str(" ORDER BY d.created_at DESC NULLS LAST LIMIT ?");
        param_values.push(Box::new(limit as i64));

        let params_ref: Vec<&dyn duckdb::ToSql> =
            param_values.iter().map(|pv| pv.as_ref()).collect();
        let mut stmt = conn
            .prepare(&sql)
            .context("failed to prepare get_decisions_filtered")?;

        let map_row = |row: &duckdb::Row| -> duckdb::Result<Decision> {
            let alts_str: String = row.get::<_, String>(6).unwrap_or_default();
            Ok(Decision {
                id: row.get(0)?,
                session_id: row.get(1)?,
                project_id: row.get(2)?,
                decision_type: row.get(3)?,
                what: row.get(4)?,
                why: row.get(5)?,
                alternatives: json_to_vec(&alts_str),
                outcome: row.get(7)?,
                created_at: row.get(8)?,
                valid_until: row.get(9)?,
            })
        };

        let rows = stmt
            .query_map(params_ref.as_slice(), map_row)
            .context("failed to query decisions")?;

        let mut decisions = Vec::new();
        for row in rows {
            decisions.push(row.context("failed to read decision row")?);
        }
        Ok(decisions)
    }

    /// Get all sessions for a specific project.
    pub fn get_project_sessions(&self, project_id: &str) -> Result<Vec<Session>> {
        self.search_sessions(None, Some(project_id), None, i32::MAX as usize)
    }

    /// Get distinct project IDs across all persisted knowledge types.
    pub fn get_project_ids(&self) -> Result<Vec<String>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT DISTINCT project_id FROM (
                    SELECT project_id FROM sessions
                    UNION ALL
                    SELECT project_id FROM memories
                    UNION ALL
                    SELECT project_id FROM decisions
                    UNION ALL
                    SELECT project_id FROM facts
                 ) AS projects
                 WHERE project_id IS NOT NULL
                 ORDER BY project_id",
            )
            .context("failed to prepare get_project_ids")?;

        let rows = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .context("failed to query project IDs")?;

        let mut ids = Vec::new();
        for row in rows {
            ids.push(row.context("failed to read project_id row")?);
        }
        Ok(ids)
    }

    /// Delete a session and all data derived from that transcript.
    ///
    /// Graph cleanup is committed before relational deletion to avoid a DuckDB
    /// foreign-key limitation around parent/child deletes in one transaction.
    pub fn delete_session(&self, session_id: &str) -> Result<bool> {
        let mut conn = self.conn.lock().expect("lock poisoned");
        let stale_graph_ids: Vec<String> = {
            let mut statement = conn
                .prepare(
                    "SELECT 'session:' || ? AS graph_id
                     UNION ALL
                     SELECT 'memory:' || id FROM memories WHERE source_session_id = ?
                     UNION ALL
                     SELECT 'decision:' || id FROM decisions WHERE session_id = ?",
                )
                .context("failed to prepare session graph lookup")?;
            let rows = statement
                .query_map(params![session_id, session_id, session_id], |row| {
                    row.get::<_, String>(0)
                })
                .context("failed to query session graph IDs")?;
            let mut ids = Vec::new();
            for row in rows {
                ids.push(row.context("failed to read session graph ID")?);
            }
            ids
        };
        for graph_id in &stale_graph_ids {
            delete_graph_node_with_edges(&mut conn, graph_id)?;
        }
        conn.execute(
            "DELETE FROM memory_tags WHERE memory_id IN (
                SELECT id FROM memories WHERE source_session_id = ?
            )",
            params![session_id],
        )
        .context("failed to delete session memory tags")?;

        let transaction = conn
            .transaction()
            .context("failed to begin session deletion transaction")?;
        transaction
            .execute(
                "DELETE FROM memories WHERE source_session_id = ?",
                params![session_id],
            )
            .context("failed to delete session memories")?;
        transaction
            .execute(
                "DELETE FROM decisions WHERE session_id = ?",
                params![session_id],
            )
            .context("failed to delete session decisions")?;
        transaction
            .execute(
                "DELETE FROM facts WHERE source_session_id = ?",
                params![session_id],
            )
            .context("failed to delete session facts")?;
        transaction
            .execute(
                "DELETE FROM tool_calls WHERE session_id = ?",
                params![session_id],
            )
            .context("failed to delete session tool calls")?;
        let affected = transaction
            .execute("DELETE FROM sessions WHERE id = ?", params![session_id])
            .context("failed to delete session")?;
        transaction
            .commit()
            .context("failed to commit session deletion transaction")?;
        Ok(affected > 0)
    }
    /// Delete old sessions before a given date. Returns number deleted.
    pub fn gc_sessions_before(&self, before: NaiveDateTime) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");

        // Graph session nodes are derived from sessions and must not survive
        // retention cleanup.
        conn.execute(
            "DELETE FROM graph_edges
             WHERE from_id IN (
                    SELECT 'session:' || id FROM sessions WHERE started_at < ?
                 ) OR to_id IN (
                    SELECT 'session:' || id FROM sessions WHERE started_at < ?
                 )",
            params![before, before],
        )
        .context("failed to delete graph edges for old sessions")?;
        conn.execute(
            "DELETE FROM graph_nodes
             WHERE id IN (SELECT 'session:' || id FROM sessions WHERE started_at < ?)",
            params![before],
        )
        .context("failed to delete graph nodes for old sessions")?;

        // First delete tool_calls for those sessions
        conn.execute(
            "DELETE FROM tool_calls WHERE session_id IN (
                SELECT id FROM sessions WHERE started_at < ?
            )",
            params![before],
        )
        .context("failed to delete old tool_calls")?;

        let affected = conn
            .execute("DELETE FROM sessions WHERE started_at < ?", params![before])
            .context("failed to delete old sessions")?;

        Ok(affected)
    }

    /// Delete tool calls whose session no longer exists.
    pub fn delete_orphaned_tool_calls(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");
        let affected = conn
            .execute(
                "DELETE FROM tool_calls
                 WHERE session_id IS NOT NULL
                   AND session_id NOT IN (SELECT id FROM sessions)",
                [],
            )
            .context("failed to delete orphaned tool_calls")?;
        Ok(affected)
    }

    /// Insert a note as a memory.
    pub fn insert_note(&self, content: &str, project: Option<&str>) -> Result<String> {
        let content = content.trim();
        if content.is_empty() {
            anyhow::bail!("note content cannot be empty");
        }
        let project = project.and_then(|project| {
            let trimmed = project.trim();
            (!trimmed.is_empty()).then_some(trimmed)
        });
        let id = uuid::Uuid::new_v4().to_string();
        let now = Utc::now().naive_utc();
        let memory = Memory {
            id: id.clone(),
            project_id: project.map(|s| s.to_string()),
            content: content.to_string(),
            memory_type: Some("note".to_string()),
            source_session_id: None,
            confidence: 1.0,
            access_count: 0,
            created_at: Some(now),
            updated_at: Some(now),
            valid_until: None,
        };
        self.insert_memory(&memory)?;
        Ok(id)
    }

    /// Insert a manual note and associate normalized tags with it.
    pub fn insert_note_with_tags(
        &self,
        content: &str,
        project: Option<&str>,
        tags: &[String],
    ) -> Result<String> {
        let normalized = normalize_tags(tags)?;
        let id = self.insert_note(content, project)?;
        self.set_memory_tags(&id, &normalized)?;
        Ok(id)
    }

    /// Replace all tags for a memory.
    pub fn set_memory_tags(&self, memory_id: &str, tags: &[String]) -> Result<()> {
        let normalized = normalize_tags(tags)?;
        let conn = self.conn.lock().expect("lock poisoned");
        conn.execute(
            "DELETE FROM memory_tags WHERE memory_id = ?",
            params![memory_id],
        )
        .context("failed to clear memory tags")?;

        let mut stmt = conn
            .prepare("INSERT INTO memory_tags (memory_id, tag) VALUES (?, ?)")
            .context("failed to prepare memory tag insert")?;
        for tag in &normalized {
            stmt.execute(params![memory_id, tag])
                .context("failed to insert memory tags")?;
        }
        Ok(())
    }

    /// Return tags for a memory in stable alphabetical order.
    pub fn get_memory_tags(&self, memory_id: &str) -> Result<Vec<String>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare("SELECT tag FROM memory_tags WHERE memory_id = ? ORDER BY tag")
            .context("failed to prepare memory tag lookup")?;
        let rows = stmt
            .query_map(params![memory_id], |row| row.get::<_, String>(0))
            .context("failed to query memory tags")?;
        let mut tags = Vec::new();
        for row in rows {
            tags.push(row.context("failed to read memory tag")?);
        }
        Ok(tags)
    }

    /// Find memories by exact (case-insensitive) tag.
    pub fn search_memories_by_tag(&self, tag: &str, limit: usize) -> Result<Vec<Memory>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT m.id, m.project_id, m.content, m.memory_type,
                        m.source_session_id, m.confidence, m.access_count,
                        m.created_at, m.updated_at, m.valid_until
                 FROM memories m
                 JOIN memory_tags t ON t.memory_id = m.id
                 WHERE lower(t.tag) = lower(?)
                 ORDER BY m.created_at DESC NULLS LAST
                 LIMIT ?",
            )
            .context("failed to prepare tag search")?;
        let rows = stmt
            .query_map(params![tag, limit as i64], |row| {
                Ok(Memory {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    content: row.get(2)?,
                    memory_type: row.get(3)?,
                    source_session_id: row.get(4)?,
                    confidence: row.get::<_, f32>(5).unwrap_or(1.0),
                    access_count: row.get::<_, i32>(6).unwrap_or(0),
                    created_at: row.get(7)?,
                    updated_at: row.get(8)?,
                    valid_until: row.get(9)?,
                })
            })
            .context("failed to query memories by tag")?;
        let mut memories = Vec::new();
        for row in rows {
            memories.push(row.context("failed to read tagged memory")?);
        }
        Ok(memories)
    }

    /// Get per-agent session counts.
    pub fn get_agent_session_counts(&self) -> Result<Vec<(String, usize)>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare("SELECT agent, COUNT(*) FROM sessions GROUP BY agent ORDER BY COUNT(*) DESC")
            .context("failed to prepare agent session counts")?;

        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .context("failed to query agent session counts")?;

        let mut counts = Vec::new();
        for row in rows {
            let (agent, count) = row.context("failed to read agent count row")?;
            counts.push((agent, count as usize));
        }
        Ok(counts)
    }

    /// Get per-project session counts.
    pub fn get_project_session_counts(&self) -> Result<Vec<(String, usize)>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT COALESCE(project_id, '(unknown)'), COUNT(*)
                 FROM sessions
                 GROUP BY project_id
                 ORDER BY COUNT(*) DESC",
            )
            .context("failed to prepare project session counts")?;

        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .context("failed to query project session counts")?;

        let mut counts = Vec::new();
        for row in rows {
            let (project, count) = row.context("failed to read project count row")?;
            counts.push((project, count as usize));
        }
        Ok(counts)
    }

    /// Full-text search over memory content using DuckDB `ILIKE`.
    pub fn search_memories(&self, query: &str) -> Result<Vec<Memory>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let pattern = format!("%{query}%");
        let mut stmt = conn
            .prepare(
                "SELECT id, project_id, content, memory_type, source_session_id,
                        confidence, access_count, created_at, updated_at, valid_until
                 FROM memories
                 WHERE content ILIKE ?
                 ORDER BY updated_at DESC NULLS LAST",
            )
            .context("failed to prepare search_memories")?;

        let rows = stmt
            .query_map(params![pattern], |row| {
                Ok(Memory {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    content: row.get(2)?,
                    memory_type: row.get(3)?,
                    source_session_id: row.get(4)?,
                    confidence: row.get::<_, f32>(5).unwrap_or(1.0),
                    access_count: row.get::<_, i32>(6).unwrap_or(0),
                    created_at: row.get(7)?,
                    updated_at: row.get(8)?,
                    valid_until: row.get(9)?,
                })
            })
            .context("failed to query memories")?;

        let mut memories = Vec::new();
        for row in rows {
            memories.push(row.context("failed to read memory row")?);
        }
        Ok(memories)
    }

    // -----------------------------------------------------------------------
    // Code analysis queries
    // -----------------------------------------------------------------------

    /// Get all symbols for a project.
    pub fn get_symbols_for_project(
        &self,
        project_id: &str,
        limit: usize,
    ) -> Result<Vec<CodeSymbol>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT id, project_id, file_path, symbol_name, symbol_kind,
                        signature, docstring, start_line, end_line, visibility,
                        parent_symbol, pagerank_score, reference_count, language,
                        content_hash, indexed_at
                 FROM code_symbols
                 WHERE project_id = ?
                 ORDER BY file_path, start_line
                 LIMIT ?",
            )
            .context("failed to prepare get_symbols_for_project")?;

        let rows = stmt
            .query_map(params![project_id, limit as i64], |row| {
                Ok(CodeSymbol {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    file_path: row.get(2)?,
                    symbol_name: row.get(3)?,
                    symbol_kind: row.get(4)?,
                    signature: row.get(5)?,
                    docstring: row.get(6)?,
                    start_line: row.get(7)?,
                    end_line: row.get(8)?,
                    visibility: row.get(9)?,
                    parent_symbol: row.get(10)?,
                    pagerank_score: row.get::<_, f64>(11).unwrap_or(0.0),
                    reference_count: row.get::<_, i32>(12).unwrap_or(0),
                    language: row.get(13)?,
                    content_hash: row.get(14)?,
                    indexed_at: row.get(15)?,
                })
            })
            .context("failed to query symbols for project")?;

        let mut symbols = Vec::new();
        for row in rows {
            symbols.push(row.context("failed to read code_symbol row")?);
        }
        Ok(symbols)
    }

    /// Get top symbols by pagerank score.
    pub fn get_top_symbols(&self, project_id: &str, limit: usize) -> Result<Vec<CodeSymbol>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT id, project_id, file_path, symbol_name, symbol_kind,
                        signature, docstring, start_line, end_line, visibility,
                        parent_symbol, pagerank_score, reference_count, language,
                        content_hash, indexed_at
                 FROM code_symbols
                 WHERE project_id = ?
                 ORDER BY pagerank_score DESC
                 LIMIT ?",
            )
            .context("failed to prepare get_top_symbols")?;

        let rows = stmt
            .query_map(params![project_id, limit as i64], |row| {
                Ok(CodeSymbol {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    file_path: row.get(2)?,
                    symbol_name: row.get(3)?,
                    symbol_kind: row.get(4)?,
                    signature: row.get(5)?,
                    docstring: row.get(6)?,
                    start_line: row.get(7)?,
                    end_line: row.get(8)?,
                    visibility: row.get(9)?,
                    parent_symbol: row.get(10)?,
                    pagerank_score: row.get::<_, f64>(11).unwrap_or(0.0),
                    reference_count: row.get::<_, i32>(12).unwrap_or(0),
                    language: row.get(13)?,
                    content_hash: row.get(14)?,
                    indexed_at: row.get(15)?,
                })
            })
            .context("failed to query top symbols")?;

        let mut symbols = Vec::new();
        for row in rows {
            symbols.push(row.context("failed to read code_symbol row")?);
        }
        Ok(symbols)
    }

    /// Get callers of a symbol (dependencies where this symbol is the target).
    pub fn get_callers_of(
        &self,
        symbol_name: &str,
        project_id: &str,
    ) -> Result<Vec<CodeDependency>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT id, project_id, from_symbol, to_symbol, relationship,
                        from_file, to_file
                 FROM code_dependencies
                 WHERE project_id = ? AND to_symbol = ?",
            )
            .context("failed to prepare get_callers_of")?;

        let rows = stmt
            .query_map(params![project_id, symbol_name], |row| {
                Ok(CodeDependency {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    from_symbol: row.get(2)?,
                    to_symbol: row.get(3)?,
                    relationship: row.get(4)?,
                    from_file: row.get(5)?,
                    to_file: row.get(6)?,
                })
            })
            .context("failed to query callers")?;

        let mut deps = Vec::new();
        for row in rows {
            deps.push(row.context("failed to read code_dependency row")?);
        }
        Ok(deps)
    }

    /// Get callees of a symbol (dependencies where this symbol is the source).
    pub fn get_callees_of(
        &self,
        symbol_name: &str,
        project_id: &str,
    ) -> Result<Vec<CodeDependency>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT id, project_id, from_symbol, to_symbol, relationship,
                        from_file, to_file
                 FROM code_dependencies
                 WHERE project_id = ? AND from_symbol = ?",
            )
            .context("failed to prepare get_callees_of")?;

        let rows = stmt
            .query_map(params![project_id, symbol_name], |row| {
                Ok(CodeDependency {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    from_symbol: row.get(2)?,
                    to_symbol: row.get(3)?,
                    relationship: row.get(4)?,
                    from_file: row.get(5)?,
                    to_file: row.get(6)?,
                })
            })
            .context("failed to query callees")?;

        let mut deps = Vec::new();
        for row in rows {
            deps.push(row.context("failed to read code_dependency row")?);
        }
        Ok(deps)
    }

    /// Get all symbols in a specific file.
    pub fn get_symbols_in_file(
        &self,
        file_path: &str,
        project_id: &str,
    ) -> Result<Vec<CodeSymbol>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT id, project_id, file_path, symbol_name, symbol_kind,
                        signature, docstring, start_line, end_line, visibility,
                        parent_symbol, pagerank_score, reference_count, language,
                        content_hash, indexed_at
                 FROM code_symbols
                 WHERE project_id = ? AND file_path = ?
                 ORDER BY start_line",
            )
            .context("failed to prepare get_symbols_in_file")?;

        let rows = stmt
            .query_map(params![project_id, file_path], |row| {
                Ok(CodeSymbol {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    file_path: row.get(2)?,
                    symbol_name: row.get(3)?,
                    symbol_kind: row.get(4)?,
                    signature: row.get(5)?,
                    docstring: row.get(6)?,
                    start_line: row.get(7)?,
                    end_line: row.get(8)?,
                    visibility: row.get(9)?,
                    parent_symbol: row.get(10)?,
                    pagerank_score: row.get::<_, f64>(11).unwrap_or(0.0),
                    reference_count: row.get::<_, i32>(12).unwrap_or(0),
                    language: row.get(13)?,
                    content_hash: row.get(14)?,
                    indexed_at: row.get(15)?,
                })
            })
            .context("failed to query symbols in file")?;

        let mut symbols = Vec::new();
        for row in rows {
            symbols.push(row.context("failed to read code_symbol row")?);
        }
        Ok(symbols)
    }

    /// Count symbols in the database.
    pub fn count_symbols(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM code_symbols", [], |row| row.get(0))
            .context("failed to count symbols")?;
        Ok(count as usize)
    }

    /// Count dependencies in the database.
    pub fn count_dependencies(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM code_dependencies", [], |row| {
                row.get(0)
            })
            .context("failed to count dependencies")?;
        Ok(count as usize)
    }

    /// Get the last analysis run for a project.
    pub fn get_last_analysis(&self, project_id: &str) -> Result<Option<AnalysisRun>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT project_id, commit_hash, files_analyzed, symbols_extracted,
                        dependencies_found, chunks_generated, duration_ms, analyzed_at
                 FROM code_analysis_runs
                 WHERE project_id = ?",
            )
            .context("failed to prepare get_last_analysis")?;

        let result = stmt.query_row(params![project_id], |row| {
            Ok(AnalysisRun {
                project_id: row.get(0)?,
                commit_hash: row.get(1)?,
                files_analyzed: row.get::<_, i32>(2).unwrap_or(0),
                symbols_extracted: row.get::<_, i32>(3).unwrap_or(0),
                dependencies_found: row.get::<_, i32>(4).unwrap_or(0),
                chunks_generated: row.get::<_, i32>(5).unwrap_or(0),
                duration_ms: row.get::<_, i32>(6).unwrap_or(0),
                analyzed_at: row.get(7)?,
            })
        });

        match result {
            Ok(run) => Ok(Some(run)),
            Err(duckdb::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(anyhow::Error::new(e).context("failed to get last analysis")),
        }
    }

    // -----------------------------------------------------------------------
    // Graph: DuckPGQ extension and property graph definition
    // -----------------------------------------------------------------------

    /// Load the DuckPGQ community extension.
    /// Returns a descriptive error if the extension is not available.
    pub fn load_duckpgq(&self) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        conn.execute_batch("INSTALL duckpgq FROM community; LOAD duckpgq;")
            .context(
                "failed to load DuckPGQ extension — \
                 make sure the extension is available (INSTALL duckpgq FROM community)",
            )?;
        Ok(())
    }

    /// Initialise the DuckPGQ property graph over `graph_nodes` / `graph_edges`.
    ///
    /// This must be called **after** `init_schema()` (which creates the tables)
    /// and is intentionally separate because the DuckPGQ extension may not be
    /// available in all environments (tests, CI).
    pub fn init_graph(&self) -> Result<()> {
        self.load_duckpgq()?;

        let conn = self.conn.lock().expect("lock poisoned");
        // DuckPGQ syntax: define vertex / edge tables for the property graph.
        // NOTE: DuckPGQ does not support `IF NOT EXISTS` on CREATE PROPERTY GRAPH,
        // so we use DROP IF EXISTS first for idempotency.
        conn.execute_batch(
            "DROP PROPERTY GRAPH IF EXISTS remembrant_graph;
             CREATE PROPERTY GRAPH remembrant_graph
             VERTEX TABLES (graph_nodes)
             EDGE TABLES (graph_edges SOURCE KEY (from_id) REFERENCES graph_nodes (id)
                                      DESTINATION KEY (to_id) REFERENCES graph_nodes (id));",
        )
        .context("failed to create property graph definition")?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Graph CRUD (standard SQL — no extension required)
    // -----------------------------------------------------------------------

    /// Insert or replace a graph node (upsert).
    pub fn insert_graph_node(
        &self,
        id: &str,
        kind: &str,
        name: &str,
        properties: &str,
    ) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        conn.execute(
            "INSERT OR REPLACE INTO graph_nodes (id, kind, name, properties)
             VALUES (?, ?, ?, ?)",
            params![id, kind, name, properties],
        )
        .context("failed to insert graph node")?;
        Ok(())
    }

    /// Insert a graph edge. Duplicates (same from_id, to_id, kind) are ignored.
    pub fn insert_graph_edge(
        &self,
        from_id: &str,
        to_id: &str,
        kind: &str,
        properties: &str,
    ) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");

        // Check for duplicate edge
        let exists: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM graph_edges
                 WHERE from_id = ? AND to_id = ? AND kind = ?",
                params![from_id, to_id, kind],
                |row| row.get(0),
            )
            .unwrap_or(false);

        if exists {
            return Ok(());
        }

        let edge_id = uuid::Uuid::new_v4().to_string();
        conn.execute(
            "INSERT INTO graph_edges (id, from_id, to_id, kind, properties)
             VALUES (?, ?, ?, ?, ?)",
            params![edge_id, from_id, to_id, kind, properties],
        )
        .context("failed to insert graph edge")?;
        Ok(())
    }

    /// Get a graph node by ID.
    pub fn get_graph_node(&self, id: &str) -> Result<Option<GraphNodeRow>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let result = conn.query_row(
            "SELECT id, kind, name, properties FROM graph_nodes WHERE id = ?",
            params![id],
            |row| {
                Ok(GraphNodeRow {
                    id: row.get(0)?,
                    kind: row.get(1)?,
                    name: row.get(2)?,
                    properties: row.get::<_, String>(3).unwrap_or_else(|_| "{}".to_string()),
                })
            },
        );
        match result {
            Ok(node) => Ok(Some(node)),
            Err(duckdb::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(anyhow::Error::new(e).context("failed to get graph node")),
        }
    }

    /// List all graph nodes in stable ID order.
    pub fn list_graph_nodes(&self) -> Result<Vec<GraphNodeRow>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare("SELECT id, kind, name, properties FROM graph_nodes ORDER BY id")
            .context("failed to prepare graph node list")?;
        let rows = stmt
            .query_map([], |row| {
                Ok(GraphNodeRow {
                    id: row.get(0)?,
                    kind: row.get(1)?,
                    name: row.get(2)?,
                    properties: row.get::<_, String>(3).unwrap_or_else(|_| "{}".to_string()),
                })
            })
            .context("failed to query graph nodes")?;

        let mut nodes = Vec::new();
        for row in rows {
            nodes.push(row.context("failed to read graph node row")?);
        }
        Ok(nodes)
    }

    /// Delete a graph node and all its incident edges. Returns `true` if the
    /// node existed.
    pub fn delete_graph_node(&self, id: &str) -> Result<bool> {
        let conn = self.conn.lock().expect("lock poisoned");

        // Delete incident edges first (to honour foreign key semantics).
        conn.execute(
            "DELETE FROM graph_edges WHERE from_id = ? OR to_id = ?",
            params![id, id],
        )
        .context("failed to delete incident graph edges")?;

        let affected = conn
            .execute("DELETE FROM graph_nodes WHERE id = ?", params![id])
            .context("failed to delete graph node")?;

        Ok(affected > 0)
    }

    /// Query all neighbors of `id`, optionally filtered by edge kind.
    /// Uses standard SQL (no PGQ extension needed).
    pub fn query_graph_neighbors(
        &self,
        id: &str,
        edge_kind: Option<&str>,
    ) -> Result<Vec<GraphNeighborRow>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let (sql, use_kind) = if edge_kind.is_some() {
            (
                "SELECT n.id, n.kind, n.name, n.properties, e.kind AS edge_kind, 'outgoing' AS direction
                 FROM graph_edges e JOIN graph_nodes n ON e.to_id = n.id
                 WHERE e.from_id = ? AND e.kind = ?
                 UNION ALL
                 SELECT n.id, n.kind, n.name, n.properties, e.kind AS edge_kind, 'incoming' AS direction
                 FROM graph_edges e JOIN graph_nodes n ON e.from_id = n.id
                 WHERE e.to_id = ? AND e.kind = ?",
                true,
            )
        } else {
            (
                "SELECT n.id, n.kind, n.name, n.properties, e.kind AS edge_kind, 'outgoing' AS direction
                 FROM graph_edges e JOIN graph_nodes n ON e.to_id = n.id
                 WHERE e.from_id = ?
                 UNION ALL
                 SELECT n.id, n.kind, n.name, n.properties, e.kind AS edge_kind, 'incoming' AS direction
                 FROM graph_edges e JOIN graph_nodes n ON e.from_id = n.id
                 WHERE e.to_id = ?",
                false,
            )
        };

        let mut stmt = conn
            .prepare(sql)
            .context("failed to prepare neighbor query")?;

        let map_row = |row: &duckdb::Row| -> duckdb::Result<GraphNeighborRow> {
            Ok(GraphNeighborRow {
                node: GraphNodeRow {
                    id: row.get(0)?,
                    kind: row.get(1)?,
                    name: row.get(2)?,
                    properties: row.get::<_, String>(3).unwrap_or_else(|_| "{}".to_string()),
                },
                edge_kind: row.get(4)?,
                direction: row.get(5)?,
            })
        };

        let rows = if use_kind {
            let ek = edge_kind.unwrap();
            stmt.query_map(params![id, ek, id, ek], map_row)
                .context("failed to query graph neighbors")?
        } else {
            stmt.query_map(params![id, id], map_row)
                .context("failed to query graph neighbors")?
        };

        let mut neighbors = Vec::new();
        for row in rows {
            neighbors.push(row.context("failed to read graph neighbor row")?);
        }
        Ok(neighbors)
    }

    /// Count graph nodes.
    pub fn count_graph_nodes(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM graph_nodes", [], |row| row.get(0))
            .context("failed to count graph nodes")?;
        Ok(count as usize)
    }

    /// Count graph edges.
    pub fn count_graph_edges(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM graph_edges", [], |row| row.get(0))
            .context("failed to count graph edges")?;
        Ok(count as usize)
    }

    /// Delete all graph nodes and edges.
    pub fn clear_graph(&self) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        conn.execute_batch("DELETE FROM graph_edges; DELETE FROM graph_nodes;")
            .context("failed to clear graph")?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Graph algorithms (persisted SQL tables; DuckPGQ is optional)
    // -----------------------------------------------------------------------

    fn load_graph_edges(&self) -> Result<Vec<(String, String, String)>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare("SELECT from_id, to_id, kind FROM graph_edges")
            .context("failed to prepare graph edge load")?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .context("failed to load graph edges")?;

        let mut edges = Vec::new();
        for row in rows {
            edges.push(row.context("failed to read graph edge")?);
        }
        Ok(edges)
    }

    /// Find the shortest path between two nodes (up to `max_depth` hops).
    /// Returns the sequence of node IDs along the path, or an empty vector if
    /// no path exists. The implementation uses the persisted graph tables and
    /// does not require the optional DuckPGQ extension.
    pub fn pgq_shortest_path(
        &self,
        from_id: &str,
        to_id: &str,
        max_depth: usize,
    ) -> Result<Vec<String>> {
        if from_id == to_id {
            return Ok(vec![from_id.to_string()]);
        }

        let edges = self.load_graph_edges()?;
        let mut adjacency: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        for (from, to, _) in edges {
            adjacency.entry(from).or_default().push(to);
        }

        let mut parents: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        let mut depths: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        depths.insert(from_id.to_string(), 0);
        let mut frontier = vec![from_id.to_string()];

        while let Some(current) = frontier.first().cloned() {
            frontier.remove(0);
            let Some(next_depth) = depths
                .get(&current)
                .map(|depth| depth + 1)
                .filter(|depth| *depth <= max_depth)
            else {
                continue;
            };

            for neighbor in adjacency.get(&current).cloned().unwrap_or_default() {
                if parents.contains_key(&neighbor) || neighbor == from_id {
                    continue;
                }
                parents.insert(neighbor.clone(), current.clone());
                depths.insert(neighbor.clone(), next_depth);
                if neighbor == to_id {
                    let mut path = vec![to_id.to_string()];
                    while let Some(parent) = parents.get(path.last().expect("path exists")) {
                        path.push(parent.clone());
                        if parent == from_id {
                            path.reverse();
                            return Ok(path);
                        }
                    }
                }
                frontier.push(neighbor);
            }
        }

        Ok(Vec::new())
    }

    /// Run PageRank on the graph and return the top `limit` nodes with scores.
    /// This deterministic implementation works directly on the persisted graph
    /// tables and does not require the optional DuckPGQ extension.
    pub fn pgq_pagerank(&self, limit: usize) -> Result<Vec<(String, f64)>> {
        let edges = self.load_graph_edges()?;
        let mut nodes: Vec<String> = Vec::new();
        let mut node_set = std::collections::HashSet::new();
        let mut outgoing: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();

        for (from, to, _) in edges {
            if node_set.insert(from.clone()) {
                nodes.push(from.clone());
            }
            if node_set.insert(to.clone()) {
                nodes.push(to.clone());
            }
            outgoing.entry(from).or_default().push(to);
        }
        if nodes.is_empty() {
            return Ok(Vec::new());
        }

        let node_count = nodes.len() as f64;
        let damping = 0.85;
        let mut scores: std::collections::HashMap<String, f64> = nodes
            .iter()
            .map(|node| (node.clone(), 1.0 / node_count))
            .collect();
        let mut incoming: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        for (from, targets) in &outgoing {
            for target in targets {
                incoming
                    .entry(target.clone())
                    .or_default()
                    .push(from.clone());
            }
        }

        for _ in 0..20 {
            let mut next = std::collections::HashMap::new();
            let dangling_sum: f64 = nodes
                .iter()
                .filter(|node| outgoing.get(*node).is_none_or(Vec::is_empty))
                .map(|node| scores[node])
                .sum();

            for node in &nodes {
                let contribution: f64 = incoming
                    .get(node)
                    .into_iter()
                    .flatten()
                    .map(|source| scores[source] / outgoing[source].len() as f64)
                    .sum();
                let score = (1.0 - damping) / node_count
                    + damping * (contribution + dangling_sum / node_count);
                next.insert(node.clone(), score);
            }
            scores = next;
        }

        nodes.sort_by(|a, b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(b))
        });
        Ok(nodes
            .into_iter()
            .take(limit)
            .map(|node| {
                let score = scores[&node];
                (node, score)
            })
            .collect())
    }

    /// Pattern match: find all nodes connected to `node_id` via a specific
    /// edge kind in the given direction ("outgoing" or "incoming").
    /// Uses standard SQL so graph exploration remains available in all builds.
    pub fn pgq_pattern_match(
        &self,
        node_id: &str,
        edge_kind: &str,
        direction: &str,
    ) -> Result<Vec<GraphNodeRow>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let sql = if direction == "outgoing" {
            "SELECT n.id, n.kind, n.name, n.properties
             FROM graph_edges e
             JOIN graph_nodes n ON n.id = e.to_id
             WHERE e.from_id = ? AND e.kind = ?
             ORDER BY n.id"
        } else {
            "SELECT n.id, n.kind, n.name, n.properties
             FROM graph_edges e
             JOIN graph_nodes n ON n.id = e.from_id
             WHERE e.to_id = ? AND e.kind = ?
             ORDER BY n.id"
        };

        let mut stmt = conn
            .prepare(sql)
            .context("failed to prepare graph pattern match")?;

        let map_row = |row: &duckdb::Row| -> duckdb::Result<GraphNodeRow> {
            Ok(GraphNodeRow {
                id: row.get(0)?,
                kind: row.get(1)?,
                name: row.get(2)?,
                properties: row.get::<_, String>(3).unwrap_or_else(|_| "{}".to_string()),
            })
        };

        let rows = if direction == "outgoing" {
            stmt.query_map(params![node_id, edge_kind], map_row)
                .context("failed to execute graph pattern match")?
        } else {
            stmt.query_map(params![node_id, edge_kind], map_row)
                .context("failed to execute graph pattern match")?
        };

        let mut results = Vec::new();
        for row in rows {
            results.push(row.context("failed to read pattern match row")?);
        }
        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Code analysis (existing)
    // -----------------------------------------------------------------------

    /// Clear all symbols for a project. Returns number of symbols deleted.
    pub fn clear_symbols_for_project(&self, project_id: &str) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");

        // Delete dependencies first
        conn.execute(
            "DELETE FROM code_dependencies WHERE project_id = ?",
            params![project_id],
        )
        .context("failed to delete code_dependencies for project")?;

        // Delete symbols
        let affected = conn
            .execute(
                "DELETE FROM code_symbols WHERE project_id = ?",
                params![project_id],
            )
            .context("failed to delete code_symbols for project")?;

        Ok(affected)
    }

    /// Aggregate tool call statistics across all sessions.
    pub fn get_tool_call_stats(&self) -> Result<Vec<(String, i64, i64, f64)>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT COALESCE(tool_name, '(unknown)') AS tn,
                    COUNT(*) AS cnt,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) AS ok,
                    AVG(duration_ms) AS avg_dur
             FROM tool_calls
             GROUP BY tn
             ORDER BY cnt DESC",
            )
            .context("failed to prepare tool call stats")?;

        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, f64>(3).unwrap_or(0.0),
                ))
            })
            .context("failed to query tool call stats")?;

        let mut stats = Vec::new();
        for row in rows {
            stats.push(row.context("failed to read tool call stats row")?);
        }
        Ok(stats)
    }

    /// Get daily session counts grouped by agent, for the last N days.
    pub fn get_session_timeline(
        &self,
        days: i64,
        agent: Option<&str>,
    ) -> Result<Vec<(String, String, i64)>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let cutoff = Utc::now().naive_utc() - chrono::Duration::days(days);

        if let Some(ag) = agent {
            let mut stmt = conn
                .prepare(
                    "SELECT CAST(started_at AS DATE) AS day, agent, COUNT(*) AS cnt
                 FROM sessions
                 WHERE started_at >= ? AND agent = ?
                 GROUP BY day, agent
                 ORDER BY day",
                )
                .context("failed to prepare session timeline")?;
            let rows = stmt
                .query_map(params![cutoff, ag], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                })
                .context("failed to query session timeline")?;
            let mut result = Vec::new();
            for row in rows {
                result.push(row.context("failed to read timeline row")?);
            }
            Ok(result)
        } else {
            let mut stmt = conn
                .prepare(
                    "SELECT CAST(started_at AS DATE) AS day, agent, COUNT(*) AS cnt
                 FROM sessions
                 WHERE started_at >= ?
                 GROUP BY day, agent
                 ORDER BY day",
                )
                .context("failed to prepare session timeline")?;
            let rows = stmt
                .query_map(params![cutoff], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                })
                .context("failed to query session timeline")?;
            let mut result = Vec::new();
            for row in rows {
                result.push(row.context("failed to read timeline row")?);
            }
            Ok(result)
        }
    }

    /// Get all facts (active + invalidated), optionally filtered by project.
    pub fn get_all_facts(&self, project: Option<&str>, limit: usize) -> Result<Vec<Fact>> {
        self.get_all_facts_filtered(project, &[], limit)
    }

    /// Get all facts (active and superseded) with optional project and
    /// source-agent filters.
    pub fn get_all_facts_filtered(
        &self,
        project: Option<&str>,
        agents: &[String],
        limit: usize,
    ) -> Result<Vec<Fact>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let mut conditions: Vec<String> = Vec::new();
        let mut param_values: Vec<Box<dyn duckdb::ToSql>> = Vec::new();
        if let Some(project) = project {
            conditions.push("project_id ILIKE ?".to_string());
            param_values.push(Box::new(format!("%{project}%")));
        }
        if !agents.is_empty() {
            conditions.push(format!(
                "source_agent IN ({})",
                vec!["?"; agents.len()].join(", ")
            ));
            for agent in agents {
                param_values.push(Box::new(agent.clone()));
            }
        }
        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };
        let sql = format!(
            "SELECT id, project_id, subject, predicate, object, confidence,
                    source_session_id, source_agent, valid_at, invalid_at,
                    superseded_by, created_at
             FROM facts
             {where_clause}
             ORDER BY created_at DESC NULLS LAST
             LIMIT ?"
        );
        param_values.push(Box::new(limit as i64));

        let params_ref: Vec<&dyn duckdb::ToSql> =
            param_values.iter().map(|pv| pv.as_ref()).collect();
        let mut stmt = conn
            .prepare(&sql)
            .context("failed to prepare get_all_facts_filtered")?;

        let map_row = |row: &duckdb::Row| -> duckdb::Result<Fact> {
            Ok(Fact {
                id: row.get(0)?,
                project_id: row.get(1)?,
                subject: row.get(2)?,
                predicate: row.get(3)?,
                object: row.get(4)?,
                confidence: row.get::<_, f32>(5).unwrap_or(1.0),
                source_session_id: row.get(6)?,
                source_agent: row.get(7)?,
                valid_at: row.get(8)?,
                invalid_at: row.get(9)?,
                superseded_by: row.get(10)?,
                created_at: row.get(11)?,
            })
        };

        let rows = stmt
            .query_map(params_ref.as_slice(), map_row)
            .context("failed to query all facts")?;

        let mut facts = Vec::new();
        for row in rows {
            facts.push(row.context("failed to read fact row")?);
        }
        Ok(facts)
    }

    /// Get per-agent aggregated stats: sessions, total tokens, average duration.
    pub fn get_agent_stats(&self) -> Result<Vec<(String, i64, i64, f64)>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT agent,
                    COUNT(*) AS sessions,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens,
                    COALESCE(AVG(duration_minutes), 0) AS avg_duration
             FROM sessions
             GROUP BY agent
             ORDER BY sessions DESC",
            )
            .context("failed to prepare agent stats")?;

        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, f64>(3).unwrap_or(0.0),
                ))
            })
            .context("failed to query agent stats")?;

        let mut stats = Vec::new();
        for row in rows {
            stats.push(row.context("failed to read agent stats row")?);
        }
        Ok(stats)
    }

    // -----------------------------------------------------------------------
    // Briefing / daily-digest helpers
    // -----------------------------------------------------------------------

    /// Per-project breakdown for the current UTC day.
    pub fn get_project_breakdown_today(
        &self,
        since: NaiveDateTime,
    ) -> Result<Vec<serde_json::Value>> {
        let sessions = self.search_sessions(None, None, Some(since), 10_000)?;

        let mut by_project: std::collections::HashMap<
            String,
            (usize, i64, Vec<String>, std::collections::HashSet<String>),
        > = std::collections::HashMap::new();

        for s in &sessions {
            let project = s
                .project_id
                .clone()
                .unwrap_or_else(|| "unknown".to_string());
            let entry = by_project
                .entry(project)
                .or_insert_with(|| (0, 0, Vec::new(), std::collections::HashSet::new()));
            entry.0 += 1;
            entry.1 += s.total_tokens.unwrap_or(0) as i64;
            for f in &s.files_changed {
                if !entry.2.contains(f) {
                    entry.2.push(f.clone());
                }
            }
            entry.3.insert(s.agent.clone());
        }

        let mut result: Vec<serde_json::Value> = by_project
            .into_iter()
            .map(|(project, (sessions, tokens, files, agents))| {
                serde_json::json!({
                    "project": project,
                    "sessions": sessions,
                    "tokens": tokens,
                    "files_changed": files,
                    "agents": agents.into_iter().collect::<Vec<_>>(),
                })
            })
            .collect();
        result.sort_by(|a, b| {
            b["sessions"]
                .as_u64()
                .unwrap_or(0)
                .cmp(&a["sessions"].as_u64().unwrap_or(0))
        });
        Ok(result)
    }

    /// Decisions created during the current local day (window starts at `since`).
    pub fn get_decisions_today(&self, since: NaiveDateTime) -> Result<Vec<serde_json::Value>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let mut stmt = conn
            .prepare(
                "SELECT what, why, created_at
                 FROM decisions
                 WHERE created_at >= ?
                 ORDER BY created_at DESC",
            )
            .context("failed to prepare get_decisions_today")?;

        let rows = stmt
            .query_map(params![since], |row| {
                let what: String = row.get(0)?;
                let why: Option<String> = row.get(1)?;
                let created_at: Option<NaiveDateTime> = row.get(2)?;
                Ok(serde_json::json!({
                    "what": what,
                    "why": why,
                    "created_at": created_at.map(|t| t.to_string()),
                }))
            })
            .context("failed to query decisions today")?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row.context("failed to read decisions_today row")?);
        }
        Ok(result)
    }

    /// Facts created or becoming valid during the current local day (window
    /// starts at `since`).
    pub fn get_new_facts_today(&self, since: NaiveDateTime) -> Result<Vec<serde_json::Value>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let mut stmt = conn
            .prepare(
                "SELECT subject, predicate, object, confidence
                 FROM facts
                 WHERE created_at >= ? OR valid_at >= ?
                 ORDER BY created_at DESC NULLS LAST",
            )
            .context("failed to prepare get_new_facts_today")?;

        let rows = stmt
            .query_map(params![since, since], |row| {
                let subject: String = row.get(0)?;
                let predicate: String = row.get(1)?;
                let object: String = row.get(2)?;
                let confidence: f32 = row.get::<_, f32>(3).unwrap_or(0.0);
                Ok(serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "confidence": confidence,
                }))
            })
            .context("failed to query new facts today")?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row.context("failed to read new_facts_today row")?);
        }
        Ok(result)
    }

    /// Top files changed during the current local day (window starts at
    /// `since`), aggregated from session `files_changed` arrays.
    pub fn get_top_files_today(
        &self,
        since: NaiveDateTime,
        limit: usize,
    ) -> Result<Vec<(String, i64)>> {
        let sessions = self.search_sessions(None, None, Some(since), 10_000)?;

        let mut counts: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
        for s in &sessions {
            for f in &s.files_changed {
                *counts.entry(f.clone()).or_insert(0) += 1;
            }
        }

        let mut pairs: Vec<(String, i64)> = counts.into_iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        pairs.truncate(limit);
        Ok(pairs)
    }

    /// Session summaries from the current local day (window starts at `since`).
    pub fn get_session_summaries_today(
        &self,
        since: NaiveDateTime,
    ) -> Result<Vec<serde_json::Value>> {
        let sessions = self.search_sessions(None, None, Some(since), 10_000)?;

        let result: Vec<serde_json::Value> = sessions
            .into_iter()
            .filter(|s| s.summary.is_some())
            .map(|s| {
                serde_json::json!({
                    "agent": s.agent,
                    "project": s.project_id,
                    "summary": s.summary,
                    "started_at": s.started_at.map(|t| t.to_string()),
                })
            })
            .collect();
        Ok(result)
    }

    /// Return items that need human attention: stale memories, contradictory
    /// facts, high-churn files, and cross-agent conflicts.
    pub fn get_attention_items(&self) -> Result<Vec<serde_json::Value>> {
        let now = Utc::now().naive_utc();
        let stale_cutoff = now - chrono::Duration::days(30);
        let churn_cutoff = now - chrono::Duration::days(7);
        let conflict_cutoff = now - chrono::Duration::days(7);
        let conn = self.conn.lock().expect("lock poisoned");
        let mut items: Vec<serde_json::Value> = Vec::new();

        // 1. Stale memories — older than 30 days
        {
            let mut stmt = conn.prepare(
                "SELECT id, project_id, content, memory_type, created_at
                 FROM memories
                 WHERE created_at < ?
                 ORDER BY created_at ASC
                 LIMIT 10",
            )?;
            let rows = stmt.query_map(params![stale_cutoff], |row| {
                let id: String = row.get(0)?;
                let project_id: Option<String> = row.get(1)?;
                let content: String = row.get(2)?;
                let memory_type: Option<String> = row.get(3)?;
                let created_at: Option<NaiveDateTime> = row.get(4)?;
                Ok((id, project_id, content, memory_type, created_at))
            })?;
            for row in rows {
                let (id, project_id, content, memory_type, created_at) = row?;
                let snippet: String = content.chars().take(120).collect();
                let title = format!(
                    "Stale {} memory in {}",
                    memory_type.as_deref().unwrap_or("unknown"),
                    project_id.as_deref().unwrap_or("unknown project"),
                );
                let ts = created_at
                    .map(|t| t.and_utc().to_rfc3339())
                    .unwrap_or_default();
                items.push(serde_json::json!({
                    "type": "stale_memory",
                    "severity": "low",
                    "title": title,
                    "detail": format!("Created {ts}, not updated in 30+ days: {snippet}..."),
                    "entity_ids": [id],
                    "created_at": ts,
                }));
            }
        }

        // 2. Contradictory facts — same subject+predicate, different object, both active
        {
            let mut stmt = conn.prepare(
                "SELECT f1.id, f2.id, f1.subject, f1.predicate, f1.object, f2.object,
                        f1.source_agent, f2.source_agent, f1.created_at
                 FROM facts f1
                 JOIN facts f2
                   ON f1.subject = f2.subject
                  AND f1.predicate = f2.predicate
                  AND f1.id < f2.id
                  AND f1.object <> f2.object
                 WHERE f1.invalid_at IS NULL
                   AND f2.invalid_at IS NULL
                 LIMIT 10",
            )?;
            let rows = stmt.query_map([], |row| {
                let id1: String = row.get(0)?;
                let id2: String = row.get(1)?;
                let subject: String = row.get(2)?;
                let predicate: String = row.get(3)?;
                let obj1: String = row.get(4)?;
                let obj2: String = row.get(5)?;
                let agent1: Option<String> = row.get(6)?;
                let agent2: Option<String> = row.get(7)?;
                let created_at: Option<NaiveDateTime> = row.get(8)?;
                Ok((
                    id1, id2, subject, predicate, obj1, obj2, agent1, agent2, created_at,
                ))
            })?;
            for row in rows {
                let (id1, id2, subject, predicate, obj1, obj2, agent1, agent2, created_at) = row?;
                let a1 = agent1.as_deref().unwrap_or("unknown");
                let a2 = agent2.as_deref().unwrap_or("unknown");
                let ts = created_at
                    .map(|t| t.and_utc().to_rfc3339())
                    .unwrap_or_default();
                items.push(serde_json::json!({
                    "type": "contradictory_fact",
                    "severity": "high",
                    "title": format!("Contradictory facts for \"{subject}\""),
                    "detail": format!("{subject} {predicate} \"{obj1}\" ({a1}) vs \"{obj2}\" ({a2})"),
                    "entity_ids": [id1, id2],
                    "created_at": ts,
                }));
            }
        }

        // 3. High-churn files — >5 sessions in last 7 days
        {
            let result: Result<(), anyhow::Error> = (|| {
                let mut stmt = conn.prepare(
                    "SELECT f.file, COUNT(*) AS freq
                     FROM sessions s,
                          LATERAL (
                              SELECT UNNEST(
                              json_extract_string(s.files_changed, '$[*]')
                              ) AS file
                          ) f
                     WHERE s.started_at >= ?
                       AND s.files_changed IS NOT NULL
                       AND s.files_changed <> '[]'
                       AND s.files_changed <> ''
                     GROUP BY f.file
                     HAVING COUNT(*) > 5
                     ORDER BY freq DESC
                     LIMIT 10",
                )?;
                let rows = stmt.query_map(params![churn_cutoff], |row| {
                    let file_path: String = row.get(0)?;
                    let freq: i64 = row.get(1)?;
                    Ok((file_path, freq))
                })?;
                for row in rows {
                    let (file_path, freq) = row?;
                    items.push(serde_json::json!({
                        "type": "high_churn",
                        "severity": "medium",
                        "title": format!("High-churn file: {file_path}"),
                        "detail": format!("{file_path} changed in {freq} sessions in the last 7 days"),
                        "entity_ids": [file_path],
                        "created_at": Utc::now().to_rfc3339(),
                    }));
                }
                Ok(())
            })();
            if let Err(e) = result {
                warn!("high-churn query failed: {e}");
            }
        }

        // 4. Cross-agent conflicts — different agents, same project+type,
        //    different content, AND topically overlapping.
        //
        //    Content inequality alone flags every unrelated memory pair as a
        //    "conflict" (issue #26), so candidates must additionally share at
        //    least MIN_SHARED_TOKENS significant tokens. Survivors are ranked
        //    by overlap size, then recency, rather than arbitrary id order.
        const MIN_SHARED_TOKENS: usize = 2;
        const CANDIDATE_LIMIT: i64 = 500;
        {
            let result: Result<(), anyhow::Error> = (|| {
                let mut stmt = conn.prepare(&format!(
                    "SELECT m1.id, m2.id, m1.project_id, m1.memory_type,
                            s1.agent, s2.agent, m1.content, m2.content, m1.created_at
                     FROM memories m1
                     JOIN memories m2
                       ON m1.project_id = m2.project_id
                      AND m1.memory_type = m2.memory_type
                      AND m1.id < m2.id
                      AND m1.content <> m2.content
                     JOIN sessions s1 ON m1.source_session_id = s1.id
                     JOIN sessions s2 ON m2.source_session_id = s2.id
                     WHERE s1.agent <> s2.agent
                       AND m1.created_at >= ?
                       AND m2.created_at >= ?
                     LIMIT {CANDIDATE_LIMIT}",
                ))?;
                let rows = stmt.query_map(params![conflict_cutoff, conflict_cutoff], |row| {
                    let id1: String = row.get(0)?;
                    let id2: String = row.get(1)?;
                    let project_id: Option<String> = row.get(2)?;
                    let memory_type: Option<String> = row.get(3)?;
                    let agent1: String = row.get(4)?;
                    let agent2: String = row.get(5)?;
                    let content1: String = row.get(6)?;
                    let content2: String = row.get(7)?;
                    let created_at: Option<NaiveDateTime> = row.get(8)?;
                    Ok((
                        id1,
                        id2,
                        project_id,
                        memory_type,
                        agent1,
                        agent2,
                        content1,
                        content2,
                        created_at,
                    ))
                })?;
                struct ConflictCandidate {
                    overlap: usize,
                    id1: String,
                    id2: String,
                    project_id: Option<String>,
                    memory_type: Option<String>,
                    agent1: String,
                    agent2: String,
                    content1: String,
                    content2: String,
                    created_at: Option<NaiveDateTime>,
                }

                let mut candidates: Vec<ConflictCandidate> = Vec::new();
                for row in rows {
                    let (
                        id1,
                        id2,
                        project_id,
                        memory_type,
                        agent1,
                        agent2,
                        content1,
                        content2,
                        created_at,
                    ) = row?;
                    let tokens1 = significant_tokens(&content1);
                    let overlap = significant_tokens(&content2).intersection(&tokens1).count();
                    if overlap < MIN_SHARED_TOKENS {
                        continue;
                    }
                    candidates.push(ConflictCandidate {
                        overlap,
                        id1,
                        id2,
                        project_id,
                        memory_type,
                        agent1,
                        agent2,
                        content1,
                        content2,
                        created_at,
                    });
                }
                // Strongest topical overlap first; newest pairs break ties.
                candidates.sort_by(|a, b| {
                    b.overlap
                        .cmp(&a.overlap)
                        .then_with(|| b.created_at.cmp(&a.created_at))
                });
                for ConflictCandidate {
                    overlap,
                    id1,
                    id2,
                    project_id,
                    memory_type,
                    agent1,
                    agent2,
                    content1,
                    content2,
                    created_at,
                } in candidates.into_iter().take(10)
                {
                    let proj = project_id.as_deref().unwrap_or("unknown");
                    let mtype = memory_type.as_deref().unwrap_or("unknown");
                    let snip1: String = content1.chars().take(80).collect();
                    let snip2: String = content2.chars().take(80).collect();
                    let ts = created_at
                        .map(|t| t.and_utc().to_rfc3339())
                        .unwrap_or_default();
                    items.push(serde_json::json!({
                        "type": "conflict",
                        "severity": "high",
                        "title": format!("Cross-agent conflict in {proj} ({mtype})"),
                        "detail": format!("{agent1}: \"{snip1}...\" vs {agent2}: \"{snip2}...\""),
                        "entity_ids": [id1, id2],
                        "shared_terms": overlap,
                        "created_at": ts,
                    }));
                }
                Ok(())
            })();
            if let Err(e) = result {
                warn!("cross-agent conflict query failed: {e}");
            }
        }

        Ok(items)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_session(id: &str) -> Session {
        Session {
            id: id.to_string(),
            project_id: Some("proj-1".into()),
            agent: "claude".into(),
            started_at: Some(Utc::now().naive_utc()),
            ended_at: None,
            duration_minutes: Some(10),
            message_count: Some(5),
            tool_call_count: Some(3),
            total_tokens: Some(1200),
            files_changed: vec!["src/main.rs".into()],
            summary: Some("refactored module".into()),
        }
    }

    fn make_memory(id: &str, content: &str) -> Memory {
        Memory {
            id: id.to_string(),
            project_id: Some("proj-1".into()),
            content: content.into(),
            memory_type: Some("insight".into()),
            source_session_id: None,
            confidence: 0.9,
            access_count: 0,
            created_at: Some(Utc::now().naive_utc()),
            updated_at: Some(Utc::now().naive_utc()),
            valid_until: None,
        }
    }

    #[test]
    fn test_open_in_memory_and_schema() {
        let store = DuckStore::open_in_memory().expect("open in-memory");
        store.init_schema().expect("re-init schema");
    }

    #[test]
    fn test_insert_and_get_sessions() {
        let store = DuckStore::open_in_memory().unwrap();
        store.insert_session(&make_session("s-1")).unwrap();
        store.insert_session(&make_session("s-2")).unwrap();

        let recent = store.get_recent_sessions(10).unwrap();
        assert_eq!(recent.len(), 2);
        assert!(recent.iter().any(|s| s.id == "s-1"));
        assert!(recent.iter().any(|s| s.id == "s-2"));
        assert_eq!(
            store.get_session("s-1").unwrap().map(|s| s.id),
            Some("s-1".into())
        );
        assert!(store.get_session("missing").unwrap().is_none());
    }

    #[test]
    fn test_session_replacement_is_idempotent_for_derived_rows() {
        let store = DuckStore::open_in_memory().unwrap();
        let session = make_session("replace-session");
        store.insert_or_replace_session(&session).unwrap();

        store
            .insert_tool_call(&ToolCall {
                id: "old-tool".into(),
                session_id: Some("replace-session".into()),
                tool_name: Some("Read".into()),
                command: None,
                success: None,
                error_message: None,
                duration_ms: None,
                timestamp: None,
            })
            .unwrap();
        let mut derived = make_memory("old-memory", "old derived memory");
        derived.source_session_id = Some("replace-session".into());
        store.insert_memory(&derived).unwrap();
        store
            .set_memory_tags("old-memory", &["derived".into()])
            .unwrap();
        store
            .insert_graph_node(
                "session:replace-session",
                "Session",
                "replace-session",
                "{}",
            )
            .unwrap();
        store
            .insert_graph_node("memory:old-memory", "Memory", "old memory", "{}")
            .unwrap();
        store
            .insert_graph_edge(
                "memory:old-memory",
                "session:replace-session",
                "DERIVED_FROM",
                "{}",
            )
            .unwrap();

        let mut replacement = make_session("replace-session");
        replacement.summary = Some("updated summary".into());
        store.insert_or_replace_session(&replacement).unwrap();
        store
            .insert_tool_call(&ToolCall {
                id: "new-tool".into(),
                session_id: Some("replace-session".into()),
                tool_name: Some("Edit".into()),
                command: None,
                success: None,
                error_message: None,
                duration_ms: None,
                timestamp: None,
            })
            .unwrap();

        assert_eq!(store.count_sessions().unwrap(), 1);
        assert_eq!(store.count_tool_calls().unwrap(), 1);
        assert_eq!(store.count_memories().unwrap(), 0);
        assert!(
            store
                .get_graph_node("session:replace-session")
                .unwrap()
                .is_none()
        );
        assert!(store.get_graph_node("memory:old-memory").unwrap().is_none());
    }

    #[test]
    fn test_delete_session_cascades_all_derived_data() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_or_replace_session(&make_session("delete-session"))
            .unwrap();
        let mut memory = make_memory("delete-memory", "derived memory");
        memory.source_session_id = Some("delete-session".into());
        store.insert_memory(&memory).unwrap();
        store
            .insert_decision(&Decision {
                id: "delete-decision".into(),
                session_id: Some("delete-session".into()),
                project_id: Some("proj-1".into()),
                decision_type: None,
                what: "delete me".into(),
                why: None,
                alternatives: Vec::new(),
                outcome: None,
                created_at: None,
                valid_until: None,
            })
            .unwrap();
        let mut fact = make_fact("delete-fact", "session", "produced", "fact");
        fact.source_session_id = Some("delete-session".into());
        store.insert_fact(&fact).unwrap();
        store
            .insert_graph_node("session:delete-session", "Session", "session", "{}")
            .unwrap();

        assert!(store.delete_session("delete-session").unwrap());
        assert_eq!(store.count_sessions().unwrap(), 0);
        assert_eq!(store.count_memories().unwrap(), 0);
        assert_eq!(store.count_decisions().unwrap(), 0);
        assert_eq!(store.count_tool_calls().unwrap(), 0);
        assert_eq!(store.count_facts().unwrap(), 0);
        assert!(
            store
                .get_graph_node("session:delete-session")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn test_gc_removes_old_session_graph_and_orphaned_tool_calls() {
        let store = DuckStore::open_in_memory().unwrap();
        let now = Utc::now().naive_utc();
        let mut old = make_session("old-session");
        old.started_at = Some(now - chrono::Duration::days(40));
        let current = make_session("current-session");
        store.insert_session(&old).unwrap();
        store.insert_session(&current).unwrap();

        store
            .insert_graph_node("session:old-session", "Session", "old-session", "{}")
            .unwrap();
        store
            .insert_graph_node(
                "session:current-session",
                "Session",
                "current-session",
                "{}",
            )
            .unwrap();
        store
            .insert_graph_edge(
                "session:old-session",
                "session:current-session",
                "RELATES_TO",
                "{}",
            )
            .unwrap();

        store
            .insert_tool_call(&ToolCall {
                id: "orphan-tool".into(),
                session_id: Some("missing-session".into()),
                tool_name: Some("Read".into()),
                command: None,
                success: None,
                error_message: None,
                duration_ms: None,
                timestamp: None,
            })
            .unwrap();
        store
            .insert_tool_call(&ToolCall {
                id: "current-tool".into(),
                session_id: Some("current-session".into()),
                tool_name: Some("Edit".into()),
                command: None,
                success: None,
                error_message: None,
                duration_ms: None,
                timestamp: None,
            })
            .unwrap();

        assert_eq!(
            store
                .gc_sessions_before(now - chrono::Duration::days(30))
                .unwrap(),
            1
        );
        assert_eq!(store.delete_orphaned_tool_calls().unwrap(), 1);
        assert!(
            store
                .get_graph_node("session:old-session")
                .unwrap()
                .is_none()
        );
        assert!(
            store
                .get_graph_node("session:current-session")
                .unwrap()
                .is_some()
        );
        assert_eq!(store.count_graph_edges().unwrap(), 0);
        assert_eq!(store.count_tool_calls().unwrap(), 1);
    }

    #[test]
    fn test_insert_decision() {
        let store = DuckStore::open_in_memory().unwrap();
        let d = Decision {
            id: "d-1".into(),
            session_id: Some("s-1".into()),
            project_id: Some("proj-1".into()),
            decision_type: Some("architecture".into()),
            what: "use DuckDB for structured store".into(),
            why: Some("embedded, fast analytics".into()),
            alternatives: vec!["SQLite".into(), "Postgres".into()],
            outcome: None,
            created_at: None,
            valid_until: None,
        };
        store.insert_decision(&d).unwrap();
    }

    #[test]
    fn test_insert_and_search_memories() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_memory(&make_memory(
                "m-1",
                "The DuckDB store handles structured data",
            ))
            .unwrap();
        store
            .insert_memory(&make_memory("m-2", "LanceDB handles vector embeddings"))
            .unwrap();

        let results = store.search_memories("duckdb").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "m-1");

        let all = store.search_memories("handles").unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_get_recent_sessions_limit() {
        let store = DuckStore::open_in_memory().unwrap();
        for i in 0..5 {
            store
                .insert_session(&make_session(&format!("s-{i}")))
                .unwrap();
        }
        let limited = store.get_recent_sessions(3).unwrap();
        assert_eq!(limited.len(), 3);
    }

    #[test]
    fn test_daily_and_today_analytics_use_provided_day_boundaries() {
        let store = DuckStore::open_in_memory().unwrap();
        let now = Utc::now().naive_utc();
        let today_start = now.date().and_hms_opt(0, 0, 0).unwrap();

        let mut current = make_session("analytics-today");
        current.started_at = Some(now);
        current.files_changed = vec!["src/today.rs".into()];

        let mut previous = make_session("analytics-previous");
        previous.started_at = Some(today_start - chrono::Duration::hours(1));
        previous.files_changed = vec!["src/yesterday.rs".into()];

        store.insert_session(&current).unwrap();
        store.insert_session(&previous).unwrap();
        store
            .insert_memory(&make_memory("analytics-memory", "today memory"))
            .unwrap();
        store
            .insert_decision(&Decision {
                id: "analytics-decision".into(),
                session_id: Some("analytics-today".into()),
                project_id: Some("proj-1".into()),
                decision_type: Some("architecture".into()),
                what: "use UTC boundaries".into(),
                why: None,
                alternatives: Vec::new(),
                outcome: None,
                created_at: Some(now),
                valid_until: None,
            })
            .unwrap();

        assert_eq!(store.count_sessions_since(today_start).unwrap(), 1);
        let todays_sessions = store.get_todays_sessions(today_start).unwrap();
        assert_eq!(todays_sessions.len(), 1);
        assert_eq!(todays_sessions[0].id, "analytics-today");
        let now_dt = now.and_utc();
        assert_eq!(
            store
                .get_daily_session_counts(0, now_dt, &Utc)
                .unwrap()
                .last()
                .copied()
                .unwrap(),
            1
        );
        assert_eq!(
            store
                .get_daily_memory_counts(0, now_dt, &Utc)
                .unwrap()
                .last()
                .copied()
                .unwrap(),
            1
        );
        assert_eq!(
            store
                .get_daily_decision_counts(0, now_dt, &Utc)
                .unwrap()
                .last()
                .copied()
                .unwrap(),
            1
        );

        let daily = store.get_daily_counts(0, now_dt, &Utc).unwrap();
        assert_eq!(daily.len(), 1);
        assert_eq!(daily[0].0, now.date().to_string());
        assert_eq!(daily[0].1, 1);

        let projects = store.get_project_breakdown_today(today_start).unwrap();
        assert_eq!(projects.len(), 1);
        assert_eq!(projects[0]["sessions"].as_u64(), Some(1));
        assert_eq!(
            projects[0]["files_changed"][0].as_str(),
            Some("src/today.rs")
        );

        let top_files = store.get_top_files_today(today_start, 10).unwrap();
        assert_eq!(top_files, vec![("src/today.rs".into(), 1)]);

        let decisions = store.get_decisions_today(today_start).unwrap();
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0]["what"].as_str(), Some("use UTC boundaries"));

        let summaries = store.get_session_summaries_today(today_start).unwrap();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0]["summary"].as_str(), Some("refactored module"));
    }

    #[test]
    fn test_manual_notes_support_tags() {
        let store = DuckStore::open_in_memory().unwrap();
        let id = store
            .insert_note_with_tags(
                "Prefer small composable parsers",
                Some("proj-1"),
                &[" rust ".into(), "RUST".into(), "architecture".into()],
            )
            .unwrap();

        assert_eq!(
            store.get_memory_tags(&id).unwrap(),
            vec!["architecture".to_string(), "rust".to_string()]
        );

        let tagged = store.search_memories_by_tag("RUST", 10).unwrap();
        assert_eq!(tagged.len(), 1);
        assert_eq!(tagged[0].id, id);
        assert_eq!(tagged[0].memory_type.as_deref(), Some("note"));
        assert_eq!(tagged[0].content, "Prefer small composable parsers");
        assert_eq!(tagged[0].project_id.as_deref(), Some("proj-1"));

        let empty = store
            .insert_note_with_tags("   ", Some("  "), &[])
            .unwrap_err();
        assert!(empty.to_string().contains("cannot be empty"));
        assert_eq!(store.get_project_ids().unwrap(), vec!["proj-1".to_string()]);
        store
            .insert_graph_node(&format!("memory:{id}"), "Memory", "note", "{}")
            .unwrap();
        store
            .insert_graph_node("project:proj-1", "Project", "proj-1", "{}")
            .unwrap();
        store
            .insert_graph_edge(&format!("memory:{id}"), "project:proj-1", "ABOUT", "{}")
            .unwrap();

        assert!(store.delete_memory(&id).unwrap());
        assert!(store.get_memory_tags(&id).unwrap().is_empty());
        assert!(store.search_memories_by_tag("rust", 10).unwrap().is_empty());
        assert!(
            store
                .get_graph_node(&format!("memory:{id}"))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn test_attention_items_include_conflicts_and_high_churn() {
        let store = DuckStore::open_in_memory().unwrap();

        for i in 0..6 {
            let mut session = make_session(&format!("churn-{i}"));
            session.files_changed = vec!["src/hot.rs".into()];
            store.insert_session(&session).unwrap();
        }

        store
            .insert_fact(&make_fact("attention-jwt", "auth", "uses", "JWT"))
            .unwrap();
        store
            .insert_fact(&make_fact("attention-oauth", "auth", "uses", "OAuth2"))
            .unwrap();

        let items = store.get_attention_items().unwrap();
        let types: Vec<_> = items
            .iter()
            .filter_map(|item| item["type"].as_str())
            .collect();

        assert!(types.contains(&"high_churn"), "items: {items:?}");
        assert!(types.contains(&"contradictory_fact"), "items: {items:?}");
        let churn = items
            .iter()
            .find(|item| item["type"] == "high_churn")
            .unwrap();
        assert_eq!(churn["entity_ids"][0].as_str(), Some("src/hot.rs"));
    }

    /// Build a session for a specific agent with a fixed start time.
    fn make_session_for(id: &str, agent: &str, started_at: NaiveDateTime) -> Session {
        Session {
            id: id.to_string(),
            project_id: Some("proj-1".into()),
            agent: agent.into(),
            started_at: Some(started_at),
            ended_at: None,
            duration_minutes: Some(10),
            message_count: Some(5),
            tool_call_count: Some(3),
            total_tokens: Some(1200),
            files_changed: vec![],
            summary: None,
        }
    }

    /// Build a memory attached to a source session.
    fn make_memory_from(id: &str, content: &str, session_id: &str) -> Memory {
        Memory {
            id: id.to_string(),
            project_id: Some("proj-1".into()),
            content: content.into(),
            memory_type: Some("insight".into()),
            source_session_id: Some(session_id.into()),
            confidence: 0.9,
            access_count: 0,
            created_at: Some(Utc::now().naive_utc()),
            updated_at: Some(Utc::now().naive_utc()),
            valid_until: None,
        }
    }

    #[test]
    fn test_significant_tokens_filters_stop_words_and_short_tokens() {
        let tokens = significant_tokens("The JWT auth middleware uses a for x!");
        assert!(tokens.contains("jwt"));
        assert!(tokens.contains("auth"));
        assert!(tokens.contains("middleware"));
        assert!(!tokens.contains("the"), "stop words must be dropped");
        assert!(!tokens.contains("a"), "short tokens must be dropped");
        assert!(!tokens.contains("x"), "short tokens must be dropped");
    }

    #[test]
    fn test_attention_conflict_requires_topical_overlap() {
        let store = DuckStore::open_in_memory().unwrap();
        let now = Utc::now().naive_utc();
        store
            .insert_session(&make_session_for("s-claude", "claude_code", now))
            .unwrap();
        store
            .insert_session(&make_session_for("s-codex", "codex", now))
            .unwrap();

        // Unrelated pair: different agents, same project/type, different
        // content — but no topical overlap. Must NOT be flagged (issue #26).
        store
            .insert_memory(&make_memory_from(
                "m-unrelated-1",
                "Prefer tower middleware for routing layers",
                "s-claude",
            ))
            .unwrap();
        store
            .insert_memory(&make_memory_from(
                "m-unrelated-2",
                "Database migrations run during deploy",
                "s-codex",
            ))
            .unwrap();

        // Topically overlapping pair: both discuss JWT auth tokens.
        store
            .insert_memory(&make_memory_from(
                "m-jwt-1",
                "JWT auth tokens expire after one hour",
                "s-claude",
            ))
            .unwrap();
        store
            .insert_memory(&make_memory_from(
                "m-jwt-2",
                "JWT auth tokens should never expire",
                "s-codex",
            ))
            .unwrap();

        let items = store.get_attention_items().unwrap();
        let conflicts: Vec<_> = items
            .iter()
            .filter(|item| item["type"] == "conflict")
            .collect();

        assert_eq!(
            conflicts.len(),
            1,
            "only the topically overlapping pair may be flagged: {conflicts:?}"
        );
        let ids: Vec<&str> = conflicts[0]["entity_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(ids.contains(&"m-jwt-1") && ids.contains(&"m-jwt-2"));
        assert!(
            conflicts[0]["shared_terms"].as_u64().unwrap_or(0) >= 2,
            "shared_terms should be reported"
        );
    }

    #[test]
    fn test_filtered_listings_by_agent() {
        let store = DuckStore::open_in_memory().unwrap();
        let now = Utc::now().naive_utc();
        store
            .insert_session(&make_session_for("s-claude", "claude_code", now))
            .unwrap();
        store
            .insert_session(&make_session_for("s-codex", "codex", now))
            .unwrap();

        store
            .insert_memory(&make_memory_from("m-c", "claude memory", "s-claude"))
            .unwrap();
        store
            .insert_memory(&make_memory_from("m-x", "codex memory", "s-codex"))
            .unwrap();
        // Manual note without a source session.
        store
            .insert_memory(&make_memory("m-note", "manual note"))
            .unwrap();

        // Memories: agent filter narrows to the agent's own rows and drops
        // manual notes; empty filter returns everything.
        let codex_only = store
            .get_memories_filtered(None, &["codex".to_string()], 100)
            .unwrap();
        assert_eq!(codex_only.len(), 1);
        assert_eq!(codex_only[0].id, "m-x");
        let all = store.get_memories_filtered(None, &[], 100).unwrap();
        assert_eq!(all.len(), 3);
        let both = store
            .get_memories_filtered(None, &["codex".to_string(), "claude_code".to_string()], 100)
            .unwrap();
        assert_eq!(both.len(), 2);

        // Facts: filtered directly on source_agent.
        let mut fact_c = make_fact("f-c", "build", "uses", "cargo");
        fact_c.source_agent = Some("claude_code".into());
        let mut fact_x = make_fact("f-x", "build", "uses", "bazel");
        fact_x.source_agent = Some("codex".into());
        store.insert_fact(&fact_c).unwrap();
        store.insert_fact(&fact_x).unwrap();
        let facts_x = store
            .get_active_facts_filtered(None, &["codex".to_string()], 100)
            .unwrap();
        assert_eq!(facts_x.len(), 1);
        assert_eq!(facts_x[0].id, "f-x");
        let all_facts = store
            .get_all_facts_filtered(None, &["claude_code".to_string()], 100)
            .unwrap();
        assert_eq!(all_facts.len(), 1);
        assert_eq!(all_facts[0].id, "f-c");

        // Decisions: filtered through the owning session's agent.
        store
            .insert_decision(&Decision {
                id: "d-c".into(),
                session_id: Some("s-claude".into()),
                project_id: Some("proj-1".into()),
                decision_type: None,
                what: "use tower".into(),
                why: None,
                alternatives: vec![],
                outcome: None,
                created_at: Some(now),
                valid_until: None,
            })
            .unwrap();
        store
            .insert_decision(&Decision {
                id: "d-x".into(),
                session_id: Some("s-codex".into()),
                project_id: Some("proj-1".into()),
                decision_type: None,
                what: "use axum".into(),
                why: None,
                alternatives: vec![],
                outcome: None,
                created_at: Some(now),
                valid_until: None,
            })
            .unwrap();
        let decisions_x = store
            .get_decisions_filtered(None, &["codex".to_string()], 100)
            .unwrap();
        assert_eq!(decisions_x.len(), 1);
        assert_eq!(decisions_x[0].id, "d-x");
        let decisions_all = store.get_decisions_filtered(None, &[], 100).unwrap();
        assert_eq!(decisions_all.len(), 2);
    }

    #[test]
    fn test_daily_counts_bucket_by_local_day() {
        use chrono::FixedOffset;

        let store = DuckStore::open_in_memory().unwrap();
        // 2026-08-23 02:00 UTC == 2026-08-22 22:00 in UTC-4.
        let ts = chrono::NaiveDate::from_ymd_opt(2026, 8, 23)
            .unwrap()
            .and_hms_opt(2, 0, 0)
            .unwrap();
        store
            .insert_session(&make_session_for("s-late", "codex", ts))
            .unwrap();

        // "Now" is 2026-08-23 03:00 UTC (still the evening of Aug 22 in UTC-4).
        let now = chrono::NaiveDate::from_ymd_opt(2026, 8, 23)
            .unwrap()
            .and_hms_opt(3, 0, 0)
            .unwrap()
            .and_utc();
        let tz_west = FixedOffset::west_opt(4 * 3600).unwrap();

        let rows = store.get_daily_counts(1, now, &tz_west).unwrap();
        assert_eq!(rows.len(), 2);
        // Local days: Aug 21 (empty) and Aug 22 (the session).
        assert_eq!(rows[0].0, "2026-08-21");
        assert_eq!(rows[0].1, 0);
        assert_eq!(rows[1].0, "2026-08-22");
        assert_eq!(rows[1].1, 1, "session must bucket into the LOCAL day");

        // Same data viewed in UTC buckets lands on Aug 23.
        let rows_utc = store.get_daily_counts(1, now, &Utc).unwrap();
        assert_eq!(rows_utc.last().unwrap().0, "2026-08-23");
        assert_eq!(rows_utc.last().unwrap().1, 1);

        // Wrapper helpers agree with the raw rows.
        let sess = store.get_daily_session_counts(1, now, &tz_west).unwrap();
        assert_eq!(sess, vec![0, 1]);
        let mem = store.get_daily_memory_counts(1, now, &tz_west).unwrap();
        assert_eq!(mem, vec![0, 0]);
        let dec = store.get_daily_decision_counts(1, now, &tz_west).unwrap();
        assert_eq!(dec, vec![0, 0]);
    }

    #[test]
    fn test_today_queries_respect_provided_window() {
        let store = DuckStore::open_in_memory().unwrap();
        let now = Utc::now().naive_utc();
        let today = now - chrono::Duration::hours(1);
        let two_days_ago = now - chrono::Duration::days(2);

        store
            .insert_session(&make_session_for("s-today", "codex", today))
            .unwrap();
        store
            .insert_session(&make_session_for("s-old", "codex", two_days_ago))
            .unwrap();
        store
            .insert_decision(&Decision {
                id: "d-today".into(),
                session_id: Some("s-today".into()),
                project_id: Some("proj-1".into()),
                decision_type: None,
                what: "ship it".into(),
                why: None,
                alternatives: vec![],
                outcome: None,
                created_at: Some(today),
                valid_until: None,
            })
            .unwrap();

        // A window starting "now minus 1 day" includes today's rows only.
        let since = now - chrono::Duration::days(1);
        let decisions = store.get_decisions_today(since).unwrap();
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0]["what"], "ship it");
        let summaries = store.get_session_summaries_today(since).unwrap();
        // s-today has no summary; ensure no panic and correct filtering.
        assert!(summaries.is_empty());
        let top_files = store.get_top_files_today(since, 10).unwrap();
        assert!(top_files.is_empty());
        let breakdown = store.get_project_breakdown_today(since).unwrap();
        assert_eq!(breakdown.len(), 1);
        assert_eq!(breakdown[0]["sessions"], 1);

        // Briefing window reaches back over both sessions.
        let recent = store
            .get_recent_sessions_for_briefing(now - chrono::Duration::days(3))
            .unwrap();
        assert_eq!(recent.len(), 2);
    }

    // -------------------------------------------------------------------
    // Graph CRUD tests (standard SQL, no PGQ extension required)
    // -------------------------------------------------------------------

    #[test]
    fn test_graph_insert_and_get_node() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_graph_node("n1", "CodeEntity", "authenticate", r#"{"lang":"rust"}"#)
            .unwrap();

        let node = store
            .get_graph_node("n1")
            .unwrap()
            .expect("node should exist");
        assert_eq!(node.id, "n1");
        assert_eq!(node.kind, "CodeEntity");
        assert_eq!(node.name, "authenticate");
        assert_eq!(node.properties, r#"{"lang":"rust"}"#);
    }

    #[test]
    fn test_graph_get_missing_node() {
        let store = DuckStore::open_in_memory().unwrap();
        assert!(store.get_graph_node("nonexistent").unwrap().is_none());
    }

    #[test]
    fn test_graph_upsert_node() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_graph_node("n1", "Concept", "old_name", "{}")
            .unwrap();
        store
            .insert_graph_node("n1", "Concept", "new_name", "{}")
            .unwrap();

        let node = store.get_graph_node("n1").unwrap().unwrap();
        assert_eq!(node.name, "new_name");
        assert_eq!(store.count_graph_nodes().unwrap(), 1);
    }

    #[test]
    fn test_graph_insert_edge_and_neighbors() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_graph_node("a", "CodeEntity", "foo", "{}")
            .unwrap();
        store
            .insert_graph_node("b", "CodeEntity", "bar", "{}")
            .unwrap();
        store.insert_graph_edge("a", "b", "CALLS", "{}").unwrap();

        // Outgoing from a
        let neighbors = store.query_graph_neighbors("a", None).unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].node.id, "b");
        assert_eq!(neighbors[0].edge_kind, "CALLS");
        assert_eq!(neighbors[0].direction, "outgoing");

        // Incoming to b (query from b's perspective)
        let neighbors_b = store.query_graph_neighbors("b", None).unwrap();
        assert_eq!(neighbors_b.len(), 1);
        assert_eq!(neighbors_b[0].node.id, "a");
        assert_eq!(neighbors_b[0].direction, "incoming");
    }

    #[test]
    fn test_graph_neighbor_edge_kind_filter() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_graph_node("a", "CodeEntity", "foo", "{}")
            .unwrap();
        store
            .insert_graph_node("b", "CodeEntity", "bar", "{}")
            .unwrap();
        store.insert_graph_node("c", "Module", "baz", "{}").unwrap();

        store.insert_graph_edge("a", "b", "CALLS", "{}").unwrap();
        store.insert_graph_edge("a", "c", "IMPORTS", "{}").unwrap();

        let calls_only = store.query_graph_neighbors("a", Some("CALLS")).unwrap();
        assert_eq!(calls_only.len(), 1);
        assert_eq!(calls_only[0].node.id, "b");

        let imports_only = store.query_graph_neighbors("a", Some("IMPORTS")).unwrap();
        assert_eq!(imports_only.len(), 1);
        assert_eq!(imports_only[0].node.id, "c");
    }

    #[test]
    fn test_graph_duplicate_edge_ignored() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_graph_node("a", "CodeEntity", "foo", "{}")
            .unwrap();
        store
            .insert_graph_node("b", "CodeEntity", "bar", "{}")
            .unwrap();

        store.insert_graph_edge("a", "b", "CALLS", "{}").unwrap();
        store.insert_graph_edge("a", "b", "CALLS", "{}").unwrap(); // duplicate

        assert_eq!(store.count_graph_edges().unwrap(), 1);
    }

    #[test]
    fn test_graph_delete_node_cascades_edges() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_graph_node("a", "CodeEntity", "foo", "{}")
            .unwrap();
        store
            .insert_graph_node("b", "CodeEntity", "bar", "{}")
            .unwrap();
        store
            .insert_graph_node("c", "CodeEntity", "baz", "{}")
            .unwrap();

        store.insert_graph_edge("a", "b", "CALLS", "{}").unwrap();
        store.insert_graph_edge("c", "a", "IMPORTS", "{}").unwrap();

        assert_eq!(store.count_graph_edges().unwrap(), 2);

        // Delete node a — both incident edges should be removed
        assert!(store.delete_graph_node("a").unwrap());
        assert!(store.get_graph_node("a").unwrap().is_none());
        assert_eq!(store.count_graph_edges().unwrap(), 0);

        // Deleting again returns false
        assert!(!store.delete_graph_node("a").unwrap());
    }

    #[test]
    fn test_graph_counts() {
        let store = DuckStore::open_in_memory().unwrap();
        assert_eq!(store.count_graph_nodes().unwrap(), 0);
        assert_eq!(store.count_graph_edges().unwrap(), 0);

        store
            .insert_graph_node("a", "Concept", "auth", "{}")
            .unwrap();
        store.insert_graph_node("b", "Concept", "db", "{}").unwrap();
        store
            .insert_graph_edge("a", "b", "RELATES_TO", "{}")
            .unwrap();

        assert_eq!(store.count_graph_nodes().unwrap(), 2);
        assert_eq!(store.count_graph_edges().unwrap(), 1);
    }

    #[test]
    fn test_graph_clear() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_graph_node("a", "Concept", "auth", "{}")
            .unwrap();
        store.insert_graph_node("b", "Concept", "db", "{}").unwrap();
        store
            .insert_graph_edge("a", "b", "RELATES_TO", "{}")
            .unwrap();

        store.clear_graph().unwrap();

        assert_eq!(store.count_graph_nodes().unwrap(), 0);
        assert_eq!(store.count_graph_edges().unwrap(), 0);
    }

    #[test]
    fn test_graph_algorithms_work_without_optional_duckpgq_extension() {
        let store = DuckStore::open_in_memory().unwrap();
        store.insert_graph_node("a", "Concept", "a", "{}").unwrap();
        store.insert_graph_node("b", "Concept", "b", "{}").unwrap();
        store.insert_graph_node("c", "Concept", "c", "{}").unwrap();
        store
            .insert_graph_edge("a", "b", "RELATES_TO", "{}")
            .unwrap();
        store
            .insert_graph_edge("b", "c", "RELATES_TO", "{}")
            .unwrap();

        assert_eq!(
            store.pgq_shortest_path("a", "c", 3).unwrap(),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
        assert!(store.pgq_shortest_path("c", "a", 3).unwrap().is_empty());
        assert_eq!(store.pgq_shortest_path("a", "a", 0).unwrap(), vec!["a"]);

        let pagerank = store.pgq_pagerank(3).unwrap();
        assert_eq!(pagerank.len(), 3);
        assert!(pagerank[0].0 == *"c" || pagerank[0].0 == *"b");
        assert!(pagerank.iter().all(|(_, score)| score.is_finite()));

        let outgoing = store
            .pgq_pattern_match("a", "RELATES_TO", "outgoing")
            .unwrap();
        assert_eq!(
            outgoing
                .iter()
                .map(|node| node.id.as_str())
                .collect::<Vec<_>>(),
            ["b"]
        );
        let incoming = store
            .pgq_pattern_match("c", "RELATES_TO", "incoming")
            .unwrap();
        assert_eq!(
            incoming
                .iter()
                .map(|node| node.id.as_str())
                .collect::<Vec<_>>(),
            ["b"]
        );
    }

    // -------------------------------------------------------------------
    // Facts (temporal knowledge graph) tests
    // -------------------------------------------------------------------

    fn make_fact(id: &str, subject: &str, predicate: &str, object: &str) -> Fact {
        Fact {
            id: id.to_string(),
            project_id: Some("proj-1".into()),
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            confidence: 0.9,
            source_session_id: Some("s-1".into()),
            source_agent: Some("claude".into()),
            valid_at: Some(Utc::now().naive_utc()),
            invalid_at: None,
            superseded_by: None,
            created_at: Some(Utc::now().naive_utc()),
        }
    }

    #[test]
    fn test_insert_and_get_facts() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_fact(&make_fact("f-1", "auth", "uses", "JWT"))
            .unwrap();
        store
            .insert_fact(&make_fact("f-2", "db", "uses", "DuckDB"))
            .unwrap();

        let facts = store.get_active_facts(None, 100).unwrap();
        assert_eq!(facts.len(), 2);
        assert_eq!(store.count_facts().unwrap(), 2);
        assert_eq!(store.count_active_facts().unwrap(), 2);
    }

    #[test]
    fn test_invalidate_fact() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_fact(&make_fact("f-1", "auth", "uses", "JWT"))
            .unwrap();

        assert!(store.invalidate_fact("f-1", Some("f-2")).unwrap());
        assert_eq!(store.count_active_facts().unwrap(), 0);
        assert_eq!(store.count_facts().unwrap(), 1); // still exists, just invalidated

        // Can't invalidate twice
        assert!(!store.invalidate_fact("f-1", None).unwrap());
    }

    #[test]
    fn test_upsert_fact_supersedes_contradicting() {
        let store = DuckStore::open_in_memory().unwrap();
        // Insert original fact: auth uses JWT
        store
            .insert_fact(&make_fact("f-1", "auth", "uses", "JWT"))
            .unwrap();

        // Upsert contradicting fact: auth uses OAuth2
        store
            .upsert_fact(&make_fact("f-2", "auth", "uses", "OAuth2"))
            .unwrap();

        // Old fact should be invalidated, new fact should be active
        let active = store.get_active_facts(None, 100).unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].object, "OAuth2");

        // Total facts should be 2 (history preserved)
        assert_eq!(store.count_facts().unwrap(), 2);
    }

    #[test]
    fn test_upsert_fact_skips_duplicate() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_fact(&make_fact("f-1", "auth", "uses", "JWT"))
            .unwrap();

        // Upsert same fact (same subject+predicate+object)
        store
            .upsert_fact(&make_fact("f-2", "auth", "uses", "JWT"))
            .unwrap();

        // Should still be 1 fact (duplicate skipped)
        assert_eq!(store.count_facts().unwrap(), 1);
    }

    #[test]
    fn test_search_facts() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_fact(&make_fact("f-1", "auth module", "uses", "JWT tokens"))
            .unwrap();
        store
            .insert_fact(&make_fact("f-2", "db layer", "uses", "DuckDB"))
            .unwrap();

        let results = store.search_facts("auth").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].subject, "auth module");

        let results = store.search_facts("DuckDB").unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_update_and_delete_memory() {
        let store = DuckStore::open_in_memory().unwrap();
        let mem = Memory {
            id: "m-upd".into(),
            project_id: None,
            content: "original content".into(),
            memory_type: Some("note".into()),
            source_session_id: None,
            confidence: 0.8,
            access_count: 0,
            created_at: Some(Utc::now().naive_utc()),
            updated_at: Some(Utc::now().naive_utc()),
            valid_until: None,
        };
        store.insert_memory(&mem).unwrap();

        // Update content
        assert!(
            store
                .update_memory("m-upd", Some("revised content"), None)
                .unwrap()
        );
        let fetched = store.get_memory("m-upd").unwrap().unwrap();
        assert_eq!(fetched.content, "revised content");
        assert_eq!(fetched.confidence, 0.8); // unchanged

        // Update confidence
        assert!(store.update_memory("m-upd", None, Some(0.95)).unwrap());
        let fetched = store.get_memory("m-upd").unwrap().unwrap();
        assert_eq!(fetched.confidence, 0.95);

        // Update nonexistent
        assert!(!store.update_memory("nope", Some("x"), None).unwrap());

        // Delete
        assert!(store.delete_memory("m-upd").unwrap());
        assert!(store.get_memory("m-upd").unwrap().is_none());
        assert!(!store.delete_memory("m-upd").unwrap()); // already gone
    }

    #[test]
    fn test_get_and_delete_fact() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_fact(&make_fact("f-del", "auth", "uses", "JWT"))
            .unwrap();

        let fact = store.get_fact("f-del").unwrap().unwrap();
        assert_eq!(fact.subject, "auth");

        assert!(store.delete_fact("f-del").unwrap());
        assert!(store.get_fact("f-del").unwrap().is_none());
        assert!(!store.delete_fact("f-del").unwrap());
    }

    #[test]
    fn test_fact_history() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_fact(&make_fact("f-1", "auth", "uses", "JWT"))
            .unwrap();
        store.invalidate_fact("f-1", Some("f-2")).unwrap();
        store
            .insert_fact(&make_fact("f-2", "auth", "uses", "OAuth2"))
            .unwrap();

        let history = store.get_fact_history("auth").unwrap();
        assert_eq!(history.len(), 2);
        // First should be the older one (sorted by valid_at ASC)
        assert_eq!(history[0].object, "JWT");
        assert!(history[0].invalid_at.is_some());
        assert_eq!(history[1].object, "OAuth2");
        assert!(history[1].invalid_at.is_none());
    }

    // -----------------------------------------------------------------------
    // Edge case tests for self-editing tools
    // -----------------------------------------------------------------------

    #[test]
    fn test_update_memory_both_fields() {
        let store = DuckStore::open_in_memory().unwrap();
        let mem = Memory {
            id: "m-both".into(),
            project_id: None,
            content: "original".into(),
            memory_type: Some("note".into()),
            source_session_id: None,
            confidence: 0.5,
            access_count: 0,
            created_at: Some(Utc::now().naive_utc()),
            updated_at: Some(Utc::now().naive_utc()),
            valid_until: None,
        };
        store.insert_memory(&mem).unwrap();

        // Update both content and confidence simultaneously
        assert!(
            store
                .update_memory("m-both", Some("new content"), Some(0.99))
                .unwrap()
        );
        let fetched = store.get_memory("m-both").unwrap().unwrap();
        assert_eq!(fetched.content, "new content");
        assert_eq!(fetched.confidence, 0.99);
    }

    #[test]
    fn test_update_memory_no_changes() {
        let store = DuckStore::open_in_memory().unwrap();
        let mem = Memory {
            id: "m-noop".into(),
            project_id: None,
            content: "unchanged".into(),
            memory_type: None,
            source_session_id: None,
            confidence: 0.7,
            access_count: 0,
            created_at: Some(Utc::now().naive_utc()),
            updated_at: Some(Utc::now().naive_utc()),
            valid_until: None,
        };
        store.insert_memory(&mem).unwrap();

        // Pass None for both — no fields to update
        let result = store.update_memory("m-noop", None, None);
        // Should either return false or succeed gracefully
        assert!(result.is_ok());
    }

    #[test]
    fn test_delete_nonexistent_fact() {
        let store = DuckStore::open_in_memory().unwrap();
        assert!(!store.delete_fact("nonexistent-id").unwrap());
    }

    #[test]
    fn test_delete_nonexistent_memory() {
        let store = DuckStore::open_in_memory().unwrap();
        assert!(!store.delete_memory("nonexistent-id").unwrap());
    }

    #[test]
    fn test_get_nonexistent_fact() {
        let store = DuckStore::open_in_memory().unwrap();
        assert!(store.get_fact("nonexistent-id").unwrap().is_none());
    }

    #[test]
    fn test_get_nonexistent_memory() {
        let store = DuckStore::open_in_memory().unwrap();
        assert!(store.get_memory("nonexistent-id").unwrap().is_none());
    }

    #[test]
    fn test_invalidate_nonexistent_fact() {
        let store = DuckStore::open_in_memory().unwrap();
        assert!(!store.invalidate_fact("nonexistent-id", None).unwrap());
    }

    #[test]
    fn test_fact_history_empty_subject() {
        let store = DuckStore::open_in_memory().unwrap();
        let history = store.get_fact_history("no-such-subject").unwrap();
        assert!(history.is_empty());
    }

    #[test]
    fn test_search_facts_no_results() {
        let store = DuckStore::open_in_memory().unwrap();
        let results = store.search_facts("zzzznonexistent").unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_upsert_fact_different_predicate_not_superseded() {
        let store = DuckStore::open_in_memory().unwrap();
        // Insert: auth "uses" JWT
        store
            .insert_fact(&make_fact("f-1", "auth", "uses", "JWT"))
            .unwrap();

        // Upsert with different predicate: auth "prefers" OAuth2
        store
            .upsert_fact(&make_fact("f-2", "auth", "prefers", "OAuth2"))
            .unwrap();

        // Both should be active (different predicates don't conflict)
        let active = store.get_active_facts(None, 100).unwrap();
        assert_eq!(active.len(), 2);
    }

    #[test]
    fn test_update_memory_confidence_bounds() {
        let store = DuckStore::open_in_memory().unwrap();
        let mem = Memory {
            id: "m-bounds".into(),
            project_id: None,
            content: "test".into(),
            memory_type: None,
            source_session_id: None,
            confidence: 0.5,
            access_count: 0,
            created_at: Some(Utc::now().naive_utc()),
            updated_at: Some(Utc::now().naive_utc()),
            valid_until: None,
        };
        store.insert_memory(&mem).unwrap();

        // Set confidence to 0.0 and 1.0 (boundary values)
        assert!(store.update_memory("m-bounds", None, Some(0.0)).unwrap());
        let fetched = store.get_memory("m-bounds").unwrap().unwrap();
        assert_eq!(fetched.confidence, 0.0);

        assert!(store.update_memory("m-bounds", None, Some(1.0)).unwrap());
        let fetched = store.get_memory("m-bounds").unwrap().unwrap();
        assert_eq!(fetched.confidence, 1.0);
    }

    #[test]
    fn test_multiple_supersessions_chain() {
        let store = DuckStore::open_in_memory().unwrap();

        // Create a chain: f1 -> f2 -> f3
        store
            .insert_fact(&make_fact("f-1", "db", "uses", "SQLite"))
            .unwrap();
        store
            .upsert_fact(&make_fact("f-2", "db", "uses", "Postgres"))
            .unwrap();
        store
            .upsert_fact(&make_fact("f-3", "db", "uses", "DuckDB"))
            .unwrap();

        // Only the latest should be active
        let active = store.get_active_facts(None, 100).unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].object, "DuckDB");

        // History should have all 3
        let history = store.get_fact_history("db").unwrap();
        assert_eq!(history.len(), 3);
    }
}
