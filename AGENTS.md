# AGENTS.md

Guidelines for AI coding agents working on the Remembrant codebase.

## Project Overview

Remembrant is a Rust-based CLI tool (`rem`) that ingests coding agent artifacts from Claude Code, Codex CLI, and Gemini CLI into DuckDB (structured rows plus a persistent property graph) and LanceDB for shared persistent memory across AI coding agents. It includes Semantic XPath, a local web dashboard/API, a stateless MCP server, optional 26-language code analysis via Infiniloom, and generic SQLite/JSONL agent adapters.

## Architecture

### Workspace Structure

This is a Cargo workspace (edition 2024) with two crates:

- **engine** (`remembrant-engine`): Core library — ingestion, storage, search, graph, Semantic XPath
- **cli** (`remembrant`): Binary crate providing the `rem` CLI tool (26 commands), web dashboard/API, and MCP stdio server

### Key Modules

**engine/src/**:
- `store/` — Database implementations
  - `duckdb.rs` — DuckStore: structured data, tags, analytics, and graph tables/algorithms (uses `Mutex<Connection>`)
  - `lance.rs` — LanceStore: vector embeddings + symbol embeddings (async)
  - `graph.rs` — GraphStoreBackend trait + in-memory GraphStore (fallback)
- `ingest/` — Agent-specific parsers
  - `claude.rs` — ClaudeIngester (JSONL transcripts)
  - `codex.rs` — CodexIngester (rollout JSONL artifacts)
  - `gemini.rs` — GeminiIngester (JSON sessions)
  - `jsonl_adapter.rs` — YAML-mapped generic JSONL adapter
  - `native_adapters.rs` — config-aware registry for native and dynamic adapters
- `semantic_tree.rs` — TreeBuilder, TreeNode, TreeNodeType, TreeSchema (hierarchical memory model)
- `xpath_query.rs` — XPathQuery parser + evaluator (weighted-set algorithm from arXiv:2603.01160)
- `semantic_scorer.rs` — SemanticScorer (embedding cache) + keyword_scorer (fallback)
- `graph_builder.rs` — `GraphBuilder<B: GraphBackend>` generic over backend (in-memory or DuckDB)
- `code_analysis.rs` — Infiniloom bridge (feature-gated: `code-analysis`)
- `repo_embed.rs` — RepoEmbedder (AST/line chunking, content-addressable IDs, update-aware replacement)
- `embed_pipeline.rs` — EmbedPipeline for batching embeddings
- `embedding.rs` — EmbedProvider trait (LmStudioEmbedder, MockEmbedder)
- `distill.rs` — Distiller for LLM-based insight extraction
- `watcher.rs` — Debounced FileWatcher for JSON/JSONL artifacts and agent MEMORY.md files
- `pipeline.rs` — IngestPipeline for orchestrating ingestion
- `config.rs` — AppConfig
- `detect.rs` — Agent detection utilities

**cli/src/**:
- `main.rs` — 26 subcommands: init, watch, stop, search, find, recent, brief, context, consolidate, patterns, decisions, related, graph, timeline, note, forget, export, embed, ingest, status, stats, gc, analyze, web, mcp, xpath
- `mcp_server.rs` — MCP `2026-07-28` stdio server with legacy initialize compatibility
- `web_dashboard.html/css/js` — embedded same-origin dashboard assets

### Feature Flags

- **default** — Enables the bundled generic SQLite adapter feature (`sqlite-adapters`)
- **code-analysis** — Enables Infiniloom integration (AST parsing, secret scanning, BLAKE3 hashing). Adds `infiniloom-engine` and `blake3`.

## Development Guidelines

### Building and Testing

```bash
# Build the entire workspace
cargo build

# Build with code-analysis feature
cargo build --features code-analysis

# Run default unit, integration, and CLI E2E tests
cargo test --workspace --all-targets

# Run feature-gated tests
cargo test --workspace --all-targets --features code-analysis

# Run specific crate tests
cargo test -p remembrant-engine
cargo test -p remembrant

# Run the CLI
cargo run --bin rem -- --help

# Format code
cargo fmt --all

# Check for issues (both feature configurations)
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --features code-analysis -- -D warnings

# Check embedded dashboard JavaScript
node --check cli/src/web_dashboard.js
```

### Code Style

- **Edition**: 2024 Rust
- **Error Handling**: Use `anyhow::Result` for application code, `thiserror` for library errors
- **Async**: Use Tokio runtime (`#[tokio::main]` or `#[tokio::test]`)
- **Logging**: Use `tracing` macros (`tracing::info!`, `tracing::error!`, etc.)
- **Serialization**: Use `serde` with `#[derive(Serialize, Deserialize)]`

### Important Patterns

#### GraphBuilder (Generic over Backend)

GraphBuilder is generic over the `GraphBackend` trait, supporting both in-memory and DuckDB backends:

```rust
pub trait GraphBackend {
    fn add_node(&self, id: &str, kind: &str, name: &str, properties: &str) -> Result<()>;
    fn add_edge(&self, from_id: &str, to_id: &str, kind: &str, properties: &str) -> Result<()>;
    fn get_node(&self, id: &str) -> Result<Option<(String, String, String, String)>>;
    fn delete_node(&self, id: &str) -> Result<bool>;
    fn query_neighbors(&self, id: &str, edge_kind: Option<&str>) -> Result<Vec<NeighborInfo>>;
    fn node_count(&self) -> Result<usize>;
    fn edge_count(&self) -> Result<usize>;
}

// In-memory (for tests and fallback)
pub type InMemoryGraphBuilder = GraphBuilder<GraphStore>;

// Persistent (DuckDB graph tables)
pub type DuckGraphBuilder = GraphBuilder<DuckStore>;
```

#### DuckStore (Synchronous + Graph Tables)

DuckStore uses `Mutex<Connection>` for thread-safe access. Graph algorithms operate directly on `graph_nodes`/`graph_edges` and do not require DuckPGQ:

```rust
impl DuckStore {
    // Structured data
    pub fn insert_session(&self, session: &Session) -> Result<()>;

    // Graph CRUD (stored in graph_nodes/graph_edges tables)
    pub fn insert_graph_node(&self, id: &str, kind: &str, name: &str, props: &str) -> Result<()>;
    pub fn insert_graph_edge(&self, from: &str, to: &str, kind: &str, props: &str) -> Result<()>;

    // Graph algorithms and optional extension loading
    pub fn load_duckpgq(&self) -> Result<()>;
    pub fn init_graph(&self) -> Result<()>;
    pub fn pgq_shortest_path(&self, from: &str, to: &str, max_depth: usize) -> Result<Vec<String>>;
    pub fn pgq_pagerank(&self, limit: usize) -> Result<Vec<(String, f64)>>;
    pub fn pgq_pattern_match(&self, id: &str, edge_kind: &str, direction: &str) -> Result<Vec<Vec<String>>>;

    // Connection accessor (for TreeBuilder)
    pub fn connection(&self) -> &Mutex<Connection>;
}
```

#### LanceStore (Async)

LanceStore is async throughout, with three explicit table schemas:

```rust
impl LanceStore {
    pub async fn open(path: &Path) -> Result<Self>;

    // Code and memory embeddings
    pub async fn insert_code_embedding(&self, /* typed fields */) -> Result<()>;
    pub async fn insert_memory_embedding(&self, /* typed fields */) -> Result<()>;
    pub async fn search_code(&self, query: &[f32], limit: usize) -> Result<Vec<CodeSearchResult>>;
    pub async fn search_memories(&self, query: &[f32], limit: usize) -> Result<Vec<MemorySearchResult>>;

    // Symbol embeddings (code analysis)
    pub async fn insert_symbol_embedding(&self, symbol: SymbolEmbedding) -> Result<()>;
    pub async fn search_symbols(&self, query: &[f32], limit: usize) -> Result<Vec<SymbolSearchResult>>;
}
```

#### EmbedProvider (Not dyn-compatible)

The `EmbedProvider` trait uses `impl Future`. Always use generics:

```rust
// CORRECT
pub async fn process<P: EmbedProvider>(provider: &P) -> Result<()> { }

// INCORRECT - will not compile
pub async fn process(provider: &dyn EmbedProvider) -> Result<()> { }
```

#### Semantic XPath

The XPath system has three layers:

1. **TreeBuilder** (`semantic_tree.rs`) — Builds hierarchical memory tree from DuckDB, lazy-loads children
2. **XPathQuery** (`xpath_query.rs`) — Recursive descent parser for XPath-like syntax
3. **SemanticScorer** (`semantic_scorer.rs`) — Embedding-based similarity for `~` operator

```rust
// Parse and evaluate a Semantic XPath query
let query = parse_xpath("//Session[node~\"auth\"]/Decision")?;
let results = evaluate_xpath(&query, &root, &scorer);
for weighted_node in results {
    println!("{}: {:.2}", weighted_node.node.name, weighted_node.weight);
}
```

### Testing

- Unit tests go in the same file (bottom of file)
- Integration tests go in `engine/tests/`
- Use `DuckStore::open_in_memory()` for tests (no cleanup needed)
- Use `MockEmbedder` for embedding tests
- Use `tempfile::tempdir()` for LanceDB paths in tests
- CLI/E2E tests must set an isolated `HOME`; never use the developer’s real home
- Add MCP protocol coverage in `cli/tests/e2e.rs` when changing `mcp_server.rs`
- Feature-gated tests: `#[cfg(feature = "code-analysis")]`

### Database Schema

**DuckDB Tables**:
- `sessions` — Session metadata (id, project_id, agent, timestamps, summary)
- `decisions` — Decisions with rationale (id, session_id, what, why, alternatives)
- `memories` — Memory notes (id, project_id, content, confidence, access_count)
- `tool_calls` — Tool call history (id, session_id, tool_name, command, success)
- `file_stats` — File statistics (file_path, project_id, language, LOC, complexity)
- `code_symbols` — AST-parsed symbols (file_path, name, kind, signature, lines)
- `code_dependencies` — Import/call dependencies between files
- `code_analysis_runs` — Analysis run metadata
- `graph_nodes` — Property graph nodes (id, kind, name, properties JSON)
- `graph_edges` — Property graph edges (from_id, to_id, kind, properties JSON)
- `memory_tags` — Persistent, case-insensitive tags for manual notes

**LanceDB Tables**:
- `code_embeddings` — Code/session/tool vector embeddings with stable IDs
- `memory_embeddings` — Memory vector embeddings with stable IDs
- `symbol_embeddings` — Code symbol embeddings including PageRank and source spans

### Common Pitfalls

1. **Don't modify test artifact directories**: Never modify `~/.claude`, `~/.codex`, or `~/.gemini` in tests
2. **Use in-memory databases**: Always use `DuckStore::open_in_memory()` for tests
3. **Handle async correctly**: LanceStore is async, DuckStore is sync — don't mix patterns
4. **EmbedProvider generics**: Never try to use `&dyn EmbedProvider`
5. **Tilde expansion**: Config paths use `~/` — expand with `dirs::home_dir()`
6. **Feature gates**: Code analysis imports must be behind `#[cfg(feature = "code-analysis")]`
7. **Dashboard is same-origin**: Do not reintroduce permissive CORS on local APIs
8. **MCP is newline JSON-RPC**: stdout must contain responses only; logs belong on stderr
9. **GraphBackend is all-string**: Node kind, properties are strings (serialized JSON for properties)

### Adding New Features

1. Add core logic to `engine/src/`
2. Add tests (unit in same file, integration in `engine/tests/`)
3. Add CLI subcommand in `cli/src/main.rs` if needed
4. Update AGENTS.md and README.md
5. Run both default and feature test/lint gates plus `node --check` when dashboard JS changed

### Dependencies

Key dependencies (see workspace `Cargo.toml`):
- `duckdb` 1.x — Embedded SQL database; optional DuckPGQ extension
- `glob` — Dynamic JSONL file matching
- `lancedb` 0.27 — Vector database
- `arrow-array`, `arrow-schema` 57 — Arrow format (must match lancedb)
- `tokio` — Async runtime
- `clap` 4 — CLI argument parsing
- `serde`, `serde_json`, `serde_norway` — Serialization (YAML uses the maintained `serde_norway` fork)
- `tracing`, `tracing-subscriber` — Logging
- `notify`, `notify-debouncer-mini` — File watching
- `reqwest` 0.12 — HTTP client (LM Studio)
- `chrono`, `uuid` — Date/time and IDs

Optional (code-analysis feature):
- `infiniloom-engine` — AST parsing, secret scanning (26 languages)
- `blake3` — Content-addressable hashing
