# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Is This

Remembrant is a Rust CLI tool (`rem`) that ingests AI coding agent artifacts (Claude Code, Codex CLI, Gemini CLI) into a triple-database (DuckDB + LanceDB + in-memory graph) for shared persistent memory. It features Semantic XPath queries and a local web dashboard/API.

## Build & Test Commands

```bash
# Build workspace (excludes desktop in CI)
cargo build

# Run all tests (~167 tests)
cargo test

# Run specific crate tests
cargo test -p remembrant-engine
cargo test -p remembrant

# Run a single test by name
cargo test -p remembrant-engine -- test_name

# Lint and format
cargo clippy --workspace -- -D warnings
cargo fmt --all

# Run the CLI locally
cargo run --bin rem -- --help
```

CI runs: `cargo check`, `cargo test`, `cargo clippy -- -D warnings`, and `cargo fmt --check` on all workspace members except `remembrant-desktop`.

## Workspace Structure

Two crates in a Cargo workspace (edition 2024):

- **engine** (`remembrant-engine`): Core library — ingestion, storage, search, graph, Semantic XPath, embedding
- **cli** (`remembrant`): Binary crate `rem` with 22+ subcommands + built-in web dashboard (axum)

## Architecture

### Storage Layer (`engine/src/store/`)

- `duckdb.rs` — **DuckStore**: Structured data + DuckPGQ graph queries. Uses `Mutex<Connection>` for thread safety. Synchronous API.
- `lance.rs` — **LanceStore**: Vector embeddings + symbol embeddings. Fully async API.
- `graph.rs` — **GraphStoreBackend** trait + in-memory **GraphStore** (fallback)

DuckStore and LanceStore have different concurrency models — don't mix sync/async patterns.

### Ingestion (`engine/src/ingest/`)

Agent-specific parsers: `claude.rs` (JSONL), `codex.rs` (SQLite), `gemini.rs` (JSON).

### Semantic XPath (`engine/src/`)

Three layers:
1. **TreeBuilder** (`semantic_tree.rs`) — Builds hierarchical memory tree from DuckDB
2. **XPathQuery** (`xpath_query.rs`) — Recursive descent parser for XPath-like syntax
3. **SemanticScorer** (`semantic_scorer.rs`) — Embedding-based similarity for `~` operator

### GraphBuilder (`engine/src/graph_builder.rs`)

Generic over `GraphBackend` trait — works with both in-memory and DuckDB backends:
```rust
pub type InMemoryGraphBuilder = GraphBuilder<GraphStore>;
pub type DuckGraphBuilder = GraphBuilder<DuckStore>;
```

## Key Patterns

### EmbedProvider Is Not dyn-Compatible

The `EmbedProvider` trait uses `impl Future`. Always use generics, never `&dyn EmbedProvider`:
```rust
// Correct
pub async fn process<P: EmbedProvider>(provider: &P) -> Result<()> { }
```

### Error Handling

`anyhow::Result` for application code, `thiserror` for library error types.

### Testing

- Unit tests at bottom of source files, integration tests in `engine/tests/`
- Use `DuckStore::open_in_memory()` — no cleanup needed
- Use `MockEmbedder` for embedding tests
- Use `tempfile::tempdir()` for LanceDB test paths
- Never modify real agent artifact dirs (`~/.claude`, `~/.codex`, `~/.gemini`) in tests

### Config Paths

Config uses `~/` tilde paths — always expand with `dirs::home_dir()`.

### GraphBackend

All-string interface: node kind, name, and properties are strings. Properties are serialized JSON.

## Dependencies to Note

- `arrow-array`/`arrow-schema` version **must match** lancedb's arrow version (currently v57)
- `protoc` is required at build time (for lance-encoding)
