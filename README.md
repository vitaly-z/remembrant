# Remembrant

**Shared Persistent Memory for AI Coding Agents**

Remembrant captures, indexes, and connects everything AI coding agents produce — sessions, decisions, tool calls, code entities — across Claude Code, Codex CLI, and Gemini CLI. It stores this in a triple-database architecture (DuckDB + LanceDB + property graph) and exposes it through a powerful CLI (`rem`), Semantic XPath queries, and a web dashboard.

No agent works in isolation anymore. Every session builds on everything that came before.

## Key Features

- **Multi-Agent Ingestion** — Native parsers for Claude Code, Codex CLI, and Gemini CLI, plus YAML-configured SQLite and JSONL adapters
- **Triple-Database Architecture** — DuckDB (structured data and a persistent property graph), LanceDB (vector search), and an in-memory graph backend for tests/fallback
- **Semantic XPath** — Tree-structured memory queries based on [arXiv:2603.01160](https://arxiv.org/abs/2603.01160), 176.7% better recall than flat RAG
- **Graph Analytics** — Directed shortest path, PageRank, and edge-kind traversal directly over DuckDB graph tables; optional DuckPGQ extension loading is supported when installed
- **Code Analysis** — AST parsing for 26 languages via Infiniloom integration (feature-gated)
- **Repository Embedding** — Embed entire codebases with content-addressable chunks and idempotent `--update` replacement
- **Security Scanning** — Secret detection and redaction before embedding (via Infiniloom)
- **LLM Distillation** — Extract insights, patterns, and decisions from raw sessions
- **File Watching** — Debounced filesystem events for JSON/JSONL artifacts and agent MEMORY.md files, with polling only for dynamic SQLite adapters
- **Web Dashboard and API** — Built-in same-origin web UI, analytics, and REST endpoints
- **MCP Server** — Stateless Model Context Protocol `2026-07-28` over stdio, with legacy initialize compatibility
- **Local-First** — All data stays local; uses LM Studio for embeddings

## Quick Start

```bash
# Build and install the CLI
cargo install --path cli

# Initialize (creates config, scans for agents)
rem init

# Ingest sessions from all detected agents
rem ingest

# Search across all sessions
rem search "authentication refactor"

# Semantic XPath query
rem xpath '//Session[node~"auth"]/Decision'

# Embed a repository for code search
rem embed /path/to/project

# View stats
rem stats
```

### With Code Analysis (26 languages)

```bash
# Build with Infiniloom integration
cargo install --path cli --features code-analysis

# Analyze a repository (AST parsing, symbol extraction, dependency graph)
rem analyze /path/to/project
```

## Architecture

```
                    +-----------+
                    |  rem CLI  |  26 commands
                    +-----+-----+
                          |
                    +-----+-----+
                    |  Engine   |  remembrant-engine
                    +-----+-----+
                          |
          +---------------+---------------+
          |               |               |
    +-----+-----+  +-----+-----+  +------+------+
    |  DuckDB   |  |  LanceDB  |  | Graph Store |
                    |  + Graph  |  |  (Vector)  |  | (In-Memory) |
    +-----------+  +-----------+  +-------------+
```

### DuckDB (Structured Data + Property Graph)

Stores all structured data and provides persistent property-graph tables and graph algorithms:

| Table | Purpose |
|-------|---------|
| `sessions` | Session metadata (agent, project, timestamps, summary) |
| `decisions` | Decisions with rationale (what, why, alternatives) |
| `memories` | Memory notes (content, confidence, access_count) |
| `tool_calls` | Tool call history (command, success/failure) |
| `file_stats` | File statistics (LOC, complexity, change frequency) |
| `code_symbols` | AST-parsed symbols (functions, classes, structs) |
| `code_dependencies` | Import/call dependencies between files |
| `graph_nodes` | Property graph nodes (kind, name, JSON properties) |
| `graph_edges` | Property graph edges (kind, JSON properties) |
| `memory_tags` | Case-insensitive tags for manual notes |

### LanceDB (Vector Embeddings)

| Table | Purpose |
|-------|---------|
| `code_embeddings` | Session/code-change/tool embeddings for semantic search |
| `memory_embeddings` | Memory embeddings for semantic search |
| `symbol_embeddings` | Code symbol embeddings with PageRank scores |

### Property graph

Graph nodes and edges are ordinary DuckDB rows, so the graph survives restarts without a second database. `DuckStore` implements deterministic BFS shortest paths, iterative PageRank with dangling-node handling, and incoming/outgoing edge-kind matching directly against those tables. `init_graph()` and `load_duckpgq()` remain available for installations that have the optional DuckPGQ extension, but graph features do not require it.

## Semantic XPath

Based on the paper ["Semantic XPath: Tree-Structured Memory Access for LLM Agents"](https://arxiv.org/abs/2603.01160), Remembrant implements a weighted-set evaluation algorithm over a hierarchical memory tree:

```
Root
├── Project "remembrant"
│   ├── Session "2026-03-20T14:30:00"
│   │   ├── Decision "persist the graph in DuckDB"
│   │   ├── Memory "PageRank handles dangling nodes"
│   │   └── ToolCall "cargo test"
│   └── Session "2026-03-21T09:00:00"
│       └── CodeEntity "graph_builder.rs"
│           ├── Symbol "GraphBackend (trait)"
│           └── Symbol "GraphBuilder (struct)"
└── Project "infiniloom"
    └── ...
```

### Query Examples

```bash
# All decisions about authentication
rem xpath '//Decision[node~"auth"]'

# Recent sessions with their tool calls
rem xpath '/Root/Project/Session[position()>last()-5]/ToolCall'

# Code entities that relate to "graph"
rem xpath '//CodeEntity[node~"graph"]/Symbol'

# Decisions in a specific project
rem xpath '/Root/Project[@name="remembrant"]/Session/Decision'

# Combined semantic + structural query
rem xpath '//Session[node~"refactor"]/Decision[node~"performance"]'
```

The `~` operator applies semantic similarity scoring, while `@attr="value"` does exact matching. The CLI evaluator uses the deterministic keyword scorer by default so XPath works without a running embedding service.

## Supported Agents

| Agent | Artifact Location | Format |
|-------|------------------|--------|
| **Claude Code** | `~/.claude/projects/*/` | JSONL transcripts + MEMORY.md |
| **Codex CLI** | `~/.codex/sessions/` | Rollout JSONL + `history.jsonl` |
| **Gemini CLI** | `~/.gemini/tmp/*/chats/` | JSON session files |
| **Configured agents** | Any local path | Generic SQLite or JSONL mappings |

## CLI Reference

| Command | Description |
|---------|-------------|
| `rem init` | Initialize config, scan for agents |
| `rem watch` | Start file watcher daemon |
| `rem stop` | Stop the watcher daemon |
| `rem ingest` | Ingest sessions from all agents |
| `rem search <query>` | Hybrid semantic search (`--project`, `--agent`, `--type`, `--since`, `--json`) |
| `rem find <text>` | Exact text search |
| `rem recent` | Show recent sessions |
| `rem brief` | Daily context briefing |
| `rem patterns [topic]` | Find cross-project patterns |
| `rem decisions` | View decision journal |
| `rem related <path>` | Find related content for a file |
| `rem graph <path>` | Show dependency graph |
| `rem timeline <topic>` | Chronological topic view |
| `rem note <text>` | Add a manual note with persistent `--tag` values |
| `rem forget --session <id>` | Remove a session |
| `rem export` | Generate agent memory files |
| `rem embed <path>` | Embed a repository; `--update` replaces stale project vectors |
| `rem xpath <query>` | Semantic XPath query |
| `rem analyze <path>` | AST code analysis (requires `code-analysis` feature) |
| `rem status` | Show daemon and database status |
| `rem stats` | Show analytics and statistics |
| `rem gc` | Garbage collect old/orphaned data |
| `rem context` | Assemble a token-budgeted project context |
| `rem consolidate` | Merge and decay related memories |
| `rem web` | Serve the local dashboard and REST API |
| `rem mcp` | Serve MCP tools over newline-delimited stdio JSON-RPC |

## Code Analysis (Feature-Gated)

When built with `--features code-analysis`, Remembrant integrates with [Infiniloom](https://github.com/Topos-Labs/infiniloom) for deep code understanding:

- **26-language AST parsing** via tree-sitter (Python, JS, TS, Rust, Go, Java, C, C++, and 18 more)
- **Symbol extraction** — functions, classes, structs, traits, interfaces
- **Dependency graph** — imports, calls, inheritance relationships
- **PageRank ranking** — identify the most important symbols in a codebase
- **BLAKE3 content hashing** — content-addressable chunk deduplication
- **Secret scanning** — detect and redact secrets before embedding

```bash
# Analyze a Rust project
rem analyze /path/to/rust-project --project my-project

# The symbols are stored in DuckDB and LanceDB for querying
rem search "GraphBuilder" --type symbol
rem xpath '//CodeEntity/Symbol[@kind="function"]'
```

## Configuration

Config lives at `~/.remembrant/config.yaml` and is created on first use. Paths accept a leading `~`:

```yaml
storage:
  duckdb_path: ~/.remembrant/remembrant.duckdb
  lancedb_path: ~/.remembrant/lancedb

agents:
  claude_code:
    enabled: true
    path: ~/.claude
  codex:
    enabled: true
    path: ~/.codex
  gemini:
    enabled: true
    path: ~/.gemini
  dynamic:
    - id: custom_jsonl_agent
      display_name: Custom JSONL Agent
      enabled: true
      path: ~/.custom-agent
      adapter_type: jsonl
      jsonl:
        file_pattern: "**/*.jsonl"
        session_id_path: session.id
        timestamp_path: timestamp
        content_path: message.text
        tool_name_path: tool.name
        tool_call_type: kind=tool

embedding:
  model: text-embedding-nomic-embed-text-v1.5@q8_0
  endpoint: http://localhost:1234/v1
  batch_size: 100
  dimensions: 768

watch:
  debounce_ms: 5000
```

Dynamic SQLite adapters use `adapter_type: sqlite` with table/column mappings under a `sqlite:` key. Unsupported adapter types fail explicitly rather than being ignored.

## Web dashboard and MCP

Run `rem web --port 7878` for the dashboard at `http://127.0.0.1:7878`. Static assets are embedded in the binary, and the API is same-origin; no permissive CORS layer is enabled.

Run `rem mcp` for an MCP stdio server. It implements the stateless `2026-07-28` protocol, including `server/discover`, deterministic cacheable `tools/list` metadata, per-request `_meta` protocol validation, and `tools/call`. Legacy `initialize` clients remain supported. Nine memory tools are exposed: search, recall, add, context, XPath, decision recording, update, delete, and fact revision.

## Project Structure

```
remembrant/
├── engine/                    # Core library (remembrant-engine)
│   ├── src/
│   │   ├── store/
│   │   │   ├── duckdb.rs      # DuckStore + persistent graph algorithms
│   │   │   ├── lance.rs       # LanceStore (vector + symbol embeddings)
│   │   │   ├── graph.rs       # In-memory GraphStore + GraphStoreBackend trait
│   │   │   └── mod.rs
│   │   ├── ingest/
│   │   │   ├── claude.rs      # Claude Code parser (JSONL)
│   │   │   ├── codex.rs       # Codex CLI parser (JSONL)
│   │   │   ├── gemini.rs      # Gemini CLI parser (JSON)
│   │   │   ├── jsonl_adapter.rs # Generic YAML-mapped JSONL adapter
│   │   │   └── native_adapters.rs # Config-aware adapter registry
│   │   ├── semantic_tree.rs   # Tree-structured memory model (TreeBuilder, TreeNode)
│   │   ├── xpath_query.rs     # Semantic XPath parser + evaluator
│   │   ├── semantic_scorer.rs # Embedding-based semantic similarity scoring
│   │   ├── graph_builder.rs   # Generic GraphBuilder<B: GraphBackend>
│   │   ├── code_analysis.rs   # Infiniloom bridge (feature-gated)
│   │   ├── repo_embed.rs      # Repository embedder (AST chunking, secret scan)
│   │   ├── embed_pipeline.rs  # Embedding batch pipeline
│   │   ├── embedding.rs       # EmbedProvider trait (LmStudio, Mock)
│   │   ├── distill.rs         # LLM distillation
│   │   ├── pipeline.rs        # Ingestion pipeline orchestrator
│   │   ├── watcher.rs         # File system watcher
│   │   ├── detect.rs          # Agent detection utilities
│   │   └── config.rs          # AppConfig
│   └── tests/                 # Integration tests
├── cli/                       # CLI binary (rem)
│   ├── src/main.rs            # 26 subcommands, REST API, MCP entry point
│   ├── src/mcp_server.rs      # MCP 2026-07-28 stdio server
│   ├── src/web_dashboard.*    # Embedded HTML/CSS/JS dashboard
│   └── tests/e2e.rs           # CLI, watcher, dashboard, and MCP E2E
├── .github/workflows/         # CI/CD
├── AGENTS.md                  # Guidelines for AI coding agents
└── Cargo.toml                 # Workspace config (edition 2024)
```

## Development

```bash
# Build the workspace
cargo build

# Run all default tests, including CLI E2E
cargo test --workspace --all-targets

# Verify the optional Infiniloom integration
cargo test --workspace --all-targets --features code-analysis

# Build with code-analysis feature
cargo build --features code-analysis

# Run specific crate tests
cargo test -p remembrant-engine
cargo test -p remembrant

# Format and lint
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --features code-analysis -- -D warnings

# Dashboard JavaScript syntax gate
node --check cli/src/web_dashboard.js
```

### Key Design Decisions

- **Edition 2024 Rust** — latest language features
- **Generic GraphBuilder** — `GraphBuilder<B: GraphBackend>` works with both in-memory and DuckDB backends
- **One DuckDB graph store** — no graph-database sync; algorithms work without optional extensions
- **Feature-gated Infiniloom** — optional `code-analysis` feature avoids heavy tree-sitter deps when not needed
- **Content-addressable chunks** — BLAKE3 with code analysis, stable SHA-256 fallback IDs otherwise
- **Secrets never embedded** — security scanning runs before chunking/embedding

## How It Works

1. **Detect** — Scan for installed agents (Claude Code, Codex, Gemini)
2. **Ingest** — Agent-specific parsers extract sessions, tool calls, decisions, memories
3. **Store** — Structured data goes to DuckDB; graph relationships persist in `graph_nodes` and `graph_edges`
4. **Embed** — LM Studio generates embeddings, stored in LanceDB
5. **Index** — Build hierarchical memory tree (Project > Session > Decision/Memory/ToolCall/CodeEntity > Symbol)
6. **Query** — CLI provides semantic search, XPath queries, graph traversal, and analytics
7. **Distill** — LLM extracts high-level insights and cross-project patterns

## License

MIT
