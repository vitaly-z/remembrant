use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Top-level config
// ---------------------------------------------------------------------------

/// Top-level application configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AppConfig {
    #[serde(default)]
    pub agents: AgentsConfig,
    #[serde(default)]
    pub storage: StorageConfig,
    #[serde(default)]
    pub embedding: EmbeddingConfig,
    #[serde(default)]
    pub distillation: DistillationConfig,
    #[serde(default)]
    pub retention: RetentionConfig,
    #[serde(default)]
    pub watch: WatchConfig,
}

impl AppConfig {
    /// Return the Remembrant configuration directory (`~/.remembrant/`).
    pub fn config_dir() -> Result<PathBuf> {
        let home = dirs::home_dir().context("could not determine home directory")?;
        Ok(home.join(".remembrant"))
    }

    /// Path to the config file (`~/.remembrant/config.yaml`).
    fn config_path() -> Result<PathBuf> {
        Ok(Self::config_dir()?.join("config.yaml"))
    }

    /// Load configuration from `~/.remembrant/config.yaml`.
    ///
    /// If the file does not exist a default configuration is written first,
    /// then returned.
    pub fn load() -> Result<Self> {
        let path = Self::config_path()?;

        if !path.exists() {
            let config = Self::default();
            config.save().context("failed to write default config")?;
            return Ok(config);
        }

        let contents = fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let config: Self = serde_norway::from_str(&contents)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        config
            .validate()
            .with_context(|| format!("invalid configuration in {}", path.display()))?;
        Ok(config)
    }

    /// Persist the current configuration to `~/.remembrant/config.yaml`.
    pub fn save(&self) -> Result<()> {
        self.validate().context("refusing to save invalid config")?;
        let dir = Self::config_dir()?;
        fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;

        let path = Self::config_path()?;
        let yaml = serde_norway::to_string(self).context("failed to serialize config")?;
        fs::write(&path, yaml).with_context(|| format!("failed to write {}", path.display()))?;
        Ok(())
    }

    /// Validate values that would otherwise panic or be silently ignored.
    pub fn validate(&self) -> Result<()> {
        if self.storage.duckdb_path.trim().is_empty() {
            anyhow::bail!("storage.duckdb_path cannot be empty");
        }
        if self.storage.lancedb_path.trim().is_empty() {
            anyhow::bail!("storage.lancedb_path cannot be empty");
        }
        if self.embedding.model.trim().is_empty() {
            anyhow::bail!("embedding.model cannot be empty");
        }
        if self.embedding.endpoint.trim().is_empty() {
            anyhow::bail!("embedding.endpoint cannot be empty");
        }
        if self.embedding.batch_size == 0 || self.embedding.batch_size > 10_000 {
            anyhow::bail!("embedding.batch_size must be between 1 and 10000");
        }
        if self.embedding.dimensions == 0 || self.embedding.dimensions > 32_768 {
            anyhow::bail!("embedding.dimensions must be between 1 and 32768");
        }
        if self.retention.raw_transcripts_days == 0 || self.retention.raw_transcripts_days > 100_000
        {
            anyhow::bail!("retention.raw_transcripts_days must be between 1 and 100000");
        }
        if self.watch.debounce_ms == 0 || self.watch.debounce_ms > 3_600_000 {
            anyhow::bail!("watch.debounce_ms must be between 1 and 3600000");
        }

        let native_ids = ["claude_code", "codex", "gemini"];
        for (name, entry) in [
            ("claude_code", &self.agents.claude_code),
            ("codex", &self.agents.codex),
            ("gemini", &self.agents.gemini),
        ] {
            if entry.path.trim().is_empty() {
                anyhow::bail!("agents.{name}.path cannot be empty");
            }
        }

        let mut dynamic_ids = std::collections::HashSet::new();
        for agent in &self.agents.dynamic {
            if agent.id.trim().is_empty() {
                anyhow::bail!("dynamic agent id cannot be empty");
            }
            if agent.id.len() > 80
                || !agent.id.chars().all(|character| {
                    character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.')
                })
            {
                anyhow::bail!(
                    "dynamic agent id '{}' may contain only ASCII letters, digits, '_', '-', or '.'",
                    agent.id
                );
            }
            if native_ids.contains(&agent.id.as_str()) || !dynamic_ids.insert(agent.id.clone()) {
                anyhow::bail!("duplicate or reserved dynamic agent id '{}'", agent.id);
            }
            if agent.display_name.trim().is_empty() {
                anyhow::bail!("dynamic agent '{}' display_name cannot be empty", agent.id);
            }
            if agent.path.trim().is_empty() {
                anyhow::bail!("dynamic agent '{}' path cannot be empty", agent.id);
            }
            if !matches!(agent.adapter_type.as_str(), "sqlite" | "jsonl") {
                anyhow::bail!(
                    "dynamic agent '{}' has unsupported adapter type '{}'",
                    agent.id,
                    agent.adapter_type
                );
            }
            if agent.adapter_type == "sqlite"
                && let Some(mapping) = &agent.sqlite
            {
                let db_path = std::path::Path::new(&mapping.db_file);
                if db_path.is_absolute()
                    || db_path
                        .components()
                        .any(|component| matches!(component, std::path::Component::ParentDir))
                {
                    anyhow::bail!(
                        "dynamic agent '{}' sqlite.db_file must be a relative path without '..'",
                        agent.id
                    );
                }
                validate_sqlite_identifier(&mapping.sessions_table, "sessions table")?;
                validate_sqlite_identifier(&mapping.session_columns.id, "session ID column")?;
                if !matches!(
                    mapping.session_columns.timestamp_format.as_str(),
                    "iso8601" | "unix_seconds" | "unix_millis"
                ) {
                    anyhow::bail!(
                        "dynamic agent '{}' has unsupported SQLite timestamp format '{}'",
                        agent.id,
                        mapping.session_columns.timestamp_format
                    );
                }
                for column in [
                    mapping.session_columns.project_id.as_deref(),
                    mapping.session_columns.started_at.as_deref(),
                    mapping.session_columns.ended_at.as_deref(),
                    mapping.session_columns.message_count.as_deref(),
                    mapping.session_columns.summary.as_deref(),
                    mapping.session_columns.content.as_deref(),
                ]
                .into_iter()
                .flatten()
                {
                    validate_sqlite_identifier(column, "session column")?;
                }
                if let Some(table) = mapping.tool_calls_table.as_deref() {
                    validate_sqlite_identifier(table, "tool-call table")?;
                }
                if let Some(columns) = &mapping.tool_call_columns {
                    validate_sqlite_identifier(&columns.id, "tool-call ID column")?;
                    for column in [
                        columns.session_id.as_deref(),
                        columns.tool_name.as_deref(),
                        columns.command.as_deref(),
                        columns.success.as_deref(),
                        columns.timestamp.as_deref(),
                    ]
                    .into_iter()
                    .flatten()
                    {
                        validate_sqlite_identifier(column, "tool-call column")?;
                    }
                }
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Agents
// ---------------------------------------------------------------------------

/// Configuration for supported AI coding agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentsConfig {
    #[serde(default)]
    pub claude_code: AgentEntry,
    #[serde(default)]
    pub codex: AgentEntry,
    #[serde(default)]
    pub gemini: AgentEntry,
    /// Additional agents defined via config (Goose, OpenCode, Cursor, etc.).
    #[serde(default)]
    pub dynamic: Vec<crate::ingest::adapter::DynamicAgentConfig>,
}

impl Default for AgentsConfig {
    fn default() -> Self {
        Self {
            claude_code: AgentEntry::new(true, "~/.claude"),
            codex: AgentEntry::new(true, "~/.codex"),
            gemini: AgentEntry::new(true, "~/.gemini"),
            dynamic: Vec::new(),
        }
    }
}

/// A single agent entry (enabled flag + transcript directory).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentEntry {
    pub enabled: bool,
    pub path: String,
}

impl AgentEntry {
    fn new(enabled: bool, path: &str) -> Self {
        Self {
            enabled,
            path: path.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

/// Paths for persistent stores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub duckdb_path: String,
    pub lancedb_path: String,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            duckdb_path: "~/.remembrant/remembrant.duckdb".to_string(),
            lancedb_path: "~/.remembrant/lancedb".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

/// Configuration for the embedding model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub model: String,
    #[serde(default = "default_embedding_endpoint")]
    pub endpoint: String,
    pub batch_size: usize,
    pub dimensions: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: "text-embedding-nomic-embed-text-v1.5@q8_0".to_string(),
            endpoint: "http://localhost:1234/v1".to_string(),
            batch_size: 100,
            dimensions: 768,
        }
    }
}

fn default_embedding_endpoint() -> String {
    "http://localhost:1234/v1".to_string()
}

fn validate_sqlite_identifier(value: &str, field: &str) -> Result<()> {
    if value.is_empty()
        || !value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_')
    {
        anyhow::bail!("SQLite {field} identifiers may contain only ASCII letters, digits, or '_'");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Distillation
// ---------------------------------------------------------------------------

/// How aggressively to distil ingested transcripts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum DistillationLevel {
    None,
    Minimal,
    #[default]
    Balanced,
    Aggressive,
    Full,
}

/// Distillation settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationConfig {
    #[serde(default)]
    pub level: DistillationLevel,
    pub llm_provider: String,
    pub llm_model: String,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            level: DistillationLevel::default(),
            llm_provider: "http://localhost:1234/v1".to_string(),
            llm_model: "qwen/qwen3-30b-a3b".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Retention
// ---------------------------------------------------------------------------

/// Data retention policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionConfig {
    pub raw_transcripts_days: u32,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        Self {
            raw_transcripts_days: 30,
        }
    }
}

// ---------------------------------------------------------------------------
// Watch
// ---------------------------------------------------------------------------

/// File-watcher settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchConfig {
    pub debounce_ms: u64,
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self { debounce_ms: 5000 }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_round_trips_through_yaml() {
        let config = AppConfig::default();
        let yaml = serde_norway::to_string(&config).expect("serialize");
        let parsed: AppConfig = serde_norway::from_str(&yaml).expect("deserialize");

        assert_eq!(
            parsed.embedding.model,
            "text-embedding-nomic-embed-text-v1.5@q8_0"
        );
        assert_eq!(parsed.embedding.endpoint, "http://localhost:1234/v1");
        assert_eq!(parsed.embedding.batch_size, 100);
        assert_eq!(parsed.embedding.dimensions, 768);
        assert_eq!(parsed.distillation.level, DistillationLevel::Balanced);
        assert_eq!(parsed.retention.raw_transcripts_days, 30);
        assert_eq!(parsed.watch.debounce_ms, 5000);
        assert!(parsed.agents.claude_code.enabled);
        assert_eq!(parsed.agents.claude_code.path, "~/.claude");
    }

    #[test]
    fn partial_yaml_fills_defaults() {
        let yaml = "retention:\n  raw_transcripts_days: 90\n";
        let config: AppConfig = serde_norway::from_str(yaml).expect("deserialize");

        // Overridden field
        assert_eq!(config.retention.raw_transcripts_days, 90);
        // Everything else should be default
        assert_eq!(config.embedding.dimensions, 768);
        assert_eq!(config.embedding.endpoint, "http://localhost:1234/v1");
        assert_eq!(config.distillation.level, DistillationLevel::Balanced);
    }

    #[test]
    fn legacy_unknown_config_keys_are_ignored() {
        let yaml =
            "watch:\n  debounce_ms: 250\n  auto_start: true\ncross_project:\n  enabled: false\n";
        let config: AppConfig = serde_norway::from_str(yaml).expect("deserialize");
        assert_eq!(config.watch.debounce_ms, 250);
    }

    #[test]
    fn invalid_values_are_rejected() {
        let mut config = AppConfig::default();
        config.embedding.dimensions = 0;
        assert!(config.validate().is_err());

        config = AppConfig::default();
        config.agents.dynamic = vec![crate::ingest::DynamicAgentConfig {
            id: "bad agent".into(),
            display_name: "Bad".into(),
            enabled: false,
            path: "/tmp/bad".into(),
            adapter_type: "jsonl".into(),
            sqlite: None,
            jsonl: None,
        }];
        assert!(config.validate().is_err());

        config = AppConfig::default();
        config.agents.dynamic = vec![crate::ingest::DynamicAgentConfig {
            id: "claude_code".into(),
            display_name: "Reserved".into(),
            enabled: false,
            path: "/tmp/reserved".into(),
            adapter_type: "jsonl".into(),
            sqlite: None,
            jsonl: None,
        }];
        let error = config.validate().unwrap_err();
        assert!(error.to_string().contains("duplicate or reserved"));
    }

    #[test]
    fn config_dir_is_under_home() {
        let dir = AppConfig::config_dir().expect("config_dir");
        assert!(dir.ends_with(".remembrant"));
    }
}
