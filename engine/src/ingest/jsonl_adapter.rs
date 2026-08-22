//! Configuration-driven adapter for newline-delimited JSON agent artifacts.
//!
//! Unlike the native adapters, this parser makes no assumptions about field
//! names. A YAML mapping describes where session IDs, timestamps, message
//! content, and tool calls live in each line.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use glob::Pattern;
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::adapter::{AgentAdapter, AgentMeta, DynamicAgentConfig, IngestOutput, JsonlMapping};
use crate::store::duckdb::{Session, ToolCall};

/// A generic JSONL adapter configured entirely from YAML.
#[derive(Debug)]
pub struct GenericJsonlAdapter {
    meta: AgentMeta,
    base_path: PathBuf,
    mapping: JsonlMapping,
    file_pattern: Pattern,
}

impl GenericJsonlAdapter {
    /// Build an adapter from a dynamic-agent definition.
    pub fn from_config(config: &DynamicAgentConfig) -> Result<Self> {
        let mapping = config.jsonl.as_ref().context("missing jsonl mapping")?;
        let file_pattern = Pattern::new(&mapping.file_pattern)
            .with_context(|| format!("invalid JSONL file pattern '{}'", mapping.file_pattern))?;
        if mapping.session_id_path.trim().is_empty() {
            anyhow::bail!("dynamic agent '{}' has an empty session_id_path", config.id);
        }
        if let Some(specification) = mapping.tool_call_type.as_deref()
            && specification
                .split_once('=')
                .is_none_or(|(field, value)| field.trim().is_empty() || value.is_empty())
        {
            anyhow::bail!(
                "dynamic agent '{}' has invalid tool_call_type '{}'; expected 'field=value'",
                config.id,
                specification
            );
        }

        Ok(Self {
            meta: AgentMeta {
                id: config.id.clone(),
                display_name: config.display_name.clone(),
                storage_format: "jsonl".into(),
                default_path: config.path.clone(),
            },
            base_path: super::adapter::expand_tilde(&config.path),
            mapping: mapping.clone(),
            file_pattern,
        })
    }

    fn matched_files(&self) -> Vec<PathBuf> {
        if !self.base_path.is_dir() {
            return Vec::new();
        }

        walk_files(&self.base_path)
            .into_iter()
            .filter(|path| {
                path.strip_prefix(&self.base_path)
                    .ok()
                    .and_then(|relative| relative.to_str())
                    .is_some_and(|relative| self.file_pattern.matches(relative))
            })
            .collect()
    }

    fn parse_files(&self, files: &[PathBuf]) -> IngestOutput {
        let mut sessions: HashMap<String, Session> = HashMap::new();
        let mut tool_calls = Vec::new();
        let mut errors = Vec::new();

        for path in files {
            let contents = match fs::read_to_string(path) {
                Ok(contents) => contents,
                Err(error) => {
                    errors.push(format!("{}: {error}", path.display()));
                    continue;
                }
            };
            let project_id = project_from_path(&self.base_path, path);

            for (line_number, line) in contents.lines().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                let event: Value = match serde_json::from_str(line) {
                    Ok(event) => event,
                    Err(error) => {
                        errors.push(format!("{}:{}: {error}", path.display(), line_number + 1));
                        continue;
                    }
                };

                let Some(session_id) = json_path(&event, &self.mapping.session_id_path)
                    .and_then(value_text)
                    .filter(|id| !id.is_empty())
                else {
                    errors.push(format!(
                        "{}:{}: missing session ID at '{}'",
                        path.display(),
                        line_number + 1,
                        self.mapping.session_id_path
                    ));
                    continue;
                };

                let timestamp = self
                    .mapping
                    .timestamp_path
                    .as_deref()
                    .and_then(|path| json_path(&event, path))
                    .and_then(parse_json_timestamp);
                let content = self
                    .mapping
                    .content_path
                    .as_deref()
                    .and_then(|path| json_path(&event, path))
                    .and_then(value_text)
                    .filter(|content| !content.trim().is_empty());
                let is_tool_call = self.is_tool_call(&event);

                let session = sessions
                    .entry(session_id.clone())
                    .or_insert_with(|| Session {
                        id: session_id.clone(),
                        project_id: project_id.clone(),
                        agent: self.meta.id.clone(),
                        started_at: timestamp,
                        ended_at: timestamp,
                        duration_minutes: None,
                        message_count: Some(0),
                        tool_call_count: Some(0),
                        total_tokens: None,
                        files_changed: Vec::new(),
                        summary: None,
                    });

                session.message_count = session.message_count.map(|count| count + 1);
                if is_tool_call {
                    session.tool_call_count = session.tool_call_count.map(|count| count + 1);
                }
                if let Some(timestamp) = timestamp {
                    if session
                        .started_at
                        .is_none_or(|existing| timestamp < existing)
                    {
                        session.started_at = Some(timestamp);
                    }
                    if session.ended_at.is_none_or(|existing| timestamp > existing) {
                        session.ended_at = Some(timestamp);
                    }
                }
                if session.summary.is_none()
                    && let Some(summary) = &content
                {
                    session.summary = Some(summary.chars().take(240).collect());
                }

                if is_tool_call {
                    let tool_name = self
                        .mapping
                        .tool_name_path
                        .as_deref()
                        .and_then(|path| json_path(&event, path))
                        .and_then(value_text);
                    let relative_source = path
                        .strip_prefix(&self.base_path)
                        .unwrap_or(path)
                        .to_string_lossy();
                    let id = deterministic_tool_id(
                        &self.meta.id,
                        &relative_source,
                        line_number,
                        &session_id,
                    );
                    tool_calls.push(ToolCall {
                        id,
                        session_id: Some(session_id),
                        tool_name,
                        command: content,
                        success: None,
                        error_message: None,
                        duration_ms: None,
                        timestamp,
                    });
                }
            }
        }

        let mut sessions: Vec<Session> = sessions.into_values().collect();
        sessions.sort_by(|a, b| a.id.cmp(&b.id));
        tool_calls.sort_by(|a, b| a.id.cmp(&b.id));

        IngestOutput {
            sessions,
            tool_calls,
            memories: Vec::new(),
            errors,
        }
    }

    fn is_tool_call(&self, event: &Value) -> bool {
        let Some(specification) = self.mapping.tool_call_type.as_deref() else {
            return self.mapping.tool_name_path.is_some()
                && self
                    .mapping
                    .tool_name_path
                    .as_deref()
                    .and_then(|path| json_path(event, path))
                    .is_some();
        };

        let Some((field, expected)) = specification.split_once('=') else {
            return false;
        };
        json_path(event, field)
            .and_then(value_text)
            .is_some_and(|actual| actual == expected)
    }
}

impl AgentAdapter for GenericJsonlAdapter {
    fn meta(&self) -> &AgentMeta {
        &self.meta
    }

    fn detect(&self) -> bool {
        self.matched_files()
            .first()
            .is_some_and(|path| path.is_file())
    }

    fn ingest(&self) -> Result<IngestOutput> {
        Ok(self.parse_files(&self.matched_files()))
    }
}

fn walk_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];

    while let Some(directory) = stack.pop() {
        let Ok(entries) = fs::read_dir(&directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                files.push(path);
            }
        }
    }

    files.sort();
    files
}

fn project_from_path(base: &Path, file: &Path) -> Option<String> {
    let relative = file.strip_prefix(base).ok()?;
    let mut components = relative.parent()?.components();
    components
        .next()
        .map(|component| component.as_os_str().to_string_lossy().to_string())
}

fn json_path<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    path.split('.')
        .try_fold(value, |current, segment| current.get(segment))
}

fn value_text(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(text) => Some(text.clone()),
        Value::Number(number) => Some(number.to_string()),
        Value::Bool(flag) => Some(flag.to_string()),
        Value::Array(items) => {
            let texts: Vec<String> = items.iter().filter_map(value_text).collect();
            (!texts.is_empty()).then(|| texts.join("\n"))
        }
        Value::Object(object) => {
            let texts: Vec<String> = object.values().filter_map(value_text).collect();
            (!texts.is_empty()).then(|| texts.join("\n"))
        }
    }
}

fn parse_json_timestamp(value: &Value) -> Option<chrono::NaiveDateTime> {
    match value {
        Value::String(text) => chrono::DateTime::parse_from_rfc3339(text)
            .ok()
            .map(|timestamp| timestamp.naive_utc())
            .or_else(|| super::parse_iso_timestamp(text)),
        Value::Number(number) => {
            let raw = number.as_f64()?;
            let seconds = if raw > 1_000_000_000_000.0 {
                raw / 1_000.0
            } else {
                raw
            };
            chrono::DateTime::from_timestamp(seconds.floor() as i64, 0)
                .map(|timestamp| timestamp.naive_utc())
        }
        _ => None,
    }
}

fn deterministic_tool_id(
    agent: &str,
    source: &str,
    line_number: usize,
    session_id: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(agent.as_bytes());
    hasher.update([0]);
    hasher.update(source.as_bytes());
    hasher.update([0]);
    hasher.update(line_number.to_le_bytes());
    hasher.update([0]);
    hasher.update(session_id.as_bytes());
    let digest = hasher.finalize();
    format!("jsonl-{}", hex_prefix(&digest))
}

fn hex_prefix(bytes: &[u8]) -> String {
    bytes
        .iter()
        .take(16)
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn adapter_config(path: &Path) -> DynamicAgentConfig {
        DynamicAgentConfig {
            id: "custom_agent".into(),
            display_name: "Custom Agent".into(),
            enabled: true,
            path: path.to_string_lossy().to_string(),
            adapter_type: "jsonl".into(),
            sqlite: None,
            jsonl: Some(JsonlMapping {
                file_pattern: "**/*.jsonl".into(),
                session_id_path: "sessionId".into(),
                timestamp_path: Some("timestamp".into()),
                content_path: Some("message.text".into()),
                tool_name_path: Some("tool.name".into()),
                tool_call_type: Some("kind=tool".into()),
            }),
        }
    }

    #[test]
    fn parses_sessions_and_tool_calls_with_stable_ids() {
        let root = TempDir::new().unwrap();
        let project = root.path().join("checkout");
        fs::create_dir_all(&project).unwrap();
        fs::write(
            project.join("session.jsonl"),
            concat!(
                "{\"sessionId\":\"s1\",\"timestamp\":\"2026-08-01T12:00:00Z\",\"kind\":\"message\",\"message\":{\"text\":\"First prompt\"}}\n",
                "{\"sessionId\":\"s1\",\"timestamp\":\"2026-08-01T12:01:00Z\",\"kind\":\"tool\",\"tool\":{\"name\":\"Edit\"},\"message\":{\"text\":\"src/lib.rs\"}}\n"
            ),
        )
        .unwrap();

        let config = adapter_config(root.path());
        let adapter = GenericJsonlAdapter::from_config(&config).unwrap();
        assert!(adapter.detect());
        let output = adapter.ingest().unwrap();

        assert_eq!(output.session_count(), 1);
        assert_eq!(output.tool_call_count(), 1);
        assert!(output.errors.is_empty());
        let session = &output.sessions[0];
        assert_eq!(session.agent, "custom_agent");
        assert_eq!(session.project_id.as_deref(), Some("checkout"));
        assert_eq!(session.message_count, Some(2));
        assert_eq!(session.tool_call_count, Some(1));
        assert_eq!(session.summary.as_deref(), Some("First prompt"));
        assert_eq!(
            session.started_at.map(|time| time.to_string()).as_deref(),
            Some("2026-08-01 12:00:00")
        );
        assert_eq!(output.tool_calls[0].tool_name.as_deref(), Some("Edit"));
        assert!(output.tool_calls[0].id.starts_with("jsonl-"));

        let rerun = adapter.ingest().unwrap();
        assert_eq!(rerun.tool_calls[0].id, output.tool_calls[0].id);
    }

    #[test]
    fn malformed_lines_are_reported_without_stopping_ingestion() {
        let root = TempDir::new().unwrap();
        fs::create_dir_all(root.path().join("project")).unwrap();
        fs::write(root.path().join("project").join("bad.jsonl"), "{bad\n").unwrap();
        fs::write(
            root.path().join("project").join("good.jsonl"),
            "{\"sessionId\":\"good\"}\n",
        )
        .unwrap();

        let adapter = GenericJsonlAdapter::from_config(&adapter_config(root.path())).unwrap();
        let output = adapter.ingest().unwrap();
        assert_eq!(output.session_count(), 1);
        assert_eq!(output.errors.len(), 1);
        assert!(output.errors[0].contains("bad.jsonl:1"));
    }

    #[test]
    fn invalid_pattern_is_rejected() {
        let root = TempDir::new().unwrap();
        let mut config = adapter_config(root.path());
        config.jsonl.as_mut().unwrap().file_pattern = "[invalid".into();
        assert!(GenericJsonlAdapter::from_config(&config).is_err());
    }

    #[test]
    fn invalid_tool_call_selector_is_rejected() {
        let root = TempDir::new().unwrap();
        let mut config = adapter_config(root.path());
        config.jsonl.as_mut().unwrap().tool_call_type = Some("tool".into());
        let error = GenericJsonlAdapter::from_config(&config).unwrap_err();
        assert!(error.to_string().contains("expected 'field=value'"));
    }

    #[test]
    fn empty_session_path_is_rejected() {
        let root = TempDir::new().unwrap();
        let mut config = adapter_config(root.path());
        config.jsonl.as_mut().unwrap().session_id_path = String::new();
        let error = GenericJsonlAdapter::from_config(&config).unwrap_err();
        assert!(error.to_string().contains("empty session_id_path"));
    }

    #[test]
    fn missing_mapping_is_rejected() {
        let root = TempDir::new().unwrap();
        let mut config = adapter_config(root.path());
        config.jsonl = None;
        let error = GenericJsonlAdapter::from_config(&config).unwrap_err();
        assert!(error.to_string().contains("missing jsonl mapping"));
    }
}
