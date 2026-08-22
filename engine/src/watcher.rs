//! File watcher for agent artifact directories.
//!
//! Monitors `~/.claude/projects/`, `~/.codex/sessions/`, and `~/.gemini/tmp/`
//! for changes using the `notify` crate with debouncing, and invokes a callback
//! when relevant files (`.json`, `.jsonl`, `MEMORY.md`) are modified.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use anyhow::{Context, Result};
use notify::RecursiveMode;
use notify_debouncer_mini::{DebouncedEvent, new_debouncer};
use tracing::{debug, info, warn};

use crate::config::AppConfig;

/// File watcher that monitors agent artifact directories for changes.
pub struct FileWatcher {
    watch_paths: Vec<PathBuf>,
    debounce_ms: u64,
    active: Arc<AtomicBool>,
}

impl FileWatcher {
    /// Create a new watcher with explicit paths and debounce duration.
    pub fn new(watch_paths: Vec<PathBuf>, debounce_ms: u64) -> Self {
        Self {
            watch_paths,
            debounce_ms,
            active: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Return the configured watch paths.
    pub fn watch_paths(&self) -> &[PathBuf] {
        &self.watch_paths
    }

    /// Return whether a `run` call currently has at least one active watch.
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::SeqCst)
    }

    /// Build watch paths from an [`AppConfig`], returning paths for enabled agents.
    ///
    /// For each enabled agent, the configured base path is expanded (`~` replaced
    /// with the actual home directory) and the relevant subdirectory is appended:
    ///
    /// - Claude Code: `<base>/projects/`
    /// - Codex: `<base>/sessions/`
    /// - Gemini: `<base>/tmp/`
    pub fn from_config(config: &AppConfig) -> Self {
        let mut paths = Vec::new();

        if config.agents.claude_code.enabled {
            let base = expand_tilde(&config.agents.claude_code.path);
            paths.push(PathBuf::from(base).join("projects"));
        }
        if config.agents.codex.enabled {
            let base = expand_tilde(&config.agents.codex.path);
            paths.push(PathBuf::from(base).join("sessions"));
        }
        if config.agents.gemini.enabled {
            let base = expand_tilde(&config.agents.gemini.path);
            paths.push(PathBuf::from(base).join("tmp"));
        }
        for agent in &config.agents.dynamic {
            if agent.enabled && agent.adapter_type == "jsonl" {
                paths.push(PathBuf::from(expand_tilde(&agent.path)));
            }
        }

        Self {
            watch_paths: paths,
            debounce_ms: config.watch.debounce_ms,
            active: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Start watching. Calls `on_change` when relevant files change.
    ///
    /// This blocks the current thread. The watcher runs until an error occurs
    /// or the channel is closed.
    pub fn run<F>(&self, on_change: F) -> Result<()>
    where
        F: Fn(Vec<DebouncedEvent>) + Send + 'static,
    {
        let (tx, rx) = mpsc::channel();

        let debounce_duration = Duration::from_millis(self.debounce_ms);
        let mut debouncer =
            new_debouncer(debounce_duration, tx).context("failed to create file debouncer")?;

        // Register each watch path, skipping directories that don't exist.
        let mut active_watches = 0;
        for path in &self.watch_paths {
            if !path.is_dir() {
                warn!("watch path does not exist, skipping: {}", path.display());
                continue;
            }
            debouncer
                .watcher()
                .watch(path, RecursiveMode::Recursive)
                .with_context(|| format!("failed to watch {}", path.display()))?;
            info!("watching: {}", path.display());
            active_watches += 1;
        }

        if active_watches == 0 {
            warn!("no valid directories to watch");
            self.active.store(false, Ordering::SeqCst);
            return Ok(());
        }

        info!(
            "file watcher started ({} directories, debounce={}ms)",
            active_watches, self.debounce_ms
        );
        self.active.store(true, Ordering::SeqCst);

        // Block and process events.
        loop {
            match rx.recv() {
                Ok(Ok(events)) => {
                    // Filter to structured artifacts and agent memory notes.
                    let relevant: Vec<DebouncedEvent> = events
                        .into_iter()
                        .filter(|e| is_relevant_file(&e.path))
                        .collect();

                    if !relevant.is_empty() {
                        debug!("detected {} relevant file change(s)", relevant.len());
                        on_change(relevant);
                    }
                }
                Ok(Err(errors)) => {
                    warn!("watch error: {errors}");
                }
                Err(e) => {
                    info!("watcher channel closed: {e}");
                    break;
                }
            }
        }

        self.active.store(false, Ordering::SeqCst);
        Ok(())
    }
}

/// Returns `true` for structured artifacts or an agent memory file.
fn is_relevant_file(path: &std::path::Path) -> bool {
    if matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("json" | "jsonl")
    ) {
        return true;
    }

    // Agent memory files are Markdown, but recursively treating every README
    // or documentation edit as an ingestion trigger would be noisy.
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("MEMORY.md"))
}

/// Expand a leading `~` in a path string to the user's home directory.
fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(rest).to_string_lossy().to_string();
    }
    if path == "~"
        && let Some(home) = dirs::home_dir()
    {
        return home.to_string_lossy().to_string();
    }
    path.to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_new_creates_watcher() {
        let watcher = FileWatcher::new(vec![PathBuf::from("/tmp")], 5000);
        assert_eq!(watcher.watch_paths.len(), 1);
        assert_eq!(watcher.debounce_ms, 5000);
    }

    #[test]
    fn test_from_config() {
        let config = AppConfig::default();
        let watcher = FileWatcher::from_config(&config);
        // Default config enables all three agents.
        assert_eq!(watcher.watch_paths.len(), 3);
        assert_eq!(watcher.debounce_ms, config.watch.debounce_ms);
    }

    #[test]
    fn test_from_config_disabled_agent() {
        let mut config = AppConfig::default();
        config.agents.codex.enabled = false;
        let watcher = FileWatcher::from_config(&config);
        assert_eq!(watcher.watch_paths.len(), 2);
    }

    #[test]
    fn test_from_config_includes_dynamic_jsonl_agents() {
        let mut config = AppConfig::default();
        config.agents.dynamic = vec![crate::ingest::DynamicAgentConfig {
            id: "custom".into(),
            display_name: "Custom".into(),
            enabled: true,
            path: "/tmp/custom-agent".into(),
            adapter_type: "jsonl".into(),
            sqlite: None,
            jsonl: None,
        }];

        let watcher = FileWatcher::from_config(&config);
        assert!(
            watcher
                .watch_paths()
                .contains(&PathBuf::from("/tmp/custom-agent"))
        );
    }

    #[test]
    fn test_expand_tilde() {
        let expanded = expand_tilde("~/.claude");
        assert!(!expanded.starts_with('~'), "tilde should be expanded");
        assert!(expanded.contains(".claude"));

        // Non-tilde paths pass through.
        assert_eq!(expand_tilde("/absolute/path"), "/absolute/path");
        assert_eq!(expand_tilde("relative/path"), "relative/path");
    }

    #[test]
    fn test_is_relevant_file() {
        assert!(is_relevant_file(std::path::Path::new("session.json")));
        assert!(is_relevant_file(std::path::Path::new("transcript.jsonl")));
        assert!(is_relevant_file(std::path::Path::new("MEMORY.md")));
        assert!(!is_relevant_file(std::path::Path::new("README.md")));
        assert!(!is_relevant_file(std::path::Path::new("data.txt")));
        assert!(!is_relevant_file(std::path::Path::new("noextension")));
    }

    #[test]
    fn test_run_with_no_valid_dirs() {
        let watcher = FileWatcher::new(vec![PathBuf::from("/nonexistent/path/abc123")], 1000);
        // Should return Ok(()) without blocking when no dirs exist.
        let result = watcher.run(|_events| {});
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_detects_jsonl_changes() {
        let directory = TempDir::new().unwrap();
        let artifact = directory.path().join("session.jsonl");
        std::fs::write(&artifact, "{}\n").unwrap();
        let sender_path = std::sync::Arc::new(artifact.clone());
        let (event_tx, event_rx) = std::sync::mpsc::channel();

        let watcher = FileWatcher::new(vec![directory.path().to_path_buf()], 100);
        let watcher_active = Arc::clone(&watcher.active);
        std::thread::spawn(move || {
            let path = sender_path;
            let _ = watcher.run(move |_events| {
                let _ = event_tx.send(path.clone());
            });
        });

        for _ in 0..100 {
            if watcher_active.load(std::sync::atomic::Ordering::SeqCst) {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(
            watcher_active.load(std::sync::atomic::Ordering::SeqCst),
            "watcher did not become active"
        );
        std::fs::write(&artifact, "{\"update\":true}\n").unwrap();
        let changed = event_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("watcher should report the changed artifact");
        assert_eq!(changed.as_path(), artifact.as_path());
    }
}
