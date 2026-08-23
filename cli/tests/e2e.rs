use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn rem(home: &std::path::Path, args: &[&str]) -> Result<std::process::Output> {
    Command::new(env!("CARGO_BIN_EXE_rem"))
        .args(args)
        .env("HOME", home)
        .env("RUST_LOG", "warn")
        .output()
        .with_context(|| format!("failed to run rem {}", args.join(" ")))
}

fn assert_rem_success(home: &std::path::Path, args: &[&str]) -> Result<std::process::Output> {
    let output = rem(home, args)?;
    if !output.status.success() {
        bail!(
            "`rem {}` failed ({})\nstdout:\n{}\nstderr:\n{}",
            args.join(" "),
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
    Ok(output)
}

fn create_claude_fixture(home: &std::path::Path) -> Result<()> {
    let project = home.join(".claude").join("projects").join("e2e-project");
    std::fs::create_dir_all(project.join("memory"))?;

    let now = chrono::Utc::now();
    let created = now.to_rfc3339();
    let modified = (now + chrono::Duration::seconds(45)).to_rfc3339();
    let index = json!({
        "version": 1,
        "entries": [{
            "sessionId": "e2e-session-001",
            "fullPath": project.join("e2e-session-001.jsonl"),
            "firstPrompt": "Refactor authentication",
            "summary": "E2E authentication refactor session",
            "messageCount": 0,
            "created": created,
            "modified": modified,
            "gitBranch": "main",
            "projectPath": "/tmp/e2e-project",
            "isSidechain": false
        }]
    });
    std::fs::write(project.join("sessions-index.json"), index.to_string())?;

    let transcript_lines = [
        json!({
            "type": "user",
            "message": {"role": "user", "content": "Refactor authentication"},
            "timestamp": created
        }),
        json!({
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "tool-write-1",
                    "name": "Write",
                    "input": {"file_path": "src/auth.rs", "content": "pub fn login() {}"}
                }],
                "usage": {"input_tokens": 120, "output_tokens": 80}
            },
            "timestamp": created
        }),
        json!({
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Refactored the auth boundary."}],
                "usage": {"input_tokens": 200, "output_tokens": 140}
            },
            "timestamp": modified
        }),
    ];
    let mut transcript = std::fs::File::create(project.join("e2e-session-001.jsonl"))?;
    for line in transcript_lines {
        writeln!(transcript, "{line}")?;
    }

    std::fs::write(
        project.join("memory").join("MEMORY.md"),
        "# Project memory\n\n- Authentication lives behind the auth boundary.\n",
    )?;
    Ok(())
}

async fn spawn_fake_embedding_server() -> Result<(String, tokio::task::JoinHandle<()>)> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let endpoint = format!("http://127.0.0.1:{}/v1", listener.local_addr()?.port());
    let server = tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                break;
            };
            tokio::spawn(async move {
                let mut request = Vec::new();
                let mut chunk = [0_u8; 4096];
                loop {
                    let read = match socket.read(&mut chunk).await {
                        Ok(0) | Err(_) => break,
                        Ok(read) => read,
                    };
                    request.extend_from_slice(&chunk[..read]);
                    let headers_end = request
                        .windows(b"\r\n\r\n".len())
                        .position(|window| window == b"\r\n\r\n");
                    if let Some(headers_end) = headers_end {
                        let headers = String::from_utf8_lossy(&request[..headers_end]).to_string();
                        let content_length = headers
                            .lines()
                            .find_map(|line| {
                                let (name, value) = line.split_once(':')?;
                                name.eq_ignore_ascii_case("content-length")
                                    .then(|| value.trim().parse::<usize>().ok())
                                    .flatten()
                            })
                            .unwrap_or(0);
                        if request.len() >= headers_end + 4 + content_length {
                            break;
                        }
                    }
                }

                let _ = request;
                let body = json!({
                    "object": "list",
                    "model": "test-embedding",
                    "data": [{
                        "object": "embedding",
                        "index": 0,
                        "embedding": [0.1, -0.2, 0.3, -0.4]
                    }]
                })
                .to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
            });
        }
    });
    Ok((endpoint, server))
}

#[tokio::test]
async fn configured_jsonl_agent_end_to_end_workflow() -> Result<()> {
    let home = tempfile::tempdir()?;
    let artifact_dir = home.path().join("agent-data").join("custom-project");
    std::fs::create_dir_all(&artifact_dir)?;
    std::fs::write(
        artifact_dir.join("session.jsonl"),
        concat!(
            "{\"sessionId\":\"dynamic-session-001\",\"timestamp\":\"2026-08-21T18:00:00Z\",\"kind\":\"message\",\"message\":{\"text\":\"Dynamic adapter works\"}}\n",
            "{\"sessionId\":\"dynamic-session-001\",\"timestamp\":\"2026-08-21T18:01:00Z\",\"kind\":\"tool\",\"tool\":{\"name\":\"Edit\"},\"message\":{\"text\":\"src/custom.rs\"}}\n"
        ),
    )?;

    let config_dir = home.path().join(".remembrant");
    std::fs::create_dir_all(&config_dir)?;
    std::fs::write(
        config_dir.join("config.yaml"),
        format!(
            r#"storage:
  duckdb_path: {home}/.remembrant/remembrant.duckdb
  lancedb_path: {home}/.remembrant/lancedb
agents:
  claude_code:
    enabled: false
    path: {home}/.claude
  codex:
    enabled: false
    path: {home}/.codex
  gemini:
    enabled: false
    path: {home}/.gemini
  dynamic:
    - id: custom_jsonl
      display_name: Custom JSONL Agent
      enabled: true
      path: {home}/agent-data
      adapter_type: jsonl
      jsonl:
        file_pattern: "**/*.jsonl"
        session_id_path: sessionId
        timestamp_path: timestamp
        content_path: message.text
        tool_name_path: tool.name
        tool_call_type: kind=tool
"#,
            home = home.path().display()
        ),
    )?;

    let initialized = assert_rem_success(home.path(), &["init"])?;
    let stdout = String::from_utf8_lossy(&initialized.stdout);
    assert!(stdout.contains("Custom JSONL Agent"), "{stdout}");
    assert!(stdout.contains("1 sessions"), "{stdout}");

    let recent = assert_rem_success(home.path(), &["recent", "--limit", "10"])?;
    let recent_stdout = String::from_utf8_lossy(&recent.stdout);
    assert!(
        recent_stdout.contains("dynamic-session-001"),
        "{recent_stdout}"
    );
    let row = recent_stdout
        .lines()
        .find(|line| line.contains("dynamic-session-001"))
        .context("dynamic session row missing")?;
    let columns: Vec<_> = row.split_whitespace().collect();
    assert_eq!(columns[1], "custom_jsonl", "{recent_stdout}");
    assert_eq!(columns[4], "2", "{recent_stdout}");
    assert_eq!(columns[5], "1", "{recent_stdout}");

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cli_and_dashboard_end_to_end_workflow() -> Result<()> {
    let home = tempfile::tempdir()?;
    create_claude_fixture(home.path())?;

    let ingest = assert_rem_success(home.path(), &["ingest", "--skip-embed", "--skip-distill"])?;
    let ingest_stdout = String::from_utf8_lossy(&ingest.stdout);
    assert!(ingest_stdout.contains("1 sessions"), "{ingest_stdout}");

    assert!(
        home.path()
            .join(".remembrant")
            .join("config.yaml")
            .is_file()
    );
    assert!(
        home.path()
            .join(".remembrant")
            .join("remembrant.duckdb")
            .is_file()
    );

    let recent = assert_rem_success(home.path(), &["recent", "--limit", "10"])?;
    let recent_stdout = String::from_utf8_lossy(&recent.stdout);
    assert!(recent_stdout.contains("e2e-session-001"), "{recent_stdout}");

    let recent_by_project = assert_rem_success(
        home.path(),
        &["recent", "--limit", "10", "--project", "e2e-project"],
    )?;
    assert!(String::from_utf8_lossy(&recent_by_project.stdout).contains("e2e-session-001"));
    let recent_wrong_agent = assert_rem_success(
        home.path(),
        &["recent", "--limit", "10", "--agent", "codex"],
    )?;
    assert!(!String::from_utf8_lossy(&recent_wrong_agent.stdout).contains("e2e-session-001"));

    let note = assert_rem_success(
        home.path(),
        &[
            "note",
            "Keep the authentication boundary explicit",
            "--project",
            "e2e-project",
            "--tag",
            "auth-pattern",
        ],
    )?;
    assert!(String::from_utf8_lossy(&note.stdout).contains("Note saved:"));
    let tagged = assert_rem_success(home.path(), &["find", "auth-pattern"])?;
    let tagged_stdout = String::from_utf8_lossy(&tagged.stdout);
    assert!(
        tagged_stdout.contains("Keep the authentication boundary explicit"),
        "{tagged_stdout}"
    );

    let search = assert_rem_success(home.path(), &["search", "authentication", "--json"])?;
    let search_value: Value = serde_json::from_slice(&search.stdout)?;
    assert!(search_value["count"].as_u64().unwrap_or(0) >= 1);
    assert!(search_value["results"].as_array().is_some_and(|results| {
        results
            .iter()
            .any(|result| result["content"] == "E2E authentication refactor session")
    }));

    let filtered = assert_rem_success(
        home.path(),
        &[
            "search",
            "authentication",
            "--json",
            "--project",
            "e2e-project",
            "--agent",
            "claude_code",
            "--type",
            "session",
            "--since",
            "1d",
        ],
    )?;
    let filtered_value: Value = serde_json::from_slice(&filtered.stdout)?;
    assert_eq!(filtered_value["count"].as_u64(), Some(1));

    let exact = assert_rem_success(
        home.path(),
        &[
            "search",
            "E2E authentication refactor session",
            "--exact",
            "--json",
            "--content-type",
            "session",
        ],
    )?;
    let exact_value: Value = serde_json::from_slice(&exact.stdout)?;
    assert_eq!(exact_value["count"].as_u64(), Some(1));

    let invalid_since = rem(
        home.path(),
        &["search", "authentication", "--since", "not-a-date"],
    )?;
    assert!(!invalid_since.status.success());
    assert!(String::from_utf8_lossy(&invalid_since.stderr).contains("invalid --since value"));

    let xpath = assert_rem_success(home.path(), &["xpath", "//Session", "--json"])?;
    let xpath_value: Value = serde_json::from_slice(&xpath.stdout)?;
    assert!(
        xpath_value["count"].as_u64().unwrap_or(0) >= 1,
        "XPath response: {xpath_value}"
    );

    let xpath_tree = assert_rem_success(
        home.path(),
        &[
            "xpath",
            "//Session",
            "--depth",
            "2",
            "--limit",
            "1",
            "--tree",
        ],
    )?;
    assert!(String::from_utf8_lossy(&xpath_tree.stdout).contains("Path: Root -> "));

    // Exercise the remaining default-build read/report commands against the
    // same isolated database.
    let status = assert_rem_success(home.path(), &["status"])?;
    assert!(String::from_utf8_lossy(&status.stdout).contains("Remembrant Status"));

    let stats = assert_rem_success(home.path(), &["stats"])?;
    assert!(String::from_utf8_lossy(&stats.stdout).contains("Sessions:    1"));

    let brief = assert_rem_success(home.path(), &["brief", "--today", "--json"])?;
    let brief_value: Value = serde_json::from_slice(&brief.stdout)?;
    assert_eq!(brief_value["sessions"].as_array().map(Vec::len), Some(1));

    let project_brief = assert_rem_success(
        home.path(),
        &["brief", "--today", "--json", "--project", "e2e-project"],
    )?;
    let project_brief_value: Value = serde_json::from_slice(&project_brief.stdout)?;
    assert_eq!(
        project_brief_value["sessions"].as_array().map(Vec::len),
        Some(1)
    );

    let agent_brief = assert_rem_success(
        home.path(),
        &["brief", "--for-agent", "--json", "--max-tokens", "500"],
    )?;
    let agent_brief_value: Value = serde_json::from_slice(&agent_brief.stdout)?;
    assert!(agent_brief_value.is_object());

    let context = assert_rem_success(
        home.path(),
        &[
            "context",
            "authentication",
            "--project",
            "e2e-project",
            "--json",
            "--max-tokens",
            "500",
        ],
    )?;
    let context_value: Value = serde_json::from_slice(&context.stdout)?;
    assert!(context_value.is_object());

    let consolidated = assert_rem_success(
        home.path(),
        &["consolidate", "--project", "e2e-project", "--json"],
    )?;
    let consolidated_value: Value = serde_json::from_slice(&consolidated.stdout)?;
    assert!(consolidated_value["decay_scores"].is_array());

    let patterns = assert_rem_success(home.path(), &["patterns", "authentication"])?;
    assert!(String::from_utf8_lossy(&patterns.stdout).contains("No patterns found."));

    let decisions = assert_rem_success(home.path(), &["decisions", "--all"])?;
    assert!(String::from_utf8_lossy(&decisions.stdout).contains("No decisions found."));

    let related = assert_rem_success(home.path(), &["related", "src/auth.rs"])?;
    assert!(String::from_utf8_lossy(&related.stdout).contains("Related to: src/auth.rs"));

    let graph = assert_rem_success(home.path(), &["graph", "src/auth.rs"])?;
    assert!(String::from_utf8_lossy(&graph.stdout).contains("[USED_IN]"));

    let timeline_since = (chrono::Utc::now() - chrono::Duration::days(1)).to_rfc3339();
    let timeline = assert_rem_success(
        home.path(),
        &["timeline", "authentication", "--since", &timeline_since],
    )?;
    assert!(String::from_utf8_lossy(&timeline.stdout).contains("Timeline: authentication"));

    let export_path = home.path().join("export.json");
    let exported = assert_rem_success(
        home.path(),
        &[
            "export",
            "--project",
            "e2e-project",
            "--format",
            "json",
            "--output",
            export_path.to_str().context("export path UTF-8")?,
        ],
    )?;
    assert!(String::from_utf8_lossy(&exported.stdout).contains("Exported to"));
    let export_value: Value = serde_json::from_str(&std::fs::read_to_string(&export_path)?)?;
    assert_eq!(export_value["sessions"].as_array().map(Vec::len), Some(1));

    let markdown = assert_rem_success(
        home.path(),
        &["export", "--project", "e2e-project", "--format", "markdown"],
    )?;
    assert!(String::from_utf8_lossy(&markdown.stdout).contains("# Project Memory"));

    let gc = assert_rem_success(home.path(), &["gc"])?;
    assert!(String::from_utf8_lossy(&gc.stdout).contains("Deleted 0 sessions"));

    let invalid_threshold = rem(
        home.path(),
        &[
            "consolidate",
            "--project",
            "e2e-project",
            "--threshold",
            "1.5",
        ],
    )?;
    assert!(!invalid_threshold.status.success());
    assert!(String::from_utf8_lossy(&invalid_threshold.stderr).contains("between 0.0 and 1.0"));

    let invalid_export = rem(home.path(), &["export", "--format", "yaml"])?;
    assert!(!invalid_export.status.success());
    assert!(String::from_utf8_lossy(&invalid_export.stderr).contains("unsupported export format"));

    let empty_note = rem(home.path(), &["note", "   "])?;
    assert!(!empty_note.status.success());
    assert!(String::from_utf8_lossy(&empty_note.stderr).contains("cannot be empty"));

    // Exercise repository embedding through a real local OpenAI-compatible
    // HTTP endpoint, including forced replacement via --update.
    let (embedding_endpoint, embedding_server) = spawn_fake_embedding_server().await?;
    let config_path = home.path().join(".remembrant").join("config.yaml");
    let config_text = std::fs::read_to_string(&config_path)?;
    let config_text = config_text
        .replace(
            "endpoint: http://localhost:1234/v1",
            &format!("endpoint: {embedding_endpoint}"),
        )
        .replace("dimensions: 768", "dimensions: 4");
    std::fs::write(&config_path, config_text)?;

    let repository = home.path().join("embedded-repo");
    std::fs::create_dir_all(&repository)?;
    std::fs::write(
        repository.join("main.rs"),
        "fn main() { println!(\"embedding e2e\"); }\n",
    )?;
    let repository = repository.to_string_lossy().to_string();
    let embedded = assert_rem_success(home.path(), &["embed", &repository])?;
    assert!(
        String::from_utf8_lossy(&embedded.stdout).contains("Chunks: 1 created, 1 embedded"),
        "embedding output missing"
    );

    std::fs::write(
        home.path().join("embedded-repo").join("main.rs"),
        "fn main() { println!(\"embedding changed\"); }\n",
    )?;
    let updated_embeddings = assert_rem_success(home.path(), &["embed", &repository, "--update"])?;
    assert!(
        String::from_utf8_lossy(&updated_embeddings.stdout)
            .contains("Chunks: 1 created, 1 embedded"),
        "updated embedding output missing"
    );
    embedding_server.abort();

    // Exercise the real event-driven watcher path, then shut it down with the
    // same SIGTERM mechanism used by `rem stop`.
    let mut watcher = Command::new(env!("CARGO_BIN_EXE_rem"))
        .args(["watch"])
        .env("HOME", home.path())
        .env("RUST_LOG", "warn")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let stdout = watcher.stdout.take().context("watcher stdout missing")?;
    let stderr = watcher.stderr.take().context("watcher stderr missing")?;
    let (line_tx, line_rx) = mpsc::channel::<String>();
    let stderr_line_tx = line_tx.clone();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if line_tx.send(line).is_err() {
                break;
            }
        }
    });
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if stderr_line_tx.send(format!("[stderr] {line}")).is_err() {
                break;
            }
        }
    });

    let mut ready = false;
    for _ in 0..100 {
        if let Ok(line) = line_rx.recv_timeout(Duration::from_millis(200))
            && line.contains("Press Ctrl+C to stop")
            && line.contains("Event watching 1 native artifact director")
        {
            ready = true;
            break;
        }
    }
    assert!(ready, "watcher did not start");
    // Give the recursive filesystem watcher time to finish registration after
    // the daemon's readiness line is printed.
    tokio::time::sleep(Duration::from_secs(1)).await;

    let append_timestamp = (chrono::Utc::now() + chrono::Duration::seconds(90)).to_rfc3339();
    let append_event = json!({
        "type": "user",
        "message": {"role": "user", "content": "Watch this authentication update"},
        "timestamp": append_timestamp
    });
    let transcript_path = home
        .path()
        .join(".claude")
        .join("projects")
        .join("e2e-project")
        .join("e2e-session-001.jsonl");
    {
        let mut transcript = std::fs::read_to_string(&transcript_path)?;
        transcript.push('\n');
        transcript.push_str(&append_event.to_string());
        transcript.push('\n');
        std::fs::write(&transcript_path, transcript)?;
    }

    let mut ingested = false;
    for _ in 0..100 {
        if let Ok(line) = line_rx.recv_timeout(Duration::from_millis(200))
            && line.contains("ingested 1 changed session(s)")
        {
            ingested = true;
            break;
        }
    }
    let watcher_diagnostics: Vec<_> = line_rx.try_iter().collect();
    assert!(
        ingested,
        "watcher did not ingest the transcript update; output so far: {watcher_diagnostics:?}"
    );

    let duplicate_watch = rem(home.path(), &["watch"])?;
    assert!(!duplicate_watch.status.success());
    assert!(String::from_utf8_lossy(&duplicate_watch.stderr).contains("already running"));

    // Reap the child concurrently so `process_is_running` does not mistake a
    // briefly unreaped zombie for a still-active watcher.
    let watcher_wait = std::thread::spawn(move || watcher.wait());
    let stopped = assert_rem_success(home.path(), &["stop"])?;
    assert!(String::from_utf8_lossy(&stopped.stdout).contains("Daemon stopped"));
    let watcher_status = watcher_wait
        .join()
        .map_err(|_| anyhow::anyhow!("watcher wait thread panicked"))??;
    assert!(watcher_status.success());
    assert!(
        !home.path().join(".remembrant").join("daemon.pid").exists(),
        "watcher should remove its PID file on SIGTERM"
    );

    let updated_recent = assert_rem_success(home.path(), &["recent", "--limit", "10"])?;
    let updated_recent_stdout = String::from_utf8_lossy(&updated_recent.stdout);
    let updated_row = updated_recent_stdout
        .lines()
        .find(|line| line.contains("e2e-session-001"))
        .context("updated session row missing")?;
    let updated_columns: Vec<_> = updated_row.split_whitespace().collect();
    assert_eq!(updated_columns[4], "4", "{updated_recent_stdout}");
    assert_eq!(updated_columns[5], "1", "{updated_recent_stdout}");

    // Start the dashboard on an OS-selected free port so parallel tests cannot collide.
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();
    drop(listener);
    let mut server = Command::new(env!("CARGO_BIN_EXE_rem"))
        .args(["web", "--port", &port.to_string()])
        .env("HOME", home.path())
        .env("RUST_LOG", "warn")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;

    let base = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;

    let mut healthy = false;
    for _ in 0..100 {
        if let Ok(response) = client.get(&base).send().await {
            healthy = response.status().is_success();
            if healthy {
                break;
            }
        }
        if let Some(status) = server.try_wait()? {
            bail!("dashboard exited early with status {status}");
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(healthy, "dashboard did not become healthy");

    let index_response = client.get(&base).send().await?.error_for_status()?;
    assert_eq!(
        index_response
            .headers()
            .get(reqwest::header::CACHE_CONTROL)
            .and_then(|value| value.to_str().ok()),
        Some("no-store")
    );
    let index_html = index_response.text().await?;
    assert!(index_html.contains("Remembrant"));
    assert!(index_html.contains("/assets/dashboard.css"));
    assert!(index_html.contains("/assets/dashboard.js"));
    for integrity in [
        "sha384-8dbf0940c6cca015338166ad7dee823800a2da58dc0dd650d8ec5ccc60376c635e59efba0c284c4c125e99e77b667bc9",
        "sha384-d45b78fc94cd6eded2ff8e15d218a6a5c8f1f987d0d5ae96e9260124cafde1b6ba8f06615a85a82da66c326b4920b82a",
        "sha384-b2c5333d05e32c72cedfd6bd022bb0c679f20f122f38c531110a6a7eb794696ee32c4743017885e94a286513b9dc5e99",
    ] {
        assert!(
            index_html.contains(integrity),
            "missing CDN integrity {integrity}"
        );
    }

    let css_response = client
        .get(format!("{base}/assets/dashboard.css"))
        .send()
        .await?
        .error_for_status()?;
    assert!(
        css_response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .is_some_and(|value| value
                .to_str()
                .is_ok_and(|value| value.starts_with("text/css")))
    );
    assert_eq!(
        css_response
            .headers()
            .get(reqwest::header::CACHE_CONTROL)
            .and_then(|value| value.to_str().ok()),
        Some("no-cache")
    );
    let js_response = client
        .get(format!("{base}/assets/dashboard.js"))
        .send()
        .await?
        .error_for_status()?;
    assert!(
        js_response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .is_some_and(|value| value
                .to_str()
                .is_ok_and(|value| value.starts_with("application/javascript")))
    );
    assert_eq!(
        js_response
            .headers()
            .get(reqwest::header::CACHE_CONTROL)
            .and_then(|value| value.to_str().ok()),
        Some("no-cache")
    );

    let stats: Value = client
        .get(format!("{base}/api/stats"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(stats["sessions"].as_u64(), Some(1));
    assert_eq!(stats["today"]["sessions"].as_u64(), Some(1));
    assert_eq!(stats["projects"].as_u64(), Some(1));
    assert_eq!(stats["facts"].as_u64(), Some(0));
    assert_eq!(stats["active_facts"].as_u64(), Some(0));

    let sessions: Value = client
        .get(format!("{base}/api/sessions"))
        .query(&[("limit", "20")])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(sessions[0]["id"].as_str(), Some("e2e-session-001"));
    assert_eq!(
        sessions[0]["files_changed"][0].as_str(),
        Some("src/auth.rs")
    );

    let bounded_sessions: Value = client
        .get(format!("{base}/api/sessions"))
        .query(&[("limit", "0")])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(bounded_sessions.as_array().map(Vec::len), Some(1));

    let detail: Value = client
        .get(format!("{base}/api/sessions/e2e-session-001"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(detail["session"]["total_tokens"].as_i64(), Some(540));
    assert_eq!(detail["tool_calls"].as_array().map(Vec::len), Some(1));

    let briefing: Value = client
        .get(format!("{base}/api/briefing"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(briefing["metrics"]["sessions"]["current"].as_u64(), Some(1));
    assert_eq!(briefing["metrics"]["tokens"]["current"].as_i64(), Some(540));
    assert_eq!(
        briefing["sparklines"]["sessions"].as_array().map(Vec::len),
        Some(7)
    );
    assert!(
        briefing["top_files"]
            .as_array()
            .is_some_and(|files| files.iter().any(|file| file["file_path"] == "src/auth.rs"))
    );
    // Issue #25: the briefing reports the local calendar date it used and
    // the resolved UTC offset, so the UI can label the period honestly.
    assert!(
        briefing["date"]
            .as_str()
            .is_some_and(|d| d.len() == 10 && d.as_bytes()[4] == b'-')
    );
    assert!(
        briefing["tz_offset"]
            .as_str()
            .is_some_and(|tz| tz.len() == 6
                && (tz.starts_with('+') || tz.starts_with('-'))
                && &tz[3..4] == ":")
    );

    // The unit-tested dashboard helpers must be served alongside the app.
    let util_response = client
        .get(format!("{base}/assets/dashboard.util.js"))
        .send()
        .await?
        .error_for_status()?;
    assert!(
        util_response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("application/javascript"))
    );
    let util_body = util_response.text().await?;
    assert!(util_body.contains("RemembrantUtil"));

    let xpath_api: Value = client
        .get(format!("{base}/api/search/xpath"))
        .query(&[("q", "//Session"), ("limit", "20")])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(xpath_api["count"].as_u64(), Some(1));
    assert!(
        xpath_api["results"]
            .as_array()
            .is_some_and(|results| results.len() == 1)
    );

    let attention: Value = client
        .get(format!("{base}/api/attention"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert!(attention["items"].is_array());

    let note: Value = client
        .post(format!("{base}/api/notes"))
        .json(&json!({
            "text": "E2E editable note",
            "project": "e2e-project",
            "tags": ["e2e-tag"]
        }))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let note_id = note["id"].as_str().context("note id missing")?.to_string();
    assert_eq!(note["tags"][0], "e2e-tag");

    let empty_api_note = client
        .post(format!("{base}/api/notes"))
        .json(&json!({"text": "   "}))
        .send()
        .await?;
    assert_eq!(empty_api_note.status(), reqwest::StatusCode::BAD_REQUEST);

    let memories: Value = client
        .get(format!("{base}/api/memories"))
        .query(&[("project", "e2e-project")])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert!(memories.as_array().is_some_and(|items| {
        items
            .iter()
            .any(|memory| memory["id"] == note_id && memory["content"] == "E2E editable note")
    }));

    let tagged_memories: Value = client
        .get(format!("{base}/api/memories"))
        .query(&[("tag", "e2e-tag"), ("project", "e2e-project")])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert!(
        tagged_memories
            .as_array()
            .is_some_and(|items| items.iter().any(|memory| memory["id"] == note_id))
    );

    let updated_response = client
        .put(format!("{base}/api/memories/{note_id}"))
        .json(&json!({
            "content": "E2E edited note",
            "confidence": 0.75,
            "tags": ["edited-tag"]
        }))
        .send()
        .await?;
    let updated_status = updated_response.status();
    let updated_body = updated_response.text().await?;
    assert!(
        updated_status.is_success(),
        "memory update failed ({updated_status}): {updated_body}"
    );
    let updated: Value = serde_json::from_str(&updated_body)?;
    assert_eq!(updated["ok"].as_bool(), Some(true));

    let updated_detail_response = client
        .get(format!("{base}/api/memories/{note_id}"))
        .send()
        .await?;
    let updated_detail_status = updated_detail_response.status();
    let updated_detail_body = updated_detail_response.text().await?;
    assert!(
        updated_detail_status.is_success(),
        "memory detail failed ({updated_detail_status}): {updated_detail_body}"
    );
    let updated_detail: Value = serde_json::from_str(&updated_detail_body)?;
    assert_eq!(updated_detail["tags"][0], "edited-tag");

    let empty_api_decision = client
        .post(format!("{base}/api/decisions"))
        .json(&json!({"what": "   "}))
        .send()
        .await?;
    assert_eq!(
        empty_api_decision.status(),
        reqwest::StatusCode::BAD_REQUEST
    );

    let deleted: Value = client
        .delete(format!("{base}/api/memories/{note_id}"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(deleted["ok"].as_bool(), Some(true));
    let missing = client
        .get(format!("{base}/api/memories/{note_id}"))
        .send()
        .await?;
    assert_eq!(missing.status(), reqwest::StatusCode::NOT_FOUND);

    let decision: Value = client
        .post(format!("{base}/api/decisions"))
        .json(&json!({
            "what": "Use UTC day boundaries",
            "why": "Stored timestamps are UTC",
            "alternatives": ["local boundaries"],
            "project": "e2e-project"
        }))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert!(decision["id"].is_string());

    let decisions: Value = client
        .get(format!("{base}/api/decisions"))
        .query(&[("project", "e2e-project")])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert!(decisions.as_array().is_some_and(|items| {
        items
            .iter()
            .any(|item| item["what"] == "Use UTC day boundaries")
    }));

    // Issue #28: memories/facts/decisions accept repeatable ?agent= filters.
    // The ingested session belongs to claude_code; the POSTed note/decision
    // have no source session, so a codex/gemini filter must exclude them.
    let memories_other_agent: Value = client
        .get(format!("{base}/api/memories"))
        .query(&[("agent", "codex,gemini")])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(memories_other_agent.as_array().map(Vec::len), Some(0));

    let memories_ingesting_agent: Value = client
        .get(format!("{base}/api/memories"))
        .query(&[("agent", "claude_code")])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert!(memories_ingesting_agent.is_array());

    let decisions_other_agent: Value = client
        .get(format!("{base}/api/decisions"))
        .query(&[("agent", "codex")])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(decisions_other_agent.as_array().map(Vec::len), Some(0));

    let decisions_unfiltered: Value = client
        .get(format!("{base}/api/decisions"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert!(
        decisions_unfiltered
            .as_array()
            .is_some_and(|items| !items.is_empty())
    );

    let facts_filtered: Value = client
        .get(format!("{base}/api/facts"))
        .query(&[("agent", "codex")])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert!(facts_filtered.is_array());

    server.kill()?;
    server.wait()?;

    let nonempty_decisions = assert_rem_success(
        home.path(),
        &["decisions", "--project", "e2e-project", "--all"],
    )?;
    assert!(String::from_utf8_lossy(&nonempty_decisions.stdout).contains("Use UTC day boundaries"));

    // Cover the MCP stdio transport end to end: modern discovery, protocol
    // negotiation, legacy compatibility, deterministic tool discovery, and a
    // mutating tool call against the same isolated database.
    let mut mcp = Command::new(env!("CARGO_BIN_EXE_rem"))
        .args(["mcp"])
        .env("HOME", home.path())
        .env("RUST_LOG", "warn")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    let mut mcp_stdin = mcp.stdin.take().context("MCP stdin missing")?;
    let mut mcp_stdout = BufReader::new(mcp.stdout.take().context("MCP stdout missing")?);

    let read_mcp_response = |reader: &mut BufReader<std::process::ChildStdout>| -> Result<Value> {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            bail!("MCP server closed stdout before responding");
        }
        serde_json::from_str(line.trim()).context("invalid MCP JSON-RPC response")
    };

    let modern_meta = json!({
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {}
    });
    writeln!(
        mcp_stdin,
        "{}",
        json!({"jsonrpc": "1.0", "id": 0, "method": "ping"})
    )?;
    let invalid_jsonrpc = read_mcp_response(&mut mcp_stdout)?;
    assert_eq!(invalid_jsonrpc["id"], 0);
    assert_eq!(invalid_jsonrpc["error"]["code"], -32600);

    writeln!(
        mcp_stdin,
        "{}",
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "server/discover",
            "params": {"_meta": modern_meta}
        })
    )?;
    let discovery = read_mcp_response(&mut mcp_stdout)?;
    let discovery_result = &discovery["result"];
    assert_eq!(discovery_result["supportedVersions"][0], "2026-07-28");
    assert_eq!(discovery_result["resultType"], "complete");
    assert_eq!(discovery_result["cacheScope"], "public");
    assert!(
        discovery_result["ttlMs"]
            .as_u64()
            .is_some_and(|ttl| ttl > 0)
    );

    writeln!(
        mcp_stdin,
        "{}",
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "server/discover",
            "params": {
                "_meta": {
                    "io.modelcontextprotocol/protocolVersion": "1900-01-01",
                    "io.modelcontextprotocol/clientCapabilities": {}
                }
            }
        })
    )?;
    let unsupported = read_mcp_response(&mut mcp_stdout)?;
    assert_eq!(unsupported["error"]["code"], -32022);
    assert_eq!(unsupported["error"]["data"]["supported"][0], "2026-07-28");

    writeln!(
        mcp_stdin,
        "{}",
        json!({
            "jsonrpc": "2.0",
            "id": 99,
            "method": "tools/list",
            "params": {
                "_meta": {
                    "io.modelcontextprotocol/protocolVersion": "2026-07-28"
                }
            }
        })
    )?;
    let invalid_metadata = read_mcp_response(&mut mcp_stdout)?;
    assert_eq!(invalid_metadata["id"], 99);
    assert_eq!(invalid_metadata["error"]["code"], -32602);

    writeln!(
        mcp_stdin,
        "{}",
        json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "legacy-client", "version": "1.0"}
            }
        })
    )?;
    let initialize = read_mcp_response(&mut mcp_stdout)?;
    assert_eq!(initialize["result"]["protocolVersion"], "2024-11-05");

    // `initialized` is a notification and MUST not produce a JSON-RPC response.
    writeln!(
        mcp_stdin,
        "{}",
        json!({"jsonrpc": "2.0", "method": "initialized"})
    )?;
    writeln!(
        mcp_stdin,
        "{}",
        json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/list",
            "params": {}
        })
    )?;
    let legacy_tools = read_mcp_response(&mut mcp_stdout)?;
    assert_eq!(legacy_tools["id"], 4);
    assert!(legacy_tools["result"]["tools"].as_array().is_some());

    writeln!(
        mcp_stdin,
        "{}",
        json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/list",
            "params": {"_meta": modern_meta}
        })
    )?;
    let modern_tools = read_mcp_response(&mut mcp_stdout)?;
    let modern_tool_result = &modern_tools["result"];
    assert_eq!(modern_tool_result["resultType"], "complete");
    assert_eq!(modern_tool_result["cacheScope"], "public");
    assert!(
        modern_tool_result["ttlMs"]
            .as_u64()
            .is_some_and(|ttl| ttl > 0)
    );
    let tool_names: Vec<&str> = modern_tool_result["tools"]
        .as_array()
        .context("MCP tools array missing")?
        .iter()
        .filter_map(|tool| tool["name"].as_str())
        .collect();
    let mut sorted_tool_names = tool_names.clone();
    sorted_tool_names.sort_unstable();
    assert_eq!(tool_names, sorted_tool_names);

    writeln!(
        mcp_stdin,
        "{}",
        json!({
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "mem_add",
                    "arguments": {
                        "type": "note",
                        "text": "MCP E2E token is durable",
                        "project": "e2e-project"
                },
                "_meta": modern_meta
            },
        })
    )?;
    let tool_call = read_mcp_response(&mut mcp_stdout)?;
    assert_eq!(tool_call["result"]["resultType"], "complete");
    assert_eq!(tool_call["result"]["isError"], Value::Null);
    assert!(
        tool_call["result"]["content"][0]["text"]
            .as_str()
            .is_some_and(|text| text.contains("MCP E2E token is durable"))
    );

    drop(mcp_stdin);
    let mcp_status = mcp.wait()?;
    assert!(mcp_status.success());
    let durable = assert_rem_success(home.path(), &["find", "MCP E2E token"])?;
    assert!(String::from_utf8_lossy(&durable.stdout).contains("MCP E2E token"));

    let forgotten = assert_rem_success(home.path(), &["forget", "--session", "e2e-session-001"])?;
    assert!(String::from_utf8_lossy(&forgotten.stdout).contains("Session e2e-session-001 deleted"));
    let recent_after_forget = assert_rem_success(home.path(), &["recent", "--limit", "10"])?;
    assert!(
        !String::from_utf8_lossy(&recent_after_forget.stdout).contains("e2e-session-001"),
        "forgotten session is still displayed"
    );
    let duplicate_forget = rem(home.path(), &["forget", "--session", "e2e-session-001"])?;
    assert!(!duplicate_forget.status.success());
    assert!(String::from_utf8_lossy(&duplicate_forget.stderr).contains("not found"));

    Ok(())
}
