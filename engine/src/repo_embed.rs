use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Language detection
// ---------------------------------------------------------------------------

/// Languages we can detect by extension.
fn detect_language(path: &Path) -> Option<&'static str> {
    match path.extension()?.to_str()? {
        "rs" => Some("rust"),
        "py" => Some("python"),
        "js" => Some("javascript"),
        "ts" => Some("typescript"),
        "go" => Some("go"),
        "java" => Some("java"),
        "c" | "h" => Some("c"),
        "cpp" | "hpp" | "cc" => Some("cpp"),
        "rb" => Some("ruby"),
        "php" => Some("php"),
        "swift" => Some("swift"),
        "kt" => Some("kotlin"),
        "scala" => Some("scala"),
        "ex" | "exs" => Some("elixir"),
        "hs" => Some("haskell"),
        "ml" | "mli" => Some("ocaml"),
        "lua" => Some("lua"),
        "r" | "R" => Some("r"),
        "tf" | "hcl" => Some("hcl"),
        "sh" | "bash" => Some("bash"),
        "zig" => Some("zig"),
        "dart" => Some("dart"),
        "cs" => Some("csharp"),
        "md" => Some("markdown"),
        "yaml" | "yml" => Some("yaml"),
        "toml" => Some("toml"),
        "json" => Some("json"),
        _ => None,
    }
}

/// Directories to skip during traversal.
const SKIP_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".next",
    ".cache",
    "vendor",
];

/// Returns `true` if the first `n` bytes of `buf` contain a null byte,
/// indicating likely binary content.
fn looks_binary(buf: &[u8]) -> bool {
    buf.contains(&0)
}

/// Stable content-derived chunk ID.
///
/// It includes project, path, line span, and content so a changed file gets a
/// new ID even when AST chunk metadata is unavailable.
fn stable_chunk_id(project_id: &str, chunk: &CodeChunk) -> String {
    let mut hasher = Sha256::new();
    hasher.update(project_id.as_bytes());
    hasher.update([0]);
    hasher.update(chunk.file_path.as_bytes());
    hasher.update([0]);
    hasher.update(chunk.start_line.to_le_bytes());
    hasher.update(chunk.end_line.to_le_bytes());
    hasher.update([0]);
    hasher.update(chunk.content.as_bytes());
    format!("{:x}", hasher.finalize())
}

// ---------------------------------------------------------------------------
// CodeChunk
// ---------------------------------------------------------------------------

/// A chunk of source code ready to be embedded.
#[derive(Debug, Clone)]
pub struct CodeChunk {
    pub file_path: String,
    pub language: Option<String>,
    pub content: String,
    pub start_line: usize,
    pub end_line: usize,
    pub granularity: String, // "file", "chunk", "symbol"
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// EmbedResult
// ---------------------------------------------------------------------------

/// Summary of an embed-and-store operation.
#[derive(Debug, Default)]
pub struct EmbedResult {
    pub files_found: usize,
    pub chunks_created: usize,
    pub chunks_embedded: usize,
    pub errors: usize,
}

// ---------------------------------------------------------------------------
// RepoEmbedder
// ---------------------------------------------------------------------------

/// Walks a repository, chunks source files, and stores embeddings.
pub struct RepoEmbedder {
    root: PathBuf,
    project_id: String,
    max_file_size: usize,
    chunk_size: usize,
    chunk_overlap: usize,
}

impl RepoEmbedder {
    /// Create a new `RepoEmbedder` rooted at the given directory.
    pub fn new(root: impl AsRef<Path>) -> Self {
        let root = root.as_ref().to_path_buf();
        let project_id = root
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        Self {
            root,
            project_id,
            max_file_size: 100_000, // 100 KB
            chunk_size: 50,         // 50 lines per chunk
            chunk_overlap: 10,      // 10 lines overlap
        }
    }

    /// Override the project ID (default: directory name).
    pub fn with_project_id(mut self, id: &str) -> Self {
        self.project_id = id.to_string();
        self
    }

    /// Return the project ID.
    pub fn project_id(&self) -> &str {
        &self.project_id
    }

    // -------------------------------------------------------------------
    // File discovery
    // -------------------------------------------------------------------

    /// Walk the repo and collect all embeddable source files.
    pub fn discover_files(&self) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        self.walk_dir(&self.root, &mut files)?;
        files.sort();
        Ok(files)
    }

    /// Recursive directory walker.
    fn walk_dir(&self, dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        let entries = fs::read_dir(dir)
            .with_context(|| format!("failed to read directory {}", dir.display()))?;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            let file_type = entry.file_type()?;

            if file_type.is_dir() {
                let dir_name = entry.file_name();
                let dir_name_str = dir_name.to_string_lossy();
                if SKIP_DIRS.contains(&dir_name_str.as_ref()) {
                    debug!(dir = %dir_name_str, "skipping directory");
                    continue;
                }
                self.walk_dir(&path, out)?;
            } else if file_type.is_file() {
                // Skip files with no recognised language extension.
                if detect_language(&path).is_none() {
                    continue;
                }

                // Skip files that are too large.
                let meta = fs::metadata(&path)?;
                if meta.len() as usize > self.max_file_size {
                    debug!(path = %path.display(), size = meta.len(), "skipping large file");
                    continue;
                }

                // Skip binary files (check first 512 bytes).
                if self.is_binary(&path)? {
                    debug!(path = %path.display(), "skipping binary file");
                    continue;
                }

                out.push(path);
            }
        }

        Ok(())
    }

    /// Returns `true` if the file appears to be binary.
    fn is_binary(&self, path: &Path) -> Result<bool> {
        let mut file = fs::File::open(path)?;
        let mut buf = [0u8; 512];
        let n = file.read(&mut buf)?;
        Ok(looks_binary(&buf[..n]))
    }

    // -------------------------------------------------------------------

    // -------------------------------------------------------------------
    // Chunking
    // -------------------------------------------------------------------

    /// Naive line-based chunker.
    ///
    /// Splits a file into fixed-size chunks of `self.chunk_size` lines with
    /// `self.chunk_overlap` lines of overlap.
    fn chunk_file_naive(
        &self,
        content: &str,
        rel_path: &str,
        language: Option<String>,
    ) -> Vec<CodeChunk> {
        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        if total_lines == 0 {
            return Vec::new();
        }

        // Small files: single chunk with "file" granularity.
        if total_lines <= self.chunk_size {
            return vec![CodeChunk {
                file_path: rel_path.to_string(),
                language,
                content: content.to_string(),
                start_line: 1,
                end_line: total_lines,
                granularity: "file".to_string(),
            }];
        }

        // Large files: overlapping chunks.
        let mut chunks = Vec::new();
        let step = self.chunk_size.saturating_sub(self.chunk_overlap).max(1);
        let mut start = 0usize;

        while start < total_lines {
            let end = (start + self.chunk_size).min(total_lines);
            let chunk_lines = &lines[start..end];

            // Prepend a header so the embedding model has file context.
            let header = format!("// File: {} (lines {}-{})", rel_path, start + 1, end);
            let mut chunk_content = String::with_capacity(
                header.len() + 1 + chunk_lines.iter().map(|l| l.len() + 1).sum::<usize>(),
            );
            chunk_content.push_str(&header);
            chunk_content.push('\n');
            for line in chunk_lines {
                chunk_content.push_str(line);
                chunk_content.push('\n');
            }

            chunks.push(CodeChunk {
                file_path: rel_path.to_string(),
                language: language.clone(),
                content: chunk_content,
                start_line: start + 1,
                end_line: end,
                granularity: "chunk".to_string(),
            });

            // Advance by step; if we've reached the end, stop.
            if end == total_lines {
                break;
            }
            start += step;
        }

        chunks
    }

    /// Read a file and split into chunks.
    pub fn chunk_file(&self, path: &Path) -> Result<Vec<CodeChunk>> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;

        let rel_path = path
            .strip_prefix(&self.root)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();

        let language = detect_language(path).map(|s| s.to_string());

        if content.is_empty() {
            return Ok(Vec::new());
        }

        Ok(self.chunk_file_naive(&content, &rel_path, language))
    }

    /// Discover files and chunk them all.
    pub fn chunk_all(&self) -> Result<(Vec<CodeChunk>, usize)> {
        self.chunk_all_with_mode(false)
    }

    /// Discover and chunk files, optionally forcing changed-file caches to be
    /// ignored. A forced scan is used by `rem embed --update`.
    pub fn chunk_all_with_mode(&self, _force_rescan: bool) -> Result<(Vec<CodeChunk>, usize)> {
        let files = self.discover_files()?;
        let file_count = files.len();
        info!(files = file_count, root = %self.root.display(), "discovered files");

        let mut all_chunks = Vec::new();

        for file in &files {
            match self.chunk_file(file) {
                Ok(chunks) => all_chunks.extend(chunks),
                Err(e) => {
                    warn!(path = %file.display(), error = %e, "failed to chunk file");
                }
            }
        }

        info!(
            chunks = all_chunks.len(),
            files = file_count,
            "chunking complete"
        );
        Ok((all_chunks, file_count))
    }

    // -------------------------------------------------------------------
    // Embed + store
    // -------------------------------------------------------------------

    /// Embed and store all chunks using the given provider and LanceDB store.
    pub async fn embed_and_store<P: crate::embedding::EmbedProvider>(
        &self,
        provider: &P,
        lance_store: &crate::store::LanceStore,
        batch_size: usize,
    ) -> Result<EmbedResult> {
        self.embed_and_store_with_update(provider, lance_store, batch_size, false)
            .await
    }

    /// Embed and store chunks, optionally replacing all prior embeddings for
    /// this project. The non-update path is idempotent for unchanged chunks.
    pub async fn embed_and_store_with_update<P: crate::embedding::EmbedProvider>(
        &self,
        provider: &P,
        lance_store: &crate::store::LanceStore,
        batch_size: usize,
        update: bool,
    ) -> Result<EmbedResult> {
        let (mut chunks, file_count) = self.chunk_all_with_mode(update)?;
        if !update {
            let existing_ids: std::collections::HashSet<String> = lance_store
                .get_code_embedding_ids_for_project(&self.project_id)
                .await?
                .into_iter()
                .collect();
            chunks.retain(|chunk| {
                let chunk_id = stable_chunk_id(&self.project_id, chunk);
                !existing_ids.contains(&chunk_id)
            });
        }

        let total_chunks = chunks.len();

        let mut result = EmbedResult {
            files_found: file_count,
            chunks_created: total_chunks,
            ..Default::default()
        };

        if chunks.is_empty() {
            return Ok(result);
        }

        // Collect text references for batch embedding.
        let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();

        // Embed in batches.
        let batch_size = batch_size.max(1);
        let n_batches = texts.len().div_ceil(batch_size);
        info!(total_chunks, batch_size, n_batches, "embedding chunks");

        let mut all_vectors: Vec<Vec<f32>> = Vec::with_capacity(total_chunks);

        for (i, text_batch) in texts.chunks(batch_size).enumerate() {
            debug!(batch = i + 1, size = text_batch.len(), "embedding batch");
            match provider.embed_texts(text_batch).await {
                Ok(vectors) => {
                    all_vectors.extend(vectors);
                }
                Err(e) => {
                    warn!(batch = i + 1, error = %e, "batch embedding failed");
                    result.errors += text_batch.len();
                    // Push empty vectors as placeholders so indices stay aligned.
                    for _ in 0..text_batch.len() {
                        all_vectors.push(Vec::new());
                    }
                }
            }
        }

        // Store each chunk with its embedding.
        if update && total_chunks > 0 && all_vectors.iter().any(|vector| !vector.is_empty()) {
            lance_store
                .delete_code_embeddings_for_project(&self.project_id)
                .await?;
        }
        for (idx, chunk) in chunks.iter().enumerate() {
            let embedding = &all_vectors[idx];
            if embedding.is_empty() {
                // Embedding failed for this chunk.
                continue;
            }

            let id = stable_chunk_id(&self.project_id, chunk);

            match lance_store
                .insert_code_embedding(
                    &id,
                    embedding,
                    &chunk.content,
                    &chunk.granularity,
                    &self.project_id,
                    Some(&chunk.file_path),
                    chunk.language.as_deref(),
                )
                .await
            {
                Ok(()) => {
                    result.chunks_embedded += 1;
                }
                Err(e) => {
                    warn!(
                        id = %id,
                        error = %e,
                        "failed to store chunk embedding"
                    );
                    result.errors += 1;
                }
            }
        }

        info!(
            embedded = result.chunks_embedded,
            errors = result.errors,
            "embed-and-store complete"
        );
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_detect_language() {
        assert_eq!(detect_language(Path::new("main.rs")), Some("rust"));
        assert_eq!(detect_language(Path::new("app.py")), Some("python"));
        assert_eq!(detect_language(Path::new("index.js")), Some("javascript"));
        assert_eq!(detect_language(Path::new("lib.ts")), Some("typescript"));
        assert_eq!(detect_language(Path::new("main.go")), Some("go"));
        assert_eq!(detect_language(Path::new("App.java")), Some("java"));
        assert_eq!(detect_language(Path::new("foo.c")), Some("c"));
        assert_eq!(detect_language(Path::new("bar.h")), Some("c"));
        assert_eq!(detect_language(Path::new("baz.cpp")), Some("cpp"));
        assert_eq!(detect_language(Path::new("x.rb")), Some("ruby"));
        assert_eq!(detect_language(Path::new("y.swift")), Some("swift"));
        assert_eq!(detect_language(Path::new("z.zig")), Some("zig"));
        assert_eq!(detect_language(Path::new("config.yaml")), Some("yaml"));
        assert_eq!(detect_language(Path::new("config.yml")), Some("yaml"));
        assert_eq!(detect_language(Path::new("Cargo.toml")), Some("toml"));
        assert_eq!(detect_language(Path::new("data.json")), Some("json"));
        assert_eq!(detect_language(Path::new("README.md")), Some("markdown"));
        // Unknown extension returns None.
        assert_eq!(detect_language(Path::new("image.png")), None);
        assert_eq!(detect_language(Path::new("no_extension")), None);
    }

    #[test]
    fn test_discover_files_skips_ignored_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        // Create some source files.
        fs::write(root.join("main.rs"), "fn main() {}").unwrap();
        fs::write(root.join("lib.py"), "print('hi')").unwrap();

        // Create directories that should be skipped.
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::write(root.join(".git/config"), "gitconfig").unwrap();

        fs::create_dir_all(root.join("node_modules/foo")).unwrap();
        fs::write(
            root.join("node_modules/foo/index.js"),
            "module.exports = {}",
        )
        .unwrap();

        fs::create_dir_all(root.join("target/debug")).unwrap();
        fs::write(root.join("target/debug/main.rs"), "// build artifact").unwrap();

        fs::create_dir_all(root.join("__pycache__")).unwrap();
        fs::write(root.join("__pycache__/mod.py"), "cached").unwrap();

        // Create a nested source file that *should* be found.
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/util.rs"), "pub fn util() {}").unwrap();

        let embedder = RepoEmbedder::new(root);
        let files = embedder.discover_files().unwrap();

        let rel_paths: Vec<String> = files
            .iter()
            .map(|p| p.strip_prefix(root).unwrap().to_string_lossy().to_string())
            .collect();

        assert!(rel_paths.contains(&"main.rs".to_string()));
        assert!(rel_paths.contains(&"lib.py".to_string()));
        assert!(rel_paths.contains(&"src/util.rs".to_string()));

        // Ignored directories should not appear.
        for p in &rel_paths {
            assert!(!p.starts_with(".git"), "should skip .git: {p}");
            assert!(
                !p.starts_with("node_modules"),
                "should skip node_modules: {p}"
            );
            assert!(!p.starts_with("target"), "should skip target: {p}");
            assert!(
                !p.starts_with("__pycache__"),
                "should skip __pycache__: {p}"
            );
        }
    }

    #[test]
    fn test_chunk_small_file() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        // A small file with 10 lines (well under the default chunk_size of 50).
        let content: String = (1..=10).map(|i| format!("line {i}\n")).collect();
        fs::write(root.join("small.rs"), &content).unwrap();

        let embedder = RepoEmbedder::new(root);
        let chunks = embedder.chunk_file(&root.join("small.rs")).unwrap();

        assert_eq!(chunks.len(), 1, "small file should produce exactly 1 chunk");
        assert_eq!(chunks[0].granularity, "file");
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 10);
        assert_eq!(chunks[0].file_path, "small.rs");
        assert_eq!(chunks[0].language.as_deref(), Some("rust"));
    }

    #[test]
    fn test_chunk_large_file() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        // A file with 120 lines (larger than the default chunk_size of 50).
        let content: String = (1..=120).map(|i| format!("line {i}\n")).collect();
        fs::write(root.join("large.py"), &content).unwrap();

        let embedder = RepoEmbedder::new(root);
        let chunks = embedder.chunk_file(&root.join("large.py")).unwrap();

        // With chunk_size=50, overlap=10, step=40:
        // Chunk 1: lines 1-50
        // Chunk 2: lines 41-90
        // Chunk 3: lines 81-120
        assert!(
            chunks.len() >= 3,
            "expected at least 3 chunks, got {}",
            chunks.len()
        );

        // All chunks should have "chunk" granularity.
        for chunk in &chunks {
            assert_eq!(chunk.granularity, "chunk");
            assert_eq!(chunk.language.as_deref(), Some("python"));
        }

        // First chunk should start at line 1.
        assert_eq!(chunks[0].start_line, 1);

        // Last chunk should end at line 120.
        let last = chunks.last().unwrap();
        assert_eq!(last.end_line, 120);

        // Chunks should have the header with file path.
        assert!(
            chunks[0].content.starts_with("// File: large.py"),
            "chunk should start with file path header"
        );

        // Verify overlap: second chunk should start before first chunk ends.
        assert!(
            chunks[1].start_line < chunks[0].end_line,
            "chunks should overlap"
        );
    }

    #[test]
    fn test_empty_file_produces_no_chunks() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        fs::write(root.join("empty.rs"), "").unwrap();

        let embedder = RepoEmbedder::new(root);
        let chunks = embedder.chunk_file(&root.join("empty.rs")).unwrap();

        assert!(chunks.is_empty(), "empty file should produce 0 chunks");
    }

    #[test]
    fn test_binary_file_is_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        // Write a file with null bytes (binary).
        let mut data = b"fn main() {}\0\0\0binary data".to_vec();
        data.extend_from_slice(&[0u8; 100]);
        // Give it a .rs extension so it would be detected by language.
        fs::write(root.join("binary.rs"), &data).unwrap();

        // Also create a valid text file.
        fs::write(root.join("valid.rs"), "fn main() {}").unwrap();

        let embedder = RepoEmbedder::new(root);
        let files = embedder.discover_files().unwrap();

        let rel_paths: Vec<String> = files
            .iter()
            .map(|p| p.strip_prefix(root).unwrap().to_string_lossy().to_string())
            .collect();

        assert!(rel_paths.contains(&"valid.rs".to_string()));
        assert!(
            !rel_paths.contains(&"binary.rs".to_string()),
            "binary file should be skipped"
        );
    }

    #[test]
    fn test_with_project_id() {
        let tmp = tempfile::tempdir().unwrap();
        let embedder = RepoEmbedder::new(tmp.path()).with_project_id("my-project");
        assert_eq!(embedder.project_id(), "my-project");
    }

    #[test]
    fn test_naive_chunker_directly() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let embedder = RepoEmbedder::new(root);

        // Test empty content.
        let chunks = embedder.chunk_file_naive("", "empty.rs", Some("rust".to_string()));
        assert!(chunks.is_empty());

        // Test small content.
        let content = "line 1\nline 2\nline 3\n";
        let chunks = embedder.chunk_file_naive(content, "small.rs", Some("rust".to_string()));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].granularity, "file");

        // Test large content (>50 lines).
        let large: String = (1..=120).map(|i| format!("line {i}\n")).collect();
        let chunks = embedder.chunk_file_naive(&large, "large.py", Some("python".to_string()));
        assert!(chunks.len() >= 3);
        for chunk in &chunks {
            assert_eq!(chunk.granularity, "chunk");
        }
    }

    #[tokio::test]
    async fn test_repository_embedding_is_idempotent_and_update_replaces() {
        use crate::embedding::MockEmbedder;
        use crate::store::LanceStore;

        let directory = tempfile::tempdir().unwrap();
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("main.rs"), "fn original() {}\n").unwrap();

        let lance = LanceStore::open_with_dim(directory.path(), 8)
            .await
            .unwrap();
        let embedder = RepoEmbedder::new(root.path());
        let provider = MockEmbedder::new(8);

        let first = embedder
            .embed_and_store(&provider, &lance, 10)
            .await
            .unwrap();
        assert_eq!(first.chunks_embedded, 1);
        assert_eq!(
            lance
                .get_code_embedding_ids_for_project(embedder.project_id())
                .await
                .unwrap()
                .len(),
            1
        );

        let unchanged = embedder
            .embed_and_store(&provider, &lance, 10)
            .await
            .unwrap();
        assert_eq!(unchanged.chunks_embedded, 0);

        fs::write(root.path().join("main.rs"), "fn replacement() {}\n").unwrap();
        let changed = embedder
            .embed_and_store(&provider, &lance, 10)
            .await
            .unwrap();
        assert_eq!(changed.chunks_embedded, 1);
        assert_eq!(
            lance
                .get_code_embedding_ids_for_project(embedder.project_id())
                .await
                .unwrap()
                .len(),
            2
        );

        let updated = embedder
            .embed_and_store_with_update(&provider, &lance, 10, true)
            .await
            .unwrap();
        assert_eq!(updated.chunks_embedded, 1);
        assert_eq!(
            lance
                .get_code_embedding_ids_for_project(embedder.project_id())
                .await
                .unwrap()
                .len(),
            1
        );
    }
}

// ---------------------------------------------------------------------------
