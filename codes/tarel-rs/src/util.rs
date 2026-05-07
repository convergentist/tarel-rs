//! Cross-module utilities shared across stages.
//!
//! Today this is just `sha256_path` (originally a private helper inside
//! `stage_one.rs`); promoting it lets the Stage-2 manifest correctly
//! populate `data_snapshot_sha256` instead of the input path string.

use std::path::Path;

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};

/// Streaming SHA-256 of a file. Returns the lowercase hex digest.
pub fn sha256_path(path: &Path) -> Result<String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)
        .with_context(|| format!("open {}", path.display()))?;
    let mut buf = [0u8; 64 * 1024];
    let mut h = Sha256::new();
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
