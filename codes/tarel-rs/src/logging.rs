//! `tracing`-based JSON-line logging wired up at the orchestration stub.
//!
//! Per `engineering_proposal.md` §III.3 (Observability), every later
//! stage emits structured `tracing` spans with `(cell, seed, phase)`
//! fields; the default subscriber writes JSON-line logs to disk and a
//! human-readable summary to stderr. `RUST_LOG=info` is the default
//! filter.

use std::path::Path;

use tracing::Level;
use tracing_subscriber::fmt::writer::BoxMakeWriter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, fmt};

/// Initialize the global tracing subscriber.
///
/// - Stderr: human-readable, ANSI-coloured.
/// - JSON-line file at `log_path`: one JSON object per event, suitable
///   for offline analysis and run-vs-run diffing.
///
/// Re-initialisation (e.g. in tests) is a soft no-op — initialising
/// the subscriber more than once returns an error which is silently
/// dropped so test threads don't fight each other.
pub fn init(log_path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(Level::INFO.to_string()));

    let stderr_layer = fmt::layer()
        .with_target(false)
        .with_writer(std::io::stderr)
        .compact();

    let json_layer = fmt::layer()
        .with_writer(BoxMakeWriter::new(file))
        .with_ansi(false)
        .json()
        .with_current_span(true)
        .with_span_list(true);

    let _ = tracing_subscriber::registry()
        .with(env_filter)
        .with(stderr_layer)
        .with(json_layer)
        .try_init();

    Ok(())
}
