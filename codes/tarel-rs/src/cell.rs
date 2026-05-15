//! Per-(exchange, symbol) "cell" identity and metadata.
//!
//! Stage 1 calibrates the [`InfoTimeBucketer`] threshold per cell and
//! persists it as `cell_metadata.json`. The bridge analysis later
//! verifies that the metadata exists for every cell in a run before
//! computing partial Spearman rank statistics.

use std::path::Path;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Identifier in `"<exchange>:<symbol>"` form (e.g. `"binance:BTC-USDT"`).
///
/// This is the canonical string used in the orchestrator manifest and
/// in the `runs/<seed>/stage_one/buckets/<cell>/...` path layout.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CellId(pub String);

impl CellId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for CellId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Parsed `(exchange, symbol)` view of a [`CellId`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cell {
    pub exchange: String,
    pub symbol: String,
}

impl Cell {
    pub fn id(&self) -> CellId {
        CellId(format!("{}:{}", self.exchange, self.symbol))
    }
}

impl FromStr for Cell {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut parts = s.splitn(2, ':');
        let exchange = parts
            .next()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| anyhow::anyhow!("cell id missing exchange: '{s}'"))?
            .to_string();
        let symbol = parts
            .next()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| anyhow::anyhow!("cell id missing symbol: '{s}'"))?
            .to_string();
        Ok(Cell { exchange, symbol })
    }
}

/// Per-cell metadata persisted as `cell_metadata.json` in the
/// run-output tree (`runs/<seed>/stage_one/buckets/<cell>/...`).
///
/// This is the artifact the Stage-1 exit gate looks for.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellMetadata {
    pub cell: CellId,
    /// Calibrated `InfoTimeBucketer.vol_threshold` for this cell.
    /// `None` if calibration failed (e.g. insufficient warmup data).
    pub info_time_threshold: Option<f64>,
    /// Number of warmup snapshots used during calibration.
    pub calibration_warmup_n: usize,
    /// Target band the calibrated threshold landed in (lower & upper
    /// bounds, in nanoseconds; pinned to [30s, 120s] in Stage 1).
    pub target_band_ns: (u64, u64),
    /// Hex-encoded SHA-256 of the data snapshot used for calibration.
    ///
    /// This is a *bytewise* hash of the consolidated parquet file as it
    /// sits on disk. It changes whenever the parquet encoding changes,
    /// even if the underlying logical data is unchanged. Use
    /// [`Self::source_corpus_sha256`] to track "did the source data
    /// change?" independently of encoding.
    pub data_snapshot_sha256: String,
    /// Hex-encoded `takens.consolidate.source_corpus_sha256` lifted from
    /// the input parquet's footer key-value metadata.
    ///
    /// This is the SHA-256 of the sorted `(filename, file_sha256)` tuple
    /// list produced at consolidation time — invariant to parquet
    /// re-encoding, tight to the raw flush-time source files.
    ///
    /// `None` if the input parquet has no `takens.consolidate.*` footer
    /// (e.g. a raw flush-time parquet was passed directly to Stage-1
    /// without first being consolidated). In that case downstream
    /// reproducibility falls back to [`Self::data_snapshot_sha256`].
    #[serde(default)]
    pub source_corpus_sha256: Option<String>,
    /// `K_BINS_PER_SIDE` in effect at calibration time. Pinned so a
    /// downstream change to the constant invalidates the metadata.
    pub k_bins_per_side: usize,
}

impl CellMetadata {
    /// Write `self` as pretty-printed JSON to `path`. Creates parent
    /// directories as needed.
    pub fn write_json(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Read `cell_metadata.json` from disk.
    pub fn read_json(path: &Path) -> anyhow::Result<Self> {
        let s = std::fs::read_to_string(path)?;
        let m: Self = serde_json::from_str(&s)?;
        Ok(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn parses_well_formed_cell_id() {
        let c: Cell = "binance:BTC-USDT".parse().unwrap();
        assert_eq!(c.exchange, "binance");
        assert_eq!(c.symbol, "BTC-USDT");
        assert_eq!(c.id().as_str(), "binance:BTC-USDT");
    }

    #[test]
    fn rejects_missing_exchange() {
        assert!(":BTC-USDT".parse::<Cell>().is_err());
    }

    #[test]
    fn rejects_missing_symbol() {
        assert!("binance:".parse::<Cell>().is_err());
        assert!("binance".parse::<Cell>().is_err());
    }

    #[test]
    fn cell_metadata_round_trips_through_json() {
        let m = CellMetadata {
            cell: CellId("binance:BTC-USDT".into()),
            info_time_threshold: Some(123.45),
            calibration_warmup_n: 4096,
            target_band_ns: (30_000_000_000, 120_000_000_000),
            data_snapshot_sha256: "abc123".into(),
            source_corpus_sha256: Some("def456".into()),
            k_bins_per_side: 25,
        };
        let dir = tempdir().unwrap();
        let path = dir.path().join("cell_metadata.json");
        m.write_json(&path).unwrap();
        let m2 = CellMetadata::read_json(&path).unwrap();
        assert_eq!(m.cell, m2.cell);
        assert_eq!(m.info_time_threshold, m2.info_time_threshold);
        assert_eq!(m.k_bins_per_side, m2.k_bins_per_side);
    }
}
