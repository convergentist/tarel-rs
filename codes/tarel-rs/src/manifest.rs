//! `manifest.json` — the run-level inventory.
//!
//! The orchestrator updates `runs/<seed>/manifest.json` after each
//! per-(cell, seed) sub-task. Stages 3-5 read this file to validate
//! that every expected artifact exists and that pre-registered score
//! specs were not mutated mid-run (Stage 5 hard-error contract).

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use crate::cell::CellId;
use crate::seed::ExperimentSeed;

/// Status of a single (cell, seed) sub-task within a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CellRunStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

/// Record for one (cell, seed) sub-task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellRunRecord {
    pub cell: CellId,
    pub seed: u64,
    pub status: CellRunStatus,
    /// Wallclock used by the sub-task in milliseconds. `None` until the
    /// sub-task finishes (success or failure).
    pub wallclock_ms: Option<u64>,
    /// Output paths the sub-task produced, relative to `run_dir`.
    pub outputs: Vec<PathBuf>,
    /// Optional human-readable failure reason.
    pub error: Option<String>,
    /// Unix-ms timestamp of the last update.
    pub updated_at_ms: u64,
}

impl CellRunRecord {
    pub fn new(cell: CellId, seed: u64) -> Self {
        Self {
            cell,
            seed,
            status: CellRunStatus::Pending,
            wallclock_ms: None,
            outputs: Vec::new(),
            error: None,
            updated_at_ms: now_ms(),
        }
    }
}

/// Top-level manifest. Pre-registered score-spec versions live here so
/// the bridge stage can refuse to mix runs that used different specs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub experiment_seed: ExperimentSeed,
    /// Hex-encoded SHA-256 of the data snapshot the run was anchored
    /// against. Recorded so Stage-5 reproducibility checks can verify
    /// that a re-run targets the same data.
    pub data_snapshot_sha256: String,
    /// Stationarity-spec version referenced by Stage 1; bridge stage
    /// hard-errors on mismatch (R10.1).
    pub stationarity_spec_version: Option<String>,
    /// Stability-spec version referenced by Stage 4; bridge stage
    /// hard-errors on mismatch.
    pub stability_spec_version: Option<String>,
    /// Tarel version that produced the run.
    pub tarel_version: String,
    /// Per-(cell, seed) records. Order is insertion-stable.
    pub runs: Vec<CellRunRecord>,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

impl Manifest {
    pub fn new(experiment_seed: ExperimentSeed, data_snapshot_sha256: impl Into<String>) -> Self {
        let now = now_ms();
        Self {
            experiment_seed,
            data_snapshot_sha256: data_snapshot_sha256.into(),
            stationarity_spec_version: None,
            stability_spec_version: None,
            tarel_version: env!("CARGO_PKG_VERSION").into(),
            runs: Vec::new(),
            created_at_ms: now,
            updated_at_ms: now,
        }
    }

    /// Append-or-replace the record for `(cell, seed)` in
    /// insertion-stable order.
    pub fn upsert(&mut self, record: CellRunRecord) {
        if let Some(slot) = self
            .runs
            .iter_mut()
            .find(|r| r.cell == record.cell && r.seed == record.seed)
        {
            *slot = record;
        } else {
            self.runs.push(record);
        }
        self.updated_at_ms = now_ms();
    }

    pub fn write_json(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let s = serde_json::to_string_pretty(self)?;
        // Atomic-ish write: write to a sibling temp then rename.
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, s)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn read_json(path: &Path) -> anyhow::Result<Self> {
        let s = std::fs::read_to_string(path)?;
        let m: Self = serde_json::from_str(&s)?;
        Ok(m)
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn upsert_replaces_in_place() {
        let mut m = Manifest::new(ExperimentSeed(1), "sha");
        let cell = CellId("a:b".into());
        m.upsert(CellRunRecord::new(cell.clone(), 1));
        let mut second = CellRunRecord::new(cell.clone(), 1);
        second.status = CellRunStatus::Completed;
        second.wallclock_ms = Some(123);
        m.upsert(second);

        assert_eq!(m.runs.len(), 1);
        assert_eq!(m.runs[0].status, CellRunStatus::Completed);
        assert_eq!(m.runs[0].wallclock_ms, Some(123));
    }

    #[test]
    fn upsert_appends_distinct() {
        let mut m = Manifest::new(ExperimentSeed(1), "sha");
        m.upsert(CellRunRecord::new(CellId("a:b".into()), 1));
        m.upsert(CellRunRecord::new(CellId("a:b".into()), 2));
        m.upsert(CellRunRecord::new(CellId("c:d".into()), 1));
        assert_eq!(m.runs.len(), 3);
    }

    #[test]
    fn manifest_round_trips_through_json() {
        let mut m = Manifest::new(ExperimentSeed(0x42), "deadbeef");
        m.stationarity_spec_version = Some("v0.1.0".into());
        m.upsert(CellRunRecord::new(CellId("binance:BTC-USDT".into()), 7));

        let dir = tempdir().unwrap();
        let path = dir.path().join("manifest.json");
        m.write_json(&path).unwrap();
        let m2 = Manifest::read_json(&path).unwrap();

        assert_eq!(m.experiment_seed, m2.experiment_seed);
        assert_eq!(m.data_snapshot_sha256, m2.data_snapshot_sha256);
        assert_eq!(m.stationarity_spec_version, m2.stationarity_spec_version);
        assert_eq!(m.runs.len(), m2.runs.len());
        assert_eq!(m.runs[0].cell, m2.runs[0].cell);
    }
}
