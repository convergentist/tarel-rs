//! Run-output directory layout.
//!
//! ```text
//! runs/<experiment_seed>/
//!   manifest.json
//!   logs/
//!     orchestrator.jsonl
//!   stage_one/
//!     buckets/<cell>/<clock>/<bucket_idx>.parquet
//!     cell_metadata.json
//!     stationarity_scores.json   # populated in Stage 3
//!   stage_two/
//!     <cell>/<seed>/
//!       checkpoints/
//!       diagnostics.msgpack
//!   figures/
//! ```
//!
//! The layout is pinned by `engineering_proposal.md` Part I. Tests in
//! `tarel verify --run <seed>` walk this tree and assert every expected
//! artifact exists. The current Stage-1 implementation only constructs
//! the directories that Stage 1 actually fills; later stages will
//! mkdir-as-needed.

use std::path::{Path, PathBuf};

use crate::cell::CellId;
use crate::seed::ExperimentSeed;

/// Helper that resolves the per-experiment run paths from a `runs_root`.
#[derive(Debug, Clone)]
pub struct RunLayout {
    /// Top-level `runs/` directory.
    pub runs_root: PathBuf,
    /// `runs/<experiment_seed>/`.
    pub run_dir: PathBuf,
    pub experiment_seed: ExperimentSeed,
}

impl RunLayout {
    /// Build a layout rooted at `runs_root`, scoped to `experiment_seed`.
    pub fn new(runs_root: impl AsRef<Path>, experiment_seed: ExperimentSeed) -> Self {
        let runs_root = runs_root.as_ref().to_path_buf();
        let run_dir = runs_root.join(format!("{:016x}", experiment_seed.0));
        Self {
            runs_root,
            run_dir,
            experiment_seed,
        }
    }

    pub fn manifest_path(&self) -> PathBuf {
        self.run_dir.join("manifest.json")
    }

    pub fn logs_dir(&self) -> PathBuf {
        self.run_dir.join("logs")
    }

    pub fn orchestrator_log(&self) -> PathBuf {
        self.logs_dir().join("orchestrator.jsonl")
    }

    pub fn stage_one_dir(&self) -> PathBuf {
        self.run_dir.join("stage_one")
    }

    pub fn stage_one_buckets_dir(&self, cell: &CellId, clock: &str) -> PathBuf {
        self.stage_one_dir()
            .join("buckets")
            .join(cell.as_str().replace(':', "_"))
            .join(clock)
    }

    pub fn cell_metadata_path(&self, cell: &CellId) -> PathBuf {
        self.stage_one_dir()
            .join("buckets")
            .join(cell.as_str().replace(':', "_"))
            .join("cell_metadata.json")
    }

    pub fn stationarity_scores_path(&self) -> PathBuf {
        self.stage_one_dir().join("stationarity_scores.json")
    }

    /// Per-(cell, clock) directory for Stage-3 statistical artifacts.
    pub fn stage_three_dir(&self, cell: &CellId, clock: &str) -> PathBuf {
        self.stage_one_dir()
            .join("recurrence")
            .join(cell.as_str().replace(':', "_"))
            .join(clock)
    }

    /// Per-(cell, clock) per-bucket RQA scalars JSONL.
    pub fn rqa_scalars_path(&self, cell: &CellId, clock: &str) -> PathBuf {
        self.stage_three_dir(cell, clock).join("rqa_scalars.jsonl")
    }

    /// Per-(cell, clock) KS-distance session-pair time series.
    pub fn ks_series_path(&self, cell: &CellId, clock: &str) -> PathBuf {
        self.stage_three_dir(cell, clock).join("ks_series.json")
    }

    /// Per-(cell, clock) Hill estimates.
    pub fn hill_path(&self, cell: &CellId, clock: &str) -> PathBuf {
        self.stage_three_dir(cell, clock).join("hill.json")
    }

    /// Per-(cell, clock) hierarchical variance ratios.
    pub fn hierarchical_path(&self, cell: &CellId, clock: &str) -> PathBuf {
        self.stage_three_dir(cell, clock).join("hierarchical.json")
    }

    /// On-disk path of the pre-registered Stationarity score spec
    /// (resolved at run time so the manifest can record the version
    /// actually in effect).
    pub fn stationarity_spec_path(&self) -> PathBuf {
        self.stage_one_dir().join("StationarityScoreSpec.json")
    }

    pub fn stage_two_dir(&self, cell: &CellId, seed: u64) -> PathBuf {
        self.run_dir
            .join("stage_two")
            .join(cell.as_str().replace(':', "_"))
            .join(format!("{seed:020}"))
    }

    pub fn figures_dir(&self) -> PathBuf {
        self.run_dir.join("figures")
    }

    /// Create the bare-minimum directories needed for Stage 1 to start
    /// writing artifacts. Idempotent; safe to call multiple times.
    pub fn ensure_stage_one_dirs(&self, cell: &CellId) -> std::io::Result<()> {
        std::fs::create_dir_all(self.logs_dir())?;
        for clock in ["physical", "event", "info_time"] {
            std::fs::create_dir_all(self.stage_one_buckets_dir(cell, clock))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn run_dir_is_hex_padded_seed() {
        let layout = RunLayout::new("/tmp/runs", ExperimentSeed(0xCAFE));
        assert!(layout
            .run_dir
            .to_string_lossy()
            .ends_with("000000000000cafe"));
    }

    #[test]
    fn ensure_stage_one_dirs_creates_all_three_clocks() {
        let dir = tempdir().unwrap();
        let layout = RunLayout::new(dir.path(), ExperimentSeed(42));
        let cell = CellId("binance:BTC-USDT".into());
        layout.ensure_stage_one_dirs(&cell).unwrap();
        for clock in ["physical", "event", "info_time"] {
            assert!(layout.stage_one_buckets_dir(&cell, clock).is_dir());
        }
        assert!(layout.logs_dir().is_dir());
    }

    #[test]
    fn cell_id_with_colon_is_safe_for_filesystem() {
        let layout = RunLayout::new("/tmp/runs", ExperimentSeed(0));
        let cell = CellId("binance:BTC-USDT".into());
        let path = layout.stage_one_buckets_dir(&cell, "physical");
        assert!(!path.to_string_lossy().contains(':'));
    }
}
