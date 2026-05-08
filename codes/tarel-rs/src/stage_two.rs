//! Stage-2 (Stage-4 in the proposal lexicon) orchestration:
//! per-(cell, seed, clock) W-GAN-GP training runs invoked through
//! the [`logan`] library and aggregated into per-cell Stability Scores.
//!
//! Per `engineering_proposal.md` Part I:
//!
//! - Loops `(cell, seed) ∈ cells × 0..n_seeds` and for each clock
//!   loads the Stage-2 normalized DP tensors produced by
//!   `tarel run-stage-one --produce-dp-rp`.
//! - Calls [`logan::train_one_seed`] *as a library* (no subprocess) so
//!   the orchestrator owns the wallclock budget and can collect
//!   diagnostics in-process.
//! - Aggregates per-(cell, clock) raw component values into per-cell
//!   `StabilityScore` records via [`logan::compute_stability_score`].
//! - Persists `StabilityScoreSpec.json` and `stability_scores.json`
//!   under the run directory.

use std::path::PathBuf;

use anyhow::Result;
use serde::Serialize;

use logan::{
    compute_stability_score,
    score::{StabilityComponentKind, StabilityRaw},
    train::{TrainOpts, TrainOutcome, WganConfig},
    StabilityScore, StabilityScoreSpec,
};

use crate::cell::{Cell, CellId};
use crate::layout::RunLayout;
use crate::manifest::{CellRunRecord, CellRunStatus, Manifest};
use crate::seed::{ExperimentSeed, derived_seed};

/// Per-clock label list. Matches `takens_rp::bucketer::Clock::as_str`.
pub const CLOCKS: &[&str] = &["physical", "event", "info_time"];

#[derive(Debug, Clone)]
pub struct StageTwoOptions {
    pub experiment: ExperimentSeed,
    pub cells: Vec<Cell>,
    pub n_seeds: u64,
    /// Per-seed wallclock budget in minutes (default 20).
    pub budget_mins: u64,
    /// Override clocks; defaults to all three.
    pub clocks: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StageTwoOutcome {
    pub n_runs: usize,
    pub n_completed: usize,
    pub n_failed: usize,
    pub stability_spec_path: PathBuf,
    pub stability_scores_path: PathBuf,
    pub scores: Vec<StabilityScore>,
}

/// Run Stage-2 (W-GAN training + diagnostics) end-to-end.
///
/// Iterates (cell, clock, seed) triples. For each, loads the Stage-2
/// normalized DP tensors from
/// `runs/<exp>/stage_one/recurrence/<cell>/<clock>/normalized/`, runs
/// [`logan::train_one_seed`] with a wallclock budget, and folds the
/// outcome's diagnostic statistics into the Stability Score raw
/// components.
pub fn run_stage_two(
    layout: &RunLayout,
    manifest: &mut Manifest,
    opts: &StageTwoOptions,
) -> Result<StageTwoOutcome> {
    let span = tracing::info_span!(
        "stage_two", experiment = opts.experiment.0, n_seeds = opts.n_seeds
    );
    let _e = span.enter();

    // Pre-register the spec.
    let spec = StabilityScoreSpec::v0_1_0();
    spec.validate()
        .map_err(|e| anyhow::anyhow!("StabilityScoreSpec validation failed: {e}"))?;
    let spec_path = layout.stability_spec_path();
    if let Some(parent) = spec_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&spec_path, spec.to_json_pretty()?)?;

    // Pick the device once.
    let device = tch::Device::cuda_if_available();
    tracing::info!(device = ?device, "tch_device");

    let clocks: Vec<String> = opts
        .clocks
        .clone()
        .unwrap_or_else(|| CLOCKS.iter().map(|s| s.to_string()).collect());

    let mut n_runs = 0_usize;
    let mut n_completed = 0_usize;
    let mut n_failed = 0_usize;

    let mut all_raws_per_cell: std::collections::HashMap<CellId, Vec<StabilityRaw>> =
        std::collections::HashMap::new();

    for cell in &opts.cells {
        let cell_id = cell.id();
        for (_clock_idx, clock) in clocks.iter().enumerate() {
            let dataset_dir = layout.stage_two_dataset_dir(&cell_id, clock);
            // Skip clocks whose Stage-2 outputs don't exist (e.g. user
            // ran Stage 1 only).
            if !dataset_dir.exists() {
                tracing::warn!(
                    cell = %cell_id, clock, dir = %dataset_dir.display(),
                    "stage_two_dataset_missing_skipping"
                );
                continue;
            }

            // Load DP tensors for this (cell, clock) once; reused
            // across all seeds.
            let dataset = match logan::dataset::load_dp_tensors(&dataset_dir) {
                Ok(d) => d,
                Err(e) => {
                    tracing::warn!(cell = %cell_id, clock, error = %e, "load_dp_tensors_failed");
                    continue;
                }
            };

            for seed_idx in 0..opts.n_seeds {
                n_runs += 1;
                let derived = derived_seed(
                    opts.experiment, cell_id.as_str(), seed_idx, "stage_two",
                );
                let out_dir = layout
                    .stage_two_dir(&cell_id, seed_idx)
                    .join(clock);
                std::fs::create_dir_all(&out_dir)?;

                // Manifest: mark running.
                let mut rec = CellRunRecord::new(cell_id.clone(), seed_idx);
                rec.status = CellRunStatus::Running;
                manifest.upsert(rec);
                manifest.write_json(&layout.manifest_path())?;

                let cfg = WganConfig::default();
                let train_opts = TrainOpts {
                    config: cfg,
                    budget_mins: opts.budget_mins,
                    seed: derived,
                    cell: format!("{cell_id}|{clock}"),
                    out_dir: out_dir.clone(),
                };

                let started = std::time::Instant::now();
                let outcome_res =
                    logan::train_one_seed(&dataset, &train_opts, device);
                let elapsed_ms = started.elapsed().as_millis() as u64;

                match outcome_res {
                    Ok(outcome) => {
                        if outcome.failed {
                            n_failed += 1;
                            let mut r = CellRunRecord::new(cell_id.clone(), seed_idx);
                            r.status = CellRunStatus::Failed;
                            r.wallclock_ms = Some(elapsed_ms);
                            r.error = outcome.failure_reason.clone();
                            r.outputs = vec![out_dir.clone()];
                            manifest.upsert(r);
                        } else {
                            n_completed += 1;
                            let mut r = CellRunRecord::new(cell_id.clone(), seed_idx);
                            r.status = CellRunStatus::Completed;
                            r.wallclock_ms = Some(elapsed_ms);
                            r.outputs = vec![
                                out_dir.clone(),
                                outcome.diagnostics_path.clone(),
                            ];
                            manifest.upsert(r);

                            // Collect raw components for the score.
                            let raws = derive_stability_raws(seed_idx as usize, &outcome);
                            all_raws_per_cell.entry(cell_id.clone()).or_default().extend(raws);
                        }
                    }
                    Err(e) => {
                        n_failed += 1;
                        let mut r = CellRunRecord::new(cell_id.clone(), seed_idx);
                        r.status = CellRunStatus::Failed;
                        r.wallclock_ms = Some(elapsed_ms);
                        r.error = Some(format!("{e}"));
                        manifest.upsert(r);
                        tracing::error!(cell = %cell_id, clock, seed_idx, error = %e, "train_failed");
                    }
                }
                manifest.write_json(&layout.manifest_path())?;
            }
        }
    }

    // Compose per-cell Stability Scores.
    let mut scores: Vec<StabilityScore> = Vec::with_capacity(all_raws_per_cell.len());
    for (cell_id, raws) in &all_raws_per_cell {
        scores.push(compute_stability_score(cell_id.as_str(), raws, &spec));
    }
    scores.sort_by(|a, b| a.cell.cmp(&b.cell));

    let scores_path = layout.stability_scores_path();
    if let Some(parent) = scores_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&scores_path, serde_json::to_string_pretty(&scores)?)?;

    Ok(StageTwoOutcome {
        n_runs,
        n_completed,
        n_failed,
        stability_spec_path: spec_path,
        stability_scores_path: scores_path,
        scores,
    })
}

/// Translate one [`TrainOutcome`] into the four Stability raw component
/// values that go into the per-cell score:
///
/// - **EM-gap**: `|em_gap_final.0|` — magnitude of the late-training
///   EM-gap mean (smaller = critic well-balanced ≈ more stable).
/// - **GP-magnitude**: stand-in `0.0` until the GP ECDF mean is
///   computed from the persisted MessagePack chunks (Stage-5 will fold
///   this in when the bridge analysis lands).
/// - **Mode coverage**: `n_mode_checkpoints` (count > 0 indicates the
///   diagnostic ran; richer aggregation when we read the chunks back).
/// - **Curvature**: `n_curvature_checkpoints` (likewise).
///
/// For Stage-4-only validation this captures whether each diagnostic
/// produced output. Stage 5 will replace these proxies with the
/// concrete time-series-mean / coverage-count statistics from the
/// MessagePack chunks.
fn derive_stability_raws(seed_idx: usize, outcome: &TrainOutcome) -> Vec<StabilityRaw> {
    let em = outcome.em_gap_final.0;
    vec![
        StabilityRaw {
            seed_idx,
            component: StabilityComponentKind::EmGap,
            value: em.abs(),
        },
        StabilityRaw {
            seed_idx,
            component: StabilityComponentKind::GpMagnitude,
            value: 0.0,
        },
        StabilityRaw {
            seed_idx,
            component: StabilityComponentKind::ModeCoverage,
            value: outcome.n_mode_checkpoints as f64,
        },
        StabilityRaw {
            seed_idx,
            component: StabilityComponentKind::Curvature,
            value: outcome.n_curvature_checkpoints as f64,
        },
    ]
}
