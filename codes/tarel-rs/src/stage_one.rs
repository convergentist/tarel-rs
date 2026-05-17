//!
//! Stage-1 orchestration entry point.
//!
//! The command is: `tarel run-stage-one --cell <id> --input <parquet>`
//! e.g. 
//! tarel run-stage-one --cell binance:BTC-USDC --input <path/to/ob_data.parquet>
//!
//! This runs the foundation pipeline for a single cell, resulting in this actions:
//!
//! 1. Read the parquet file via [`takens_rp::ingest::read_deltas`].
//! 2. Reconstruct LOB snapshots via [`takens_rp::book::reconstruct`].
//! 3. Calibrate the [`InfoTimeBucketer`] threshold for the cell using
//!    the first day of the input as a warmup slice; persist to
//!    `cell_metadata.json`.
//! 4. Run the three bucketers (Physical 60s, Event N=64, InfoTime
//!    calibrated) over the full snapshot stream and write an inventory
//!    of bucketed snapshot streams under
//!    `runs/<seed>/stage_one/buckets/<cell>/<clock>/`.
//! 5. Update the run [`Manifest`].
//!

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};

use takens_rp::book::reconstruct;
use takens_rp::bucketer::{
    Bucket, EventBucketer, InfoTimeBucketer, PhysicalBucketer, calibrate_info_time_threshold,
    run_bucketer,
};
use takens_rp::distance::DistanceMetric;
use takens_rp::ingest::read_deltas;
use takens_rp::represent::K_BINS_PER_SIDE;
use takens_rp::types::LobSnapshot;

use crate::cell::{Cell, CellId, CellMetadata};
use crate::layout::RunLayout;
use crate::manifest::{CellRunRecord, CellRunStatus, Manifest};
use crate::recurrence::{produce_recurrence_artifacts, RecurrenceOutcome};
use crate::stage_three::{
    compute_and_write_rqa_scalars, run_stage_three_for_clock,
    write_and_verify_stationarity_spec, write_stationarity_scores,
};
use crate::util::sha256_path;

// Locked Stage-1 calibration band per engineering_proposal.md Part I.
pub const INFO_TIME_TARGET_LO_NS: u64 = 30 * 1_000_000_000;
pub const INFO_TIME_TARGET_HI_NS: u64 = 120 * 1_000_000_000;
pub const CALIBRATION_MAX_ITERS: usize = 64;
/// Max snapshots used for InfoTime calibration warmup. Roughly 1 day of
/// 100 ms snapshots = 864k; we cap at 200k to bound calibration cost.
pub const CALIBRATION_WARMUP_CAP: usize = 200_000;

/// Result summary written to the manifest after a Stage-1 run.
#[derive(Debug, Clone)]
pub struct StageOneOutcome {
    pub cell: CellId,
    pub n_snapshots: usize,
    pub n_buckets_physical: usize,
    pub n_buckets_event: usize,
    pub n_buckets_info_time: usize,
    pub info_time_threshold: Option<f64>,
    pub wallclock_ms: u64,
    pub written_paths: Vec<PathBuf>,
    /// Per-clock recurrence-layer outcomes when `produce_dp_rp == true`.
    /// `None` when Stage 2 outputs were not requested.
    pub recurrence: Option<Vec<RecurrenceOutcome>>,
    /// Stationarity-score artifact path when `full == true`.
    pub stationarity_scores_path: Option<PathBuf>,
    /// Spec version used when `full == true`. Recorded so Stage-5 can
    /// validate it across runs.
    pub stationarity_spec_version: Option<String>,
}

/// Extra flags controlling how `run_stage_one` behaves.
#[derive(Debug, Clone, Copy)]
pub struct StageOneOptions {
    /// Produce Stage-2 recurrence-layer artifacts (DPs + RPs + 64×64
    /// normalized tensors + PNG sanity grids) for every (clock, bucket).
    pub produce_dp_rp: bool,
    /// Run the full Stage-3 statistical pass: per-bucket RQA scalars,
    /// KS over 8h sessions, hierarchical variance ratio, Hill estimator,
    /// per-cell Stationarity Score. Implies `produce_dp_rp = false`
    /// for the Stage-3 work (Stage-3 reuses the bucket-level pipeline
    /// internally). When both are true, Stage 2 artifacts are also
    /// emitted alongside the Stage-3 outputs.
    pub full: bool,
    /// Distance metric to use when building DPs. Defaults to L2.
    pub metric: DistanceMetric,
    /// α for the price/volume subspace combination. Defaults to 0.5.
    pub alpha: f64,
}

impl Default for StageOneOptions {
    fn default() -> Self {
        Self {
            produce_dp_rp: false,
            full: false,
            metric: DistanceMetric::Euclidean,
            alpha: 0.5,
        }
    }
}

/// Run Stage 1 against a single parquet file for one cell. Defaults to
/// the no-Stage-2 behaviour; see [`run_stage_one_with_opts`] for the
/// `--produce-dp-rp` variant.
pub fn run_stage_one(
    layout: &RunLayout,
    manifest: &mut Manifest,
    cell: &Cell,
    seed: u64,
    parquet_path: &Path,
) -> Result<StageOneOutcome> {
    run_stage_one_with_opts(layout, manifest, cell, seed, parquet_path, StageOneOptions::default())
}

/// Run Stage 1 against a single parquet file for one cell with explicit
/// options. The parquet file is read once and the resulting snapshot
/// stream is shared across all three bucketers (cheap clones — only the
/// per-bucket snapshot vectors are duplicated).
pub fn run_stage_one_with_opts(
    layout: &RunLayout,
    manifest: &mut Manifest,
    cell: &Cell,
    seed: u64,
    parquet_path: &Path,
    opts: StageOneOptions,
) -> Result<StageOneOutcome> {
    let cell_id = cell.id();
    let span = tracing::info_span!(
        "stage_one",
        cell = %cell_id,
        seed,
        parquet = %parquet_path.display(),
    );
    let _enter = span.enter();

    layout.ensure_stage_one_dirs(&cell_id)?;

    // Mark the run as running before doing any work so a hard kill
    // leaves a non-pending status behind.
    let mut record = CellRunRecord::new(cell_id.clone(), seed);
    record.status = CellRunStatus::Running;
    manifest.upsert(record);
    manifest.write_json(&layout.manifest_path())?;

    let started = Instant::now();
    match run_stage_one_inner(layout, cell, seed, parquet_path, opts) {
        Ok(outcome) => {
            let mut rec = CellRunRecord::new(cell_id.clone(), seed);
            rec.status = CellRunStatus::Completed;
            rec.wallclock_ms = Some(started.elapsed().as_millis() as u64);
            rec.outputs = outcome.written_paths.clone();
            manifest.upsert(rec);
            manifest.write_json(&layout.manifest_path())?;
            tracing::info!(
                snapshots = outcome.n_snapshots,
                physical = outcome.n_buckets_physical,
                event = outcome.n_buckets_event,
                info_time = outcome.n_buckets_info_time,
                "stage_one_complete"
            );
            Ok(outcome)
        }
        Err(e) => {
            let mut rec = CellRunRecord::new(cell_id, seed);
            rec.status = CellRunStatus::Failed;
            rec.wallclock_ms = Some(started.elapsed().as_millis() as u64);
            rec.error = Some(format!("{e:#}"));
            manifest.upsert(rec);
            manifest.write_json(&layout.manifest_path())?;
            tracing::error!(error = %e, "stage_one_failed");
            Err(e)
        }
    }
}

fn run_stage_one_inner(
    layout: &RunLayout,
    cell: &Cell,
    _seed: u64,
    parquet_path: &Path,
    opts: StageOneOptions,
) -> Result<StageOneOutcome> {
    let cell_id = cell.id();

    // 1. Ingest.
    let batches = read_deltas(parquet_path).with_context(|| {
        format!("read_deltas({})", parquet_path.display())
    })?;
    tracing::info!(batches = batches.len(), "ingest_complete");

    // 2. Reconstruct.
    let snapshots = reconstruct(&batches, K_BINS_PER_SIDE)
        .with_context(|| "book::reconstruct")?;
    let n_snapshots = snapshots.len();
    tracing::info!(snapshots = n_snapshots, k_bins = K_BINS_PER_SIDE, "reconstruct_complete");
    anyhow::ensure!(
        n_snapshots >= 16,
        "stage 1 requires ≥ 16 valid snapshots, got {n_snapshots}"
    );

    // 3. Calibrate InfoTime threshold against a warmup slice.
    let warmup_n = n_snapshots.min(CALIBRATION_WARMUP_CAP);
    let warmup = &snapshots[..warmup_n];
    let info_time_threshold = calibrate_info_time_threshold(
        warmup,
        INFO_TIME_TARGET_LO_NS,
        INFO_TIME_TARGET_HI_NS,
        CALIBRATION_MAX_ITERS,
    );
    tracing::info!(
        warmup_n,
        info_time_threshold = ?info_time_threshold,
        "calibration_complete"
    );

    // 4. Persist cell_metadata.json (R6.x).
    //
    // Two SHAs:
    //   - data_snapshot_sha256: bytewise hash of the consolidated parquet
    //     as it sits on disk (sensitive to parquet re-encoding).
    //   - source_corpus_sha256: lifted from the parquet's footer KV (set
    //     by takens_consolidate), invariant to encoding, tight to the
    //     raw flush-time source files. `None` if the input was not
    //     produced by takens_consolidate.
    let snapshot_sha = sha256_path(parquet_path)?;
    let source_corpus_sha256 =
        takens_rp::consolidate::read_source_corpus_sha256(parquet_path).with_context(|| {
            format!(
                "lift takens.consolidate.source_corpus_sha256 from {}",
                parquet_path.display()
            )
        })?;
    let metadata = CellMetadata {
        cell: cell_id.clone(),
        info_time_threshold,
        calibration_warmup_n: warmup_n,
        target_band_ns: (INFO_TIME_TARGET_LO_NS, INFO_TIME_TARGET_HI_NS),
        data_snapshot_sha256: snapshot_sha,
        source_corpus_sha256,
        k_bins_per_side: K_BINS_PER_SIDE,
    };
    metadata.write_json(&layout.cell_metadata_path(&cell_id))?;

    // 5. Run the three bucketers.
    let mut written_paths = vec![layout.cell_metadata_path(&cell_id)];

    // PhysicalBucketer (60s).
    let mut physical = PhysicalBucketer::default_60s();
    let buckets_phys = run_bucketer(&mut physical, &snapshots);
    write_bucket_inventory(layout, &cell_id, "physical", &buckets_phys, &mut written_paths)?;

    // EventBucketer (N=64).
    let mut event = EventBucketer::default_64();
    let buckets_evt = run_bucketer(&mut event, &snapshots);
    write_bucket_inventory(layout, &cell_id, "event", &buckets_evt, &mut written_paths)?;

    // InfoTimeBucketer (calibrated; falls back to median magnitude * 64
    // if calibration failed so the pipeline still produces buckets we
    // can inspect).
    let info_threshold = info_time_threshold.unwrap_or_else(|| {
        let mut mags: Vec<f64> = snapshots
            .windows(2)
            .map(|p| {
                (snapshot_total(&p[1]) - snapshot_total(&p[0])).abs()
            })
            .collect();
        mags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = mags.get(mags.len() / 2).copied().unwrap_or(1.0);
        (med * 64.0).max(1.0)
    });
    let mut info_time = InfoTimeBucketer::new(info_threshold);
    let buckets_it = run_bucketer(&mut info_time, &snapshots);
    write_bucket_inventory(layout, &cell_id, "info_time", &buckets_it, &mut written_paths)?;

    // 6. Stage-2 recurrence-layer artifacts (optional).
    let recurrence = if opts.produce_dp_rp {
        use takens_rp::bucketer::Clock;
        let outs = vec![
            (Clock::Physical, &buckets_phys),
            (Clock::Event, &buckets_evt),
            (Clock::InfoTime, &buckets_it),
        ];
        let mut all = Vec::with_capacity(outs.len());
        for (clock, buckets) in outs {
            let span = tracing::info_span!("stage_two", clock = clock.as_str());
            let _e = span.enter();
            let outcome = produce_recurrence_artifacts(
                layout,
                &cell_id,
                clock,
                buckets,
                opts.metric,
                opts.alpha,
            )?;
            tracing::info!(
                clock = clock.as_str(),
                in_buckets = outcome.n_buckets_in,
                processed = outcome.n_processed,
                skipped_below_min = outcome.n_skipped_below_min,
                skipped_no_baseline = outcome.n_skipped_no_baseline,
                "recurrence_complete"
            );
            for p in &outcome.written {
                written_paths.push(p.clone());
            }
            all.push(outcome);
        }
        Some(all)
    } else {
        None
    };

    // 7. Stage-3 statistics (optional: --full).
    let (stationarity_scores_path, stationarity_spec_version) = if opts.full {
        use takens_rp::bucketer::Clock;

        let spec = write_and_verify_stationarity_spec(layout)?;
        written_paths.push(layout.stationarity_spec_path());

        let cell_idx = 0_usize; // single-cell pilot
        let mut all_raws: Vec<takens_stats::score::RawScalar> = Vec::new();

        for (clock_idx, (clock, buckets)) in [
            (Clock::Physical, &buckets_phys),
            (Clock::Event, &buckets_evt),
            (Clock::InfoTime, &buckets_it),
        ]
        .iter()
        .enumerate()
        {
            let span = tracing::info_span!("stage_three", clock = clock.as_str());
            let _e = span.enter();
            let records = compute_and_write_rqa_scalars(
                layout,
                &cell_id,
                *clock,
                buckets,
                opts.metric,
                opts.alpha,
            )?;
            tracing::info!(
                clock = clock.as_str(),
                records = records.len(),
                "rqa_scalars_complete"
            );
            written_paths.push(layout.rqa_scalars_path(&cell_id, clock.as_str()));

            let outcome = run_stage_three_for_clock(
                layout,
                &cell_id,
                cell_idx,
                *clock,
                clock_idx,
                &records,
                &spec,
                _seed,
            )?;
            for p in &outcome.written {
                written_paths.push(p.clone());
            }
            let n_raws = outcome.raw_scalars.len();
            all_raws.extend(outcome.raw_scalars);
            tracing::info!(
                clock = clock.as_str(),
                raw_components = n_raws,
                "stage_three_complete"
            );
        }

        // Combine into per-cell Stationarity Score and persist.
        let scores = write_stationarity_scores(layout, &all_raws, &spec)?;
        let path = layout.stationarity_scores_path();
        written_paths.push(path.clone());
        tracing::info!(
            n_cells = scores.len(),
            spec_version = %spec.version,
            "stationarity_scores_written"
        );
        (Some(path), Some(spec.version))
    } else {
        (None, None)
    };

    Ok(StageOneOutcome {
        cell: cell_id,
        n_snapshots,
        n_buckets_physical: buckets_phys.len(),
        n_buckets_event: buckets_evt.len(),
        n_buckets_info_time: buckets_it.len(),
        info_time_threshold,
        wallclock_ms: 0, // filled in by caller
        written_paths,
        recurrence,
        stationarity_scores_path,
        stationarity_spec_version,
    })
}

/// Persist a per-clock bucket inventory as an `inventory.json` listing
/// the buckets, their windows, sizes, durations, and truncation flags.
///
/// Stage-2 work will replace this with full per-bucket parquet writes;
/// for Stage 1 the inventory is enough to verify the bucketer behaves
/// correctly on real data.
fn write_bucket_inventory(
    layout: &RunLayout,
    cell: &CellId,
    clock: &str,
    buckets: &[Bucket],
    written: &mut Vec<PathBuf>,
) -> Result<()> {
    use serde::Serialize;

    #[derive(Serialize)]
    struct InventoryEntry {
        idx: usize,
        window: u64,
        idx_in_window: usize,
        n_snapshots: usize,
        first_ts_ns: u64,
        last_ts_ns: u64,
        duration_ns: u64,
        truncated_at_window_boundary: bool,
        dropped_below_min: bool,
        baseline_total_bid_vol_first: Option<f64>,
        baseline_total_ask_vol_first: Option<f64>,
    }

    #[derive(Serialize)]
    struct Inventory {
        cell: String,
        clock: String,
        n_buckets: usize,
        n_dropped: usize,
        n_truncated: usize,
        entries: Vec<InventoryEntry>,
    }

    let entries: Vec<InventoryEntry> = buckets
        .iter()
        .enumerate()
        .map(|(idx, b)| InventoryEntry {
            idx,
            window: b.window.0,
            idx_in_window: b.idx_in_window,
            n_snapshots: b.snapshots.len(),
            first_ts_ns: b.snapshots.first().map(|s| s.timestamp_ns).unwrap_or(0),
            last_ts_ns: b.snapshots.last().map(|s| s.timestamp_ns).unwrap_or(0),
            duration_ns: b.duration_ns(),
            truncated_at_window_boundary: b.truncated_at_window_boundary,
            dropped_below_min: b.dropped_below_min,
            baseline_total_bid_vol_first: b.baseline.map(|x| x.total_bid_vol_first),
            baseline_total_ask_vol_first: b.baseline.map(|x| x.total_ask_vol_first),
        })
        .collect();

    let inventory = Inventory {
        cell: cell.as_str().into(),
        clock: clock.into(),
        n_buckets: buckets.len(),
        n_dropped: buckets.iter().filter(|b| b.dropped_below_min).count(),
        n_truncated: buckets
            .iter()
            .filter(|b| b.truncated_at_window_boundary)
            .count(),
        entries,
    };

    let dir = layout.stage_one_buckets_dir(cell, clock);
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("inventory.json");
    let json = serde_json::to_string_pretty(&inventory)?;
    std::fs::write(&path, json)?;
    written.push(path);
    Ok(())
}

fn snapshot_total(s: &LobSnapshot) -> f64 {
    s.bids.iter().map(|&(_, v)| v).sum::<f64>()
        + s.asks.iter().map(|&(_, v)| v).sum::<f64>()
}
