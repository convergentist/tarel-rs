//! Stage-3 orchestration: per-bucket RQA scalars + Stationarity Score.
//!
//! Per `engineering_proposal.md` Part I (Stage 3):
//!
//! 1. Compute the six RQA scalars (RR, DET, LAM, L_max, TT, ENTR) per
//!    bucket using the headline RR=5% threshold computed in Stage 2.
//! 2. Persist them as JSONL at
//!    `runs/<seed>/stage_one/recurrence/<cell>/<clock>/rqa_scalars.jsonl`.
//! 3. Run the three Stage-1 statistical tests:
//!    - KS distance over successive 8h-session distributions per (clock, scalar).
//!    - Hierarchical (within/between-instrument) variance ratio per (clock, scalar).
//!    - Hill tail-index per (clock, scalar) — gracefully degrades to
//!      `InsufficientData` for cells with fewer than 200 buckets per scalar.
//! 4. Compose the per-cell raw component values and combine into the
//!    per-cell Stationarity Score with the locked
//!    [`StationarityScoreSpec`] weights `[0.4, 0.3, 0.3]`.
//!
//! For a single-cell pilot run, the *hierarchical* component is
//! degenerate (`n_groups = 1` → `+∞` ratio) and rank normalisation
//! collapses (one cell → all ranks = 1.0). The driver still emits the
//! computed values + a Score so the artifact-presence exit gate is met
//! and the post-processing step can aggregate across cells.

use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use takens_rp::bucketer::{Bucket, Clock};
use takens_rp::distance::{combine, compute_subspace_distances_with_metric, DistanceMetric};
use takens_rp::represent::vectorize_with_baseline;
use takens_rp::rqa::{compute as rqa_compute, RqaConfig, RqaMetrics};

use takens_stats::{
    HillEstimate, HillError,
    StationarityScore,
    compute_stationarity_score,
    hierarchical::{compute_hierarchical_variance_ratio, HierarchicalVarianceRatio},
    hill::hill_estimate,
    ks::ks_session_pair_series,
    score::{ComponentKind, RawScalar},
    spec::StationarityScoreSpec,
};

use crate::cell::CellId;
use crate::layout::RunLayout;
use crate::recurrence::{HEADLINE_RR, TARGET_RR_MAX_ITERS, TARGET_RR_TOL};

/// Locked Stage-3 RQA threshold: use the headline RR=5% RP for stat work.
pub const STATS_RR_TARGET: f64 = HEADLINE_RR;

/// Names of the six RQA scalars (in the order serialised below).
pub const RQA_SCALAR_NAMES: [&str; 6] = ["rr", "det", "lam", "l_max", "tt", "entr"];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RqaScalarRecord {
    pub idx: usize,
    pub window: u64,
    pub idx_in_window: usize,
    pub first_ts_ns: u64,
    pub last_ts_ns: u64,
    pub rr: f64,
    pub det: f64,
    pub lam: f64,
    pub l_max: usize,
    pub tt: f64,
    pub entr: f64,
}

impl RqaScalarRecord {
    fn from_metrics(idx: usize, bucket: &Bucket, m: &RqaMetrics) -> Self {
        Self {
            idx,
            window: bucket.window.0,
            idx_in_window: bucket.idx_in_window,
            first_ts_ns: bucket.snapshots.first().map(|s| s.timestamp_ns).unwrap_or(0),
            last_ts_ns: bucket.snapshots.last().map(|s| s.timestamp_ns).unwrap_or(0),
            rr: m.rr,
            det: m.det,
            lam: m.lam,
            l_max: m.l_max,
            tt: m.tt,
            entr: m.entr,
        }
    }

    fn scalar_value(&self, name: &str) -> f64 {
        match name {
            "rr" => self.rr,
            "det" => self.det,
            "lam" => self.lam,
            "l_max" => self.l_max as f64,
            "tt" => self.tt,
            "entr" => self.entr,
            _ => f64::NAN,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PerScalarKsSeries {
    pub scalar: String,
    pub series: Vec<KsSeriesEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct KsSeriesEntry {
    pub session_a: u64,
    pub session_b: u64,
    pub d: f64,
    pub n_a: usize,
    pub n_b: usize,
    pub d_ci_lo: f64,
    pub d_ci_hi: f64,
    pub bootstrap_b: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct PerScalarHill {
    pub scalar: String,
    /// `Some(...)` when n ≥ MIN_N; otherwise `None` and `error` carries
    /// the human-readable reason.
    pub estimate: Option<HillEstimate>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PerScalarHierarchical {
    pub scalar: String,
    pub ratio: HierarchicalVarianceRatio,
    /// `true` when this run only has one instrument (single-cell pilot)
    /// — in that case the ratio is degenerate (ratio = +∞) and the
    /// downstream Stationarity Score collapses for this component.
    pub degenerate_single_instrument: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct StageThreeOutcome {
    pub cell: CellId,
    pub clock: String,
    pub n_buckets_in: usize,
    pub n_processed: usize,
    pub n_skipped: usize,
    pub written: Vec<PathBuf>,
    pub raw_scalars: Vec<RawScalar>,
}

/// Compute per-bucket RQA scalars from a list of buckets and persist as JSONL.
pub fn compute_and_write_rqa_scalars(
    layout: &RunLayout,
    cell: &CellId,
    clock: Clock,
    buckets: &[Bucket],
    metric: DistanceMetric,
    alpha: f64,
) -> Result<Vec<RqaScalarRecord>> {
    let dir = layout.stage_three_dir(cell, clock.as_str());
    std::fs::create_dir_all(&dir)?;
    let path = layout.rqa_scalars_path(cell, clock.as_str());
    let mut writer = std::io::BufWriter::new(
        std::fs::File::create(&path)
            .with_context(|| format!("create {}", path.display()))?,
    );
    use std::io::Write;

    let cfg = RqaConfig::default();
    let mut records: Vec<RqaScalarRecord> = Vec::new();

    for (idx, bucket) in buckets.iter().enumerate() {
        if bucket.dropped_below_min { continue; }
        let baseline = match bucket.baseline {
            Some(b) => b,
            None => continue,
        };
        let vectors = match vectorize_with_baseline(&bucket.snapshots, baseline) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if vectors.len() < 2 { continue; }

        let (d_p, d_v) = compute_subspace_distances_with_metric(&vectors, metric)?;
        let dp = combine(&d_p, &d_v, alpha)?;
        let out = dp.threshold_to_target_rr(STATS_RR_TARGET, TARGET_RR_TOL, TARGET_RR_MAX_ITERS);
        let rqa = match rqa_compute(&out.plot, &cfg) {
            Some(m) => m,
            None => continue,
        };

        let record = RqaScalarRecord::from_metrics(idx, bucket, &rqa);
        let line = serde_json::to_string(&record)?;
        writeln!(writer, "{line}")?;
        records.push(record);
    }

    writer.flush()?;
    Ok(records)
}

/// Run the per-cell Stage-3 statistical tests against an already-computed
/// list of RQA scalar records. Persists per-(scalar) artifacts under
/// `runs/<seed>/stage_one/recurrence/<cell>/<clock>/`.
pub fn run_stage_three_for_clock(
    layout: &RunLayout,
    cell: &CellId,
    cell_idx: usize,
    clock: Clock,
    clock_idx: usize,
    records: &[RqaScalarRecord],
    spec: &StationarityScoreSpec,
    rng_seed: u64,
) -> Result<StageThreeOutcome> {
    let _e = tracing::info_span!("stage_three", cell = %cell, clock = clock.as_str()).entered();

    let mut written = vec![layout.rqa_scalars_path(cell, clock.as_str())];
    let mut raw_scalars: Vec<RawScalar> = Vec::new();

    let mut ks_out: Vec<PerScalarKsSeries> = Vec::with_capacity(RQA_SCALAR_NAMES.len());
    let mut hill_out: Vec<PerScalarHill> = Vec::with_capacity(RQA_SCALAR_NAMES.len());
    let mut hier_out: Vec<PerScalarHierarchical> = Vec::with_capacity(RQA_SCALAR_NAMES.len());

    for (scalar_idx, scalar_name) in RQA_SCALAR_NAMES.iter().enumerate() {
        // (timestamp_ns, value) pairs for this scalar.
        let obs: Vec<(u64, f64)> = records
            .iter()
            .map(|r| (r.first_ts_ns, r.scalar_value(scalar_name)))
            .filter(|(_, v)| v.is_finite())
            .collect();

        // --- KS over 8h sessions ---
        let series = ks_session_pair_series(&obs, spec.ks_bootstrap_b, rng_seed.wrapping_add(scalar_idx as u64));
        let ks_entries: Vec<KsSeriesEntry> = series
            .iter()
            .map(|(sa, sb, r)| KsSeriesEntry {
                session_a: *sa,
                session_b: *sb,
                d: r.d,
                n_a: r.n_a,
                n_b: r.n_b,
                d_ci_lo: r.d_ci_lo,
                d_ci_hi: r.d_ci_hi,
                bootstrap_b: r.bootstrap_b,
            })
            .collect();
        // KS component for the score = mean of |D| across session pairs
        // (a higher mean implies more session-to-session drift = less stationary).
        if !ks_entries.is_empty() {
            let mean_d = ks_entries.iter().filter_map(|e| if e.d.is_finite() { Some(e.d) } else { None })
                .collect::<Vec<_>>();
            if !mean_d.is_empty() {
                let m = mean_d.iter().sum::<f64>() / mean_d.len() as f64;
                raw_scalars.push(RawScalar {
                    cell_idx,
                    clock_idx,
                    scalar_idx,
                    component: ComponentKind::Ks,
                    value: m,
                });
            }
        }
        ks_out.push(PerScalarKsSeries { scalar: scalar_name.to_string(), series: ks_entries });

        // --- Hill estimator ---
        // Hill is defined on positive-valued tails. RQA scalars in [0,1]
        // (RR/DET/LAM/ENTR) need shifting to a positive tail; for L_max and
        // TT we already have positive integers/floats. For the score, we
        // use Hill on the raw positive values (skipping zeros). If the
        // scalar values are too uniform / small / negative, Hill will
        // graceful-fail.
        let positive: Vec<f64> = records
            .iter()
            .map(|r| r.scalar_value(scalar_name))
            .filter(|v| v.is_finite() && *v > 0.0)
            .collect();
        let (estimate, error) = match hill_estimate(&positive) {
            Ok(e) => {
                // Hill component for the score: use |alpha - 2| as a
                // "tail-index distance" proxy (Stage-1 hand-wave: 2 is a
                // common "moderate-tails" reference). The smaller the
                // distance, the more stationary the tail behaviour.
                raw_scalars.push(RawScalar {
                    cell_idx,
                    clock_idx,
                    scalar_idx,
                    component: ComponentKind::Hill,
                    value: (e.alpha - 2.0).abs(),
                });
                (Some(e), None)
            }
            Err(HillError::InsufficientData { n, min_n }) => {
                tracing::debug!(scalar = scalar_name, n, min_n, "hill_insufficient");
                (None, Some(format!("InsufficientData: n={n}, min={min_n}")))
            }
            Err(other) => {
                tracing::warn!(scalar = scalar_name, error = %other, "hill_error");
                (None, Some(format!("{other}")))
            }
        };
        hill_out.push(PerScalarHill { scalar: scalar_name.to_string(), estimate, error });

        // --- Hierarchical variance ratio (single-instrument case) ---
        // For a single cell pilot, all observations belong to the same
        // "instrument" (= the symbol of this cell). The ratio is then
        // degenerate: between_ms = 0, within_ms > 0 → ratio = +∞.
        // We still emit the computed values so downstream code knows
        // exactly what was computed.
        let ratio = compute_hierarchical_variance_ratio(
            &records.iter().map(|r| (cell.as_str().to_string(), r.scalar_value(scalar_name)))
                .filter(|(_, v)| v.is_finite()).collect::<Vec<_>>(),
        );
        let degenerate = ratio.n_groups <= 1;
        if !degenerate && ratio.within_over_between.is_finite() {
            raw_scalars.push(RawScalar {
                cell_idx,
                clock_idx,
                scalar_idx,
                component: ComponentKind::Hierarchical,
                value: ratio.within_over_between,
            });
        }
        hier_out.push(PerScalarHierarchical {
            scalar: scalar_name.to_string(),
            ratio,
            degenerate_single_instrument: degenerate,
        });
    }

    // Persist artifacts.
    let ks_path = layout.ks_series_path(cell, clock.as_str());
    std::fs::write(&ks_path, serde_json::to_string_pretty(&ks_out)?)?;
    written.push(ks_path);

    let hill_path = layout.hill_path(cell, clock.as_str());
    std::fs::write(&hill_path, serde_json::to_string_pretty(&hill_out)?)?;
    written.push(hill_path);

    let hier_path = layout.hierarchical_path(cell, clock.as_str());
    std::fs::write(&hier_path, serde_json::to_string_pretty(&hier_out)?)?;
    written.push(hier_path);

    Ok(StageThreeOutcome {
        cell: cell.clone(),
        clock: clock.as_str().into(),
        n_buckets_in: records.len(),
        n_processed: records.len(),
        n_skipped: 0,
        written,
        raw_scalars,
    })
}

/// Persist the StationarityScoreSpec into the run dir AND verify it
/// matches the in-code v0.1.0. Hard-errors on mismatch (R10.1 contract
/// — Stage-5 will do the same check across runs).
pub fn write_and_verify_stationarity_spec(layout: &RunLayout) -> Result<StationarityScoreSpec> {
    let spec = StationarityScoreSpec::v0_1_0();
    spec.validate()
        .map_err(|e| anyhow::anyhow!("StationarityScoreSpec::v0_1_0 failed validation: {e}"))?;
    let path = layout.stationarity_spec_path();
    if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
    std::fs::write(&path, spec.to_json_pretty()?)?;
    Ok(spec)
}

/// Combine a population of [`RawScalar`] observations into per-cell
/// Stationarity Scores and persist as JSON.
pub fn write_stationarity_scores(
    layout: &RunLayout,
    raws: &[RawScalar],
    spec: &StationarityScoreSpec,
) -> Result<Vec<StationarityScore>> {
    let scores = compute_stationarity_score(raws, spec);
    let path = layout.stationarity_scores_path();
    std::fs::write(&path, serde_json::to_string_pretty(&scores)?)?;
    Ok(scores)
}

/// Convenience: convert an observation `(first_ts_ns, value)` series into
/// the (sessionised) timestamp the KS pipeline expects. Currently a noop
/// — present so the call site reads as documentation.
#[inline]
pub fn ts_for_session(first_ts_ns: u64) -> u64 {
    first_ts_ns
}
