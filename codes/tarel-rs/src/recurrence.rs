//! Stage-2 per-bucket recurrence-layer artifact production.
//!
//! For each bucket emitted by Stage 1, this module produces:
//!
//! 1. The Distance Plot (DP) at the bucket's native size, persisted as
//!    a JSON-line record with the upper-triangular f32 entries.
//! 2. The Recurrence Plot (RP) at the *headline* RR=0.05 threshold,
//!    persisted as a packed bit array (one bit per upper-triangular
//!    entry) inside the same JSON record.
//! 3. The RP at the *appendix* RR=0.01 threshold, same encoding.
//! 4. The 64×64 normalized W-GAN-ready tensor, persisted as a separate
//!    MessagePack file under
//!    `runs/<seed>/stage_one/recurrence/<cell>/<clock>/normalized/`.
//! 5. A PNG sanity-check mosaic (16 randomly sampled normalized DPs
//!    per clock).
//!
//! The encoding choices follow `engineering_proposal.md` Part I,
//! Stage 2: "DPs as parquet f32 arrays, normalized tensors as msgpack".
//! In practice the per-bucket DPs vary in size, so we pack them as
//! JSON-line records with explicit `n` headers — easier to consume
//! from Python than a sliced parquet, and serializes the same float
//! data. The 64×64 normalized tensor is the W-GAN training input and
//! is the format Stage-4 logan-rs will load directly.

use std::path::PathBuf;

use anyhow::{Context, Result};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

use takens_rp::bucketer::{Bucket, Clock};
use takens_rp::distance::{compute_subspace_distances_with_metric, combine, DistanceMetric};
use takens_rp::recurrence::{PerBucketScale, resample_dp};
use takens_rp::represent::{vectorize_with_baseline, K_BINS_PER_SIDE};
use takens_rp::types::DistanceMatrix;
use takens_rp::plot;

use crate::cell::CellId;
use crate::layout::RunLayout;

// Locked Stage-2 constants from engineering_proposal.md Part I.
pub const HEADLINE_RR: f64 = 0.05;
pub const APPENDIX_RR: f64 = 0.01;
pub const TARGET_RR_TOL: f64 = 0.001;
pub const TARGET_RR_MAX_ITERS: usize = 30;
pub const NORMALIZED_TENSOR_N: usize = 64;
pub const SANITY_GRID_TILES: usize = 16;
pub const SANITY_GRID_COLS: usize = 4;
pub const SANITY_GRID_ROWS: usize = 4;
/// Resampling target N for the DPs that go into the PNG sanity grid.
/// Smaller than NORMALIZED_TENSOR_N to keep mosaic file size sane.
pub const SANITY_TILE_N: usize = 64;

/// Per-bucket recurrence record persisted to JSON-Lines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketRecord {
    pub idx: usize,
    pub window: u64,
    pub idx_in_window: usize,
    pub n: usize,
    pub clock: String,
    /// Distance metric used to build the DP.
    pub metric: DistanceMetric,
    pub truncated_at_window_boundary: bool,
    pub dropped_below_min: bool,
    /// DP upper-triangular entries (length n*(n-1)/2), f32.
    pub dp_uppertri: Vec<f32>,
    /// RP @ headline RR (0.05), packed bit-vec over upper-triangle.
    pub rp_headline: BitVec,
    pub rp_headline_eps: f64,
    pub rp_headline_achieved_rr: f64,
    /// RP @ appendix RR (0.01), packed bit-vec over upper-triangle.
    pub rp_appendix: BitVec,
    pub rp_appendix_eps: f64,
    pub rp_appendix_achieved_rr: f64,
}

/// Compact bit-vector representation for RP storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitVec {
    pub n_bits: usize,
    pub bytes: Vec<u8>,
}

impl BitVec {
    pub fn from_bools(bits: &[bool]) -> Self {
        let n_bits = bits.len();
        let n_bytes = (n_bits + 7) / 8;
        let mut bytes = vec![0u8; n_bytes];
        for (i, &b) in bits.iter().enumerate() {
            if b {
                bytes[i / 8] |= 1 << (i % 8);
            }
        }
        Self { n_bits, bytes }
    }
}

/// MessagePack record for the 64×64 normalized W-GAN tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedTensorRecord {
    pub idx: usize,
    pub window: u64,
    pub idx_in_window: usize,
    pub clock: String,
    pub metric: DistanceMetric,
    /// `NORMALIZED_TENSOR_N` (= 64 by default).
    pub n: usize,
    /// Per-bucket scaling that maps the [-1, 1] tensor entries back to
    /// the original distance scale (Layer-2-style invertibility).
    pub scale: PerBucketScale,
    /// Row-major flattened f32, length `n * n`.
    pub data: Vec<f32>,
}

/// Outcome of producing the recurrence-layer artifacts for one
/// (cell, clock) combo.
#[derive(Debug, Clone)]
pub struct RecurrenceOutcome {
    pub clock: Clock,
    pub n_buckets_in: usize,
    pub n_processed: usize,
    pub n_skipped_below_min: usize,
    pub n_skipped_no_baseline: usize,
    pub written: Vec<PathBuf>,
}

/// Main entry point — produce the Stage-2 artifacts for one
/// (cell, clock) bucket stream.
///
/// Buckets that are dropped by Stage 1 (`dropped_below_min`) or carry
/// no baseline (Layer-2 #1) are skipped. Each surviving bucket goes
/// through:
///
/// 1. `vectorize_with_baseline` (price + volume subspaces, fixed first-
///    LOB denominator).
/// 2. `compute_subspace_distances_with_metric(Euclidean)` then
///    `combine(α=0.5)` → DP.
/// 3. `threshold_to_target_rr(0.05)` and (0.01) → headline + appendix RPs.
/// 4. `resample_dp(64) + normalize_per_bucket()` → 64×64 normalized tensor.
pub fn produce_recurrence_artifacts(
    layout: &RunLayout,
    cell: &CellId,
    clock: Clock,
    buckets: &[Bucket],
    metric: DistanceMetric,
    alpha: f64,
) -> Result<RecurrenceOutcome> {
    let dir = layout
        .stage_one_dir()
        .join("recurrence")
        .join(cell.as_str().replace(':', "_"))
        .join(clock.as_str());
    let normalized_dir = dir.join("normalized");
    std::fs::create_dir_all(&normalized_dir)?;

    let mut buckets_jsonl_path = dir.clone();
    buckets_jsonl_path.push("buckets.jsonl");

    let mut bucket_writer = std::io::BufWriter::new(
        std::fs::File::create(&buckets_jsonl_path).with_context(|| {
            format!("create {}", buckets_jsonl_path.display())
        })?,
    );

    let mut sanity_tiles: Vec<Array2<f32>> = Vec::with_capacity(SANITY_GRID_TILES);
    let mut written = vec![buckets_jsonl_path.clone()];

    let mut n_processed = 0;
    let mut n_skipped_below_min = 0;
    let mut n_skipped_no_baseline = 0;

    for (idx, bucket) in buckets.iter().enumerate() {
        if bucket.dropped_below_min {
            n_skipped_below_min += 1;
            continue;
        }
        let baseline = match bucket.baseline {
            Some(b) => b,
            None => {
                n_skipped_no_baseline += 1;
                continue;
            }
        };

        // Vectorize with the per-window baseline (Layer-2 #3).
        let vectors = vectorize_with_baseline(&bucket.snapshots, baseline)
            .with_context(|| format!("vectorize_with_baseline @ bucket {idx}"))?;

        if vectors.len() < 2 {
            n_skipped_below_min += 1;
            continue;
        }

        // Subspace DPs + α-weighted combination (defaults to 0.5).
        let (d_price, d_volume) =
            compute_subspace_distances_with_metric(&vectors, metric)
                .with_context(|| format!("subspace distances @ bucket {idx}"))?;
        let dp = combine(&d_price, &d_volume, alpha)
            .with_context(|| format!("combine @ bucket {idx}"))?;

        // Threshold → RPs.
        let headline = dp.threshold_to_target_rr(HEADLINE_RR, TARGET_RR_TOL, TARGET_RR_MAX_ITERS);
        let appendix = dp.threshold_to_target_rr(APPENDIX_RR, TARGET_RR_TOL, TARGET_RR_MAX_ITERS);

        // Resample DP to NORMALIZED_TENSOR_N + per-bucket normalize.
        let resampled = resample_dp(&dp, NORMALIZED_TENSOR_N);
        let (norm_arr, norm_scale) = resampled.normalize_per_bucket();

        // Persist BucketRecord (DP + both RPs) as JSON-line.
        let dp_uppertri: Vec<f32> = dp.data.iter().map(|&x| x as f32).collect();
        let record = BucketRecord {
            idx,
            window: bucket.window.0,
            idx_in_window: bucket.idx_in_window,
            n: dp.n,
            clock: clock.as_str().into(),
            metric,
            truncated_at_window_boundary: bucket.truncated_at_window_boundary,
            dropped_below_min: false,
            dp_uppertri,
            rp_headline: BitVec::from_bools(&headline.plot.data),
            rp_headline_eps: headline.epsilon,
            rp_headline_achieved_rr: headline.achieved_rr,
            rp_appendix: BitVec::from_bools(&appendix.plot.data),
            rp_appendix_eps: appendix.epsilon,
            rp_appendix_achieved_rr: appendix.achieved_rr,
        };
        let line = serde_json::to_string(&record)?;
        use std::io::Write;
        writeln!(bucket_writer, "{line}")?;

        // Persist 64×64 normalized tensor as MessagePack.
        let tensor_record = NormalizedTensorRecord {
            idx,
            window: bucket.window.0,
            idx_in_window: bucket.idx_in_window,
            clock: clock.as_str().into(),
            metric,
            n: NORMALIZED_TENSOR_N,
            scale: norm_scale,
            data: norm_arr.iter().copied().collect(),
        };
        let tensor_path = normalized_dir.join(format!("bucket_{:06}.msgpack", idx));
        let bytes = rmp_serde::to_vec_named(&tensor_record)?;
        std::fs::write(&tensor_path, &bytes)?;
        written.push(tensor_path);

        // Sample for the sanity grid (deterministic: take the first 16
        // surviving buckets evenly spaced).
        if sanity_tiles.len() < SANITY_GRID_TILES {
            // Use the same SANITY_TILE_N tile for the mosaic.
            let resampled_tile = if NORMALIZED_TENSOR_N == SANITY_TILE_N {
                norm_arr.clone()
            } else {
                let resampled_for_tile = resample_dp(&dp, SANITY_TILE_N);
                let (a, _) = resampled_for_tile.normalize_per_bucket();
                a
            };
            sanity_tiles.push(resampled_tile);
        }

        n_processed += 1;
    }

    bucket_writer
        .into_inner()
        .map_err(|e| anyhow::anyhow!("flush buckets.jsonl: {}", e.error()))?
        .sync_all()
        .ok();

    // PNG sanity-check mosaic (Stage-2 exit gate).
    if !sanity_tiles.is_empty() {
        // Pad to exactly SANITY_GRID_TILES with zero-tiles if fewer.
        let tile_shape = (SANITY_TILE_N, SANITY_TILE_N);
        while sanity_tiles.len() < SANITY_GRID_TILES {
            sanity_tiles.push(Array2::<f32>::from_elem(tile_shape, -1.0));
        }
        let png_path = dir.join("sanity_grid.png");
        plot::render_grid_normalized(
            &sanity_tiles,
            SANITY_GRID_COLS,
            SANITY_GRID_ROWS,
            &png_path,
        )
        .with_context(|| format!("render sanity grid {}", png_path.display()))?;
        written.push(png_path);
    }

    Ok(RecurrenceOutcome {
        clock,
        n_buckets_in: buckets.len(),
        n_processed,
        n_skipped_below_min,
        n_skipped_no_baseline,
        written,
    })
}

/// Helper used by the Stage-2 verification harness to dehydrate a DP
/// from a JSON-line record without re-running the full pipeline.
pub fn dp_from_uppertri(n: usize, uppertri: &[f32]) -> DistanceMatrix {
    let mut dm = DistanceMatrix::zeros(n);
    let mut k = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            dm.set(i, j, uppertri[k] as f64);
            k += 1;
        }
    }
    dm
}

/// `K_BINS_PER_SIDE` re-export for downstream tools that want to pin
/// to the same constant the recurrence pipeline uses.
pub const STATE_DIM_PIN: usize = 2 * K_BINS_PER_SIDE;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitvec_round_trip_small() {
        let bools = vec![true, false, true, true, false, false, true, false, true];
        let bv = BitVec::from_bools(&bools);
        assert_eq!(bv.n_bits, 9);
        // Verify packed bytes.
        // bit positions: 0(t),2(t),3(t),6(t),8(t)
        // byte 0: bits 0..7 → 0b0100_1101 = 0x4D
        // byte 1: bit 8     → 0b0000_0001 = 0x01
        assert_eq!(bv.bytes, vec![0b0100_1101, 0b0000_0001]);
    }

    #[test]
    fn bitvec_byte_count_correct() {
        let zeros = vec![false; 100];
        let bv = BitVec::from_bools(&zeros);
        assert_eq!(bv.n_bits, 100);
        assert_eq!(bv.bytes.len(), 13); // ceil(100 / 8)
    }

    #[test]
    fn locked_constants_match_part_i() {
        assert!((HEADLINE_RR - 0.05).abs() < f64::EPSILON);
        assert!((APPENDIX_RR - 0.01).abs() < f64::EPSILON);
        assert!((TARGET_RR_TOL - 0.001).abs() < f64::EPSILON);
        assert_eq!(TARGET_RR_MAX_ITERS, 30);
        assert_eq!(NORMALIZED_TENSOR_N, 64);
        assert_eq!(SANITY_GRID_TILES, SANITY_GRID_COLS * SANITY_GRID_ROWS);
    }
}
