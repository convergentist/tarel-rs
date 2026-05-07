# HFT-RP-WGAN — Implementation Requirements

**Audience:** Research lead (PI) and implementing engineers.
**Purpose:** Translate the combined research proposal (Proposal 1 + Sub-proposal 1.A) into an implementable Rust / tch-rs system. Identify under-specified decisions, propose concrete defaults that will be assumed in absence of explicit pushback, and define the contracts between subsystems so that work can parallelize.
**Convention:** Each subsystem section ends with an **Open Questions** block. Items marked **[ASSUME]** carry a default the engineer will adopt if no answer is forthcoming by phase start. Items marked **[BLOCKING]** must be answered before that subsystem can be started.

---

## Part I — System Specification

### I.1 End-to-End Data Flow

```
raw parquet (Layer 1)
    │  hft_data
    ▼
bps + volume-scaled snapshot stream (Layer 2)
    │  hft_layer2
    ▼
bucket stream × 3 windowing schemes
    │  hft_bucketing
    ▼
distance plot / recurrence plot per bucket
    │  hft_recurrence
    ├──────────────────────────────────────────────┐
    ▼                                              ▼
RQA scalars per bucket                  64×64 normalized DP tensor
    │  hft_rqa                              │  hft_recurrence::resample
    ▼                                              ▼
Stage-One statistical tests             W-GAN-GP training (30 seeds)
    │  hft_stats                            │  hft_wgan + hft_diagnostics
    ▼                                              ▼
per-cell Stationarity Score             per-cell Stability Score
    └──────────────────────┬───────────────────────┘
                           ▼
              bridge analysis (Spearman, partial Spearman, bootstrap)
                           │  hft_bridge
                           ▼
                   reportable outputs (figures, tables)
```

### I.2 Top-Level Crate Layout

A Cargo workspace with one binary crate and ten library crates. Each library crate exposes a narrow, typed API and depends only on crates strictly above it in the topological order below.

```
hft_rp_wgan/
├── crates/
│   ├── hft_types/           # shared Newtypes, error enum, time types
│   ├── hft_data/            # parquet I/O, schema validation
│   ├── hft_layer2/          # bps aggregation, volume scaling
│   ├── hft_bucketing/       # three windowing schemes
│   ├── hft_recurrence/      # DP, RP, resampling, normalization
│   ├── hft_rqa/             # six RQA scalars
│   ├── hft_stats/           # KS, Hill, hierarchical model, bootstrap
│   ├── hft_wgan/            # generator, critic, GP training loop
│   ├── hft_diagnostics/     # four stability instruments
│   ├── hft_bridge/          # Spearman + partial Spearman + bootstrap
│   └── hft_cli/             # binary: subcommands per phase
└── Cargo.toml
```

The strict topological dependency rule ensures no subsystem can implicitly leak across boundaries; the wgan crate cannot reach into the data layer, and the bridge analysis sees only typed score artifacts.

### I.3 Scale of the Experimental Grid

Total runs: 3 clocks × 3 exchanges × 4 symbols × 30 seeds = 1,080 W-GAN-GP training jobs for Stage Two, plus 36 cell-level RQA evaluations for Stage One. The orchestration layer must treat 1,080 runs as a first-class workload, not as a one-off scripted loop.

---

## Part II — Requirements by Subsystem

### II.1 `hft_types` — Shared Domain Types

**Purpose.** Eliminate primitive obsession across the workspace. Every subsystem speaks the same vocabulary.

**Required types (non-exhaustive):**

```rust
// Strongly-typed identifiers
pub struct ExchangeId(SmolStr);          // "binance" | "coinbase" | "kraken"
pub struct SymbolId(SmolStr);            // e.g. "BTC-USDT"
pub struct ClockKind { Physical60s, FixedEvent { n: u32 }, InfoTime { vol_threshold: f64 } }

// Time
pub struct EventTimeNs(i64);             // exchange-side timestamp, nanoseconds since Unix epoch
pub struct IngestTimeNs(i64);            // local capture timestamp
pub struct BucketIdx(u64);               // monotonic per (cell, run)

// Geometry
pub const K_BINS_PER_SIDE: usize = 25;   // 10-bps spacing × 25 = 250 bps half-width
pub const STATE_DIM: usize = 2 * K_BINS_PER_SIDE;  // 50

pub struct LobSnapshot { ... }           // bps-keyed, volume-scaled, length 50
pub struct PtSnapshot { signed_volume: f64, mid_shift_bps: f64, ts: EventTimeNs }

// Plots
pub struct DistancePlot { data: ndarray::Array2<f32>, n: usize }   // n × n, f32
pub struct RecurrencePlot { data: ndarray::Array2<u8>, n: usize, eps: f32, achieved_rr: f32 }

// Errors
pub enum HftError { Io(...), Schema(...), Bucketing(...), Numeric(...), Tch(...) }
```

**Requirements.**

R1.1 All fallible operations return `Result<T, HftError>`; no `panic!` in non-test code.
R1.2 All public types implement `Serialize + Deserialize` (serde) and `Debug`. The serialization format is JSON for inspection, MessagePack (rmp-serde) for hot paths.
R1.3 Numeric work uses `f64` in computation, `f32` in storage and on GPU. Conversions are explicit at boundaries.

**Open Questions.**

Q1.1 [ASSUME] Symbol set is `{"BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT"}` unless told otherwise.
Q1.2 [BLOCKING] Confirm the canonical exchange set as `{binance, coinbase, kraken}` and confirm we are treating the per-exchange spelling of each symbol (e.g. Coinbase uses `BTC-USD` not `BTC-USDT`) as opaque from the analysis perspective once normalized.
Q1.3 [ASSUME] `EventTimeNs` is the exchange-reported timestamp where available, falling back to `IngestTimeNs` only when the exchange doesn't provide one.

---

### II.2 `hft_data` — Parquet Ingestion (Layer 1 Reader)

**Purpose.** Read the existing per-exchange / per-instrument 60-minute Parquet files and yield typed snapshot streams.

**Required API (sketch):**

```rust
pub struct DatasetReader { /* root path, schema cache */ }
impl DatasetReader {
    pub fn iter_lob(&self, exchange: ExchangeId, symbol: SymbolId, window: TimeRange)
        -> impl Iterator<Item = Result<RawLobSnapshot, HftError>>;
    pub fn iter_pt(&self, ...) -> impl Iterator<Item = Result<RawPtEvent, HftError>>;
}
```

**Requirements.**

R2.1 The reader must validate parquet schema on first file open per (exchange, symbol) and cache the schema. Schema mismatch is a hard error, not a warning.
R2.2 The iterator yields snapshots in monotonic event-time order. Out-of-order events within a 250 ms bucket are tolerated (sorted on read); out-of-order events across buckets fail the iterator with `HftError::Schema`.
R2.3 The reader supports zero-copy reads via Arrow where possible; the Layer 2 transform consumes Arrow `RecordBatch` and emits `LobSnapshot` without intermediate `Vec` allocation per snapshot.
R2.4 The reader exposes a `WindowSet` query that returns the union of all available 60-minute Parquet windows per (exchange, symbol), which the orchestrator uses to plan the seed grid.

**Open Questions.**

Q2.1 [BLOCKING] Provide the exact Parquet schema for each of the four file types (LOB-Binance, LOB-Coinbase, LOB-Kraken, and the symmetric three for PT). Specifically: column names, types, units (price in quote currency vs ticks; volume in base currency vs raw lot count; signed-volume convention for PT — buyer-initiated positive or sell-aggressor positive).
Q2.2 [BLOCKING] Confirm the LOB snapshot encoding — is it (a) full depth-25 bid/ask arrays per row, (b) one row per level with a snapshot id, or (c) delta-encoded? The Layer 2 transform's complexity depends entirely on this.
Q2.3 [ASSUME] PT events arrive in the file in event-time order; we will sort on read if violated and log a counter.
Q2.4 [BLOCKING] Are timestamps in nanoseconds, microseconds, or milliseconds? Default assumption is nanoseconds; deviation requires a cast and a monotonicity recheck.

---

### II.3 `hft_layer2` — bps Aggregation and Volume Scaling

**Purpose.** Convert raw LOB snapshots to the bps-keyed, volume-scaled state vector x_t ∈ ℝ^{50}.

**Algorithm (as specified by research):**

Given a raw LOB snapshot with absolute prices and volumes:
1. Compute best bid `bb` and best ask `ba`.
2. For each bid level at price p with volume v, compute `bps_offset = (bb - p) * 10000 / bb`. Negative offsets (above best bid) are an error.
3. Bin by `floor(bps_offset / 10.0)` into bins `0, 1, ..., K-1`. Levels with bin index ≥ K are dropped.
4. Sum volumes within each bin. Bins with no level present hold 0.
5. Mirror for ask side using `ba`.
6. Concatenate: `x_t = [bid_bins, ask_bins]`, length `2K = 50`.
7. **Volume scaling**: for the first snapshot in each Parquet window, compute `total_bid_vol_first` and `total_ask_vol_first`. For all subsequent snapshots in the window, divide bid bins by `total_bid_vol_first` and ask bins by `total_ask_vol_first`.

**Requirements.**

R3.1 The transform is pure and stateful only within a single Parquet window. Crossing windows resets the volume-scaling baseline, which means buckets must not span window boundaries (see II.4).
R3.2 The `(total_bid_vol_first, total_ask_vol_first)` baselines are persisted alongside the transformed snapshots so that downstream analyses can recover absolute volumes if needed.
R3.3 Transform is benchmarked: target throughput ≥ 200,000 snapshots/sec/core on the reference workstation. Any approach slower than this is rejected.

**Open Questions.**

Q3.1 [BLOCKING] What happens if the first LOB snapshot of a Parquet file has zero total volume on one side (illiquid moment, halted trading)? Three options: (a) propagate forward to find the first non-zero, (b) drop the entire window, (c) use a fallback constant. The choice affects how many windows survive into Stage One.
Q3.2 [ASSUME] Levels falling outside the K-bin range (>= 250 bps from best) are silently dropped. The drop count is logged per window and surfaced in the data-quality report.
Q3.3 [BLOCKING] When a level exactly equals the best price (bps_offset == 0), is its volume credited to bin 0 or excluded? Exact-best handling matters at moments of price improvement.
Q3.4 [BLOCKING] Confirm the volume-scaling denominator is `total side volume of the first LOB`, not e.g. `mean of first N LOBs` or `top-of-book volume only`.

---

### II.4 `hft_bucketing` — Three Windowing Schemes

**Purpose.** Group the Layer-2 snapshot stream into buckets according to one of three rules, exposing a uniform bucket iterator.

**Required API:**

```rust
pub trait Bucketer {
    fn next_bucket(&mut self) -> Option<Bucket>;
}
pub struct Bucket {
    pub clock: ClockKind,
    pub idx: BucketIdx,
    pub lob_snapshots: Vec<LobSnapshot>,
    pub pt_events: Vec<PtSnapshot>,
    pub start_ts: EventTimeNs,
    pub end_ts: EventTimeNs,
}

pub struct PhysicalBucketer { window_ns: i64, /* state */ }      // 60_000_000_000 default
pub struct EventBucketer    { n: u32, /* state */ }               // N=64 default
pub struct InfoTimeBucketer { vol_threshold: f64, /* state */ }   // per-cell calibrated
```

**Requirements.**

R4.1 Buckets must not cross 60-minute Parquet window boundaries (per R3.1). A bucket whose accumulation would cross a boundary is closed at the boundary and a new bucket begins on the other side. The closed bucket is tagged `truncated_at_window_boundary: true` and excluded from analysis if its snapshot count is below a configurable minimum (default 8 for fixed-physical, 8 for info-time).
R4.2 The info-time `vol_threshold` is calibrated per (exchange, symbol) so that the median bucket duration approximates the 60-second physical baseline, ensuring rough comparability of bucket counts across clocks. Calibration is performed on a held-out warm-up week of data and persisted as part of the cell metadata.
R4.3 Each `Bucket` carries enough metadata for the recurrence layer to construct DP+RP without reaching back into the snapshot store.
R4.4 The bucketer is deterministic given the same input stream and same configuration. Determinism is asserted by a property test in CI.

**Open Questions.**

Q4.1 [BLOCKING] For `EventBucketer`, confirm `N = 64` is the default. The choice ties the W-GAN architecture (64×64) to native-shape DPs and is load-bearing for the resampling-ablation in §5.2.
Q4.2 [BLOCKING] For `InfoTimeBucketer`, the calibration target — "approximate the 60-second median" — needs a tolerance band. Proposal: calibrate so that the median info-time bucket lasts between 30 s and 120 s, rejecting calibrations outside this range.
Q4.3 [ASSUME] Minimum snapshots per bucket for inclusion is 8. Buckets below this are dropped from RQA and from W-GAN training inputs but logged.
Q4.4 [BLOCKING] PT events: do they belong to the bucket whose LOB snapshots cover their event time, or do they form an independent parallel bucketing? The proposal says "parallel set of plots" for PT — confirm that PT bucketing uses the same boundaries as LOB bucketing within each clock, just with PT events as the embedded sequence.

---

### II.5 `hft_recurrence` — Distance Plot, Recurrence Plot, Resampling

**Purpose.** Compute DP[i,j] = ||x_i − x_j||_p for each bucket, threshold to RP, and resample to the W-GAN's 64×64 input shape.

**Required API:**

```rust
pub enum DistanceMetric { Euclidean, WassersteinOne }
pub fn compute_dp(bucket: &Bucket, metric: DistanceMetric) -> DistancePlot;
pub fn threshold_to_rp(dp: &DistancePlot, target_rr: f32) -> RecurrencePlot;
pub fn resample_dp(dp: &DistancePlot, target_n: usize, kind: ResampleKind) -> DistancePlot;
pub fn normalize_for_wgan(dp: &DistancePlot) -> ndarray::Array2<f32>;  // -> [-1, 1]
```

**Requirements.**

R5.1 Euclidean DP uses pairwise SIMD-accelerated computation (manual or via `wide` / `packed_simd` / `ndarray-linalg` BLAS depending on benchmarking). Target: a 256×256 DP (worst-case bucket size) computes in < 5 ms on a single core.
R5.2 Wasserstein-1 DP treats each LOB snapshot as a discrete measure on the 50 bins. Implementation uses 1D-EMD via cumulative-difference formula since bins are ordered: `W1(u, v) = sum_b |U_b - V_b|` where U, V are CDFs. This is O(50) per pair, dramatically cheaper than general OT.
R5.3 RP thresholding to a target recurrence rate uses bisection on ε with at most 30 iterations and an absolute RR tolerance of 0.001. The achieved RR is reported in the `RecurrencePlot` struct.
R5.4 Resampling preserves the symmetry and zero-diagonal of DP. The bicubic implementation must be tested against a reference (scipy or matlab) on a small benchmark set, with max abs error < 1e-4.
R5.5 Normalization to [−1, 1] uses per-bucket min-max of the DP — explicit, not learned. The min and max are persisted per bucket so that normalized→raw inversion is possible.

**Open Questions.**

Q5.1 [ASSUME] Default norm for Stage One headline is Euclidean; Wasserstein-1 is the ablation reported in the appendix.
Q5.2 [BLOCKING] Recurrence rate targets — confirm both `RR = 1%` and `RR = 5%` are reported. If only one, which is the headline?
Q5.3 [ASSUME] Per-bucket min-max normalization for W-GAN inputs. Alternative (global normalization across the corpus) would change cross-cell comparability and is rejected unless requested.
Q5.4 [BLOCKING] For PT plots, the embedding is `(signed_volume, mid_shift_bps)` per snapshot — confirm this is the dimensionality (2-D) and confirm the distance metric (Euclidean by default, with both dimensions standardized to unit variance per cell over the warm-up week).
Q5.5 [BLOCKING] For resampling: should bucket-shape DPs that are *smaller* than 64×64 be upsampled, or should such buckets be dropped from W-GAN training? Upsampling small DPs may introduce artifacts; dropping reduces effective training set.

---

### II.6 `hft_rqa` — Six RQA Scalars

**Purpose.** From a thresholded RP, compute the six standard RQA scalars used as Stationarity-Score components.

**Required outputs per bucket:**

```rust
pub struct RqaScalars {
    pub recurrence_rate: f64,         // RR
    pub determinism: f64,             // DET
    pub laminarity: f64,              // LAM
    pub trapping_time: f64,           // TT (mean vertical line length)
    pub l_max: u32,                   // longest diagonal (excluding LOI)
    pub diag_entropy: f64,            // ENTR (Shannon entropy of diagonal lengths)
}
```

**Requirements.**

R6.1 Diagonal and vertical line extraction excludes the line-of-identity (main diagonal) for L_max and ENTR computations. Minimum line length parameter `l_min = 2` is the standard convention; this is the only RQA hyperparameter and is fixed across the study.
R6.2 The implementation must match a reference (PyRQA or CRP-toolbox) on a 10-bucket fixture set with absolute error < 1e-9 for RR/DET/LAM, exact match for L_max, and < 1e-6 for TT/ENTR. CI fails on regression.
R6.3 Computation runs at ≥ 1,000 buckets/sec/core on representative bucket sizes (avg 100×100 RP). Linear scan over the upper triangle is sufficient; no need for FFT-based methods at these sizes.

**Open Questions.**

Q6.1 [ASSUME] `l_min = 2` for both diagonal and vertical line definitions, per literature standard.
Q6.2 [BLOCKING] ENTR base of logarithm — natural log or base 2? Both are used in the literature; pick one and lock it.
Q6.3 [ASSUME] Diagonals of all lengths contribute to ENTR; we do not truncate to a `theiler_window` since the line-of-identity exclusion subsumes this.

---

### II.7 `hft_stats` — Statistical Tests for Stage One

**Purpose.** Compute the three Stage-One statistical tests and the per-cell Stationarity Score.

**Required tests:**

1. **Two-sample KS distance** between RQA distributions of successive trading sessions, per cell, per scalar. Output: a session-pair time series of KS distances per (cell, scalar).
2. **Hierarchical variance decomposition** under a two-level model with instrument as grouping factor; output: ratio of within-instrument to between-instrument variance per (clock, scalar). Implementation may use a simple ANOVA-style decomposition (no MCMC needed for this scale).
3. **Hill estimator tail index** on upper and lower tails of each RQA distribution per (cell, scalar). Output: tail index estimates with bootstrap CIs.

**Stationarity Score aggregation:** A weighted sum of normalized test outputs with weights set in advance and frozen before any inference is run. Default weights: `[0.4 KS, 0.3 hierarchical, 0.3 Hill]`. Each test output is rank-normalized within the (clock, scalar) population before weighting.

**Requirements.**

R7.1 All tests support bootstrap resampling with B = 1000 by default. Bootstrap is parallelized via Rayon.
R7.2 Pre-registration: the weight vector and rank-normalization rule are persisted as a versioned `StationarityScoreSpec` struct, serialized to JSON, and committed to the repo before Stage Two begins. Any change to the spec invalidates Stage One's results and the bridge analysis.
R7.3 Numerical stability: Hill estimator implementations notoriously misbehave on small samples; minimum sample size of 200 is enforced, with smaller samples returning a typed `InsufficientData` variant.

**Open Questions.**

Q7.1 [BLOCKING] Definition of "trading session" for the KS test — calendar day in UTC, calendar day in exchange-local time, or 24-hour rolling windows? Crypto markets don't close, so this is purely a partitioning convention with no natural answer.
Q7.2 [BLOCKING] Hill estimator tail fraction `k` — a fixed fraction of sample size (e.g. top 10%) or selected via the Hill plot's stability region per sample? Fixed is reproducible; selected is better-calibrated. Recommend fixed `k = floor(0.1 * n)`.
Q7.3 [BLOCKING] Confirm the Stationarity Score weight vector. If different weights are preferred, lock them now.

---

### II.8 `hft_wgan` — W-GAN-GP Generator, Critic, Training Loop

**Purpose.** Implement the architecture and training procedure exactly as specified in §5 of the proposal, using `tch-rs`.

**Architecture (frozen, no per-clock variation):**

Generator: 128-d latent → ConvT(128→512, k4 s1 p0) → BN → ReLU → ConvT(512→256, k4 s2 p1) → BN → ReLU → ConvT(256→128, k4 s2 p1) → BN → ReLU → ConvT(128→64, k4 s2 p1) → BN → ReLU → ConvT(64→1, k4 s2 p1) → Tanh. Output: 1×64×64.

Critic: Conv(1→64, k4 s2 p1) → LeakyReLU(0.2) → Conv(64→128, k4 s2 p1) → LeakyReLU(0.2) → Conv(128→256, k4 s2 p1) → LeakyReLU(0.2) → Conv(256→512, k4 s2 p1) → LeakyReLU(0.2) → Conv(512→1, k4 s1 p0). Output: scalar per sample. No BN (per-sample GP requirement). LeakyReLU hand-rolled as `clamp_min(0) + 0.2 * clamp_max(0)` per the implementation note.

**Training loop:**

For each generator step, perform 5 critic steps. Each critic step samples a real batch x and a fake batch x̃ = G(z), draws ε ∼ U(0,1)^B, forms x̂ = εx + (1−ε)x̃, computes D(x̂), backprops through input to obtain ∇_x̂ D(x̂), and computes loss = D(x̃).mean() − D(x).mean() + λ_GP * (||∇_x̂ D(x̂)||_2 − 1)².mean(). Generator step minimizes −D(G(z)).mean().

**Hyperparameters (frozen):** Adam β1=0, β2=0.9; learning rate 1e-4 for both networks; λ_GP = 10; batch size 64; latent dim 128; critic ratio 5:1.

**Requirements.**

R8.1 The generator uses `nn::SequentialT` (not `nn::Sequential`) since BN implements `ModuleT`. The critic uses `nn::Sequential` since it has no train/eval-aware layers. Verified by `cargo expand` in CI.
R8.2 The W-GAN training loop runs for a fixed wallclock budget per seed (default 30 minutes on a single A100, parameter-overridable). Wallclock-based termination is fairer than epoch-based given that the three clocks produce different bucket counts per data hour.
R8.3 Determinism: seeded `tch::Cuda::manual_seed_all(seed)` and a separate seed for the data sampler, both derived from a single experiment-level seed via SHA-256 hash → u64. Two runs with the same experiment-level seed must produce bit-identical critic loss curves on a single GPU; tested in CI on a tiny configuration.
R8.4 Mixed precision: NOT used in the headline experiments. The gradient-penalty term is numerically sensitive and the small architecture doesn't benefit much. FP32 throughout.
R8.5 Single-GPU per training run. Multi-GPU parallelism is at the orchestration layer (multiple seeds in parallel across multiple GPUs), never within a single seed.
R8.6 Failure handling: if critic loss returns NaN within the first 1000 steps, the run is recorded as a failure with reason `early_nan`, the seed and configuration are persisted, and the run terminates. Failures are first-class data points (proposal §9, fifth risk).

**Open Questions.**

Q8.1 [ASSUME] Wallclock budget: 30 minutes per seed on A100, scaled proportionally for other GPUs. Total Stage Two: ~540 GPU-hours under the proposal's nominal estimate.
Q8.2 [BLOCKING] Confirm Adam learning rate of 1e-4 for both networks. The proposal does not state this explicitly; 1e-4 is the W-GAN-GP literature default.
Q8.3 [BLOCKING] Should the generator's BatchNorm be retained as-is, or pre-emptively swapped to GroupNorm/LayerNorm in anticipation of future conditional training (Sub-proposal 1.C)? Decision now affects whether the architecture is "frozen" across the trajectory.
Q8.4 [ASSUME] Checkpoint frequency: every 5,000 generator steps, plus a final checkpoint at termination. Checkpoints store generator+critic+optimizer state and the four diagnostics' rolling buffers.

---

### II.9 `hft_diagnostics` — Four Stability Instruments

**Purpose.** Implement the four-axis instrumentation that defines Stage Two's measurement.

**Required diagnostics (per training run, persisted as time series + summary statistics):**

1. **EM-gap proxy.** Per generator step, record `D(real).mean() − D(fake).mean()`. Smoothed over a 100-step rolling window. Output: full time series + final-100-step mean and standard deviation.
2. **Gradient-penalty magnitude.** Per critic step, record the scalar `(||∇_x̂ D(x̂)||_2 − 1)²` averaged over the batch. Output: full time series + empirical CDF aggregated over the run.
3. **Mode coverage.** At each checkpoint, generate 10,000 samples; for each, find its nearest training-set neighbor under L2 in pixel space; count unique training points that are the nearest neighbor of at least one sample; report this count and the full nearest-neighbor distance distribution. The L2 cutoff for "near match" is reported as a CDF rather than a single threshold (per proposal §9, fourth risk).
4. **Critic-loss curvature.** Approximated via finite-difference second-order along the realized generator update direction. At each checkpoint: compute D-loss at three points along the realized last generator step (− δ, 0, + δ), where δ = 0.1 × ||Δθ_G|| for the realized step Δθ_G. Curvature estimate = (L(−δ) − 2L(0) + L(+δ)) / δ². Report distribution across checkpoints.

**Requirements.**

R9.1 Each diagnostic is computed inline with training (no separate post-hoc pass), with the exception of mode coverage which is checkpoint-only due to its cost.
R9.2 Diagnostic outputs are streamed to disk in MessagePack-encoded chunks of 1,000 entries to bound memory.
R9.3 The Stability Score aggregation rule is pre-registered analogously to the Stationarity Score: a fixed weight vector over normalized diagnostic outputs, frozen before bridge inference. Default weights: `[0.25 EM-gap variance, 0.25 GP-magnitude IQR, 0.25 mode coverage count, 0.25 |curvature| median]`. Each diagnostic is rank-normalized within the (clock, exchange, symbol) population (across seeds) before weighting.
R9.4 The mode-coverage nearest-neighbor search uses an exact GPU brute-force scan (10k generated × N training, with N ≤ 10k typically). Approximate NN methods are forbidden because they introduce a hyperparameter that confounds the comparison across clocks.

**Open Questions.**

Q9.1 [BLOCKING] Curvature finite-difference step δ — is `0.1 × ||Δθ_G||` reasonable, or should this be an independently swept hyperparameter? Proposal says "the more involved diagnostic" but does not specify δ.
Q9.2 [BLOCKING] Confirm Stability Score weights. Equal weights are the proposed default; if a different weighting reflects the research intent better, fix it now.
Q9.3 [ASSUME] Mode-coverage L2 cutoff is reported as a curve; the headline metric for the Stability Score is the count at the median per-cell training-point pairwise L2 distance.

---

### II.10 `hft_bridge` — Spearman Correlation, Partial Correlation, Bootstrap

**Purpose.** Test H_bridge by correlating per-cell Stationarity Scores with per-cell Stability Scores across the 36-cell grid.

**Required analyses:**

1. Spearman ρ between Stationarity Score and Stability Score across all 36 cells. Bootstrap CI via B = 10,000 resamples.
2. Partial Spearman ρ conditioning on instrument identity. Implementation: rank-residualize each score against instrument dummy variables, then compute Spearman on the residuals. Bootstrap CI similarly.
3. Permutation null distribution: permute cell labels independently within each clock (preserving clock-level marginals) B = 10,000 times; report the permutation p-value for the observed ρ.

**Requirements.**

R10.1 The bridge analysis runs only after both Stationarity Scores and Stability Scores are finalized and their `Spec` versions match. Mismatched specs cause a hard error.
R10.2 All reported correlations include both point estimate and 95% bootstrap CI.
R10.3 The output is a small structured report (JSON + a single PDF figure showing the 36-cell scatter with marginal histograms by clock).

**Open Questions.**

Q10.1 [BLOCKING] Confirm that the bridge test is one-sided (predicting positive correlation) or two-sided. Pre-registration should pick one.
Q10.2 [ASSUME] Partial correlation conditions only on instrument identity, not on exchange identity. If exchange should also be conditioned on, the cell count drops the residual variance further and the test becomes more conservative — flag if this is preferred.

---

## Part III — Cross-Cutting Requirements

### III.1 Reproducibility

X1.1 A single experiment-level seed (u64) determines all derived seeds in the system. Two runs of the entire pipeline with the same experiment seed and the same data snapshot produce bit-identical Stage One reports and bit-identical Stage Two diagnostic time series, modulo nondeterministic CUDA kernels (cuDNN deterministic flag is set; see X1.4).
X1.2 The data snapshot is identified by a content hash of the canonical Parquet file set, computed and stored at orchestration start.
X1.3 The codebase is pinned to specific versions of `tch-rs`, LibTorch, CUDA, and cuDNN; these versions are recorded in the run manifest. A version-mismatch on rerun produces a warning, not a failure, but the warning is surfaced in the report.
X1.4 cuDNN is configured for determinism (`use_deterministic_algorithms`) at the cost of throughput. The throughput cost is measured and reported in the appendix; if it exceeds 30%, the deterministic flag becomes optional and a non-deterministic mode is added with appropriate caveats in the paper.

### III.2 Observability

X2.1 All long-running operations emit `tracing` spans with structured fields (cell identifier, seed, phase). The default subscriber writes JSON-line logs to disk and a human-readable summary to stderr.
X2.2 A run manifest is produced per orchestrator invocation, listing every (cell, seed) pair, its status, its wall-clock duration, and the path to its output artifacts.
X2.3 A live progress dashboard (terminal-based, via `indicatif` or similar) shows orchestration progress; no web UI is built — this is research code, not production.

### III.3 Testing

X3.1 Unit tests on every pure function in `hft_layer2`, `hft_recurrence`, `hft_rqa`, and `hft_stats`. Coverage target: 90% line coverage in these crates.
X3.2 Property tests (proptest) for: bucketer determinism (R4.4), DP symmetry and zero-diagonal (R5.4), RP recurrence rate within tolerance (R5.3), and RQA invariance under RP transposition (since RP is symmetric).
X3.3 Reference-comparison tests for RQA against PyRQA on a fixed 10-bucket fixture set (R6.2) and for resampling against scipy on a fixed DP fixture set (R5.4).
X3.4 A "tiny configuration" integration test that runs the full pipeline (read → bucket → DP → RQA → Stage One stats → 1 W-GAN seed for 100 generator steps → Stage Two diagnostics → bridge analysis) end-to-end on a 5-MB synthetic dataset, completing in < 60 seconds on CPU. This is the gate for every PR.

### III.4 Performance Budgets

X4.1 Layer 1 ingestion: ≥ 200,000 raw snapshots/sec/core (R3.3).
X4.2 DP computation: 256×256 DP in < 5 ms on a single core (R5.1).
X4.3 RQA computation: ≥ 1,000 buckets/sec/core on average bucket size (R6.3).
X4.4 W-GAN training: ≥ 80% GPU utilization measured by `nvidia-smi` during steady-state training (CI gate via a 5-minute synthetic run).
X4.5 Total wall-clock for the full Stage Two grid (1,080 runs) on a 4-GPU node: ≤ 7 days.

### III.5 Configuration

X5.1 All configuration is TOML. The orchestrator accepts a single top-level config file that references per-cell config overrides if needed.
X5.2 The config schema is validated at load time via `serde` + a `validator` crate. Misconfiguration is a hard error before any computation starts.
X5.3 The default config (`config/default.toml`) reflects the assumed defaults documented in this requirements file. Engineers can override via CLI flag.

### III.6 Output Artifacts

The system produces a deterministic directory layout:

```
runs/<experiment_seed>/
├── manifest.json
├── data_snapshot.sha256
├── stage_one/
│   ├── rqa_per_bucket.parquet
│   ├── stationarity_scores.json
│   └── figures/
├── stage_two/
│   └── <cell>/<seed>/
│       ├── diagnostics.msgpack
│       ├── checkpoints/
│       └── failure.json   (only if failed)
├── bridge/
│   ├── correlations.json
│   └── figure_36cell_scatter.pdf
└── logs/
```

Every reported result in the paper traces back to a file under this layout via a manifest entry.

### III.7 Error Handling

X7.1 No `unwrap()` or `expect()` in non-test code outside of well-justified invariant assertions (e.g. "this index is bounded by construction"). Each justified case is annotated with `// SAFETY:` or equivalent comment.
X7.2 The error type `HftError` is a non-exhaustive enum so that crates can introduce variants without breaking dependents. Conversion via `thiserror`.
X7.3 At the orchestration layer, individual cell or seed failures do not abort the whole run; they're recorded and the orchestrator continues.

---

## Part IV — Build, Toolchain, Dependencies

### IV.1 Toolchain Pinning (`rust-toolchain.toml`)

```
channel = "1.84.0"
components = ["rustfmt", "clippy", "rust-src"]
```

(Or whichever stable version is current at kickoff; the point is to pin.)

### IV.2 Key Crate Dependencies (proposed)

| Domain | Crate | Notes |
|---|---|---|
| Tensor / DL | `tch = "0.17"` | Pin LibTorch + CUDA versions in build.rs |
| Arrow / Parquet | `arrow = "54"`, `parquet = "54"` | Or `arrow2`/`parquet2` if perf demands |
| Numerics | `ndarray = "0.16"`, `ndarray-stats`, `statrs` | |
| Stats | hand-rolled or `linregress`, `nalgebra` for hierarchical | KS via `kolmogorov_smirnov` if maintained |
| Parallelism | `rayon = "1"` | for stats bootstraps and per-cell parallelism |
| Serialization | `serde`, `serde_json`, `rmp-serde` | |
| Logging | `tracing`, `tracing-subscriber` | |
| Config | `toml`, `validator` | |
| Errors | `thiserror`, `anyhow` (CLI only) | |
| CLI | `clap = "4"` (derive macros) | |
| Testing | `proptest`, `insta` (snapshot), `criterion` (bench) | |
| Hashing | `sha2`, `blake3` | for data snapshots |
| Progress | `indicatif` | |

### IV.3 LibTorch / CUDA Pinning

L1 LibTorch version pinned to a specific release matching the `tch-rs` minor version requirement.
L2 CUDA version pinned to one of {11.8, 12.1} based on host environment; documented in `BUILD.md`.
L3 The build script (`build.rs` for `hft_wgan`) verifies LibTorch and CUDA presence and aborts with a clear message if missing. CI builds against a Docker image with the pinned versions.

---

## Part V — Phasing and Acceptance Criteria

### Phase 0 — Foundation (week 1)

Deliverable: `hft_types`, `hft_data`, `hft_layer2` implemented and tested. Tiny-config integration test green.

Acceptance: Reading one Parquet window for one (exchange, symbol) and emitting Layer-2 snapshots passes a fixed-fixture comparison with a Python reference implementation.

### Phase 1 — Stage One (weeks 2–4)

Deliverable: `hft_bucketing`, `hft_recurrence`, `hft_rqa`, `hft_stats`. Full Stage One pipeline runs end-to-end on the full data corpus and produces the per-cell Stationarity Scores plus the figures motivating H1.

Acceptance: (a) RQA matches PyRQA within the tolerances of R6.2 on the fixture set; (b) Stationarity Scores produced for all 36 cells; (c) figures rendered and a draft of the Stage One paper section fillable from the outputs alone.

### Phase 2 — Stage Two infrastructure (weeks 5–8)

Deliverable: `hft_wgan`, `hft_diagnostics`, orchestration. A single W-GAN training run on a single cell completes successfully end-to-end and produces all four diagnostics. The 30-seed grid for one cell completes within 24 hours on a single A100.

Acceptance: (a) Determinism test (R8.3) passes — bit-identical losses on repeated runs of the tiny configuration; (b) all four diagnostics produce the expected output formats; (c) failure handling (R8.6) catches an injected NaN and records correctly.

### Phase 3 — Full grid execution (weeks 9–12)

Deliverable: 1,080 W-GAN runs completed, Stability Scores computed for all 36 cells, bridge analysis run, paper-ready figures and tables generated.

Acceptance: (a) ≤ 7 days wallclock on the target hardware (X4.5); (b) failure rate < 5% across runs; (c) bridge analysis produces both Spearman and partial Spearman with bootstrap CIs; (d) every figure in the paper traces to a file in the output layout.

### Phase 4 — Hardening, replication, paper (weeks 13–16)

Deliverable: Cleanup, README, replication script, public-release-quality code, and the NeurIPS-ready paper draft.

Acceptance: A fresh clone on a fresh machine, given the data corpus and the experiment seed, reproduces all figures and tables in < 7 days wallclock.

---

## Part VI — Consolidated Researcher Clarifications

The following items are flagged **[BLOCKING]** in the body and need answers before the corresponding subsystem can begin. Listed in approximate order of urgency.

1. **Q2.1, Q2.2, Q2.4** — Exact Parquet schema, snapshot encoding, and timestamp resolution. Blocks all of Phase 0.
2. **Q3.1, Q3.3, Q3.4** — Layer 2 edge cases: zero-volume first snapshots, exact-best level binning, volume-scaling denominator. Blocks Phase 0.
3. **Q4.1, Q4.2, Q4.4** — Bucketing parameters: confirm N=64, info-time calibration tolerance, PT bucketing semantics. Blocks Phase 1.
4. **Q5.2, Q5.4, Q5.5** — Recurrence layer: RR target(s), PT embedding choice, small-bucket policy. Blocks Phase 1.
5. **Q6.2** — ENTR logarithm base. Blocks Phase 1 (RQA implementation).
6. **Q7.1, Q7.2, Q7.3** — Statistical test choices: trading-session definition, Hill `k`, Stationarity Score weights. Blocks Phase 1 finalization and the entire bridge analysis.
7. **Q8.2, Q8.3** — W-GAN learning rate; BatchNorm-vs-GroupNorm decision now or later. Blocks Phase 2.
8. **Q9.1, Q9.2** — Curvature step size; Stability Score weights. Blocks Phase 2 finalization.
9. **Q10.1** — One-sided vs two-sided bridge test pre-registration. Blocks Phase 3 publication.

Items marked **[ASSUME]** in the body will be implemented per the documented default if no answer is provided by the start of the relevant phase. Each assumption is mechanically extractable from this document and will be reproduced verbatim in the run manifest, so any later disagreement is straightforward to identify and resolve.

---

## Appendix A — Engineering Risks (Distinct from Research Risks)

The research-side risks are catalogued in the proposal §9. The engineering-side risks below are different in kind and need separate mitigation.

E1. **tch-rs / LibTorch version drift.** LibTorch releases break ABI compatibility with `tch-rs` regularly. Mitigation: pinned Docker image for CI and recommended local development; explicit build-time check.

E2. **CUDA non-determinism in transposed convolutions.** Even with `use_deterministic_algorithms`, certain ops fall back to non-deterministic kernels. Mitigation: assert determinism in the tiny-config integration test on the actual GPU model used for production runs, not a development laptop.

E3. **Parquet schema drift in upstream pipeline.** The data is being collected continuously; the Layer 1 pipeline could change format. Mitigation: schema validation at read time (R2.1) and a contract test against a frozen sample.

E4. **Memory pressure in mode-coverage NN search.** Brute-force 10k × 10k × 4096-d L2 is 1.6 GB of distance values in fp32. Mitigation: tiled computation on GPU with a 1k-row tile, never materializing the full matrix.

E5. **Disk I/O bottleneck in the 1,080-run grid.** Each run writes ~100 MB of diagnostics; total ~100 GB. Mitigation: orchestration writes to a local NVMe scratch and rsyncs to durable storage in the background.

E6. **Bootstrap memory in `hft_stats`.** B=10,000 bootstrap iterations across 36 cells with full RQA distributions risks blowing the heap if naively held in memory. Mitigation: streaming bootstrap with online quantile estimators (`tdigest` crate).

E7. **Floating-point drift in cross-machine reproduction.** Even with all flags set, different CPU instruction sets (AVX2 vs AVX-512) can produce divergent results. Mitigation: document that the reproducibility guarantee is per-machine; a separate "spirit of reproducibility" guarantee covers cross-machine within a small numerical tolerance, asserted by tests in the paper repo.
