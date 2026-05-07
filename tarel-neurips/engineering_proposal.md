# HFT-RP-WGAN — Five-Stage Development Plan

**Audience:** Solo research engineer executing the NeurIPS 2027 submission of the proposal in `research_proposal.md`.

**Purpose:** Translate the locked design decisions from the readiness audit into a sequenced engineering plan. Each stage produces a tangible pilot artifact on the 1-week test data slice, validates one layer of the pipeline, and unblocks the next stage. The production 720-run grid is a follow-on after stage 5 completes.

**Convention:** Each stage section is structured as Goal → Deliverables → Pilot Artifact → Exit Gate. Stages are sequential; sub-tasks within a stage may overlap. Calendar estimates assume single-engineer execution with ~25% solopreneur slack for context-switching.

---

## Part I — Locked Design Decisions

The readiness audit and grilling resolved every blocking decision in `implementation_proposal.md`. The decisions below are inputs to the plan and are not re-debated here.

**Strategy.** Extend `takens-rs` and `logan-rs` in-place. Add one new sibling crate `takens-stats` inside the takens-rs workspace. Rebuild `tarel-rs` (currently a stale logan-rs clone) as the orchestration + bridge driver.

**Stage One — LOB only.** PT (Public Trade) parallel construction is deferred to follow-up. K = 25 bins per side at 10-bps spacing, state vector x_t ∈ ℝ^50. Three windowing schemes via a `Bucketer` trait: PhysicalBucketer (60s), EventBucketer (N=64), InfoTimeBucketer (vol_threshold per cell, calibrated so median bucket duration ∈ [30s, 120s]). Buckets must not cross 60-minute parquet window boundaries. Trading session = 8-hour fixed shifts (Asia/EU/US) for the KS test. Recurrence rate: RR=5% headline, RR=1% appendix ablation. Wasserstein-1 distance: appendix-only ablation. Hill estimator k = floor(0.1 · n) with a minimum sample size of 200. ENTR uses natural log (matches PyRQA reference). Stationarity Score weights = [0.4 KS, 0.3 hierarchical-variance-ratio, 0.3 Hill-tail-index-distance], rank-normalized within (clock, scalar).

**Stage Two — full architecture frozen.** Generator and critic per `research_proposal.md` §5.1 (BatchNorm retained, no per-clock variation). Adam(β1=0, β2=0.9), lr=1e-4, λ_GP=10, batch 64, latent 128, critic ratio 5:1. Strict cuDNN determinism on the headline grid; SHA-256-derived seeds. Wallclock budget compressed to 20 minutes per seed × 20 seeds per cell = 720-run production grid (~10 GPU-days serial). Curvature finite-difference δ = 0.1 · ‖Δθ_G‖. Mode-coverage headline scalar = count at median per-cell training-pairwise L2 distance. Stability Score weights = [0.25, 0.25, 0.25, 0.25], rank-normalized within (cell) population across seeds.

**Bridge.** One-sided Spearman test predicting positive ρ. Partial Spearman conditioning on instrument identity only (not exchange). Bootstrap CIs at B=10,000.

**Data.** atelier-sdk is capture-only; takens-rs reads parquet directly via Arrow. atelier_data stays in workspace deps for schema sharing only — no runtime coupling. Production data collection runs as a parallel workstream starting at stage 1 day 1.

---

## Part II — The Five Stages

### II.1 Stage 1 — Foundation: Data Ingest, Bucketing, Orchestration Stub

**Goal.** Stand up the data pipeline from parquet to bucketed snapshot streams under all three clocks. Establish the orchestration scaffold every later stage uses.

**Deliverables.**

In `takens-rp`: flip `K_BINS_PER_SIDE` default from 20 to 25 in `represent.rs`. Add 60-minute parquet window-boundary detection in `ingest.rs` so the iterator emits a window-boundary marker. Persist `(total_bid_vol_first, total_ask_vol_first)` baselines per window for downstream recovery. Lock the answers to the [BLOCKING] Layer-2 edge cases as inline doc + unit tests: zero-volume first snapshot drops the window; exact-best price (bps_offset == 0) credits to bin 0; volume-scaling denominator is total side volume of the first LOB.

In `takens-rs`: introduce a `Bucketer` trait with `next_bucket(&mut self) -> Option<Bucket>` and three implementations. `PhysicalBucketer { window_ns: 60_000_000_000 }` is a refactor of the existing `split_into_windows`. `EventBucketer { n: 64 }` is mechanical. `InfoTimeBucketer { vol_threshold: f64 }` is the substantive new piece, paired with a `calibrate_info_time_threshold` helper that scans a warm-up day per (exchange, symbol) and returns the threshold whose median bucket duration lands in [30s, 120s]. Calibration is persisted as `cell_metadata.json`. Enforce R4.1: buckets close at 60-minute parquet boundaries with `truncated_at_window_boundary: true`; minimum-snapshot floor of 8 buckets are dropped from analysis but logged.

In `tarel-rs`: wipe the current logan-rs-clone scaffold. Rebuild as a minimal driver: experiment_seed (u64) → SHA-256 → derived per-(cell, seed) seeds via stable hash; the `runs/<experiment_seed>/` directory layout from `implementation_proposal.md` §III.6; a `manifest.json` writer recording cell, seed, status, wallclock, output paths; tracing setup writing JSON-line logs to `logs/`; a `tarel run-stage-one --cell <id>` CLI subcommand stub that accepts a cell triple and runs through Stage One on a single cell, calling into takens-rs as a library.

**Pilot Artifact.** Run `tarel run-stage-one --cell binance:BTC-USDT` on the 1-week slice. Output: bucketed snapshot streams persisted as parquet under `runs/<seed>/stage_one/buckets/<cell>/<clock>/` for all three clocks, plus `cell_metadata.json` with the calibrated info-time threshold for that cell.

**Exit Gate.** Pilot artifact exists for one cell across all three clocks. Bucket-count ratios across clocks are within reason (info-time bucket count comparable to physical-60s; event-N=64 different by construction). Property-based determinism test (R4.4) passes: feeding the same input stream twice produces bit-identical bucket sequences. Unit test confirms parquet-window-boundary handling on a synthetic 70-minute snapshot stream that splits cleanly at 60 minutes with the truncation flag set.

---

### II.2 Stage 2 — Recurrence Layer: DP, RP, Resampling, Normalization

**Goal.** Convert bucketed snapshot streams into Distance Plots, Recurrence Plots, and 64×64-normalized W-GAN-ready tensors, with the Wasserstein-1 ablation included.

**Deliverables.**

In `takens-rp`: add `RecurrenceMatrix::threshold_to_target_rr(target_rr: f32) -> RecurrencePlot` using bisection on ε with at most 30 iterations and 0.001 RR tolerance, recording the achieved RR in the `RecurrencePlot` struct. Add `resample_dp(&DistancePlot, target_n: usize) -> DistancePlot` with bicubic interpolation and a scipy reference-comparison test (R5.4, max abs error < 1e-4). Add `resample_rp(&RecurrencePlot, target_n: usize) -> RecurrencePlot` with nearest-neighbor. Add `DistancePlot::normalize_per_bucket(&self) -> ndarray::Array2<f32>` performing per-bucket min-max to [-1, 1] and persisting (min, max) for inversion. In `distance.rs`, add a `DistanceMetric` enum with `Euclidean` and `WassersteinOne` variants. Implement Wasserstein-1 via the cumulative-difference formula `W1(u, v) = sum_b |U_b − V_b|` with U, V the per-snapshot CDFs over the 50 bins (~30 lines).

In `takens-rs` CLI: extend the binary with subcommands that produce DPs and 64×64 normalized tensors per clock, output as parquet (DPs as `f32` arrays) and msgpack (normalized tensors).

**Pilot Artifact.** Run `tarel run-stage-one --cell binance:BTC-USDT --produce-dp-rp` on the 1-week slice. Output: per-bucket DP, RP at RR=5% and RR=1%, and the 64×64 normalized W-GAN-ready tensor, all persisted under `runs/<seed>/stage_one/recurrence/<cell>/<clock>/`. Output a sanity-check PNG grid (16 randomly sampled DPs per clock) for visual inspection.

**Exit Gate.** Achieved RR within 0.001 of target on 100 random buckets. Resampling preserves DP symmetry and zero-diagonal (proptest R5.4). Bicubic implementation matches scipy on a fixed 10-DP fixture set within max abs error 1e-4. Wasserstein-1 distance is non-negative and zero iff inputs are identical (proptest). Visual sanity-check PNG grid shows recognizable structure (no all-zero or all-noise images).

---

### II.3 Stage 3 — Stage-One Statistics: takens-stats Crate, Stationarity Score

**Goal.** Build the new `takens-stats` sibling crate. Produce a Stationarity Score per cell on the 1-week slice. Freeze the `StationarityScoreSpec` as pre-registration.

**Deliverables.**

New crate `takens-stats` inside the takens-rs workspace with three modules.

`stats::ks` — two-sample Kolmogorov-Smirnov distance between RQA distributions of successive 8-hour sessions (Asia/EU/US fixed shifts). Returns a session-pair time series of KS distances per (cell, scalar). B=1000 Rayon-parallelized bootstraps for CIs.

`stats::hierarchical` — two-level ANOVA-style variance decomposition with instrument as grouping factor. Output: ratio of within-instrument to between-instrument variance per (clock, scalar). Closed-form on the 36-cell scale; no MCMC.

`stats::hill` — Hill estimator with k = floor(0.1 · n), minimum sample size 200, returning typed `InsufficientData` below threshold. Streaming bootstrap via `tdigest` to bound memory (E6).

A `StationarityScoreSpec` struct with weights `[0.4 KS, 0.3 hierarchical, 0.3 Hill]`, rank-normalization within (clock, scalar), serde JSON serialization. The first action of stage 3 is to commit `stage_one/StationarityScoreSpec_v0_1_0.json` to the repo with these locked weights. Pre-registration per R7.2.

In `takens-rp::rqa`: confirm ENTR uses natural log. Add a fixture-based reference-comparison test against PyRQA outputs on a 10-bucket synthetic set with the R6.2 tolerances (RR/DET/LAM < 1e-9, L_max exact, TT/ENTR < 1e-6).

In `tarel-rs`: extend `tarel run-stage-one` to compute the six RQA scalars per bucket, run the three statistical tests, and emit the per-cell Stationarity Score to `runs/<seed>/stage_one/stationarity_scores.json`.

**Pilot Artifact.** Run `tarel run-stage-one --cell binance:BTC-USDT --full` on the 1-week slice. Output: 6 RQA scalars per bucket persisted as parquet; per-cell KS distance time series, hierarchical variance ratio, Hill tail index per scalar, and the Stationarity Score scalar — all under `runs/<seed>/stage_one/`.

**Exit Gate.** PyRQA fixture comparison passes within tolerance. Stationarity Score produced for at least one cell across all three clocks. The committed `StationarityScoreSpec.json` causes a hard error in the bridge stage if it is mutated. KS test sample size on 1 week of 8h sessions is ~21 per cell — borderline, so the gate accepts "computed and serialized correctly" rather than "statistically convincing." Hill tail index gracefully returns `InsufficientData` for any RQA scalar with fewer than 200 buckets in the 1-week slice.

---

### II.4 Stage 4 — Stage-Two Hardening: logan-rs W-GAN-GP and Four-Axis Diagnostics

**Goal.** Make logan-rs produce reproducible 30-seed-grid-style outputs with full four-axis diagnostics. Run the stage-4 pilot of 9 W-GAN training runs (3 clocks × 3 seeds, fixed exchange = Binance, fixed symbol = BTC-USDT) on 1-week DPs.

**Deliverables.**

Determinism. At process start, `tch::manual_seed(seed)` + `tch::Cuda::manual_seed_all(seed)`. Separate sampler seed via SHA-256 of `(experiment_seed, "sampler", cell, seed)`. cuDNN deterministic flag enabled. CI test: tiny-config (latent 8, 100 generator steps) bit-identical critic-loss curves on two reruns.

Wallclock-bounded training loop. Replace the `for epoch in 0..cfg.epochs` outer loop with a `while wallclock_remaining()` loop scheduling generator + critic steps. Default 20 minutes per seed, parameter-overridable per `--budget-mins` flag.

NaN / early-failure handling. Wrap the critic step in NaN detection; if first 1000 steps produce NaN, write `failure.json` with `{seed, config, step, reason: "early_nan"}` and terminate cleanly. Test by injecting NaN via a malformed input batch.

Four-axis diagnostics module. *EM-gap*: per-step `D(real).mean() − D(fake).mean()` with 100-step rolling smooth, full time series + final-100-step mean and stdev. *GP magnitude*: per-step batch-mean of `(||∇_x̂ D(x̂)||_2 − 1)²`, full time series + empirical CDF. *Mode coverage at 5k-step checkpoints*: generate 10k samples; GPU brute-force tiled L2 reverse-NN against training set in 1k-row tiles (E4); unique-count + full distance distribution; headline scalar = count at median per-cell training-pairwise L2. *Critic-loss curvature at each checkpoint*: three-point finite difference along the realized last generator step with δ = 0.1 · ‖Δθ_G‖, store `(L(−δ), L(0), L(+δ))` and the curvature estimate.

Diagnostic streaming. Persist time-series chunks of 1k entries each to MessagePack under `runs/<seed>/stage_two/<cell>/<seed>/diagnostics.msgpack` (R9.2).

DP-tensor data path. Replace `dataset::load_recurrence_plots` (PNG/JPG) with `dataset::load_dp_tensors(path)` reading the msgpack-encoded normalized tensors written in stage 2. Keep the PNG path behind a `--legacy-png` flag for single-image debugging.

Per-5k-step checkpointing. Generator + critic + optimizer state + diagnostic rolling buffers under `checkpoints/`.

A `StabilityScoreSpec` struct with weights `[0.25, 0.25, 0.25, 0.25]`, rank-normalization within (cell) population across seeds. Commit `stage_two/StabilityScoreSpec_v0_1_0.json` to the repo at the start of stage 4.

A `Generator::sample(z: &Tensor) -> Tensor` direct in-Rust inference path so tarel-rs and ablations can call the generator without going through the TorchScript export.

In `tarel-rs`: extend the orchestration driver with `tarel run-stage-two --cell <id> --seeds <n>` that loops through (cell, seed) pairs, invokes logan-rs as a library (not a subprocess), enforces wallclock per seed, and collects diagnostics.

**Pilot Artifact.** Run `tarel run-stage-two --cells "binance:BTC-USDT" --seeds 3` for all three clocks. Total: 9 W-GAN training runs at 20 minutes each = ~3 hours wallclock on the workstation GPU. Output: per-(cell, seed) generator and critic checkpoints, four-axis diagnostic time series, a `failure.json` if any run early-NaN'd. Stability Score computed per cell across the 3 seeds.

**Exit Gate.** cuDNN-deterministic bit-identical critic-loss curves on the tiny-config CI test, measured on the actual GPU model (E2). All four diagnostics produce non-empty output for every seed. NaN injection test triggers correct `failure.json` write. Mode-coverage GPU tile pipeline never exceeds 2 GB peak VRAM (E4). Generator outputs visually look like DP-shaped images (sanity-check PNG grid). The 9-run pilot produces a Stability Score scalar per cell. Throughput penalty from cuDNN-deterministic measured and recorded; if >30%, dual-mode flag is added per X1.4.

---

### II.5 Stage 5 — Bridge Analysis, tarel-rs Full Build, Paper-Ready Output Layout

**Goal.** Complete `tarel-rs` with bridge analysis. Run end-to-end on the 9-run pilot. Pin paper-ready figure and table templates so the production grid run drops in cleanly.

**Deliverables.**

Bridge module. `bridge::spearman` (Spearman ρ across cells with B=10,000 bootstrap CI, one-sided test for positive ρ); `bridge::partial_spearman` (rank-residualize each score against instrument dummy variables, then Spearman on residuals, conditioning on instrument identity only per Q10.2); `bridge::permutation_null` (B=10,000 cell-label permutations within each clock for the null distribution).

Spec validation. At bridge run-time, load both `StationarityScoreSpec.json` and `StabilityScoreSpec.json`, verify versions match the run manifests, and hard-error if any mismatch (R10.1).

Figure generation. `figures::scatter_36cell` (36-cell scatter with marginal histograms by clock — for the pilot it's a 9-cell scatter with placeholder text noting "pilot, n=9"). `figures::stage_one_panels` (RQA distribution comparisons across clocks). `figures::stage_two_diagnostics` (per-clock distribution panels for the four diagnostics with bootstrap CIs). All figures emit PDFs + SVGs to `runs/<seed>/figures/` with consistent typography, axes, colors. The pilot figures are paper-ready in *layout*; production grid output will fill in the data.

Top-level CLI. `tarel run-all --pilot --cells <list>` runs stages 1→5 end-to-end. `tarel report` regenerates all figures and tables from existing `runs/<seed>/` artifacts. `tarel verify --run <seed>` walks the directory layout and asserts all expected artifacts exist, score specs are unchanged, and per-cell metadata matches the orchestrator manifest. This becomes the per-PR CI gate going forward.

Documentation. Update README and add `BUILD.md` describing how to reproduce a pilot run from a fresh clone.

**Pilot Artifact.** Run `tarel run-all --pilot --cells "binance:BTC-USDT"` on the 1-week slice. Output: complete `runs/<pilot-seed>/` directory with manifest, data snapshot SHA, Stage One artifacts for one cell, Stage Two artifacts for 9 (cell, seed) pairs, bridge correlation outputs (statistically thin: 3-point Spearman per clock), all figures rendered to PDF.

**Exit Gate.** `tarel run-all --pilot` produces every artifact in §III.6 from a fresh clone in under 4 hours of wallclock (~1 hour Stage One, ~3 hours Stage Two, <1 minute bridge analysis). `tarel verify --run <pilot-seed>` exits zero. The 36-cell scatter figure renders correctly even with only 9 points (fewer empty marginals visible). The score spec mismatch error fires if `StationarityScoreSpec.json` is mutated and the bridge stage is re-run. Pilot reproducibility: two runs with the same `experiment_seed` and the same data snapshot SHA produce bit-identical artifacts (verified by `sha256sum runs/*/manifest.json` plus `cmp` on the float arrays — modulo cuDNN nondeterminism caveats).

---

## Part III — Cross-Cutting Concerns

### III.1 Continuous Integration

A tiny-configuration synthetic integration test (X3.4) is the per-PR gate from stage 1 onward. It exercises read → bucket → DP → RQA → Stage-One stats → 1 W-GAN seed for 100 generator steps → Stage-Two diagnostics → bridge analysis end-to-end on a 5-MB synthetic dataset and completes in under 60 seconds on CPU. Reference-comparison fixtures (PyRQA for RQA, scipy for bicubic resampling) are added in their respective stages and run as part of the per-PR gate from then on. Property tests for bucketer determinism, DP symmetry, and RP recurrence-rate tolerance are added incrementally as the relevant code lands.

### III.2 Pre-Registration

Two specifications are pre-registered as JSON in the repo before any inference touches real data: `stage_one/StationarityScoreSpec_v0_1_0.json` is committed at the start of stage 3; `stage_two/StabilityScoreSpec_v0_1_0.json` is committed at the start of stage 4. The bridge stage hard-errors on any spec-version mismatch with the run manifest. Any change to either spec invalidates the prior Stage-One or Stage-Two outputs and requires re-running. Spec versions are bumped (v0.1.1, v0.2.0, etc.) on any change so that the bridge analysis can refuse to mix them.

### III.3 Observability

`tracing` spans with structured fields (cell, seed, phase) are wired in stage 1 with the orchestration stub and used by every later stage. The default subscriber writes JSON-line logs to disk and a human-readable summary to stderr. A live progress bar via `indicatif` shows orchestration progress. No web UI; this is research code.

### III.4 Parallel Workstream — atelier-sdk Continuous Capture

Continuous LOB data capture across all 12 (exchange × symbol) streams must start at stage 1 day 1 and run uninterrupted until the production data window is sufficient (~6–8 weeks minimum for the 720-run grid; longer is better). atelier-sdk reliability is the single most important infrastructure investment outside the 5 stages. Failure modes to monitor: exchange-side rate limits, websocket disconnects, parquet write contention, disk-full conditions, schema drift from upstream. Recommended: a small daemon per (exchange, symbol) stream with restart-on-failure, plus a daily integrity check that compares row counts and schema hashes against the prior day. This capture work is independent of the 5 development stages and runs on different hardware (capture host, not GPU workstation).

---

## Part IV — Calendar Estimate

Single-engineer execution with ~25% solopreneur slack for context-switching:

Stage 1 — 1.5 weeks of focused engineering, ~2 weeks calendar.
Stage 2 — 1 week of focused engineering, ~1.5 weeks calendar.
Stage 3 — 2 weeks of focused engineering, ~2.5 weeks calendar.
Stage 4 — 3.5 weeks of focused engineering, ~4.5 weeks calendar. The bulk of new code lives here.
Stage 5 — 1.5 weeks of focused engineering, ~2 weeks calendar.

Total dev calendar: 11–13 weeks. Production data accumulation runs in parallel and is the gating constraint for the production grid that follows.

---

## Part V — Out of Scope (Deliberately)

The following are not part of these 5 stages and are explicit follow-ons:

The production 720-run grid execution on multi-month data; paper writing and revision; ablation sweeps over λ_GP, batch size, and critic-to-generator ratio per `research_proposal.md` §7; the deferred PT (Public Trade) parallel pipeline; Sub-proposal 1.C conditional training; cross-machine reproducibility hardening beyond the per-machine guarantee (E7).

All five are credible NeurIPS 2027 work but cannot start until stage 5 is complete and the production data window is sufficient.

---

## Appendix A — Stage Dependency Diagram

```
[atelier-sdk capture, parallel workstream] ───────────────────────────────────►
                                                 (gates production grid only)

Stage 1 ──► Stage 2 ──► Stage 3 ──► Stage 4 ──► Stage 5
foundation  recurrence  takens-     logan-rs    bridge +
+ orchstub  layer       stats       hardening   figures
                        + spec      + spec
                        v0.1.0      v0.1.0
                        commit      commit
```

Stage N's exit gate must pass before stage N+1 starts. Within a stage, sub-tasks (e.g., Wasserstein-1 implementation in stage 2 vs the bicubic resampler) can be developed in parallel by the same engineer with feature branches.

---

## Appendix B — Risk Register Specific to the Plan

R1. Stage 4 wallclock overrun. cuDNN-deterministic mode penalty plus consumer-GPU throughput could push 20-min-per-seed to 30-min-per-seed. Mitigation: stage 4 pilot is only 9 runs; if those take 4.5 hours instead of 3 hours, adjust the production grid budget rather than the dev plan.

R2. Stage 3 statistical-test sample size on 1 week. ~21 sessions per cell at 8h shifts is borderline for stable KS distributions. The Hill estimator's 200-sample floor will trigger `InsufficientData` for some RQA scalars. Mitigation: gate accepts "computed and serialized correctly," not "statistically convincing." Production data will resolve this naturally.

R3. atelier-sdk capture interruption. A multi-day gap in the production capture window degrades cross-regime claims. Mitigation: monitoring + restart-on-failure daemon, daily integrity checks, redundant capture for at least one (exchange, symbol) stream as ground truth.

R4. tch-rs / LibTorch ABI drift mid-development. A point-release bump in tch-rs could break CUDA pinning. Mitigation: lock tch-rs and LibTorch versions in `Cargo.toml` and `BUILD.md` at stage 1, verify at stage 4 start before substantial GPU work begins.

R5. Pre-registration spec drift. An accidental edit to a pre-registered spec invalidates downstream artifacts. Mitigation: spec files are committed to git with explicit version numbers in the filename; the bridge stage refuses to run on spec-version mismatch with the run manifest.
