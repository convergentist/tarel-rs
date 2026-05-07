# Endogenous Time-Changes as Implicit Regularizers
## A Recurrence-Quantification Bridge from Microstructure Subordination to W-GAN Saddle-Point Stability

---

## Abstract

This proposal advances a two-step empirical argument connecting two literatures that have not previously spoken to each other: stochastic-clock subordination in market microstructure (Clark 1973; Mandelbrot & Taylor 1967; Easley, López de Prado & O'Hara 2012) and saddle-point training dynamics in generative adversarial networks (Goodfellow et al. 2014; Arjovsky, Chintala & Bottou 2017; Gulrajani et al. 2017).

The first step constructs an empirical diagnostic — Recurrence Quantification Analysis (RQA) over Limit Order Book and Public Trade distance plots — and uses it to test whether information-time data exhibits more Gaussian, more stationary, and more cross-regime-comparable behavior than physical-time data. The second step uses the same data, partitioned by the same three windowing schemes, to train identical Wasserstein GANs with gradient penalty (W-GAN-GP) and measure whether the data-distribution stationarity established in step one propagates into measurable improvements in saddle-point optimization stability.

The combined contribution is a falsifiable claim — that endogenous time-changes act as implicit regularizers of adversarial training — together with a reference implementation in Rust / tch-rs, a benchmark dataset spanning 3 exchanges × 4 symbols × 3 clocks, and a methodological template that downstream microstructure-generative work can adopt without rederivation.

---

## 1. Motivation

The microstructure literature has, since Clark (1973) and Mandelbrot & Taylor (1967), maintained that physical time is the wrong clock for the analysis of asset returns. Subordinating a Gaussian process to a stochastic time-change driven by traded volume yields the heavy-tailed, volatility-clustered, leverage-asymmetric return distributions that are observed empirically. Easley, López de Prado & O'Hara (2012) operationalized this insight via VPIN, and a substantial empirical literature has since shown that returns measured in volume-time are closer to Gaussian, more stationary in their conditional moments, and more amenable to standard inferential tools.

Independently, the GAN literature has documented that adversarial training is fragile. The min-max objective is non-convex / non-concave, alternating SGD provides no convergence guarantees in the asymptotic sense, and practical stability rests on a combination of architectural priors (W-GAN, spectral normalization, gradient penalty) and empirical heuristics (critic-to-generator update ratios, learning-rate balancing, batch composition). The community has converged on the recognition that the *data distribution itself* — not just the architecture — shapes what kind of saddle-point landscape the optimizer must navigate.

These two literatures have not been brought into contact. The bridge is straightforward to state but, to the best of our knowledge, has never been tested: if a stochastic clock makes the data distribution more stationary, then training a generative model on data drawn from that clock should produce a measurably more tractable optimization. This proposal tests that bridge directly. The first stage measures the data-distribution property using a recurrence-geometric diagnostic suite. The second stage measures the training-dynamics property using a fixed W-GAN-GP architecture and a four-axis stability instrumentation. A consistent positive result in both stages, combined with a demonstrable correspondence between the two, would constitute an empirical demonstration that endogenous time-changes act as implicit regularizers of GAN training — a result that connects subordination theory to saddle-point optimization in a way neither field has previously articulated.

---

## 2. Combined Hypothesis

The proposal carries three nested hypotheses.

**H1 — Diagnostic (Stage One).** Distance Plots and Recurrence Plots constructed over information-time buckets exhibit RQA-statistical distributions that are (i) closer to a stable parametric form across successive trading sessions, (ii) more concentrated across instruments under a hierarchical pooling model, and (iii) less heavy-tailed in the way that breaks downstream inference, when compared to those constructed over fixed-physical-time and fixed-event-time buckets.

**H2 — Training (Stage Two).** Identical W-GAN-GPs trained on Distance Plots from each windowing scheme, holding architecture, hyperparameters, batch composition, and seed distribution fixed, exhibit measurably improved saddle-point stability under information-time as quantified by the EM-gap proxy variance, gradient-penalty magnitude distribution, mode-coverage incidence, and critic-loss curvature along realized generator updates.

**H_bridge — Causal Correspondence.** The improvement in training stability claimed in H2 is *predicted by* the improvement in data-distribution stationarity established in H1. Operationally, this means that across the (clock × exchange × symbol) grid, the per-cell RQA-stationarity score (Stage One) and the per-cell training-stability score (Stage Two) are positively rank-correlated with statistical significance, with the dependence surviving control for instrument-level idiosyncrasy.

The bridge hypothesis is the contribution-bearing one. H1 alone restates a known finding in modern dress; H2 alone is a single-architecture stability result. H_bridge connects them and operationalizes the claim that subordination is a generative principle, not just a descriptive convenience.

---

## 3. Data and Pipeline

The dataset is collected and processed through the existing pipeline whose two layers are summarized here for completeness.

**Layer 1 — Extract.** Limit Order Book snapshots (depth = 25) and Public Trade events are streamed from Binance, Coinbase, and Kraken at 250 ms sampling for four selected symbols, yielding 12 streams. Snapshots are flushed to per-exchange / per-instrument Parquet files in 60-minute windows, with LOB and PT files held separately within each instrument folder.

**Layer 2 — Transform.** Each LOB snapshot is converted from absolute price levels to bps offsets relative to the best bid (for bid levels) and the best ask (for ask levels), then aggregated into 10-bps bins so that level volumes are summed within each bin and zero where no level falls. Volumes are scaled per Parquet file by the total side volume of the file's first LOB snapshot, producing a normalized vector x_t ∈ ℝ^{2K} with K = 25 bins per side. PT registries are passed through unmodified.

**Three windowing schemes.** From the Layer 2 stream, buckets are constructed under three rules: (i) *fixed physical time* of 60 seconds, the literature-standard HFT microstructure window and the comparability baseline; (ii) *fixed event time* of N consecutive LOB snapshots, which yields uniformly-shaped distance matrices and is the GPU-friendly computational baseline (N = 64 is the natural choice given the W-GAN architecture, see §5); (iii) *variable physical time* defined by a cumulative-traded-volume threshold, which deforms time to equalize information content per bucket and is the methodological contribution being tested.

The three windowing schemes are the manipulated variable across both stages. Every claim in this proposal is structured as a comparison among the three.

---

## 4. Stage One — RP/DP-Based Diagnostic Suite for Clock-Conditional Stylized Facts

### 4.1 Construction

For every bucket produced under each windowing scheme, the Distance Plot is computed as DP[i,j] = ||x_i − x_j||_p with x_t the bps-aggregated, volume-scaled LOB snapshot vector. The default norm is Euclidean (p = 2); a Wasserstein-1 distance treating each LOB snapshot as a discrete measure over the 50 bins is included as an ablation, since it respects the natural geometry of the order book. Recurrence Plots are obtained by thresholding DP at ε chosen by the fixed-recurrence-rate convention (RR ∈ [1%, 5%]) so that cross-bucket comparisons are not confounded by ε itself; both RR = 1% and RR = 5% are reported. For PT, a parallel construction is applied to per-snapshot (signed-volume, mid-shift) pairs.

### 4.2 RQA Scalars

For each bucket, six interpretable scalars are computed: Recurrence Rate (RR), Determinism (DET), Laminarity (LAM), Trapping Time (TT), maximum diagonal length (L_max), and Shannon entropy of the diagonal-length distribution (ENTR). Each maps to a known stylized fact: LAM and TT to volatility clustering and regime persistence; DET to short-horizon predictability; ENTR to multifractal complexity; L_max and RR to qualitative recurrence intensity. The mapping between RQA scalars and stylized facts is what makes Stage One interpretable to a finance reviewer; the fact that the scalars are continuous, unitless, and bucket-additive is what makes Stage One quantitative for an ML reviewer.

### 4.3 Statistical Tests

For each clock c ∈ {physical, N-event, info-time} and each instrument-exchange pair (s, e), the empirical distribution of each RQA scalar is constructed across all buckets in the available data window. Three statistical tests evaluate H1: (i) two-sample Kolmogorov-Smirnov distance between RQA distributions of successive trading sessions, with the prediction that information-time minimizes session-to-session KS distance — this tests temporal stability; (ii) variance decomposition under a two-level hierarchical model with instrument as the grouping factor, with the prediction that information-time minimizes the ratio of within-instrument to between-instrument variance — this tests cross-asset comparability; (iii) Hill estimator tail indices on the upper and lower tails of each RQA distribution, with the prediction that information-time produces tail indices closer to those of well-behaved (Gaussian-tailed) reference distributions — this tests heavy-tail attenuation. All tests are reported with bootstrap confidence intervals over 1000 resamples.

### 4.4 What Stage One Produces

The artifact of Stage One is a panel of figures showing RQA distribution comparisons across clocks, plus a summary table of the three statistical tests with bootstrap confidence intervals. These outputs serve three roles in the paper: they establish H1; they motivate the choice of windowing scheme adopted in Stage Two; and the per-cell RQA-stationarity scores feed directly into the bridge correlation tested in §6.

---

## 5. Stage Two — W-GAN-GP Saddle-Point Stability under Each Clock

### 5.1 Architecture (Fixed across Conditions)

The architecture is held fixed across all conditions to isolate the effect of the windowing scheme. The generator maps a 128-dimensional latent z ∼ N(0, I) to a 1-channel 64×64 grayscale image via five transposed convolutions: an initial projection (k=4, s=1, p=0) from latent to a 4×4 grid at 512 channels, followed by four upsampling deconvolutions (k=4, s=2, p=1) doubling spatial resolution while halving channels (512→256→128→64→1). Each non-final layer is followed by BatchNorm and ReLU; the final layer uses Tanh to bound output to [−1, 1] following standard W-GAN normalization. The critic mirrors the generator with five strided convolutions collapsing 1×64×64 to a single unbounded scalar, with no BatchNorm (the per-sample gradient penalty forbids the cross-sample correlations that BN introduces), LeakyReLU(0.2) activations to avoid dead neurons under the 5:1 critic-to-generator update ratio, and no terminal sigmoid since the Wasserstein loss operates on raw score differences.

### 5.2 Input Representation

Distance Plots from each windowing scheme serve as W-GAN training inputs. The fixed-event-time scheme with N = 64 produces DPs that are natively 64×64 and require no resampling. The 60-second physical and variable info-time schemes produce DPs of variable shape and are resampled to 64×64 via bicubic interpolation for the DP and nearest-neighbor for the binary RP, with both interpolation choices ablated to demonstrate robustness. DP values are normalized to [−1, 1] to match the Tanh output range. This standardization is explicit in the paper and treated as a deliberate methodological choice rather than a hidden preprocessing step.

### 5.3 Training Protocol

Training is W-GAN-GP with λ_GP = 10, Adam optimizer (β1 = 0, β2 = 0.9), critic-to-generator update ratio of 5:1, batch size 64, latent dimension 128. Each (clock × exchange × symbol) cell is trained from 30 random seeds to enable bootstrap confidence intervals on all reported diagnostics. Training proceeds for a fixed wallclock budget (not a fixed epoch count), with all four stability diagnostics tracked online and dumped at fixed checkpoints. Holding wallclock fixed rather than epoch count is the more honest comparison given that information-time produces a different bucket count per data hour.

### 5.4 Four-Axis Stability Instrumentation

Four diagnostics are tracked per training run, all of which contribute to the H2 evaluation.

The *EM-gap proxy* is computed as the critic's score difference between real and generated batches, smoothed over 100-step windows. This quantity is read by the literature as a Wasserstein-1 estimate. The headline plot for Stage Two is the variance of this quantity across seeds, by clock; the prediction is that information-time minimizes seed-to-seed variance, indicating that the saddle-point trajectory is more reproducible.

The *gradient-penalty magnitude* (||∇_x̂ D(x̂)|| − 1)² is recorded per step, and its full empirical distribution per clock is reported. The prediction is that information-time produces a tighter, more concentrated distribution because the data manifold has more uniform local Lipschitz structure. This is a stronger claim than mean-magnitude differences and is testable via two-sample tests on the empirical distributions.

*Mode coverage* is measured at fixed checkpoints by generating 10,000 samples and running reverse nearest-neighbor search against the training set, counting unique near-matches under a fixed L2 cutoff. Mode collapse is the central failure mode of GANs; the prediction is that information-time reduces its incidence both in mean count of unique modes recovered and in seed-to-seed variance of that count.

The *critic-loss curvature along the realized generator update direction* is approximated via finite-difference second-order along each realized generator step. This is the most involved diagnostic and the one that supports a genuinely saddle-point-geometric claim rather than a vague stability claim. The prediction is that information-time produces a curvature distribution with smaller magnitude in the dimension orthogonal to the realized update — meaning the optimization is operating on a flatter, less hostile saddle locally.

### 5.5 What Stage Two Produces

The artifact of Stage Two is a per-clock distribution for each of the four diagnostics, with bootstrap confidence intervals across 30 seeds. The H2 evaluation is positive if information-time wins on at least three of the four diagnostics by a margin that exceeds the bootstrap CI overlap. Each (clock × exchange × symbol) cell receives a composite stability score that aggregates the four diagnostics; this score feeds the bridge correlation in §6.

---

## 6. The Bridge — Operationalizing H_bridge

The bridge hypothesis claims that the data-distribution stationarity established in Stage One is the *cause* of the training-stability advantage measured in Stage Two. Operationalizing this without overclaiming requires care.

For each (clock × exchange × symbol) cell, two scalar scores are constructed: a *Stationarity Score* aggregating the three Stage-One statistical tests (KS distance, hierarchical variance ratio, Hill tail index distance) into a single composite via a fixed weighted sum with weights set in advance and frozen, and a *Stability Score* aggregating the four Stage-Two diagnostics analogously. The bridge claim is tested via Spearman rank correlation between the two scores across the 36-cell grid, with bootstrap confidence intervals and a null distribution generated by permuting cell labels independently within each clock.

The bridge claim is supported if Spearman ρ is positive and its bootstrap CI excludes zero. To control for instrument-level idiosyncrasy (some symbols may be both more stationary and more trainable for unrelated reasons), a partial Spearman correlation is also reported, conditioning on instrument identity. A surviving partial correlation is the strongest version of the claim.

This is deliberately a *correlational* test, not a *causal* one. The proposal does not claim that intervening on data stationarity by some means other than clock choice would necessarily produce a corresponding training-stability change. The claim is more limited and more defensible: across the natural experiment provided by the three windowing schemes, the cells where the data is more stationary are the cells where training is more stable, and the magnitude of this association is large enough to suggest a shared underlying mechanism.

---

## 7. Experimental Design Summary

The full experimental grid comprises 3 clocks × 3 exchanges × 4 symbols × 30 seeds = 1,080 W-GAN-GP training runs, plus the corresponding RQA computations on the same 36 (clock × exchange × symbol) cells. The Rust / tch-rs implementation makes this scale tractable on a single workstation with 1–2 GPUs over a few weeks of wallclock, which is the throughput edge worth mentioning in the paper's appendix as a reproducibility argument. PyTorch researchers attempting the same study would face either a substantially larger compute bill or a substantially smaller seed budget, which would weaken their ability to make bootstrap-based stability claims of the kind H2 requires.

All hyperparameters not explicitly listed are held at their literature-standard W-GAN-GP defaults, with no per-clock tuning. Per-clock tuning would dissolve the experimental contrast and is therefore deliberately avoided. The paper's appendix should include a sensitivity analysis around the three most consequential fixed choices — λ_GP, critic-to-generator update ratio, and batch size — to demonstrate that the headline results are robust within a reasonable hyperparameter neighborhood.

---

## 8. Expected Results and Their Interpretation

Three result patterns are possible.

**Pattern A — Both stages positive, bridge significant.** H1, H2, and H_bridge are all supported. This is the strongest result and the basis for the paper's headline claim. The interpretation is that endogenous time-changes act as implicit regularizers of GAN training, with the regularization mechanism being measurable as RQA-statistical stationarity of the training distribution. This pattern would justify the paper's title and the subordination-as-generative-principle framing.

**Pattern B — Both stages positive, bridge weak.** H1 and H2 are supported but the per-cell correlation is noisy. The interpretation is that information-time helps both data-distribution properties and training dynamics, but the connection between the two is mediated by additional factors not captured in the current scoring. This is still publishable, with the paper reframed around the two independent improvements rather than the bridge mechanism. The bridge result becomes an open question for follow-up.

**Pattern C — Only one stage positive.** Either H1 or H2 fails. This is the least likely outcome given the strength of the prior literature on subordination, but if it occurred it would be informative: H1 failing would suggest the bps + volume-scaling normalization is not preserving the relevant subordination structure; H2 failing would suggest that data-distribution stationarity does not propagate to the W-GAN-GP architecture as cleanly as the prior would predict, perhaps due to the gradient-penalty term dominating the loss landscape regardless of input distribution. Either failure mode is itself a contribution to a methodological appendix.

The paper is robust to Patterns A and B and informative even under Pattern C, which is the property a well-designed empirical study should have.

---

## 9. Risks and Mitigations

The principal risks are five. First, the effect size in Stage Two could be small relative to seed noise; the mitigation is the 30-seed protocol with bootstrap CIs and the explicit pre-registration of the four-axis evaluation rule (3-of-4 win threshold). Second, the resampling of variable-shape DPs to 64×64 could itself introduce artifacts that confound the clock comparison; the mitigation is the bicubic / nearest-neighbor ablation and a separate experiment on the fixed-event-time scheme alone where no resampling occurs. Third, the bridge correlation could be dominated by instrument identity; the mitigation is the partial Spearman correlation conditioning on instrument. Fourth, mode coverage measured by reverse-NN search is sensitive to the L2 cutoff; the mitigation is to report the full coverage curve as a function of cutoff rather than a single number. Fifth, GAN training instability could produce per-seed failures that are not informative about the claim; the mitigation is to define a failure criterion in advance (e.g., NaN in critic loss within first 1000 steps) and report failure rates per clock as a separate diagnostic, which itself becomes evidence for or against the implicit-regularization claim.

---

## 10. Implementation Notes (Rust / tch-rs)

The W-GAN-GP is implemented against `tch-rs` using `nn::SequentialT` rather than `nn::Sequential` because the BatchNorm layers in the generator implement `ModuleT` (train / eval awareness), not `Module`. The critic's LeakyReLU(0.2) is hand-rolled as `clamp_min(0) + 0.2 * clamp_max(0)` because `tch-rs`'s auto-generated bindings may not expose the slope parameter of the C++ binding directly; this is a known minor quirk and the hand-rolled form is equivalent and correct.

The per-sample gradient penalty is computed by drawing a uniform interpolation coefficient ε ∼ U(0, 1) per batch element, forming x̂ = εx + (1 − ε)x̃, computing D(x̂), backpropagating through the input to obtain ∇_x̂ D(x̂), and forming the penalty (||∇_x̂ D(x̂)||_2 − 1)². The gradient-magnitude diagnostic for Stage Two is read off this same computation at no extra cost.

For the cross-clock comparison to be fair, the same RNG sequence must drive the seed grid across all three clocks. The implementation should use seeded `tch::Cuda::manual_seed_all` plus a separate seed for the data sampler, with both seeds derived deterministically from a single experiment-level seed via a hash. Logging the experiment-level seed makes the entire grid reproducible from a single integer.

The 30-seed grid per cell, at the throughput typical of this architecture on a single A100 (roughly 30 minutes per seed for the configured wallclock budget), implies approximately 540 GPU-hours for Stage Two. This is tractable on a small in-house cluster or a moderate cloud budget. Stage One's RQA computation is CPU-bound and parallelizes trivially across cells; budget approximately 24 CPU-hours total.

---

## 11. Contribution Summary

The paper's contributions, in order of decreasing novelty:

The first contribution is the bridge claim: that endogenous time-changes act as implicit regularizers of W-GAN-GP saddle-point dynamics, established by a positive Spearman correlation between RQA-stationarity scores and four-axis training-stability scores across a 36-cell experimental grid. To the best of our knowledge this connection between subordination theory and adversarial training dynamics has not previously been articulated, let alone tested.

The second contribution is the diagnostic suite itself: a reproducible RQA pipeline over LOB and PT distance plots, with the six-scalar summary and three-test statistical evaluation, applied uniformly across 12 (exchange × symbol) streams under three windowing schemes. The suite is independently useful for any future microstructure-generative work and is the natural evaluation substrate for follow-on diffusion or flow-matching models.

The third contribution is the open-source Rust / tch-rs implementation of the W-GAN-GP under the three windowing schemes, with the seed grid scaled to enable bootstrap inference, and the published benchmark of training-stability metrics that other architectures can be compared against.

The fourth contribution is a methodological template: the practice of constructing two scoring layers (data-distribution stationarity and training-dynamics stability) and testing the correspondence between them. This template generalizes beyond microstructure data to any setting where the generative-modeling community might want to measure whether a representation choice helps training in a way that is mechanistically traceable.

---

## 12. Scope and Sequencing

Stage One is implementable in approximately one month given the existing pipeline and standard RQA libraries; the bulk of the work is the statistical-test infrastructure and the figure-generation code. Stage Two is approximately three months, with the W-GAN-GP implementation occupying the first month and the 30-seed grid occupying the remaining two. The bridge analysis in §6 is a final two weeks of statistical work on the merged outputs of the two stages. A four-month total timeline is realistic for the experimental work, with paper writing and revision occupying the subsequent two months — placing the work on track for a NeurIPS 2026 deadline if started promptly.

A note on data: the four-month estimate assumes continuous data collection in parallel with method development, ideally beginning before Stage One starts so that by Stage Two's seed grid there is several weeks of data spanning both calm and high-volatility regimes. Cross-regime stability is implicit in every claim this proposal makes, and the data window must support that claim.
