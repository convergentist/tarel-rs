"""Plotting routines for the four article-2 figures.

Figures produced (300 DPI PNG):

  bucket_duration.png       Per-bucket duration distributions, 3 cells x 3 clocks.
  rqa_violins.png           Violin distributions of DET and LAM across buckets,
                            faceted by clock with one violin per cell per panel.
  mean_rp_per_clock.png     Mean recurrence-plot heatmap (binary RP, headline
                            RR=5%) per (cell, clock).
  em_gap_trajectories.png   Stage-2 EM-gap trajectories per trained (cell, clock).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .aggregate import CellClockSummary, rqa_array  # noqa: E402
from .io import CELLS, CLOCKS, iter_bucket_records, unpack_rp  # noqa: E402
from .stage_two import StageTwoTrajectory  # noqa: E402

DPI = 300
NS_PER_SEC = 1_000_000_000

_CELL_COLOR = {
    "binance_BTC-USDC": "#EE6677",
    "kraken_BTC-USDC": "#228833",
}


def _venue_title(cell_us: str) -> str:
    return cell_us.replace("_", ":")


# ---------------------------------------------------------------------------
# Figure 1 — Bucket duration histograms across 3 cells x 3 clocks
# ---------------------------------------------------------------------------


def plot_bucket_duration(
    summaries: dict[tuple[str, str], CellClockSummary],
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(CELLS), len(CLOCKS), figsize=(10.5, 6.8),
                              sharey=False)
    for r, cell_us in enumerate(CELLS):
        for c, clock in enumerate(CLOCKS):
            ax = axes[r, c]
            s = summaries.get((cell_us, clock))
            if s is None or s.durations_ns.size == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="#666666")
            else:
                d_sec = s.durations_ns.astype(np.float64) / NS_PER_SEC
                d_sec = d_sec[d_sec > 0]
                if d_sec.size > 0:
                    bins = np.logspace(
                        np.log10(max(d_sec.min(), 1e-3)),
                        np.log10(max(d_sec.max(), 1.0)),
                        30,
                    )
                    ax.hist(d_sec, bins=bins, color=_CELL_COLOR[cell_us],
                            edgecolor="white", linewidth=0.3, alpha=0.85)
                    ax.set_xscale("log")
                    ax.text(0.98, 0.96, f"n={d_sec.size}",
                            transform=ax.transAxes, ha="right", va="top",
                            fontsize=7, color="#444444")
            if r == 0:
                ax.set_title(f"clock = {clock}", fontsize=10)
            if c == 0:
                ax.set_ylabel(_venue_title(cell_us), fontsize=9)
            if r == len(CELLS) - 1:
                ax.set_xlabel("bucket duration (s, log)", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.25, linestyle=":")
    fig.suptitle("Bucket duration distribution (Stage-1)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — RQA scalar violins: DET (top) + LAM (bottom) x clock columns
# ---------------------------------------------------------------------------


def plot_rqa_violins(
    summaries: dict[tuple[str, str], CellClockSummary],
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scalars = (("DET", r"$DET$"), ("LAM", r"$LAM$"))
    fig, axes = plt.subplots(2, len(CLOCKS), figsize=(10.5, 5.6), sharey="row")
    for r, (scalar, label) in enumerate(scalars):
        for c, clock in enumerate(CLOCKS):
            ax = axes[r, c]
            data = []
            colors = []
            xticks = []
            xlabels = []
            for k, cell_us in enumerate(CELLS):
                s = summaries.get((cell_us, clock))
                if s is None:
                    continue
                vals = rqa_array(s.rqa, scalar)
                vals = vals[np.isfinite(vals)]
                pos = k + 1
                xticks.append(pos)
                xlabels.append(_venue_title(cell_us).split(":")[0])
                if vals.size >= 2:
                    data.append((pos, vals, _CELL_COLOR[cell_us]))
                elif vals.size == 1:
                    data.append((pos, vals, _CELL_COLOR[cell_us]))
                # else: empty -> ticks remain but no violin
            for pos, vals, color in data:
                if vals.size >= 2:
                    parts = ax.violinplot(
                        [vals], positions=[pos],
                        showmeans=True, showmedians=True, widths=0.7,
                    )
                    for body in parts["bodies"]:
                        body.set_facecolor(color)
                        body.set_alpha(0.55)
                else:
                    ax.plot(pos, vals[0], marker="o", color=color, markersize=6)
                ax.text(pos, ax.get_ylim()[1] if ax.get_ylim()[1] < 1.05 else 1.0,
                        f"n={vals.size}", ha="center", va="bottom",
                        fontsize=6, color="#444444")
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels, fontsize=8)
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, alpha=0.25, linestyle=":")
            if r == 0:
                ax.set_title(f"clock = {clock}", fontsize=10)
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
    fig.suptitle("Per-bucket RQA scalar distributions at $RR^{*}=5\\%$",
                  fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Mean recurrence plot per (cell, clock)
# ---------------------------------------------------------------------------


# Per-(cell, clock) bucket count cap for the mean RP heatmap, used to
# bound wallclock when re-reading the giant info_time jsonl on the
# binance cell.
_MEAN_RP_CAP: dict[tuple[str, str], int] = {
    ("binance_BTC-USDC", "info_time"): 8,
}


def plot_mean_rp_per_clock(
    run_dir: str,
    summaries: dict[tuple[str, str], CellClockSummary],
    out_path: str | Path,
    max_buckets_per_cell_clock: int = 200,
) -> None:
    """Mean RP heatmap per (cell, clock), averaged over (up to) the first N
    buckets per pair to keep wallclock bounded. RPs are nearest-neighbour
    resampled to a common 64 x 64 grid before averaging.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target = 64
    fig, axes = plt.subplots(len(CELLS), len(CLOCKS), figsize=(9.5, 8.0))
    for r, cell_us in enumerate(CELLS):
        for c, clock in enumerate(CLOCKS):
            ax = axes[r, c]
            cap = _MEAN_RP_CAP.get((cell_us, clock), max_buckets_per_cell_clock)
            accum = np.zeros((target, target), dtype=np.float64)
            count = 0
            for rec in iter_bucket_records(run_dir, cell_us, clock,
                                            max_records=cap):
                n = int(rec["n"])
                if n < 16:
                    continue
                rp_headline = rec.get("rp_headline")
                if not rp_headline:
                    continue
                rp_bytes = rp_headline.get("bytes")
                if rp_bytes is None:
                    continue
                R = unpack_rp(rp_bytes, n).astype(np.float32)
                idx = (np.arange(target) * n / target).astype(np.int64)
                idx = np.clip(idx, 0, n - 1)
                accum += R[idx[:, None], idx[None, :]].astype(np.float64)
                count += 1
                if count >= cap:
                    break
            if count == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="#666666")
                ax.set_xticks([]); ax.set_yticks([])
            else:
                mean_rp = accum / count
                ax.imshow(mean_rp, cmap="magma", origin="lower",
                          vmin=0.0, vmax=1.0, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                ax.text(0.02, 0.95, f"n={count}", color="white",
                        transform=ax.transAxes, fontsize=8, va="top")
            if r == 0:
                ax.set_title(f"clock = {clock}", fontsize=10)
            if c == 0:
                ax.set_ylabel(_venue_title(cell_us).split(":")[0], fontsize=9)
    fig.suptitle(
        "Population-mean recurrence plot per cell $\\times$ clock "
        "($RR^{*}=5\\%$)",
        fontsize=11,
    )
    cax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    sm = plt.cm.ScalarMappable(cmap="magma",
                                norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label="P(recurrent)")
    fig.tight_layout(rect=(0, 0, 0.9, 0.96))
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — Loss-function evolution aggregated by cell across clocks
# ---------------------------------------------------------------------------

# The W-GAN-GP critic minimises L_D = -EM_gap + lambda_GP * GP_penalty.
# We approximate that loss by combining the per-step EM-gap and GP signals
# with the locked lambda_GP = 10. The generator's loss (-E[D(x_fake)]) is
# not reconstructible from these diagnostics because only the difference
# D(x_real) - D(x_fake) is recorded (as EM-gap), not the two terms
# separately. We therefore plot EM-gap (Wasserstein-1 estimator), GP
# magnitude (Lipschitz-violation signal), and the critic-loss
# approximation as three rows, with cells across columns.

LAMBDA_GP = 10.0

_CLOCK_LINESTYLE = {
    "physical": "-",
    "event": "--",
    "info_time": ":",
}


def plot_loss_evolution_by_cell(
    trajectories: dict[tuple[str, str], StageTwoTrajectory],
    out_path: str | Path,
) -> None:
    """Loss-function evolution per cell, with clocks overlaid.

    Three rows (EM-gap, GP magnitude, critic-loss approximation),
    two columns (one per cell). Each panel overlays the trained
    clocks of that cell, distinguishing them by line style.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, len(CELLS), figsize=(11.0, 8.4),
                              sharex=True)
    row_labels = ("EM-gap", "GP magnitude", r"$\hat{L}_D \approx -\text{EM} + \lambda_{\text{GP}} \cdot \text{GP}$")
    for c, cell_us in enumerate(CELLS):
        color = _CELL_COLOR[cell_us]
        any_data = False
        legend_handles = []
        for clock in CLOCKS:
            t = trajectories.get((cell_us, clock))
            if t is None or t.em_raw.size == 0:
                continue
            any_data = True
            ls = _CLOCK_LINESTYLE[clock]
            # Align em_step and gp_step grids (they share the same
            # critic-step domain; index-align by step value).
            # We plot the rolling-100 EM-gap when available.
            em_step = t.em_step
            em_smooth = t.em_roll if np.isfinite(t.em_roll).any() else t.em_raw
            axes[0, c].plot(em_step, em_smooth, color=color, linewidth=1.4,
                            linestyle=ls, alpha=0.95, label=clock)
            axes[1, c].plot(t.gp_step, t.gp_val, color=color, linewidth=1.0,
                            linestyle=ls, alpha=0.6)
            # Critic-loss approximation: -EM_smooth(step) + lambda * GP(step)
            # The two signals share the per-critic-step domain, so
            # element-wise on the shorter array.
            n_min = min(em_smooth.size, t.gp_val.size)
            if n_min > 0:
                em_a = em_smooth[:n_min]
                gp_a = t.gp_val[:n_min]
                step_a = em_step[:n_min]
                ld = -em_a + LAMBDA_GP * gp_a
                axes[2, c].plot(step_a, ld, color=color, linewidth=1.2,
                                linestyle=ls, alpha=0.85)
        if not any_data:
            for r in range(3):
                axes[r, c].text(0.5, 0.5, "no trained clock",
                                ha="center", va="center",
                                transform=axes[r, c].transAxes,
                                color="#999999", fontsize=9)
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        else:
            axes[0, c].legend(fontsize=8, loc="upper right", framealpha=0.85)
        axes[0, c].set_title(_venue_title(cell_us), fontsize=11)
        for r in range(3):
            axes[r, c].grid(True, alpha=0.25, linestyle=":")
            axes[r, c].tick_params(labelsize=8)
        axes[1, c].set_yscale("symlog", linthresh=0.1)
        axes[2, c].set_xlabel("critic step", fontsize=9)
    for r, lbl in enumerate(row_labels):
        axes[r, 0].set_ylabel(lbl, fontsize=10)
    fig.suptitle(
        "Stage-2 loss-function evolution by cell "
        "(EM-gap, GP magnitude, and $\\hat{L}_D$ approximation)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — Stage-2 EM-gap trajectories across the 6 trained (cell, clock)
# ---------------------------------------------------------------------------


def plot_em_gap_trajectories(
    trajectories: dict[tuple[str, str], StageTwoTrajectory],
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(CELLS), len(CLOCKS), figsize=(11.0, 6.8),
                              sharex=True, sharey=False)
    for r, cell_us in enumerate(CELLS):
        for c, clock in enumerate(CLOCKS):
            ax = axes[r, c]
            t = trajectories.get((cell_us, clock))
            if t is None:
                ax.text(0.5, 0.5, "training not run",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=8, color="#999999")
                ax.set_xticks([]); ax.set_yticks([])
            else:
                color = _CELL_COLOR[cell_us]
                ax.plot(t.em_step, t.em_raw, color=color, alpha=0.30,
                        linewidth=0.6, label="raw")
                if np.isfinite(t.em_roll).any():
                    ax.plot(t.em_step, t.em_roll, color=color, linewidth=1.5,
                            label="rolling-100")
                ax.axhline(t.em_final_mean, color="#222222", linewidth=0.6,
                           linestyle="--", alpha=0.6)
                ax.text(0.98, 0.96,
                        f"EM̅={t.em_final_mean:.1f}±{t.em_final_std:.1f}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=7, color="#222222")
                ax.grid(True, alpha=0.25, linestyle=":")
                ax.tick_params(labelsize=8)
            if r == 0:
                ax.set_title(f"clock = {clock}", fontsize=10)
            if c == 0:
                ax.set_ylabel(_venue_title(cell_us).split(":")[0], fontsize=9)
            if r == len(CELLS) - 1:
                ax.set_xlabel("critic step", fontsize=9)
    fig.suptitle(
        "Stage-2 EM-gap convergence per trained (cell, clock); "
        "dashed line = final-100-step mean",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
