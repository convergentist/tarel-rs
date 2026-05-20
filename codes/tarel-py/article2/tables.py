"""Emit booktabs LaTeX tables from aggregate summaries.

Tables produced:

  yield.tex  (Table 1): bucket construction yield per cell x clock (9 rows).
  rqa.tex    (Table 2): aggregate RQA statistics per cell x clock at RR* = 5%
                        (9 rows; cells without admitted RQA buckets show "--").

Both tables assume the article class already loads `booktabs`.
"""

from __future__ import annotations

from pathlib import Path

from .aggregate import CellClockSummary, rqa_summary_row
from .io import CELLS, CLOCKS
from .stage_two import StageTwoTrajectory


_VENUE_LABEL = {
    "binance_BTC-USDC": r"\texttt{binance:BTC\nobreakdash-USDC}",
    "kraken_BTC-USDC": r"\texttt{kraken:BTC\nobreakdash-USDC}",
}

_CLOCK_LABEL = {
    "physical": "physical",
    "event": "event",
    "info_time": r"info\_time",
}


def _fmt_int(x: int) -> str:
    return f"{x:,}"


def _fmt_pct(x: float) -> str:
    if x != x:  # NaN
        return "--"
    return f"{x:.1f}\\%"


def _pm(mean: float, std: float, decimals: int) -> str:
    if mean != mean:
        return "--"
    if std != std:
        return f"${mean:.{decimals}f}$"
    return f"${mean:.{decimals}f} {{\\scriptstyle\\pm {std:.{decimals}f}}}$"


def write_stage_two_summary_table(
    trajectories: dict[tuple[str, str], StageTwoTrajectory],
    out_path: str | Path,
) -> None:
    """Stage-2 execution summary table.

    One row per (cell, clock) in the article-2 cell list. Trained pairs
    report wallclock seconds, generator-step count, critic-step count,
    final-100-step EM-gap mean+/-std, GP-magnitude mean, GP-magnitude
    max. Pairs not trained in this run show "--" across all columns.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Stage-2 execution summary for run "
        r"\texttt{00000000c002daaaX} at $J{=}1$ seed (\texttt{seed\_idx}$=0$)"
        r" per (cell, clock). Wallclock is from the last checkpoint's "
        r"\texttt{diagnostics\_meta.json}; gen-steps and crit-steps are "
        r"the locked counters at that checkpoint. $\overline{\text{EM}}$ "
        r"is the mean of the last 100 raw EM-gap samples; "
        r"$\overline{\text{GP}}$ is the mean of all per-step gradient-"
        r"penalty values; $\overline{\text{GP}}_{\max}$ is their "
        r"maximum. Rows marked ``--'' were not launched in this run.}"
    )
    lines.append(r"\label{tab:stage_two_summary}")
    lines.append(r"\begin{tabular}{llrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Cell & Clock & Wallclock (s) & Gen-steps & Crit-steps & "
        r"$\overline{\text{EM}}_{-100}$ & $\overline{\text{GP}}$ & "
        r"$\overline{\text{GP}}_{\max}$ \\"
    )
    lines.append(r"\midrule")
    for cell_us in CELLS:
        for i, clock in enumerate(CLOCKS):
            cell_cell = _VENUE_LABEL[cell_us] if i == 0 else ""
            t = trajectories.get((cell_us, clock))
            if t is None:
                lines.append(
                    " & ".join([
                        cell_cell, _CLOCK_LABEL[clock],
                        "--", "--", "--", "--", "--", "--",
                    ]) + r" \\"
                )
                continue
            wc = (f"{t.meta.wallclock_secs:.1f}"
                  if t.meta and t.meta.wallclock_secs == t.meta.wallclock_secs
                  else "--")
            gs = f"{t.meta.generator_steps:,}" if t.meta else "--"
            cs = f"{t.meta.critic_steps:,}" if t.meta else "--"
            lines.append(
                " & ".join([
                    cell_cell, _CLOCK_LABEL[clock],
                    wc, gs, cs,
                    _pm(t.em_final_mean, t.em_final_std, 2),
                    f"${t.gp_mean:.3f}$",
                    f"${t.gp_max:.2f}$",
                ]) + r" \\"
            )
        if cell_us != CELLS[-1]:
            lines.append(r"\addlinespace")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    out_path.write_text("\n".join(lines) + "\n")


def write_yield_table(
    summaries: dict[tuple[str, str], CellClockSummary],
    out_path: str | Path,
) -> None:
    """Table 1: bucket construction yield (6 rows: 2 cells x 3 clocks).

    Uses table* (two-column-spanning) because the 7-column layout would
    overflow a single column at 10pt in tex-article.cls.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Stage-1 bucket construction yield per cell and clock for "
        r"run \texttt{00000000c002daaaX}. $N_{\text{rec}}$ is the count of "
        r"buckets that reached the recurrence-plot writer; "
        r"$N_{\text{rqa}}$ is the count admitted to RQA aggregation "
        r"($n{\ge}16$ snapshots).}"
    )
    lines.append(r"\label{tab:stage_one_yield}")
    lines.append(r"\begin{tabular}{llrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Cell & Clock & $N_{\text{buckets}}$ & $N_{\text{dropped}}$ & "
        r"$N_{\text{trunc}}$ & $N_{\text{rec}}$ & Yield \\"
    )
    lines.append(r"\midrule")
    for cell_us in CELLS:
        for i, clock in enumerate(CLOCKS):
            s = summaries.get((cell_us, clock))
            if s is None:
                continue
            cell_cell = _VENUE_LABEL[cell_us] if i == 0 else ""
            lines.append(
                " & ".join([
                    cell_cell,
                    _CLOCK_LABEL[clock],
                    _fmt_int(s.n_buckets),
                    _fmt_int(s.n_dropped_below_min),
                    _fmt_int(s.n_truncated),
                    _fmt_int(s.n_recurrence),
                    _fmt_pct(s.yield_pct),
                ]) + r" \\"
            )
        if cell_us != CELLS[-1]:
            lines.append(r"\addlinespace")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    out_path.write_text("\n".join(lines) + "\n")


def write_rqa_table(
    summaries: dict[tuple[str, str], CellClockSummary],
    out_path: str | Path,
) -> None:
    """Table 2: aggregate RQA statistics (mean +/- std) at RR* = 5%."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Aggregate Recurrence Quantification Analysis at the "
        r"headline target $RR^{*}{=}5\%$ across the bucket population per "
        r"cell and clock for run \texttt{00000000c002daaaX}. Values are "
        r"mean$\pm$std over buckets with $n{\ge}16$ snapshots. "
        r"$\overline{L_{\max}}$ rounded to the nearest integer.}"
    )
    lines.append(r"\label{tab:stage_one_rqa}")
    lines.append(r"\begin{tabular}{llrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Cell & Clock & $N_{\text{rqa}}$ & "
        r"$\overline{RR}$ & $\overline{DET}$ & $\overline{LAM}$ & "
        r"$\overline{L_{\max}}$ & $\overline{TT}$ & $\overline{ENTR}$ \\"
    )
    lines.append(r"\midrule")
    for cell_us in CELLS:
        for i, clock in enumerate(CLOCKS):
            s = summaries.get((cell_us, clock))
            if s is None:
                continue
            stats = rqa_summary_row(s.rqa)
            cell_cell = _VENUE_LABEL[cell_us] if i == 0 else ""
            row_vals = [
                cell_cell,
                _CLOCK_LABEL[clock],
                _fmt_int(s.n_rqa_admitted),
                _pm(stats["RR_mean"], stats["RR_std"], 3),
                _pm(stats["DET_mean"], stats["DET_std"], 3),
                _pm(stats["LAM_mean"], stats["LAM_std"], 3),
                _pm(stats["L_max_mean"], stats["L_max_std"], 1),
                _pm(stats["TT_mean"], stats["TT_std"], 2),
                _pm(stats["ENTR_mean"], stats["ENTR_std"], 2),
            ]
            lines.append(" & ".join(row_vals) + r" \\")
        if cell_us != CELLS[-1]:
            lines.append(r"\addlinespace")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    out_path.write_text("\n".join(lines) + "\n")
