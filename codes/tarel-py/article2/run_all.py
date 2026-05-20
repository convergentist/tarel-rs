"""CLI orchestrator: regenerate article-2 tables + figures from a run.

Usage:
    python -m article2.run_all \
        --run-dir /Users/franciscome/git/research/tarel/artifacts/runs/00000000c002daaaX \
        --out-dir /Users/franciscome/git/research/tarel/tex/content/sections-v2/04-results-discussion

Outputs (relative to --out-dir):
    tables/yield.tex
    tables/rqa.tex
    figures/bucket_duration.png
    figures/rqa_violins.png
    figures/mean_rp_per_clock.png
    figures/em_gap_trajectories.png
    figures/partial_stability_scores.json   (auxiliary; used in the discussion text)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .aggregate import aggregate_run
from .plots import (
    plot_bucket_duration,
    plot_em_gap_trajectories,
    plot_loss_evolution_by_cell,
    plot_mean_rp_per_clock,
    plot_rqa_violins,
)
from .stage_two import load_all_trajectories, partial_stability_score
from .tables import (
    write_rqa_table,
    write_stage_two_summary_table,
    write_yield_table,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=str)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"[article2] run_dir = {run_dir}")
    print(f"[article2] out_dir = {out_dir}")

    t0 = time.time()
    print("[article2] aggregating stage_one (RQA across all admitted buckets)...")
    summaries = aggregate_run(str(run_dir))
    print(f"[article2]   {time.time() - t0:.1f}s elapsed")
    for (cell, clock), s in summaries.items():
        print(f"    ({cell}, {clock}): n_buckets={s.n_buckets} "
              f"n_rec={s.n_recurrence} n_rqa={s.n_rqa_admitted}")

    print("[article2] loading stage_two trajectories...")
    trajectories = load_all_trajectories(str(run_dir))
    print(f"[article2]   {len(trajectories)} trained (cell, clock) pairs")
    for (cell, clock), t in trajectories.items():
        print(f"    ({cell}, {clock}): em_final={t.em_final_mean:.2f}"
              f"+/-{t.em_final_std:.2f}, gp_mean={t.gp_mean:.3f}")

    print("[article2] writing tables")
    write_yield_table(summaries, tables_dir / "yield.tex")
    write_rqa_table(summaries, tables_dir / "rqa.tex")
    write_stage_two_summary_table(trajectories,
                                   tables_dir / "stage_two_summary.tex")

    print("[article2] writing figures")
    plot_bucket_duration(summaries, figures_dir / "bucket_duration.png")
    plot_rqa_violins(summaries, figures_dir / "rqa_violins.png")
    plot_mean_rp_per_clock(str(run_dir), summaries,
                            figures_dir / "mean_rp_per_clock.png")
    plot_em_gap_trajectories(trajectories,
                              figures_dir / "em_gap_trajectories.png")
    plot_loss_evolution_by_cell(trajectories,
                                 figures_dir / "loss_evolution_by_cell.png")

    # Auxiliary: 2-axis partial Stability Score for the discussion text.
    scores = partial_stability_score(trajectories)
    aux = {
        "spec": "2-axis partial (em_gap, gp_magnitude), rank-normalised across "
                "trained (cell, clock) pairs at J=1. mode_coverage and curvature "
                "unavailable in this run (training stopped at step 1000).",
        "scores": [
            {"cell": c, "clock": clk,
             "em_final_mean": trajectories[(c, clk)].em_final_mean,
             "em_final_std": trajectories[(c, clk)].em_final_std,
             "gp_mean": trajectories[(c, clk)].gp_mean,
             "gp_max": trajectories[(c, clk)].gp_max,
             "partial_score": score}
            for (c, clk), score in sorted(scores.items())
        ],
    }
    (figures_dir / "partial_stability_scores.json").write_text(
        json.dumps(aux, indent=2) + "\n"
    )

    print(f"[article2] done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
