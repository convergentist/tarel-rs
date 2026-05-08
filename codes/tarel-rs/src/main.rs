//! `tarel` — orchestration + bridge driver CLI.
//!
//! Stage-1 CLI surface (per `engineering_proposal.md` Part I):
//!
//! ```text
//! tarel run-stage-one --cell binance:BTC-USDT \
//!     --input  /path/to/lob.parquet \
//!     --runs   ./runs \
//!     --experiment-seed 0xC001D00D \
//!     --seed   0
//! ```
//!
//! Stages 2-5 will introduce additional subcommands (`run-stage-two`,
//! `run-all`, `report`, `verify`); they share this scaffolding.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use tarel::cell::Cell;
use tarel::layout::RunLayout;
use tarel::logging;
use tarel::manifest::Manifest;
use tarel::seed::ExperimentSeed;
use tarel::stage_one::{StageOneOptions, run_stage_one_with_opts};
use tarel::stage_two::{StageTwoOptions, run_stage_two};
use tarel::util::sha256_path;

#[derive(Parser, Debug)]
#[command(
    name = "tarel",
    version,
    about = "Orchestration + bridge driver for the HFT-RP-WGAN pipeline"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Run Stage 1 (Foundation: ingest → reconstruct → bucket) for a
    /// single cell against a parquet file.
    RunStageOne {
        /// Cell identifier in `<exchange>:<symbol>` form.
        #[arg(long)]
        cell: String,

        /// Parquet file with the LOB delta stream for the cell.
        #[arg(short, long)]
        input: PathBuf,

        /// `runs/` root directory. Defaults to `./runs`.
        #[arg(long, default_value = "runs")]
        runs: PathBuf,

        /// Top-level experiment seed (decimal or `0x`-prefixed hex).
        #[arg(long, default_value = "0")]
        experiment_seed: String,

        /// Per-cell seed within the experiment. Defaults to 0.
        #[arg(long, default_value_t = 0)]
        seed: u64,

        /// Also produce Stage-2 recurrence-layer artifacts (per-bucket
        /// DPs, RPs at RR=5% and RR=1%, 64×64 normalized W-GAN-ready
        /// tensors, PNG sanity grids).
        #[arg(long, default_value_t = false)]
        produce_dp_rp: bool,

        /// Run the full Stage-3 statistical pass: per-bucket RQA scalars,
        /// KS over 8h sessions, hierarchical variance ratio, Hill
        /// estimator, per-cell Stationarity Score.
        #[arg(long, default_value_t = false)]
        full: bool,
    },

    /// Run Stage-2 (Stage-4 in the proposal lexicon) — W-GAN-GP
    /// training across (cell, seed, clock) triples with four-axis
    /// diagnostics, then aggregate per-cell Stability Scores.
    RunStageTwo {
        /// Comma-separated list of cell identifiers.
        #[arg(long)]
        cells: String,

        /// Number of seeds per cell. Production grid uses 20 seeds; the
        /// pilot uses 3.
        #[arg(long, default_value_t = 3)]
        seeds: u64,

        /// Per-seed wallclock budget in minutes (default 20).
        #[arg(long, default_value_t = 20)]
        budget_mins: u64,

        /// `runs/` root directory.
        #[arg(long, default_value = "runs")]
        runs: PathBuf,

        /// Top-level experiment seed (decimal or `0x`-prefixed hex).
        #[arg(long, default_value = "0")]
        experiment_seed: String,

        /// Optional comma-separated list of clocks
        /// (default: physical,event,info_time).
        #[arg(long)]
        clocks: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::RunStageOne {
            cell,
            input,
            runs,
            experiment_seed,
            seed,
            produce_dp_rp,
            full,
        } => {
            let exp = parse_seed(&experiment_seed)?;
            let layout = RunLayout::new(&runs, exp);
            std::fs::create_dir_all(&layout.run_dir)?;

            // Initialise tracing → JSON-line file under runs/<exp>/logs/.
            logging::init(&layout.orchestrator_log())?;

            tracing::info!(
                experiment_seed = exp.0,
                runs_root = %runs.display(),
                produce_dp_rp,
                full,
                "tarel_start"
            );

            let cell: Cell = cell.parse()?;

            // Load or initialise the manifest. The data-snapshot SHA is
            // the SHA-256 of the input parquet — Stage-5 reproducibility
            // checks compare this across reruns to guarantee the same
            // input produced the same outputs.
            let manifest_path = layout.manifest_path();
            let mut manifest = if manifest_path.exists() {
                Manifest::read_json(&manifest_path)
                    .with_context(|| format!("read {}", manifest_path.display()))?
            } else {
                let snap_sha = sha256_path(&input)
                    .with_context(|| format!("sha256({})", input.display()))?;
                Manifest::new(exp, snap_sha)
            };

            let opts = StageOneOptions {
                produce_dp_rp,
                full,
                ..Default::default()
            };
            let outcome = run_stage_one_with_opts(
                &layout, &mut manifest, &cell, seed, &input, opts,
            )?;

            tracing::info!(
                cell = %outcome.cell,
                snapshots = outcome.n_snapshots,
                physical = outcome.n_buckets_physical,
                event = outcome.n_buckets_event,
                info_time = outcome.n_buckets_info_time,
                info_time_threshold = ?outcome.info_time_threshold,
                "tarel_done"
            );
        }

        Cmd::RunStageTwo {
            cells,
            seeds,
            budget_mins,
            runs,
            experiment_seed,
            clocks,
        } => {
            let exp = parse_seed(&experiment_seed)?;
            let layout = RunLayout::new(&runs, exp);
            std::fs::create_dir_all(&layout.run_dir)?;
            logging::init(&layout.orchestrator_log())?;

            tracing::info!(
                experiment_seed = exp.0,
                runs_root = %runs.display(),
                seeds,
                budget_mins,
                "tarel_stage_two_start"
            );

            // Parse cell list.
            let cell_objs: Result<Vec<Cell>, _> = cells
                .split(',')
                .map(|s| s.trim().parse::<Cell>())
                .collect();
            let cell_objs = cell_objs?;

            let manifest_path = layout.manifest_path();
            let mut manifest = if manifest_path.exists() {
                Manifest::read_json(&manifest_path)
                    .with_context(|| format!("read {}", manifest_path.display()))?
            } else {
                Manifest::new(exp, "(stage_two)")
            };

            let opts = StageTwoOptions {
                experiment: exp,
                cells: cell_objs,
                n_seeds: seeds,
                budget_mins,
                clocks: clocks.map(|s| {
                    s.split(',').map(|p| p.trim().to_string()).collect()
                }),
            };
            let outcome = run_stage_two(&layout, &mut manifest, &opts)?;
            tracing::info!(
                n_runs = outcome.n_runs,
                n_completed = outcome.n_completed,
                n_failed = outcome.n_failed,
                stability_scores_path = %outcome.stability_scores_path.display(),
                "tarel_stage_two_done"
            );
        }
    }
    Ok(())
}

fn parse_seed(s: &str) -> Result<ExperimentSeed> {
    let s = s.trim();
    let value = if let Some(rest) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(rest, 16)
            .with_context(|| format!("invalid hex seed: '{s}'"))?
    } else {
        s.parse::<u64>()
            .with_context(|| format!("invalid decimal seed: '{s}'"))?
    };
    Ok(ExperimentSeed(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_seed_decimal_and_hex() {
        assert_eq!(parse_seed("0").unwrap().0, 0);
        assert_eq!(parse_seed("42").unwrap().0, 42);
        assert_eq!(parse_seed("0xCAFE").unwrap().0, 0xCAFE);
        assert_eq!(parse_seed("0xCAFEBABEDEADBEEF").unwrap().0, 0xCAFEBABEDEADBEEF);
        assert!(parse_seed("not a number").is_err());
    }
}
