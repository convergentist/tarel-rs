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
use tarel::stage_one::run_stage_one;

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
        } => {
            let exp = parse_seed(&experiment_seed)?;
            let layout = RunLayout::new(&runs, exp);
            std::fs::create_dir_all(&layout.run_dir)?;

            // Initialise tracing → JSON-line file under runs/<exp>/logs/.
            logging::init(&layout.orchestrator_log())?;

            tracing::info!(
                experiment_seed = exp.0,
                runs_root = %runs.display(),
                "tarel_start"
            );

            let cell: Cell = cell.parse()?;

            // Load or initialise the manifest. We treat the parquet
            // path as the data-snapshot identifier when no prior
            // manifest exists; later stages compute the rolling SHA.
            let manifest_path = layout.manifest_path();
            let mut manifest = if manifest_path.exists() {
                Manifest::read_json(&manifest_path)
                    .with_context(|| format!("read {}", manifest_path.display()))?
            } else {
                Manifest::new(exp, input.display().to_string())
            };

            let outcome = run_stage_one(&layout, &mut manifest, &cell, seed, &input)?;

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
