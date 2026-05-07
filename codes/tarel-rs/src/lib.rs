//! Tarel — orchestration + bridge driver for the HFT-RP-WGAN pipeline.
//!
//! `tarel` is the top-level driver that fans out the per-cell, per-seed
//! work across [`takens-rp`] (Stage 1 + Stage 2) and (eventually)
//! `logan-rs` (Stage 4). It owns:
//!
//! - The deterministic seeding scheme rooted at `experiment_seed: u64`
//!   ([`seed`]). Every per-(cell, seed) sub-task pulls a SHA-256
//!   derivation so rerunning with the same `experiment_seed` produces
//!   bit-identical artifacts.
//! - The on-disk run layout under `runs/<experiment_seed>/`
//!   ([`layout`]).
//! - The run-level [`manifest`] that records cell, seed, status,
//!   wallclock, and output paths.
//! - A `tracing`-based JSON-line logger ([`logging`]) wired in at the
//!   orchestration stub so every later stage gets structured logs for
//!   free.
//!
//! The `runs/` directory layout, the manifest schema, and the
//! pre-registered score-spec policy are pinned by
//! `engineering_proposal.md` Part I.

pub mod cell;
pub mod layout;
pub mod logging;
pub mod manifest;
pub mod recurrence;
pub mod seed;
pub mod stage_one;
pub mod stage_three;
pub mod util;

pub use cell::{Cell, CellId, CellMetadata};
pub use layout::RunLayout;
pub use manifest::{CellRunRecord, Manifest};
pub use seed::{ExperimentSeed, derived_seed};
