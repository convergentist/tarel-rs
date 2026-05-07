//! Deterministic seeding rooted at a single `experiment_seed: u64`.
//!
//! Per `engineering_proposal.md` Part I, every per-(cell, seed) sub-task
//! must derive its randomness from a SHA-256 of the experiment seed +
//! the (cell, seed, purpose) triple. Re-running with the same
//! `experiment_seed` and the same data snapshot SHA must produce
//! bit-identical artifacts (Stage-5 exit gate).
//!
//! ## Derivation
//!
//! ```text
//!   derived = SHA-256("tarel|v1" || experiment_seed_le8
//!                     || purpose_len_le8 || purpose
//!                     || cell_len_le8 || cell
//!                     || seed_le8)
//! ```
//!
//! The first 8 bytes of `derived` are interpreted little-endian as the
//! `u64` returned to the caller. The hash domain string `"tarel|v1"`
//! buys us future flexibility — bumping it invalidates every previous
//! derivation, which is the right behaviour if the spec changes.

use sha2::{Digest, Sha256};

/// Domain-separation tag for tarel-derived seeds. Bump to `v2` if the
/// derivation contract ever changes.
pub const SEED_DOMAIN_V1: &[u8] = b"tarel|v1";

/// Newtype wrapper around the user-facing experiment seed (logged on
/// every run so the manifest is self-describing).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExperimentSeed(pub u64);

impl ExperimentSeed {
    /// Hex-encoded SHA-256 of just the experiment seed; useful as a
    /// stable directory-name fragment in `runs/<experiment_seed>/`.
    pub fn hex_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(SEED_DOMAIN_V1);
        h.update(self.0.to_le_bytes());
        format!("{:x}", h.finalize())
    }
}

/// Compute a per-(cell, seed, purpose) derived `u64` seed.
///
/// `purpose` is a free-form short string identifying which sub-task is
/// asking for randomness — `"sampler"`, `"init"`, `"dropout"`, …. Callers
/// are responsible for using stable purpose strings; renaming a purpose
/// invalidates determinism for that channel.
pub fn derived_seed(experiment: ExperimentSeed, cell: &str, seed: u64, purpose: &str) -> u64 {
    let mut h = Sha256::new();
    h.update(SEED_DOMAIN_V1);
    h.update(experiment.0.to_le_bytes());
    h.update((purpose.len() as u64).to_le_bytes());
    h.update(purpose.as_bytes());
    h.update((cell.len() as u64).to_le_bytes());
    h.update(cell.as_bytes());
    h.update(seed.to_le_bytes());
    let digest = h.finalize();
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn experiment_seed_hex_digest_is_stable() {
        let s = ExperimentSeed(0xCAFEBABE_DEADBEEF);
        // Stable across runs.
        assert_eq!(s.hex_digest(), s.hex_digest());
        // 64 hex chars = 32 bytes of SHA-256.
        assert_eq!(s.hex_digest().len(), 64);
    }

    #[test]
    fn derived_seed_is_deterministic() {
        let exp = ExperimentSeed(42);
        let a = derived_seed(exp, "binance:BTC-USDT", 7, "sampler");
        let b = derived_seed(exp, "binance:BTC-USDT", 7, "sampler");
        assert_eq!(a, b);
    }

    #[test]
    fn derived_seed_changes_with_each_input() {
        let exp = ExperimentSeed(42);
        let base = derived_seed(exp, "binance:BTC-USDT", 7, "sampler");

        assert_ne!(base, derived_seed(ExperimentSeed(43), "binance:BTC-USDT", 7, "sampler"));
        assert_ne!(base, derived_seed(exp, "binance:ETH-USDT", 7, "sampler"));
        assert_ne!(base, derived_seed(exp, "binance:BTC-USDT", 8, "sampler"));
        assert_ne!(base, derived_seed(exp, "binance:BTC-USDT", 7, "init"));
    }

    #[test]
    fn derivation_distinguishes_collision_attacks() {
        // Without length-prefixing, ("ab", "c") and ("a", "bc") would
        // collide. Verify the length-prefix protects us.
        let exp = ExperimentSeed(1);
        let a = derived_seed(exp, "ab", 0, "c");
        let b = derived_seed(exp, "a", 0, "bc");
        assert_ne!(a, b);
    }
}
