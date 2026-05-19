# syntax=docker/dockerfile:1.7
# ──────────────────────────────────────────────────────────────────────────────
# m4-stage-two.Dockerfile — Stage-2 demo image (Apple Silicon).
#
# Requires BuildKit (for the `COPY <<EOF ... EOF` heredoc and for the
# `additional_contexts` build feature wired in from docker-compose.yml).
#
# What this image does at runtime:
#   Runs `tarel run-stage-two` for the (cell, seeds, clocks) triple
#   produced by stage-one, training the W-GAN-GP across the configured
#   wallclock budget per seed. Reads from and writes into the same
#   bind-mounted ./artifacts/runs that stage-one populated.
#
# Why this Dockerfile is so short:
#   The build environment, sibling crates, Rust toolchain, PyTorch and
#   compiled `tarel` binary are all identical to stage-one. To avoid
#   building the whole stack twice (≈5 GB of cargo + torch), this image
#   uses the stage-one image as its base. docker-compose wires the
#   dependency via `additional_contexts: stage_one_image: service:stage-one`
#   (see docker-compose.yml) so the stage-one image is guaranteed to be
#   built first.
#
# Manual build (without compose):
#   First build the stage-one image, then this one with --build-context:
#     docker build --platform linux/arm64 -t tarel-rs:m4-stage-one \
#         -f m4-stage-one.Dockerfile .
#     docker build --platform linux/arm64 -t tarel-rs:m4-stage-two \
#         --build-context stage_one_image=docker-image://tarel-rs:m4-stage-one \
#         -f m4-stage-two.Dockerfile .
# ──────────────────────────────────────────────────────────────────────────────

# `stage_one_image` is a build context wired in by docker-compose
# (additional_contexts) or by `--build-context` on a manual build. Using a
# named build context — rather than a hard-coded image tag — keeps this
# Dockerfile decoupled from how the stage-one image got tagged.
FROM stage_one_image

# ── Entrypoint ───────────────────────────────────────────────────────────────
# Runtime env vars (CELL, SEED, RUN_MINS, RUNS, EXP, CLOCKS) are supplied by
# docker-compose.yml's `environment:` block (with `.env` overrides via the
# `${VAR:-default}` pattern). The Dockerfile intentionally does NOT bake
# defaults for them, so docker-compose.yml stays the single source of truth
# for runtime configuration. If you run this image directly without compose,
# pass them with `-e VAR=...` on the `docker run` command line.
#
# Notes on flag semantics (from `tarel run-stage-two --help`):
#   --cells        comma-separated <exchange>:<symbol> identifiers
#   --seeds        NUMBER of seeds (not a specific seed value); pilot=3, prod=20
#   --budget-mins  wallclock budget per (cell, seed) in minutes
#   --clocks       comma-separated subset of {physical,event,info_time}

WORKDIR /work/tarel
COPY <<'EOF' /usr/local/bin/run-stage-two.sh
#!/usr/bin/env bash
set -euo pipefail

echo "[stage-two] cells=${CELL} seeds=${SEED} budget_mins=${RUN_MINS}"
echo "[stage-two] runs=${RUNS} exp=${EXP} clocks=${CLOCKS}"

if [[ ! -d "${RUNS}" ]] || [[ -z "$(ls -A "${RUNS}" 2>/dev/null || true)" ]]; then
    echo "[stage-two] ERROR: runs dir '${RUNS}' is missing or empty." >&2
    echo "[stage-two]        Did stage-one complete successfully?" >&2
    exit 1
fi

exec tarel run-stage-two \
    --cells "${CELL}" \
    --seeds "${SEED}" \
    --budget-mins "${RUN_MINS}" \
    --runs "${RUNS}" \
    --experiment-seed "${EXP}" \
    --clocks "${CLOCKS}"
EOF
RUN chmod +x /usr/local/bin/run-stage-two.sh

ENTRYPOINT ["/usr/local/bin/run-stage-two.sh"]
