# syntax=docker/dockerfile:1.7
# ──────────────────────────────────────────────────────────────────────────────
# m4-stage-one.Dockerfile — standalone Stage-1 demo image (Apple Silicon).
#
# Requires BuildKit (Docker 18.09+ with DOCKER_BUILDKIT=1, or Docker
# Desktop 2.4+, or `docker buildx`). The `# syntax=` directive above and
# the `COPY <<EOF ... EOF` heredoc below both depend on BuildKit's
# dockerfile-frontend 1.4+.
#
# What this image does at runtime:
#   Runs `tarel run-stage-one --produce-dp-rp` for one (cell, seed) pair
#   against a parquet bind-mounted from the host. Outputs the run directory
#   tree (manifest, buckets, DPs, RPs, 64x64 W-GAN-ready tensors, PNG grids)
#   into the bind-mounted ./artifacts/runs on the host, ready for stage-two
#   to consume.
#
# Build context vs. sibling crates:
#   tarel-rs's Cargo.toml depends on sibling crates `takens-rs/` and
#   `logan-rs/` via relative paths (`../../../takens-rs/takens-rp`,
#   `../../../logan-rs`). To make this image self-contained — buildable from
#   any clone of just the tarel/ repo — we `git clone` those siblings into
#   /work at build time. The build context is the tarel/ repo itself
#   (`context: .` in docker-compose).
#
# Platform:
#   linux/arm64 (Apple Silicon). PyTorch is the CPU build — stage-one is
#   I/O- and CPU-bound, no GPU needed. Stage-two GPU training is not the
#   target of this demo image; for CUDA see the ultradian_* scripts.
#
# Build (via docker-compose):
#   docker compose build stage-one
#
# Run (via docker-compose):
#   docker compose run --rm stage-one
#
# Manual build/run (without compose):
#   docker build --platform linux/arm64 -t tarel-rs:m4-stage-one \
#       -f m4-stage-one.Dockerfile .
#   docker run --rm --platform linux/arm64 \
#       -v "$PWD/datasets":/work/tarel/datasets:ro \
#       -v "$PWD/artifacts/runs":/work/tarel/artifacts/runs \
#       tarel-rs:m4-stage-one
# ──────────────────────────────────────────────────────────────────────────────

FROM --platform=linux/arm64 python:3.12-slim-bookworm

# ── System dependencies ──────────────────────────────────────────────────────
# build-essential + cmake + pkg-config: the Rust build needs a C/C++ toolchain
# and pkg-config to discover libtorch via PyTorch's installed share/cmake files.
# git: needed to clone takens-rs and logan-rs at build time.
# libssl-dev + ca-certificates: TLS for cargo, rustup, pip indexes.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        pkg-config \
        curl \
        git \
        ca-certificates \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Rust toolchain ───────────────────────────────────────────────────────────
# Pinned to 1.85.0 to match tarel-rs Cargo.toml `rust-version = "1.85.0"`.
# `--profile minimal` keeps the image small (no docs / no clippy by default).
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain 1.85.0 --profile minimal

# ── PyTorch (CPU) for libtorch linkage ───────────────────────────────────────
# tch 0.18 is pinned in tarel-rs/Cargo.toml and is the stable pair with
# PyTorch 2.5.x. We install torch into the system Python (the base image is
# already python:3.12), which is simpler than a venv inside an image whose
# only Python is the one we care about. LIBTORCH_USE_PYTORCH tells tch's
# build script to discover libtorch via the active Python install.
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
RUN pip install --upgrade pip \
 && pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu

# tch build/runtime knobs
ENV LIBTORCH_USE_PYTORCH=1 \
    LIBTORCH_BYPASS_VERSION_CHECK=1
# Runtime linker path: when the built binary launches it loads
# libtorch_cpu.so etc. from inside the site-packages torch install.
# The path is deterministic for python:3.12-slim-bookworm + a pip install
# into the system Python, so we hard-code it. If we ever switch base
# image / Python version, this must change in lock-step.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/torch/lib

# CUDA-stub symlinks. torch-sys's build.rs emits unconditional
# `cargo:rustc-link-lib=torch_cuda` / `c10_cuda` and `-Wl,--no-as-needed`,
# even when LIBTORCH_USE_PYTORCH points at a CPU-only wheel that lacks
# those .so files. Without these symlinks the final cargo link step
# dies with `cannot find -ltorch_cuda` / `-lc10_cuda`.
#
# Pointing the missing CUDA libs at the CPU equivalents satisfies the
# linker (no unresolved symbols are referenced because we never enter
# CUDA code paths on this image — stage-two's `Device::cuda_if_available()`
# falls back to CPU at runtime, see tarel-rs/src/stage_two.rs:88).
RUN cd /usr/local/lib/python3.12/site-packages/torch/lib \
 && ln -sf libtorch_cpu.so libtorch_cuda.so \
 && ln -sf libc10.so       libc10_cuda.so

# ── Sibling crates (git clone) ───────────────────────────────────────────────
# tarel-rs's Cargo.toml uses `path = "../../../takens-rs/takens-rp"` etc.
# Relative to /work/tarel/codes/tarel-rs, three levels up is /work, so
# takens-rs and logan-rs must sit at /work/takens-rs and /work/logan-rs.
#
# Refs are exposed as build args so the demo is pin-able later (e.g.
# `docker compose build --build-arg TAKENS_REF=<branch-or-tag> stage-one`).
# Limitation: `--branch <ref>` accepts branches and tags but NOT commit
# SHAs. If you ever need SHA-pinning for reproducibility, switch the RUN
# block below to:
#     git clone <url> repo && cd repo && git checkout <sha>
# (and drop --depth 1, which is incompatible with arbitrary SHAs).
ARG TAKENS_REPO=https://github.com/convergentist/takens-rs.git
ARG TAKENS_REF=main
ARG LOGAN_REPO=https://github.com/convergentist/logan-rs.git
ARG LOGAN_REF=main

WORKDIR /work
RUN git clone --depth 1 --branch "${TAKENS_REF}" "${TAKENS_REPO}" takens-rs \
 && git clone --depth 1 --branch "${LOGAN_REF}"  "${LOGAN_REPO}"  logan-rs

# ── tarel sources ────────────────────────────────────────────────────────────
# Copy the whole tarel/ build context into /work/tarel. A .dockerignore at
# the repo root keeps target/, artifacts/, .git/, large datasets/ etc. out
# of the build context so this stays fast.
COPY . /work/tarel

# ── Build the release binary ─────────────────────────────────────────────────
WORKDIR /work/tarel/codes/tarel-rs
RUN cargo build --release

# Convenience: put the built binary on PATH so the entrypoint is short.
ENV PATH=/work/tarel/codes/tarel-rs/target/release:$PATH

# ── Entrypoint ───────────────────────────────────────────────────────────────
# All inputs are configurable via env vars. Defaults match the available
# dataset (kraken:BTC-USDC) so `docker compose run stage-one` works
# out-of-the-box with the parquet that ships in datasets/consolidated/.
ENV CELL=kraken:BTC-USDC \
    INPUT_PARQUET=/work/tarel/datasets/consolidated/kraken/BTC-USDC.parquet \
    RUNS=/work/tarel/artifacts/runs \
    EXP=0xC001DFFF \
    SEED=4

WORKDIR /work/tarel
COPY <<'EOF' /usr/local/bin/run-stage-one.sh
#!/usr/bin/env bash
set -euo pipefail

echo "[stage-one] cell=${CELL} seed=${SEED} exp=${EXP}"
echo "[stage-one] input=${INPUT_PARQUET}"
echo "[stage-one] runs=${RUNS}"

mkdir -p "${RUNS}"

exec tarel run-stage-one \
    --cell "${CELL}" \
    --input "${INPUT_PARQUET}" \
    --runs "${RUNS}" \
    --experiment-seed "${EXP}" \
    --seed "${SEED}" \
    --produce-dp-rp
EOF
RUN chmod +x /usr/local/bin/run-stage-one.sh

ENTRYPOINT ["/usr/local/bin/run-stage-one.sh"]
