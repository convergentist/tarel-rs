# ──────────────────────────────────────────────────────────────────────────────
# Dockerfile — isolated build/run environment for tarel-rs on Apple Silicon.
#
# Why this exists:
#   The Mac dev path (scripts/setup_mac_venv.sh + scripts/mac_env.sh) works
#   when your Xcode CLT + libc++ + PyTorch 2.5.1 line up. When they don't —
#   e.g. on a freshly-set-up M-series machine with the latest CLT — this
#   Dockerfile gives you a Linux ARM64 environment where libc++ is replaced
#   by libstdc++ and the PyTorch 2.5.x strong_type.h issue cannot trigger.
#
# What it builds:
#   - debian:bookworm-slim (ARM64 — pulled natively on Apple Silicon)
#   - rust toolchain matching the workspace's rust-version (1.85)
#   - python 3.12 + torch 2.5.1 (CPU build)
#   - the tarel-rs workspace, compiled in --release
#
# Build:
#   docker build --platform linux/arm64 -t tarel-rs:dev -f Dockerfile ../../..
#   (the build context is the *repo root* so the Docker daemon can see
#    sibling crates takens-rs/ and logan-rs/ that tarel-rs depends on via
#    relative paths in Cargo.toml.)
#
# Run:
#   docker run --rm -it --platform linux/arm64 \
#       -v "$PWD/datasets":/work/tarel/datasets \
#       -v "$PWD/runs":/work/tarel/runs \
#       tarel-rs:dev bash
#
# ──────────────────────────────────────────────────────────────────────────────

FROM --platform=linux/arm64 debian:bookworm-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config curl ca-certificates \
        python3.11 python3.11-venv python3-pip \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Rust toolchain pinned to match workspace rust-version (1.85.0).
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo
ENV PATH="$CARGO_HOME/bin:$PATH"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain 1.85.0 --profile minimal

# Python venv with PyTorch 2.5.1 (matches tch 0.22 ABI). Python 3.12
RUN python3.12 -m venv /opt/m4_tarel_venv \
 && /opt/tarel_venv/bin/pip install --upgrade pip \
 && /opt/tarel_venv/bin/pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu
ENV VIRTUAL_ENV=/opt/m4_tarel_venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV LIBTORCH_USE_PYTORCH=1
ENV LIBTORCH_BYPASS_VERSION_CHECK=1
ENV LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH}"

# Workspace layout: this Dockerfile sits at tarel/Dockerfile
# but tarel-rs depends on sibling crates via `path = "../takens-rs/..."`
# and `path = "../logan-rs"`. So the build context must be the repo
# root containing tarel/, takens-rs/, logan-rs/ side-by-side. The COPY
# pattern below mirrors that layout inside the image.
WORKDIR /work
COPY tarel       /work/tarel
COPY takens-rs   /work/takens-rs
COPY logan-rs    /work/logan-rs

WORKDIR /work/tarel/codes/tarel-rs
RUN cargo build --release

# Default command drops you into a shell with cargo + python ready.
CMD ["bash"]
