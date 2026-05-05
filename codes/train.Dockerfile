
# ============================================================================
# Train for the TAReL Project — multi-stage Docker build
#
# Build context: Rust codes root (codes/tarel-rs)
#
#   docker build -t tarel-train -f tarel/codes/train.Dockerfile .
# ============================================================================

# ── Stage 1: builder ─────────────────────────────────────────────────────────

FROM rust:1.88-bookworm AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        protobuf-compiler libprotobuf-dev cmake build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN cargo run -- train \
    --data src/datasets/rp/train \
    --output generator.pt \
    --epochs 200 \
    --batch-size 10 \
    --latent-dim 128 \
    --img-size 64

