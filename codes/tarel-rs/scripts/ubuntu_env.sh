# ──────────────────────────────────────────────────────────────────────────────
# ubuntu_env.sh — source this in your shell before `cargo build` on Ubuntu
# x86_64 (the Ultradian Linux box).
#
#   pyenv shell ubuntu_tarel_venv
#   source scripts/ubuntu_env.sh
#   cargo build --release
#
# Intentionally not executable: must be sourced, not run, so env vars
# survive into the current shell.
#
# Differences vs scripts/mac_env.sh:
#   - LD_LIBRARY_PATH instead of DYLD_LIBRARY_PATH (Linux dynamic linker)
#   - No CXXFLAGS workaround (Linux uses libstdc++; libc++'s
#     `is_arithmetic` specialization issue is macOS-only)
#   - CUDA_HOME + nvcc on PATH if /usr/local/cuda exists
# ──────────────────────────────────────────────────────────────────────────────

# Tell torch-sys's build script to discover libtorch via the active
# Python venv (rather than expecting a separately-downloaded libtorch).
export LIBTORCH_USE_PYTORCH=1

# Defensive: skip torch-sys's libtorch version check. tch 0.18 + PyTorch
# 2.5.1 is the design pair, but bypass the check in case a patch release
# of either side tightens it spuriously.
export LIBTORCH_BYPASS_VERSION_CHECK=1

# Runtime linker path: when the built binary launches it loads
# libtorch_cpu.so / libtorch_cuda.so etc. from inside the venv's torch
# install. Resolve once via the active python.
TORCH_LIB_DIR="$(python -c 'import os, torch; print(os.path.dirname(torch.__file__))')/lib"
export LD_LIBRARY_PATH="$TORCH_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# CUDA toolkit discovery. If CUDA is installed in the canonical location
# at /usr/local/cuda, make `nvcc`, the CUDA libraries, and CUPTI visible
# to the build and runtime. Skip silently if CUDA is system-managed
# elsewhere or not installed (the cargo build will still link against
# the PyTorch-bundled CUDA runtime).
if [ -d /usr/local/cuda ]; then
    export CUDA_HOME=/usr/local/cuda
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    if [ -d "$CUDA_HOME/extras/CUPTI/lib64" ]; then
        export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
    fi
fi

# Optional: explicitly select a GPU on multi-GPU boxes. The Ultradian
# has a single 1080 Ti so this is a no-op, but uncomment if you ever
# add a second card and want to pin Stage-2 training to it.
# export CUDA_VISIBLE_DEVICES=0

echo "[ubuntu_env] LIBTORCH_USE_PYTORCH=$LIBTORCH_USE_PYTORCH"
echo "[ubuntu_env] LIBTORCH_BYPASS_VERSION_CHECK=$LIBTORCH_BYPASS_VERSION_CHECK"
echo "[ubuntu_env] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
if [ -n "${CUDA_HOME:-}" ]; then
    echo "[ubuntu_env] CUDA_HOME=$CUDA_HOME"
fi
