#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup_ubuntu_venv.sh
#
# One-time setup of a pyenv-managed virtualenv for tarel-rs on the Ultradian
# Linux box (Ubuntu x86_64, GTX 1080 Ti). Creates `ubuntu_tarel_venv` with
# Python 3.12 and PyTorch 2.5.1 + CUDA — pinned to match `tch = "0.18"` in
# the tarel-rs and logan-rs Cargo.toml files.
#
# Why these exact versions:
#
# - tarel-rs and logan-rs are pinned to tch 0.18, which targets the
#   PyTorch 2.5.x ATen C++ surface. Later tch releases chase an unstable
#   moving target; 2.5.1 is the stable pair.
# - GTX 1080 Ti is Pascal-architecture (compute capability 6.1). PyTorch
#   2.5.1 still supports Pascal; CUDA 11.8 is the safest CUDA toolkit
#   pairing (cu121/cu124 also work if your driver is new enough, but
#   cu118 has the widest driver compatibility for older boxes).
# - Python 3.12 is in PyTorch 2.5's wheel support window (3.9–3.12).
#
# Usage:
#   ./scripts/setup_ubuntu_venv.sh
#
# Override defaults via env:
#   CUDA_VARIANT=cu121 ./scripts/setup_ubuntu_venv.sh   # if your driver supports CUDA 12.1
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PY_VERSION="${PY_VERSION:-3.12.7}"
VENV_NAME="${VENV_NAME:-ubuntu_tarel_venv}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
CUDA_VARIANT="${CUDA_VARIANT:-cu118}"   # cu118 | cu121 | cu124 | cpu

# Sanity-check pyenv + pyenv-virtualenv are available.
command -v pyenv >/dev/null || {
    echo "[error] pyenv is not on PATH." >&2
    echo "        Install via:  curl https://pyenv.run | bash" >&2
    echo "        Then add to your shell rc:" >&2
    echo '          export PYENV_ROOT="$HOME/.pyenv"' >&2
    echo '          [[ -d "$PYENV_ROOT/bin" ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >&2
    echo '          eval "$(pyenv init -)"' >&2
    echo '          eval "$(pyenv virtualenv-init -)"' >&2
    exit 1
}
if ! pyenv help virtualenv >/dev/null 2>&1; then
    echo "[error] pyenv-virtualenv plugin not installed." >&2
    echo "        Install via:" >&2
    echo "          git clone https://github.com/pyenv/pyenv-virtualenv \\" >&2
    echo "              \"\$(pyenv root)/plugins/pyenv-virtualenv\"" >&2
    exit 1
fi

# Driver / GPU sanity-check (non-blocking — we still install if no GPU).
if command -v nvidia-smi >/dev/null; then
    echo "[gpu ] $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
else
    echo "[warn] nvidia-smi not found; CUDA wheel will install but training will fall back to CPU."
fi

echo "[1/4] Ensure Python $PY_VERSION is installed via pyenv"
pyenv install --skip-existing "$PY_VERSION"

echo "[2/4] Create virtualenv $VENV_NAME (force-recreate if exists)"
if pyenv virtualenvs --bare | grep -qx "$VENV_NAME"; then
    echo "       removing existing $VENV_NAME"
    pyenv uninstall -f "$VENV_NAME"
fi
pyenv virtualenv "$PY_VERSION" "$VENV_NAME"

echo "[3/4] Install PyTorch $TORCH_VERSION ($CUDA_VARIANT)"
PYENV_VERSION="$VENV_NAME" pyenv exec pip install --upgrade pip
PYENV_VERSION="$VENV_NAME" pyenv exec pip install \
    "torch==$TORCH_VERSION" \
    --index-url "https://download.pytorch.org/whl/$CUDA_VARIANT"

echo "[4/4] Verify"
PYENV_VERSION="$VENV_NAME" pyenv exec python - <<'PY'
import torch, os
print(f"torch.__version__        = {torch.__version__}")
print(f"torch install path       = {os.path.dirname(torch.__file__)}")
print(f"torch.cuda.is_available  = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch.cuda.device_count  = {torch.cuda.device_count()}")
    print(f"torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
    print(f"torch.version.cuda       = {torch.version.cuda}")
    cap = torch.cuda.get_device_capability(0)
    print(f"compute capability       = {cap[0]}.{cap[1]}")
PY

cat <<EOF

────────────────────────────────────────────────────────────────────────────
Setup complete.

Per-shell usage (run these before \`cargo build\` in tarel-rs or logan-rs):

    pyenv shell $VENV_NAME
    source $(pwd)/scripts/ubuntu_env.sh
    cargo build --release

After build, training picks the GPU automatically via
\`tch::Device::cuda_if_available()\` (already wired in stage_two.rs:88).

────────────────────────────────────────────────────────────────────────────
EOF
