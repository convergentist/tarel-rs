#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# m4_setup_venv.sh
#
# One-time setup of a pyenv-managed virtualenv for tarel-rs development on
# Apple Silicon (M4) Mac. 
# Creates `m4_tarel_venv` with Python 3.12 and PyTorch 2.5.1 + CUDA
# These are necessary to match `tch = "0.18"` in both tarel-rs and logan-rs 
# Cargo.toml files.
#
# Why these exact versions:
#
# - tarel-rs and logan-rs are pinned to tch 0.18 (see their Cargo.toml).
#   tch 0.18 was generated against PyTorch 2.5.x; later tch releases chase
#   an unstable moving target in PyTorch's private ATen API, with no
#   released PyTorch wheel exactly matching tch 0.20/0.21/0.22.
# - PyTorch 2.5.x has been GA since October 2024 (mature wheels for
#   Apple Silicon), supports Python 3.9–3.12, and ships the strong_type.h
#   libc++ fix so Xcode 16 CLT doesn't error on `is_arithmetic`
#   specialization.
# - We use Python 3.12 for the venv: stable, fast, and avoids your
#   existing 3.13.9 (where pip would force-install a newer torch).
# - The libc++ `is_arithmetic` specialization error comes from newer
#   Xcode CLT (16+) enforcing `_LIBCPP_NO_SPECIALIZATIONS` on standard
#   traits. PyTorch 2.5's bundled `c10/util/strong_type.h` predates the
#   fix; we suppress the warning via CXXFLAGS in `mac_env.sh` so the
#   error becomes a no-op warning.
#
# Usage:
#   ./scripts/setup_mac_venv.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PY_VERSION="${PY_VERSION:-3.12.7}"
VENV_NAME="${VENV_NAME:-m4_tarel_venv}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"

# Sanity-check pyenv + pyenv-virtualenv are available.
command -v pyenv >/dev/null || {
    echo "[error] pyenv is not on PATH. Install via brew install pyenv pyenv-virtualenv." >&2
    exit 1
}
if ! pyenv help virtualenv >/dev/null 2>&1; then
    echo "[error] pyenv-virtualenv plugin not installed. Install via brew install pyenv-virtualenv." >&2
    exit 1
fi

echo "[1/4] Ensure Python $PY_VERSION is installed via pyenv"
pyenv install --skip-existing "$PY_VERSION"

echo "[2/4] Create virtualenv $VENV_NAME (force-recreate if exists)"
if pyenv virtualenvs --bare | grep -qx "$VENV_NAME"; then
    echo "       removing existing $VENV_NAME"
    pyenv uninstall -f "$VENV_NAME"
fi
pyenv virtualenv "$PY_VERSION" "$VENV_NAME"

echo "[3/4] Install PyTorch $TORCH_VERSION (CPU build — Mac is dev/build only, GPU training runs on the Linux box)"
# Force the venv as the active python for this script's shell only.
PYENV_VERSION="$VENV_NAME" pyenv exec pip install --upgrade pip
PYENV_VERSION="$VENV_NAME" pyenv exec pip install "torch==$TORCH_VERSION"

echo "[4/4] Verify"
PYENV_VERSION="$VENV_NAME" pyenv exec python - <<'PY'
import torch, os
print(f"torch.__version__   = {torch.__version__}")
print(f"torch install path  = {os.path.dirname(torch.__file__)}")
print(f"libtorch lib dir    = {os.path.dirname(torch.__file__)}/lib")
PY

cat <<EOF

────────────────────────────────────────────────────────────────────────────
Setup complete.

Per-shell usage (run these before \`cargo build\` in tarel-rs or logan-rs):

    pyenv shell $VENV_NAME
    source $(pwd)/scripts/m4_tarel_env.sh
    cargo build --release

────────────────────────────────────────────────────────────────────────────
EOF
