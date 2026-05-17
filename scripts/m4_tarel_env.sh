# ──────────────────────────────────────────────────────────────────────────────
# m4_tarel_env.sh — source this in your shell before `cargo build` on macOS.
#
#   pyenv shell m4_tarel_venv
#   source scripts/m4_tarel_env.sh
#   cargo build --release
#
# Intentionally not executable: must be sourced, not run, so env vars
# survive into the current shell.
# ──────────────────────────────────────────────────────────────────────────────

# Tell tch's build script to discover libtorch via the active Python venv
# (rather than expecting a separately-downloaded libtorch directory).
export LIBTORCH_USE_PYTORCH=1

# tch hard-checks libtorch version against its own internal pin. We pin
# PyTorch 2.5.1 in setup_mac_venv.sh to match tch 0.22, so this is purely
# defensive — but keeps us from failing if a future tch patch tightens the
# check.
export LIBTORCH_BYPASS_VERSION_CHECK=1

# Suppress the libc++ specialization error in PyTorch 2.5.x's
# `c10/util/strong_type.h`. Newer Xcode CLT (16+) marks `std::is_arithmetic`
# with `_LIBCPP_NO_SPECIALIZATIONS`, which PyTorch 2.5's strong_type
# header specializes. Both warning *and* error-as-warning are required:
# the first prevents the warning from being emitted; the second
# prevents -Werror upgrades downstream.
export CXXFLAGS="${CXXFLAGS:-} -Wno-error=invalid-specialization -Wno-invalid-specialization"

# Runtime linker path: when the built binary launches it loads
# libtorch_cpu.dylib etc. from inside the venv's torch install.
TORCH_LIB_DIR="$(python -c 'import os, torch; print(os.path.dirname(torch.__file__))')/lib"
export DYLD_LIBRARY_PATH="$TORCH_LIB_DIR${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

echo "[mac_env] LIBTORCH_USE_PYTORCH=$LIBTORCH_USE_PYTORCH"
echo "[mac_env] LIBTORCH_BYPASS_VERSION_CHECK=$LIBTORCH_BYPASS_VERSION_CHECK"
echo "[mac_env] CXXFLAGS=$CXXFLAGS"
echo "[mac_env] DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH"
