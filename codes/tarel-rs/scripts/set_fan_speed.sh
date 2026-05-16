#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# set_fan_speed.sh — set manual fan control to a fixed % on every NVIDIA GPU.
#
# Usage:
#   ./scripts/set_fan_speed.sh           # default 75%
#   ./scripts/set_fan_speed.sh 85        # explicit 85%
#
# Notes:
#   - The setting persists for the lifetime of the current X session.
#     A reboot, or stopping the display manager, resets fans to "auto".
#   - To switch back to automatic control:
#       nvidia-settings -a "[gpu:0]/GPUFanControlState=0"
#       nvidia-settings -a "[gpu:1]/GPUFanControlState=0"
#   - Requires the X server's "Coolbits" option to allow manual fan
#     control. Most desktop Ubuntu installs already have it; if the
#     `GPUTargetFanSpeed` command errors with "Attribute not available",
#     you need to enable Coolbits=4 (or 28) in xorg.conf and restart X.
# ──────────────────────────────────────────────────────────────────────────────

set -u
TARGET="${1:-75}"

# Sanity-check the target value.
if ! [[ "$TARGET" =~ ^[0-9]+$ ]] || (( TARGET < 30 || TARGET > 100 )); then
    echo "[error] fan speed must be an integer in [30, 100], got '$TARGET'" >&2
    exit 1
fi

# Discover an X display. Most servers run Xorg on :0.
if [ -z "${DISPLAY:-}" ]; then
    if pgrep -x Xorg >/dev/null 2>&1 || pgrep -x X >/dev/null 2>&1; then
        export DISPLAY=:0
        echo "[info] DISPLAY was unset; using :0"
    else
        echo "[error] No X server running. Headless fan control needs Xvfb." >&2
        echo "        Quick path:" >&2
        echo "          sudo apt install xvfb" >&2
        echo "          Xvfb :99 -screen 0 1024x768x24 &" >&2
        echo "          DISPLAY=:99 $0 $TARGET" >&2
        exit 1
    fi
fi

# Some display managers stash the auth cookie outside \$HOME. Probe a few
# canonical locations; harmless if none match.
if [ -z "${XAUTHORITY:-}" ]; then
    for auth in "$HOME/.Xauthority" \
                /var/run/lightdm/root/:0 \
                /run/user/$UID/gdm/Xauthority; do
        if [ -e "$auth" ]; then
            export XAUTHORITY="$auth"
            break
        fi
    done
fi

NGPUS=$(nvidia-smi -L | wc -l)
echo "[info] $NGPUS GPU(s) found; target fan speed = ${TARGET}%"

for ((i=0; i<NGPUS; i++)); do
    nvidia-settings \
        -a "[gpu:$i]/GPUFanControlState=1" \
        -a "[fan:$i]/GPUTargetFanSpeed=$TARGET" \
        >/dev/null
    echo "  gpu:$i → manual, $TARGET%"
done

echo
echo "[verify]"
nvidia-smi --query-gpu=index,name,fan.speed,temperature.gpu \
           --format=csv,noheader | sed 's/^/  /'
