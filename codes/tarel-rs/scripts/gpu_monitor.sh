#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# gpu_monitor.sh — live NVIDIA GPU monitor for the Ultradian
#
# Refreshes every $INTERVAL seconds (default 5). Per-device row shows
# utilization, VRAM (used/free/total), temperature, power draw + limit,
# fan, graphics + memory clocks, and PCIe link generation + width. The
# bottom block is per-process compute usage from `nvidia-smi pmon`.
#
# Usage:
#   ./scripts/gpu_monitor.sh
#   INTERVAL=2 ./scripts/gpu_monitor.sh   # faster refresh
#
# To quit: Ctrl+C.
# ──────────────────────────────────────────────────────────────────────────────

set -u
INTERVAL="${INTERVAL:-5}"

command -v nvidia-smi >/dev/null || {
    echo "[error] nvidia-smi not found on PATH" >&2
    exit 1
}

# Clean exit on Ctrl+C / SIGTERM.
trap 'echo; echo "exit"; exit 0' INT TERM

# Color helpers (ANSI). Falls back to plain text if NO_COLOR is set.
if [[ -z "${NO_COLOR:-}" ]]; then
    C_HDR="\e[1;36m"     # bold cyan
    C_RULE="\e[1;90m"    # bold dim grey
    C_OFF="\e[0m"
else
    C_HDR=""; C_RULE=""; C_OFF=""
fi

RULE="────────────────────────────────────────────────────────────────────────────────────────────────────"

while true; do
    clear

    printf "${C_HDR}GPU Monitor${C_OFF}  —  %s UTC  —  refresh ${INTERVAL}s  —  Ctrl+C to quit\n\n" \
        "$(date -u '+%Y-%m-%d %H:%M:%S')"

    printf "${C_RULE}── Devices ${RULE:0:88}${C_OFF}\n"
    {
        echo "Idx,Name,Util%,Mem%,Used(MiB),Free(MiB),Total(MiB),Temp(°C),Power(W),Limit(W),Fan%,GfxClk(MHz),MemClk(MHz),PCIe-Gen,PCIe-W"
        nvidia-smi \
            --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.free,memory.total,temperature.gpu,power.draw,power.limit,fan.speed,clocks.current.graphics,clocks.current.memory,pcie.link.gen.current,pcie.link.width.current \
            --format=csv,noheader,nounits 2>/dev/null \
        | sed 's/, /,/g'
    } | column -t -s ','

    echo
    printf "${C_RULE}── Active CUDA processes (pid, type=C/G, sm%%, mem%%, enc%%, dec%%, command) ${RULE:0:18}${C_OFF}\n"
    nvidia-smi pmon -c 1 2>/dev/null | head -20

    sleep "$INTERVAL"
done
