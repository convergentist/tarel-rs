#!/usr/bin/env python3
import msgpack, json
from pathlib import Path
import matplotlib.pyplot as plt

RUNS = Path.home() / "git/research/tarel/runs/00000000c001d00d/stage_two"
CELLS = ["kraken_BTC-USDC", "coinbase_BTC-USDC"]
COLORS = {"kraken_BTC-USDC": "#2563eb", "coinbase_BTC-USDC": "#dc2626"}

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
metrics = [("em_gap", "EM-gap (W-distance proxy)"),
           ("gp_magnitude", "Gradient-penalty magnitude"),
           ("mode_coverage", "Mode coverage (reverse-NN)"),
           ("critic_curvature", "Critic-loss curvature")]

for cell in CELLS:
    diag_path = RUNS / cell / "0" / "diagnostics.msgpack"
    if not diag_path.exists():
        print(f"MISSING: {diag_path}"); continue
    raw = msgpack.unpackb(diag_path.read_bytes(), raw=False)
    print(f"\n── {cell}  diagnostics keys: {list(raw.keys()) if isinstance(raw, dict) else type(raw)}")
    for ax, (key, title) in zip(axes.flat, metrics):
        ys = raw.get(key, raw.get("metrics", {}).get(key)) if isinstance(raw, dict) else None
        if not ys: continue
        xs = list(range(len(ys)))
        ax.plot(xs, ys, color=COLORS[cell], label=cell, alpha=0.85)
        ax.set_title(title); ax.grid(alpha=0.3); ax.legend()

axes[1, 0].set_xlabel("step / 100"); axes[1, 1].set_xlabel("step / 100")
fig.suptitle("Stage-2 contrast: kraken (1408 RPs) vs coinbase (375 RPs), physical clock", fontsize=13)
fig.tight_layout()
out = Path.home() / "stage_two_contrast.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"\nFigure written: {out}")
