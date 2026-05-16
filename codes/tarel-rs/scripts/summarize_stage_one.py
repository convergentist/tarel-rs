# scripts/summarize_stage_one.py — paste somewhere
from pathlib import Path
import json, statistics as st

RUNS = Path.home() / "git/research/tarel/runs/00000000c001d00d"
CELLS = ["kraken_BTC-USDC", "coinbase_BTC-USDC"]
CLOCKS = ["physical", "event", "info_time"]
SCALARS = ["rr", "det", "lam", "l_max", "tt", "entr"]

for cell in CELLS:
    for clk in CLOCKS:
        p = RUNS / "stage_one/recurrence" / cell / clk / "rqa_scalars.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if not rows: continue
        print(f"\n── {cell} / {clk}  n_buckets={len(rows)}")
        for s in SCALARS:
            vals = [r[s] for r in rows if s in r and r[s] is not None]
            if vals:
                print(f"    {s:6s}  mean={st.mean(vals):.4f}  std={st.stdev(vals):.4f}  n={len(vals)}")
