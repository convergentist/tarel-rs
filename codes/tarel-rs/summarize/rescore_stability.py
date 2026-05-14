"""Re-score stability across the pilot using the corrected
'rank across cells' policy. Reads existing diagnostics.msgpack files;
no Rust rebuild or retraining required.

Output schema mirrors stability_scores.json + the v0.1.0 weights, but
computes per-cell mean_rank_norm by ranking each component's per-cell
*statistic* across the 3-cell population (not within-cell)."""

import msgpack, struct, glob, json, statistics, os
from collections import defaultdict

ROOT = '/home/franciscome/git/research/tarel/codes/tarel-rs/runs_pilot/00000000c001d00d'
WEIGHTS = {"em_gap": 0.25, "gp_magnitude": 0.25, "mode_coverage": 0.25, "curvature": 0.25}

# Per-(cell, component): list of seed-level statistics (one per seed).
# The "statistic" for each component is what we want to compare across cells.
# - em_gap: stdev across seeds of final-100-step rolling mean (lower = more stable)
# - gp_magnitude: median across seeds of GP-time-series mean (lower = better Lipschitz)
# - mode_coverage: mean across seeds of coverage_count (higher = better mode coverage)
# - curvature: stdev across seeds of curvature mean (lower = more consistent landscape)

per_seed = defaultdict(lambda: defaultdict(dict))  # per_seed[cell][seed][metric] = value

for path in sorted(glob.glob(f'{ROOT}/stage_two/**/diagnostics.msgpack', recursive=True)):
    parts = path.split('/'); cell, seed, clock = parts[-4], parts[-3], parts[-2]
    em_roll, gp, mode_cov, curv = [], [], [], []
    with open(path, 'rb') as f:
        while True:
            h = f.read(4)
            if not h: break
            n, = struct.unpack('<I', h)
            c = msgpack.unpackb(f.read(n), raw=False)
            em_roll.extend(s['rolling100'] for s in c['em_gap'])
            gp.extend(s['gp']  for s in c['gp'])
            mode_cov.extend(m['coverage_count'] for m in c['mode_coverage'])
            curv.extend(cv['curvature'] for cv in c['curvature'] if cv['curvature'] == cv['curvature'])  # filter NaN

    if len(em_roll) >= 100:
        per_seed[cell][seed].setdefault('em_finals', []).append(statistics.mean(em_roll[-100:]))
    if gp:
        per_seed[cell][seed].setdefault('gp_means', []).append(statistics.mean(gp))
    if mode_cov:
        per_seed[cell][seed].setdefault('mode_covs', []).append(statistics.mean(mode_cov))
    if curv:
        per_seed[cell][seed].setdefault('curv_means', []).append(statistics.mean(curv))

# Aggregate to one statistic per (cell, component) across the 3 seeds × 3 clocks.
def agg(values, fn):
    return fn(values) if len(values) >= 2 else (values[0] if values else float('nan'))

per_cell = {}
for cell, seeds in per_seed.items():
    em_finals  = [v for s in seeds.values() for v in s.get('em_finals', [])]
    gp_means   = [v for s in seeds.values() for v in s.get('gp_means', [])]
    mode_covs  = [v for s in seeds.values() for v in s.get('mode_covs', [])]
    curv_means = [v for s in seeds.values() for v in s.get('curv_means', [])]
    per_cell[cell] = {
        # "Stability" metrics: invert when "lower is more stable" so higher = better.
        "em_gap":       -statistics.stdev(em_finals)  if len(em_finals)  >= 2 else 0,  # lower stdev = more stable → higher score
        "gp_magnitude": -agg(gp_means, statistics.median),                              # lower GP = better → higher score
        "mode_coverage": agg(mode_covs, statistics.mean),                               # higher coverage = better
        "curvature":    -statistics.stdev(curv_means) if len(curv_means) >= 2 else 0,   # lower curvature stdev = better
    }

# Now rank-normalise each component ACROSS CELLS.
def rank_norm_across(values):
    """Average-rank tie-breaking, return rank/n in [0,1] (higher = bigger value)."""
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and indexed[j][1] == indexed[i][1]:
            j += 1
        avg = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[indexed[k][0]] = avg
        i = j
    return [r / n for r in ranks]

cells = sorted(per_cell.keys())
print(f"\n{'cell':22s}  {'em_gap':>10s}  {'gp_mag':>10s}  {'mode_cov':>10s}  {'curv':>10s}  {'score':>8s}")
print('-' * 80)

components = ["em_gap", "gp_magnitude", "mode_coverage", "curvature"]
component_rns = {}
for comp in components:
    raw = [per_cell[c][comp] for c in cells]
    rns = rank_norm_across(raw)
    for cell, rn in zip(cells, rns):
        component_rns.setdefault(cell, {})[comp] = rn

scores_out = []
for cell in cells:
    rns = component_rns[cell]
    score = sum(WEIGHTS[c] * rns[c] for c in components)
    scores_out.append({
        "cell": cell.replace('_', ':', 1),
        "score": score,
        "component_rank_norms": rns,
        "raw_statistics": per_cell[cell],
        "n_seeds": 3,
        "spec_version": "v0.1.0-rescore",
    })
    print(f"{cell:22s}  {rns['em_gap']:>10.4f}  {rns['gp_magnitude']:>10.4f}  "
          f"{rns['mode_coverage']:>10.4f}  {rns['curvature']:>10.4f}  {score:>8.4f}")

out_path = f'{ROOT}/stage_two/stability_scores_rescored.json'
with open(out_path, 'w') as f:
    json.dump(scores_out, f, indent=2)
print(f"\nwrote {out_path}")
