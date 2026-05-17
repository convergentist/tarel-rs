import json
ROOT='/home/franciscome/git/research/tarel/codes/tarel-rs/runs_pilot/00000000c001d00d'
stab = {s['cell']: s['score'] for s in json.load(open(f'{ROOT}/stage_two/stability_scores.json'))}
stat = {s.get('cell_idx', i): s['score'] for i, s in enumerate(json.load(open(f'{ROOT}/stage_one/stationarity_scores.json')))}
print(f'{'cell':25s}  {'stationarity':>13s}  {'stability':>10s}')
for cell in sorted(set(stab.keys()) | set(stat.keys())):
    print(f'{str(cell):25s}  {stat.get(cell, float(\"nan\")):>13.4f}  {stab.get(cell, float(\"nan\")):>10.4f}')
