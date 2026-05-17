import json
m = json.load(open('$RUN/manifest.json'))
for r in m.get('cell_runs', []):
    print(r['cell'], r['seed'], r['status'], r.get('wallclock_ms'))
