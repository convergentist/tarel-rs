import msgpack, struct, glob, statistics
ROOT='/home/franciscome/git/research/tarel/codes/tarel-rs/runs_pilot/00000000c001d00d'
agg = {}
for path in glob.glob(f'{ROOT}/stage_two/*/*/*/diagnostics.msgpack'):
    parts = path.split('/')
    cell, seed, clock = parts[-4], parts[-3], parts[-2]
    em = []
    with open(path,'rb') as f:
        while True:
            h=f.read(4)
            if not h: break
            n,=struct.unpack('<I',h)
            c=msgpack.unpackb(f.read(n),raw=False)
            em.extend(s['rolling100'] for s in c['em_gap'])
    if len(em) >= 100:
        final = statistics.mean(em[-100:])
        agg.setdefault((cell, clock), []).append(final)

print(f'{'cell':25s} {'clock':10s}  {'final_em_gap_mean':>18s}  {'seeds':>6s}  {'stdev':>8s}')
for (cell, clock), vals in sorted(agg.items()):
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0
    print(f'{cell:25s} {clock:10s}  {m:>18.5f}  {len(vals):>6d}  {s:>8.5f}')
