# summarize.py — at the tarel-rs root
import msgpack, struct, glob, statistics

ROOT = '/home/franciscome/git/research/tarel/codes/tarel-rs/runs_pilot/00000000c001d00d'

print(f"{'cell':22s} {'seed':22s} {'clock':9s}  {'em_n':>6s}  {'em_mean':>10s}  {'em_std':>8s}  {'gp_n':>6s}  {'gp_mean':>8s}  {'mode':>4s}  {'curv':>4s}")
print('-' * 120)

# Recursive glob — much simpler and robust to layout changes
for path in sorted(glob.glob(f'{ROOT}/stage_two/**/diagnostics.msgpack', recursive=True)):
    parts = path.split('/')
    # Layout: .../stage_two/<cell>/<seed_padded>/<clock>/diagnostics.msgpack
    cell, seed, clock = parts[-4], parts[-3], parts[-2]
    em, gp, mc, cv = [], [], 0, 0
    with open(path, 'rb') as f:
        while True:
            h = f.read(4)
            if not h: break
            n, = struct.unpack('<I', h)
            chunk = msgpack.unpackb(f.read(n), raw=False)
            em.extend(s['raw'] for s in chunk['em_gap'])
            gp.extend(s['gp']  for s in chunk['gp'])
            mc += len(chunk['mode_coverage'])
            cv += len(chunk['curvature'])
    em_mean = statistics.mean(em) if em else float('nan')
    em_std  = statistics.stdev(em) if len(em) > 1 else float('nan')
    gp_mean = statistics.mean(gp) if gp else float('nan')
    print(f"{cell:22s} {seed:22s} {clock:9s}  {len(em):>6d}  {em_mean:>10.4f}  {em_std:>8.4f}  {len(gp):>6d}  {gp_mean:>8.4f}  {mc:>4d}  {cv:>4d}")
