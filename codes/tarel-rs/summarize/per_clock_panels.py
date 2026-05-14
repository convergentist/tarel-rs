# per_clock_panels.py — at tarel-rs root
import msgpack, struct, sys, glob, os
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("usage: python3 per_clock_panels.py <cell_dir>")
    print("example: python3 per_clock_panels.py runs_pilot/00000000c001d00d/stage_two/binance_SOL-USDC")
    sys.exit(1)

cell_dir = sys.argv[1].rstrip('/')
print(f"cell_dir: {cell_dir}")
print(f"exists?  {os.path.isdir(cell_dir)}")
if not os.path.isdir(cell_dir):
    print("ABORT: cell_dir is not a directory")
    sys.exit(2)

# Diagnostic: what's directly under cell_dir?
top_entries = sorted(os.listdir(cell_dir))
print(f"contents ({len(top_entries)} entries):")
for e in top_entries[:10]:
    full = os.path.join(cell_dir, e)
    kind = 'dir' if os.path.isdir(full) else 'file'
    print(f"  [{kind}] {e}")

clocks = ['physical', 'event', 'info_time']
# Use os.listdir + isdir filter — more robust than the trailing-slash glob
seed_dirs = sorted(
    os.path.join(cell_dir, e) for e in top_entries
    if os.path.isdir(os.path.join(cell_dir, e))
)[:3]
print(f"\nseed_dirs ({len(seed_dirs)}):")
for d in seed_dirs:
    print(f"  {d}")
    for clock in clocks:
        path = os.path.join(d, clock, 'diagnostics.msgpack')
        if os.path.exists(path):
            sz = os.path.getsize(path)
            print(f"    {clock}: {sz} bytes")
        else:
            print(f"    {clock}: MISSING")

if not seed_dirs:
    print("\nABORT: no seed dirs under cell_dir — nothing to plot")
    sys.exit(3)

fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex='col')

for col, clock in enumerate(clocks):
    for row, seed_dir in enumerate(seed_dirs):
        path = os.path.join(seed_dir, clock, 'diagnostics.msgpack')
        em_step, em_roll = [], []
        n_chunks = 0
        try:
            with open(path, 'rb') as f:
                while True:
                    h = f.read(4)
                    if not h: break
                    n, = struct.unpack('<I', h)
                    c = msgpack.unpackb(f.read(n), raw=False)
                    n_chunks += 1
                    em_step.extend(s['step'] for s in c['em_gap'])
                    em_roll.extend(s['rolling100'] for s in c['em_gap'])
        except FileNotFoundError:
            axes[row, col].text(0.5, 0.5, 'missing', transform=axes[row, col].transAxes, ha='center')
            continue

        if not em_step:
            axes[row, col].text(0.5, 0.5, f'no em_gap data\n({n_chunks} chunks)',
                               transform=axes[row, col].transAxes, ha='center')
        else:
            axes[row, col].plot(em_step, em_roll, lw=1.2)
        axes[row, col].set_title(f'{clock} seed={row}')
        axes[row, col].grid(True, alpha=0.3)
        print(f"  {clock} row={row}: {len(em_step)} em_gap samples across {n_chunks} chunks")

fig.suptitle(f'EM-gap rolling-100 across (clock × seed) for {os.path.basename(cell_dir)}')
fig.supylabel('EM-gap')
fig.supxlabel('critic step')
fig.tight_layout()

out_path = '/home/franciscome/git/research/tarel/codes/tarel-rs/summarize/per_clock_panels.png'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, dpi=120)
print(f"\nsaved {out_path}")
