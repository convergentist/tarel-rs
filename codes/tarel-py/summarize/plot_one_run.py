import msgpack, struct, sys
import matplotlib.pyplot as plt

path = sys.argv[1]   # path to diagnostics.msgpack

em_step, em_raw, em_roll = [], [], []
gp_step, gp_val = [], []
with open(path, 'rb') as f:
    while True:
        h = f.read(4)
        if not h: break
        n, = struct.unpack('<I', h)
        c = msgpack.unpackb(f.read(n), raw=False)
        for s in c['em_gap']:
            em_step.append(s['step']); em_raw.append(s['raw']); em_roll.append(s['rolling100'])
        for s in c['gp']:
            gp_step.append(s['step']); gp_val.append(s['gp'])

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
axes[0].plot(em_step, em_raw, alpha=0.3, label='raw')
axes[0].plot(em_step, em_roll, lw=2, label='rolling-100')
axes[0].set_ylabel('EM-gap')
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(gp_step, gp_val, alpha=0.5)
axes[1].set_yscale('symlog', linthresh=0.01)
axes[1].set_ylabel('GP magnitude (symlog)')
axes[1].set_xlabel('critic step')
axes[1].grid(True, alpha=0.3)
fig.suptitle(path.split('/')[-4:-1])
fig.tight_layout()
fig.savefig('/home/franciscome/git/research/tarel/codes/tarel-rs/summarize/diagnostics_plot.png', dpi=120)
print('saved diagnostics_plot.png')
