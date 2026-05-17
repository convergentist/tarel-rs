
import msgpack, json, sys, os

p = os.path.expanduser(
    "~/git/research/tarel/runs/00000000c001d00d/"
    f"stage_two/kraken_BTC-USDC/{os.environ.get('SEED','00000000000000000000')}/"
    f"{os.environ.get('CLK','physical')}/diagnostics.msgpack")

d = msgpack.unpack(open(p,'rb'), raw=False)

out = {}

for k, v in d.items():
    if isinstance(v, list) and len(v) > 6:
        out[k] = {"len": len(v), "first": v[:3], "last": v[-3:]}
    else:
        out[k] = v

print(json.dumps(out, indent=2, default=str))

