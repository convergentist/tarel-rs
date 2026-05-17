# Artifacts 

# Inventory

```bash
000...c001daaa/
manifest.json                  # run-level record of every (cell, seed) try

logs/
    orchestrator.jsonl         # structured event stream

stage_one/
    buckets/$CELL              # Descriptive stats for cell:clock combination
        event/inventory.json   
        info_time
        physical
    recurrence/$CELL
        event
        info_time
        physical

stage_two/
  StabilityScoreSpec.json      # spec actually in effect for this build
  stability_scores.json        # per-cell headline score (rank-normalised)
  <cell>/<seed_hex>/<clock>/
    diagnostics.msgpack        # EM-gap, GP mag, mode-cov, curvature traces
    checkpoints/               # periodic generator/critic snapshots
    failure.json               # only present if training failed
```

## Successful run
---

```bash
EXP=00000000c001d00d
RUN=~/git/research/tarel/artifacts/runs/$EXP
CELL=kraken_BTC-USDC
```

how many diagnostics files exist (expect S*K)

```bash
find $RUN/stage_two/$CELL -name diagnostics.msgpack | wc -l
```
how many failed (expect 0 for clean run)

```bash
find $RUN/stage_two/$CELL -name failure.json | wc -l
```

per-clock breakdown

```bash
for clk in physical event info_time; do
    n=$(find $RUN/stage_two/$CELL -path "*/$clk/diagnostics.msgpack" | wc -l)
    echo "$clk: $n"
done
```

If a (seed, clock) combination is missing both diagnostics.msgpack
and failure.json, the run was interrupted mid-budget — that seed
contributes nothing to the headline score.

## Stats
---

### JSON Orchestration

```bash
LOG=$RUN/logs/orchestrator.jsonl
```

completion banner

```bash
grep tarel_stage_two_done $LOG
```

any failures

```bash
grep -E 'train_failed|stage_two_dataset_missing' $LOG
```

per-clock load lines (sanity: should appear once per clock per process)

```bash
grep load_dp_tensors_complete $LOG
```

warn lines only

```bash
grep '"level":"WARN"' $LOG
```


