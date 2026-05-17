# Tarel Code Guide

There are three components for this project: 

- tarel-stats: Standalone Rust crate to generate statistics about the data files.
- tarel-py: For inference and interaction with the HuggingFace platform.
- tare-rs: Heavy lifting for Training.

# From cloned repo to trained model

Done in three stages: 

- Stage 0: Compile binary
- Stage 1: Generate Recurrence Plots
- Stage 2: Train Model
- Stage 3: Performance Metrics
- Stage 4: Provision .pt model file

## Stage 0: Compile Binary
---

All is done within the root folder `/tarel`, then prepare the environment with dependencies and 
variables necessary for a successfull compilation of the binary.

```bash
cd /Users/franciscome/git/research/tarel
```

1. Create virtual environment (10 secs)

Do this one-time: create the venv with python 3.12 + torch 2.5.1.

```bash
./scripts/ultradian_setup_venv.sh
```

2. Prepare environment (10 secs)

Do this at every shell session: activate virtual environment + source env vars.

```bash
pyenv shell ultradian_tarel_venv
source scripts/ultradian_tarel_env.sh
```

3. Compile Binary (1-5 mins)

Change folder to the one where the binary is, and execute the compilation with a `release`
profile, which will make it lighter in space and faster in excution.

```bash
cd codes/tarel-rs
cargo build --release
```

If this executed successfully, the binary will be named `tarel` and should be located 
at `target/release/tarel-rs`

4. Already Compiled Binaries

- x86_gpu_pascal (linux-old-nvidia-gpu-only): 
- arm_cpu_mps (mac-cpu-only): 

## Stage-One: Generate Recurrence Plots
---

It executes the code in `tarel-rs/src/stage_one.rs`, for the case of processing
orderbooks of the BTC-USDC symbol in the BInance exchange, taking the data from
the file named `binance_BTC-USDT_ob_sync_YYYYMMDD.parquet`, store results in `/runs` 
folder and will produce the respective Distance-Plots based Recurrence-Plots.
In the same file there are more details about this process.

Run the following command (with the tarel binary compiled). Produce artifacts

```bash
CELL=coinbase:BTC-USDC
INPUT_PARQUET=./datasets/consolidated/coinbase/BTC-USDC.parquet
RUNS=./artifacts/runs
EXP=0xC001DFFF
SEED=4

cargo run --release \
    --manifest-path .codes/tarel-rs/Cargo.toml \
    -- run-stage-one \
    --cell $CELL \
    --input $INPUT_PARQUET \
    --runs $RUNS \
    --experiment-seed $EXP \
    --seed $SEED \
    --produce-dp-rp
```

Full generation of the metrics.

```bash
CELL=kraken:BTC-USDC
INPUT_PARQUET=./datasets/consolidated/kraken/BTC-USDC.parquet
RUNS=./artifacts/runs
EXP=0xC001DAAA
SEED=4

./codes/tarel-rs/target/release/tarel run-stage-one \
    --cell "$CELL" \
    --input "$INPUT_PARQUET" \
    --runs "$RUNS" \
    --experiment-seed "$EXP" \
    --seed "$SEED" \
    --full
```

## Stage-Two: Model Training
---

Trainin with the data .msgpack already generated from stage-one. 

For a machine with Nvidia GPU: 

```bash
EXP=0xC001DAAA
CELL=coinbase:BTC-USDC
SEED=3
RUN_MINS=25
RUNS=./artifacts/runs
CLOCKS=event
CUDA_DEVICE=0

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE ./codes/tarel-rs/target/release/tarel run-stage-two \
    --cells "$CELL" \
    --seeds "$SEED" \
    --budget-mins "$RUN_MINS" \
    --runs "$RUNS" \
    --experiment-seed "$EXP" \
    --clocks "$CLOCKS"
```

Apple silicon (CPU Only)

```bash
EXP=0xC001DAAA
CELL=coinbase:BTC-USDC
SEED=3
RUN_MINS=25
RUNS=./artifacts/runs
CLOCKS=event

./codes/tarel-rs/target/release/tarel run-stage-two \
    --cells "$CELL" \
    --seeds "$SEED" \
    --budget-mins "$RUN_MINS" \
    --runs "$RUNS" \
    --experiment-seed "$EXP" \
    --clocks "$CLOCKS"
```

## Stage-Three: Metrics
---

### Compute of metrics.

```bash
python3 trace.py
```

### Generate model weights file

```bash
cargo run --release -- generate \
    --model generator_traced.pt \
    --n 16 \
    --output samples/
```

## Complementary

Consolidate .parquet files. Build the new binary.

```bash
cd /Users/franciscome/git/research/takens-rs
cargo build --release -p takens_rs --bin takens_consolidate
```

Run on one exchange to spot-check.

```bash
./target/release/takens_consolidate \
    --input-dir  datasets/raw/kraken/orderbooks \
    --output     datasets/consolidated/kraken/BTC-USDC.parquet
```

Expect: one JSON line on stdout, exit 0, file appears at the output path,
no .tmp left behind. kraken is the smallest cell (~53 MB, 5M rows) so this
is the fastest validation.

Now the  Full sweep

```bash
./scripts/consolidate_all.sh
```

