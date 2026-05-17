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

Run the following command (with the tarel binary compiled).

```bash
EXP=0xC001DAAA
RUNS=./runs/stage_one

tarel run-stage-one \
  --cell binance:BTC-USDC \
  --input  /path/to/BTC-USDC.parquet \
  --runs   $RUNS \
  --experiment-seed $EXP \
  --seed 0 \
  --produce-dp-rp full
```

It executes the code in `tarel-rs/src/stage_one.rs`, for the case of processing
orderbooks of the BTC-USDC symbol in the BInance exchange, taking the data from
the file named `binance_BTC-USDT_ob_sync_YYYYMMDD.parquet`, store results in `/runs` 
folder and will produce the respective Distance-Plots based Recurrence-Plots.
In the same file there are more details about this process.

### Stage-Two: Model Training

cd /Users/franciscome/git/research/logan-rs
cargo run --release -- train \
  --data    ./runs/<exp>/datasets/dp_tensors \
  --out-dir ./checkpoints/<exp> \
  --seed 0 --budget-mins 20

### Stage-Three

while in `codes/tarel-rs` ... 

```bash
cargo run run-stage-one --cell binance:BTC-USDC \
    --input ../../datasets/raw/binance/orderbooks/binance_BTC-USDC_ob_sync_20260507_041711.731.parquet
```

### Train

```bash
cargo run --release -- train \
    --data datasets/rp/train \
    --output generator.pt \
    --epochs 200 \
    --batch-size 10 \
    --latent-dim 128 \
    --img-size 64
```

### Trace + generate

```bash
python3 trace.py
cargo run --release -- generate \
    --model generator_traced.pt \
    --n 16 \
    --output samples/
```


## Consolidate .parquet files

# (1) Build the new binary.
cd /Users/franciscome/git/research/takens-rs
cargo build --release -p takens_rs --bin takens_consolidate

# (2) Run on one exchange to spot-check.
./target/release/takens_consolidate \
    --input-dir  datasets/raw/kraken/orderbooks \
    --output     datasets/consolidated/kraken/BTC-USDC.parquet

# Expect: one JSON line on stdout, exit 0, file appears at the output path,
# no .tmp left behind. kraken is the smallest cell (~53 MB, 5M rows) so this
# is the fastest validation.

# (3) Full sweep.
./scripts/consolidate_all.sh


