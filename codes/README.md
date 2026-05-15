
# Stage-one

while in `codes/tarel-rs` ... 

```bash
cargo run run-stage-one --cell binance:BTC-USDC \
    --input ../../datasets/raw/binance/orderbooks/binance_BTC-USDC_ob_sync_20260507_041711.731.parquet
```


# tarel-rs (rust)

The `template-rs`

# tarel-py (python)

The `template-py`

```bash
export LIBTORCH=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
export DYLD_LIBRARY_PATH=$LIBTORCH/lib:$DYLD_LIBRARY_PATH  # macOS
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

