# tarel-stats

Standalone descriptive report over the raw parquet datasets used by the
TAReL pipeline. 

It walks `datasets/raw/<exchange>/<orderbooks|trades>/*.parquet` and prints, for each
dataset kind:

- total number of parquet files;
- unique exchanges and markets observed (cross-checked against the expected
  `{binance, coinbase, kraken}` and `{BTC-USDC, SOL-USDC, SHIB-USDC, TON-USDC}`);
- number of `exchange × market` cells covered (expected `12`);
- per-cell file count, total registries, average registries/file, and elapsed
  unix time from the first registry of the first file to the last registry of
  the last file (filenames are lexicographically sortable on the embedded
  UTC timestamp);
- per-market roll-up with the average elapsed time across the 3 exchanges.

## Usage

from the tarel/ folder

```bash
cargo run --release --manifest-path codes/tarel-stats/Cargo.toml -- datasets/raw
```
or, from this crate

```bash
cd codes/tarel-stats
cargo run --release -- ../../datasets/raw
```
