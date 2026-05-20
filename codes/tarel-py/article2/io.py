"""I/O for Stage-1 run artifacts.

Layout consumed:
    <run_dir>/stage_one/buckets/<cell_us>/cell_metadata.json
    <run_dir>/stage_one/buckets/<cell_us>/<clock>/inventory.json
    <run_dir>/stage_one/recurrence/<cell_us>/<clock>/buckets.jsonl
    <run_dir>/stage_one/recurrence/<cell_us>/<clock>/sanity_grid.png
    <run_dir>/stage_one/recurrence/<cell_us>/<clock>/normalized/bucket_NNNNNN.msgpack

The packed recurrence-plot bits in `rp_headline.bytes` / `rp_appendix.bytes`
encode the strict upper triangle of an n x n binary matrix (no diagonal),
n*(n-1)/2 = n_bits, little-endian bit order within each byte.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

# Coinbase is excluded from article-2: the stage_one RR-bisection on this
# run's coinbase distance plots collapsed to RR_achieved = 0 for 100% of
# coinbase/physical and coinbase/info_time admitted buckets, and 99.92% of
# coinbase/event, producing degenerate (all-zero off-diagonal) binary
# recurrence plots that cannot be RQA-aggregated meaningfully. Binance and
# kraken decoded normally with achieved RR in [0.04, 0.06]. The defect is
# tracked separately as a tarel-rs stage_one issue and is not a property
# of the architecture under study.
CELLS = ("binance_BTC-USDC", "kraken_BTC-USDC")
CLOCKS = ("physical", "event", "info_time")

# Default run id used when neither --run-dir nor explicit path is supplied.
DEFAULT_RUN_ID = "00000000c002daaaX"


@dataclass(frozen=True)
class CellSpec:
    cell: str          # e.g. "coinbase:BTC-USDC"
    cell_us: str       # e.g. "coinbase_BTC-USDC" (filesystem-safe form)
    venue: str         # e.g. "coinbase"
    symbol: str        # e.g. "BTC-USDC"


def parse_cell(cell_us: str) -> CellSpec:
    venue, symbol = cell_us.split("_", 1)
    return CellSpec(
        cell=f"{venue}:{symbol}",
        cell_us=cell_us,
        venue=venue,
        symbol=symbol,
    )


def run_root(run_dir: str | os.PathLike) -> Path:
    p = Path(run_dir)
    if not p.exists():
        raise FileNotFoundError(f"run_dir does not exist: {p}")
    return p


def load_cell_metadata(run_dir: str | os.PathLike, cell_us: str) -> dict:
    p = run_root(run_dir) / "stage_one" / "buckets" / cell_us / "cell_metadata.json"
    with open(p, "r") as f:
        return json.load(f)


def load_inventory(run_dir: str | os.PathLike, cell_us: str, clock: str) -> dict:
    p = run_root(run_dir) / "stage_one" / "buckets" / cell_us / clock / "inventory.json"
    with open(p, "r") as f:
        return json.load(f)


_DP_KEY_B = b'"dp_uppertri":['


def _strip_dp_uppertri_bytes(line: bytes) -> bytes:
    """Excise the dp_uppertri array from a single JSONL line (byte-level).

    dp_uppertri scales O(n^2) per bucket and grows to ~1 GB per line for
    info_time clocks with thousands of snapshots. The array contains only
    numeric literals (no nested brackets or strings), so the first `]`
    after the opening `[` is the matching close. Operating on `bytes`
    avoids the expensive UTF-8 -> Python str decode of the full line.
    """
    start = line.find(_DP_KEY_B)
    if start < 0:
        return line
    open_idx = start + len(_DP_KEY_B)  # offset *after* the opening '['
    close_idx = line.find(b"]", open_idx)
    if close_idx < 0:
        raise ValueError("malformed dp_uppertri: opening [ has no matching ]")
    # Splice [open_idx:close_idx] out; preserve the closing ']' so the
    # resulting JSON has an empty array literal "[]" in place of the data.
    return line[:open_idx] + line[close_idx:]


def iter_bucket_records(
    run_dir: str | os.PathLike,
    cell_us: str,
    clock: str,
    skip_dp_uppertri: bool = True,
    max_records: int | None = None,
) -> Iterator[dict]:
    """Yield dict-per-bucket records from buckets.jsonl.

    By default, the heavy `dp_uppertri` field is excised on the fly so
    that the parser does not allocate ~1 GB Python lists for info_time
    buckets. RQA consumes only the packed `rp_headline.bytes`, so this
    is lossless for the article-2 pipeline. The file is read in binary
    mode and `json.loads` is given a `bytes` object directly, which
    avoids the full-line UTF-8 decode.

    `max_records`, if set, stops the iteration after that many lines have
    been yielded. This is used to cap wallclock on the binance/info_time
    cell whose 34 GB jsonl is otherwise prohibitive in a sandboxed run.
    """
    p = (
        run_root(run_dir)
        / "stage_one"
        / "recurrence"
        / cell_us
        / clock
        / "buckets.jsonl"
    )
    if not p.exists():
        return
    yielded = 0
    with open(p, "rb") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if skip_dp_uppertri:
                line = _strip_dp_uppertri_bytes(line)
            yield json.loads(line)
            yielded += 1
            if max_records is not None and yielded >= max_records:
                return


def unpack_rp(rp_bytes: list[int] | bytes, n: int) -> np.ndarray:
    """Reconstruct symmetric n x n binary RP from packed upper-triangle bytes.

    `rp_bytes` encodes the strict upper triangle (no diagonal), n_bits =
    n*(n-1)/2, little-endian bit order. The diagonal (LOI) is forced to 1
    in the returned matrix.

    Implementation note: for large n (info_time buckets can have n in the
    10^4 range) we avoid `np.triu_indices(n)` because it allocates two
    int64 arrays of size n(n-1)/2 each (e.g. 768 MB at n=9808). Instead
    we walk the rows of the upper triangle and copy contiguous slices of
    the bit array directly into each row.
    """
    n_bits = n * (n - 1) // 2
    expected_bytes = (n_bits + 7) // 8
    if isinstance(rp_bytes, (list, tuple)):
        buf = bytes(rp_bytes)
    else:
        buf = bytes(rp_bytes)
    if len(buf) != expected_bytes:
        raise ValueError(
            f"packed RP bytes length {len(buf)} != expected {expected_bytes} "
            f"for n={n} (n_bits={n_bits})"
        )
    bits = np.unpackbits(
        np.frombuffer(buf, dtype=np.uint8),
        bitorder="little",
    )[:n_bits].astype(np.uint8)
    rp = np.zeros((n, n), dtype=np.uint8)
    # Row-by-row scatter into the strict upper triangle (no giant index arrays).
    offset = 0
    for i in range(n - 1):
        row_len = n - 1 - i
        rp[i, i + 1 :] = bits[offset : offset + row_len]
        offset += row_len
    # Mirror the upper triangle to the lower triangle (R symmetric); the
    # in-place max keeps the temporary footprint bounded at one n x n copy.
    rp_t = rp.T.copy()
    np.maximum(rp, rp_t, out=rp)
    np.fill_diagonal(rp, 1)
    return rp


def unpack_dp_uppertri(dp_uppertri: list[float], n: int) -> np.ndarray:
    """Reconstruct symmetric n x n distance plot from upper-tri (with diagonal=0).

    The on-disk `dp_uppertri` for a record with size n is length n*(n-1)/2
    (strict upper triangle, no diagonal) when emitted by the current writer,
    OR n*(n+1)/2 (upper triangle including diagonal) on older emitters. We
    detect which by length and reconstruct accordingly.
    """
    arr = np.asarray(dp_uppertri, dtype=np.float64)
    strict = n * (n - 1) // 2
    full = n * (n + 1) // 2
    dp = np.zeros((n, n), dtype=np.float64)
    if arr.size == strict:
        iu = np.triu_indices(n, k=1)
        dp[iu] = arr
    elif arr.size == full:
        iu = np.triu_indices(n, k=0)
        dp[iu] = arr
    else:
        # Heuristic fallback: if the JSONL writer concatenated the full n x n
        # row-major upper-tri-including-diag with repeats (we observed an
        # n*(n-1) length once), try to reshape; otherwise raise.
        if arr.size == n * n:
            return arr.reshape(n, n)
        raise ValueError(
            f"dp_uppertri length {arr.size} not in {{n(n-1)/2={strict}, "
            f"n(n+1)/2={full}, n^2={n*n}}} for n={n}"
        )
    dp = dp + dp.T
    np.fill_diagonal(dp, 0.0)
    return dp


def load_normalized_rp(
    run_dir: str | os.PathLike,
    cell_us: str,
    clock: str,
    bucket_idx: int,
) -> np.ndarray | None:
    """Load a 64x64 normalized recurrence-plot tensor from msgpack.

    Returns None if the bucket has no normalized output (sub-floor buckets).
    Best-effort: tries common msgpack shapes; returns None on parse failure.
    """
    try:
        import msgpack  # type: ignore
    except ImportError:
        return None
    p = (
        run_root(run_dir)
        / "stage_one"
        / "recurrence"
        / cell_us
        / clock
        / "normalized"
        / f"bucket_{bucket_idx:06d}.msgpack"
    )
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            obj = msgpack.unpackb(f.read(), raw=False)
    except Exception:
        return None
    # Try common shapes: {"tensor": [...flat 64*64...], "shape": [1,64,64]}
    # or top-level list. Fall back to None if we can't parse.
    if isinstance(obj, dict):
        for key in ("tensor", "data", "values", "rp"):
            if key in obj:
                arr = np.asarray(obj[key], dtype=np.float64)
                if arr.size == 64 * 64:
                    return arr.reshape(64, 64)
                if arr.size == 1 * 64 * 64:
                    return arr.reshape(64, 64)
        # tensor + shape pairing
        if "shape" in obj and "data" in obj:
            arr = np.asarray(obj["data"], dtype=np.float64)
            shape = tuple(obj["shape"])
            try:
                return arr.reshape(shape).squeeze()
            except ValueError:
                return None
    if isinstance(obj, list):
        arr = np.asarray(obj, dtype=np.float64)
        if arr.size == 64 * 64:
            return arr.reshape(64, 64)
    return None
