"""Stage-2 W-GAN-GP diagnostics ingestion.

Reads `diagnostics.msgpack` produced by tarel stage_two and exposes a
trajectory record per (cell, clock, seed). The file format is a sequence
of length-prefixed (little-endian u32) MessagePack chunks; each chunk is
a dict with keys:

    em_gap       : list of {step, raw, rolling100} samples
    gp           : list of {step, gp} samples
    mode_coverage: list of checkpoint records (may be empty for short runs)
    curvature    : list of checkpoint records (may be empty for short runs)

For this run the recorded training horizon is one checkpoint (step 1000),
so `mode_coverage` and `curvature` are universally empty. The 2-axis
partial Stability Score uses only `em_gap` and `gp_magnitude`.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    import msgpack  # type: ignore
    _HAVE_MSGPACK = True
except ImportError:
    _HAVE_MSGPACK = False


SEED0 = "00000000000000000000"


@dataclass
class StageTwoMeta:
    step: int                  # last logged generator-step at checkpoint
    wallclock_secs: float
    generator_steps: int
    critic_steps: int


@dataclass
class StageTwoTrajectory:
    cell_us: str
    clock: str
    seed_dir: str
    n_chunks: int
    em_step: np.ndarray
    em_raw: np.ndarray
    em_roll: np.ndarray
    gp_step: np.ndarray
    gp_val: np.ndarray
    mode_coverage_count: int
    curvature_count: int
    meta: StageTwoMeta | None = None  # populated when diagnostics_meta.json exists

    @property
    def em_final_mean(self) -> float:
        if self.em_raw.size < 100:
            return float(self.em_raw.mean()) if self.em_raw.size > 0 else float("nan")
        return float(self.em_raw[-100:].mean())

    @property
    def em_final_std(self) -> float:
        if self.em_raw.size < 100:
            return float(self.em_raw.std(ddof=1)) if self.em_raw.size > 1 else float("nan")
        return float(self.em_raw[-100:].std(ddof=1))

    @property
    def gp_mean(self) -> float:
        return float(self.gp_val.mean()) if self.gp_val.size > 0 else float("nan")

    @property
    def gp_max(self) -> float:
        return float(self.gp_val.max()) if self.gp_val.size > 0 else float("nan")


def stage_two_root(run_dir: str | Path) -> Path:
    return Path(run_dir) / "stage_two"


def stage_two_paths(run_dir: str | Path) -> dict[tuple[str, str], Path]:
    """Map (cell_us, clock) -> diagnostics.msgpack path for every present file.

    Returns only entries where the diagnostics file exists on disk; missing
    cells / clocks / seeds are simply absent from the result. We restrict
    to seed_dir = "00000000000000000000" (seed_idx 0), which is the only
    seed actually trained in this run.
    """
    from .io import CELLS  # local import to honour the article-2 cell filter

    root = stage_two_root(run_dir)
    out: dict[tuple[str, str], Path] = {}
    if not root.exists():
        return out
    for cell_dir in sorted(root.iterdir()):
        if not cell_dir.is_dir() or not cell_dir.name.endswith("BTC-USDC"):
            continue
        if cell_dir.name not in CELLS:
            # Skip cells excluded from article-2 (e.g., coinbase under
            # the stage_one defect documented in io.py).
            continue
        cell_us = cell_dir.name
        seed_dir = cell_dir / SEED0
        if not seed_dir.exists():
            continue
        for clock_dir in sorted(seed_dir.iterdir()):
            if not clock_dir.is_dir():
                continue
            diag = clock_dir / "diagnostics.msgpack"
            if diag.exists():
                out[(cell_us, clock_dir.name)] = diag
    return out


def _iter_chunks(path: Path) -> Iterator[dict]:
    """Yield each length-prefixed MessagePack chunk in the diagnostics file."""
    if not _HAVE_MSGPACK:
        raise RuntimeError(
            "msgpack is required to read stage_two diagnostics. "
            "Install with: pip install msgpack"
        )
    with open(path, "rb") as f:
        while True:
            hdr = f.read(4)
            if len(hdr) < 4:
                return
            (n,) = struct.unpack("<I", hdr)
            body = f.read(n)
            if len(body) < n:
                return
            yield msgpack.unpackb(body, raw=False)


def _meta_path_for(diag_path: Path) -> Path:
    """Locate diagnostics_meta.json for a diagnostics.msgpack path.

    Layout: .../<cell>/<seed>/<clock>/diagnostics.msgpack
    Meta:   .../<cell>/<seed>/<clock>/checkpoints/step_*/diagnostics_meta.json
    """
    ck_dir = diag_path.parent / "checkpoints"
    if not ck_dir.exists():
        return diag_path.parent / "checkpoints" / "step_0001000" / "diagnostics_meta.json"
    steps = sorted([d for d in ck_dir.iterdir() if d.is_dir()])
    if not steps:
        return diag_path.parent / "checkpoints" / "step_0001000" / "diagnostics_meta.json"
    return steps[-1] / "diagnostics_meta.json"


def _load_meta(path: Path) -> StageTwoMeta | None:
    if not path.exists():
        return None
    try:
        import json
        d = json.loads(path.read_text())
        return StageTwoMeta(
            step=int(d.get("step", 0)),
            wallclock_secs=float(d.get("wallclock_secs", float("nan"))),
            generator_steps=int(d.get("generator_steps", 0)),
            critic_steps=int(d.get("critic_steps", 0)),
        )
    except Exception:
        return None


def load_trajectory(cell_us: str, clock: str, path: Path) -> StageTwoTrajectory:
    em_step: list[int] = []
    em_raw: list[float] = []
    em_roll: list[float] = []
    gp_step: list[int] = []
    gp_val: list[float] = []
    mc_count = 0
    cv_count = 0
    n_chunks = 0
    for chunk in _iter_chunks(path):
        n_chunks += 1
        for s in chunk.get("em_gap", []) or []:
            em_step.append(int(s.get("step", -1)))
            em_raw.append(float(s.get("raw", float("nan"))))
            em_roll.append(float(s.get("rolling100", float("nan"))))
        for s in chunk.get("gp", []) or []:
            gp_step.append(int(s.get("step", -1)))
            gp_val.append(float(s.get("gp", float("nan"))))
        mc_count += len(chunk.get("mode_coverage", []) or [])
        cv_count += len(chunk.get("curvature", []) or [])
    meta = _load_meta(_meta_path_for(path))
    return StageTwoTrajectory(
        cell_us=cell_us,
        clock=clock,
        seed_dir=SEED0,
        n_chunks=n_chunks,
        em_step=np.asarray(em_step, dtype=np.int64),
        em_raw=np.asarray(em_raw, dtype=np.float64),
        em_roll=np.asarray(em_roll, dtype=np.float64),
        gp_step=np.asarray(gp_step, dtype=np.int64),
        gp_val=np.asarray(gp_val, dtype=np.float64),
        mode_coverage_count=mc_count,
        curvature_count=cv_count,
        meta=meta,
    )


def load_all_trajectories(
    run_dir: str | Path,
) -> dict[tuple[str, str], StageTwoTrajectory]:
    out: dict[tuple[str, str], StageTwoTrajectory] = {}
    for (cell_us, clock), p in stage_two_paths(run_dir).items():
        out[(cell_us, clock)] = load_trajectory(cell_us, clock, p)
    return out


def partial_stability_score(
    trajectories: dict[tuple[str, str], StageTwoTrajectory],
) -> dict[tuple[str, str], float]:
    """2-axis partial Stability Score.

    Article-1's locked 4-axis score weights {em_gap, gp_magnitude,
    mode_coverage, curvature} at 0.25 each, with rank-normalisation
    across the seed population within each cell. With mode_coverage and
    curvature unavailable in this run (training stopped at step 1000),
    we report the 2-axis projection {em_gap, gp_magnitude} weighted
    0.5 each, **rank-normalised across (cell, clock) trajectories**
    (single-seed: J=1) rather than across seeds. This is honest about
    what the present run supports; the Score is comparable across the
    six trained (cell, clock) pairs but is not the locked-spec score.

    Lower em_gap_mean (closer to zero from above) is "better"; lower
    gp_magnitude_mean (closer to 1 from above) is "better". We use the
    raw magnitudes here for transparency and let the Section 4 narrative
    interpret directionality.

    Returns the partial score on [0, 1] for each (cell, clock); 1.0 is
    the best of the trained set, 0.0 is the worst.
    """
    if not trajectories:
        return {}
    keys = list(trajectories.keys())
    em_means = np.asarray([trajectories[k].em_final_mean for k in keys], dtype=np.float64)
    gp_means = np.asarray([trajectories[k].gp_mean for k in keys], dtype=np.float64)

    # Lower-is-better rank on |em_mean| (distance from 0)
    em_rank = _lower_better_rank(np.abs(em_means))
    # Lower-is-better rank on |gp_mean - 0| (we report raw GP; for a partial
    # score we treat smaller |gp| as better; deviation from 1 would be the
    # Lipschitz-violation framing, but with this short training horizon GP
    # has not yet stabilised to 1, so we report raw magnitude.)
    gp_rank = _lower_better_rank(np.abs(gp_means))

    score = 0.5 * em_rank + 0.5 * gp_rank
    return {keys[i]: float(score[i]) for i in range(len(keys))}


def _lower_better_rank(x: np.ndarray) -> np.ndarray:
    """Return rank in [0, 1] where smaller x gets a higher rank value."""
    if x.size == 0:
        return x
    finite = np.isfinite(x)
    out = np.zeros_like(x)
    idx = np.argsort(x[finite])  # ascending
    ranks = np.empty(idx.size, dtype=np.float64)
    if idx.size > 1:
        ranks[idx] = np.linspace(1.0, 0.0, idx.size)  # smallest -> 1.0
    else:
        ranks[idx] = np.array([0.5])
    out_idx = np.flatnonzero(finite)
    out[out_idx] = ranks
    out[~finite] = float("nan")
    return out
