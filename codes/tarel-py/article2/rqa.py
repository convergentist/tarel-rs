"""Recurrence Quantification Analysis (Marwan convention).

References: Marwan et al. (2007) "Recurrence plots for the analysis of
complex systems", Phys. Rep. 438, 237-329.

Scalars computed per binary symmetric RP R in {0,1}^(n x n) with R[i,i]=1:

    RR    : recurrence rate, sum(R - I) / (n*(n-1))                 (LOI excluded)
    DET   : determinism, fraction of off-diagonal recurrence points
            lying on diagonals of length >= l_min
    LAM   : laminarity, same for vertical lines of length >= v_min
    L_max : longest diagonal line length excluding the LOI
    TT    : trapping time, mean of vertical-line-lengths >= v_min
    ENTR  : Shannon entropy (natural log) of the diagonal-line-length
            histogram, restricted to lines of length >= l_min, computed
            on the LOI-excluded set of diagonal lines

l_min = v_min = 2 (Marwan default).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

L_MIN = 2
V_MIN = 2


@dataclass(frozen=True)
class RqaScalars:
    n: int
    RR: float
    DET: float
    LAM: float
    L_max: int
    TT: float
    ENTR: float


def _runs_of_ones(arr: np.ndarray) -> np.ndarray:
    """Return run lengths of consecutive 1s in a 1D 0/1 array (vectorised)."""
    if arr.size == 0:
        return np.zeros(0, dtype=np.int64)
    # Pad with 0 on both ends so every run is bounded by a 0 -> 1 (start) and
    # 1 -> 0 (end) transition. starts[i] is the index in `padded` of the i-th
    # transition into a run; ends[i] is the index of the matching transition
    # out. The length of the i-th run is ends[i] - starts[i].
    padded = np.concatenate(([0], arr.astype(np.int8), [0]))
    diffs = np.diff(padded)
    starts = np.flatnonzero(diffs == 1)
    ends = np.flatnonzero(diffs == -1)
    return (ends - starts).astype(np.int64)


def _diag_line_lengths_excluding_loi(R: np.ndarray) -> np.ndarray:
    """Return all diagonal recurrence-line lengths, LOI excluded (vectorised)."""
    n = R.shape[0]
    chunks: list[np.ndarray] = []
    # Off-diagonal diagonals only: k != 0. Both upper (+k) and lower (-k) are
    # counted, matching the Marwan-toolbox convention on a symmetric RP.
    for k in range(1, n):
        chunks.append(_runs_of_ones(np.diagonal(R, offset=k)))
        chunks.append(_runs_of_ones(np.diagonal(R, offset=-k)))
    if not chunks:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(chunks)


def _vert_line_lengths(R: np.ndarray) -> np.ndarray:
    """Vertical line lengths, LOI element of each column zeroed (vectorised).

    Builds an n x n int8 working copy of R with R[j,j] := 0 for every j, then
    extracts run-of-ones lengths column by column.
    """
    n = R.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    work = R.astype(np.int8, copy=True)
    np.fill_diagonal(work, 0)
    chunks: list[np.ndarray] = []
    for j in range(n):
        chunks.append(_runs_of_ones(work[:, j]))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)


def rqa_from_rp(R: np.ndarray) -> RqaScalars:
    """Compute the six RQA scalars from a symmetric binary RP."""
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"R must be square, got shape {R.shape}")
    n = R.shape[0]
    if n < 2:
        return RqaScalars(n=n, RR=float("nan"), DET=float("nan"),
                          LAM=float("nan"), L_max=0, TT=float("nan"),
                          ENTR=float("nan"))

    # --- RR (LOI excluded) -------------------------------------------------
    # Total off-diagonal cells = n*(n-1); recurrence count excludes diagonal.
    R_bin = (R > 0).astype(np.int64)
    off_diag_recurrences = int(R_bin.sum() - np.trace(R_bin))
    denom = n * (n - 1)
    RR = off_diag_recurrences / denom if denom > 0 else float("nan")

    # --- Diagonal-line statistics (LOI excluded) ---------------------------
    diag_lengths = _diag_line_lengths_excluding_loi(R_bin)
    if diag_lengths.size == 0:
        DET = 0.0
        L_max = 0
        ENTR = float("nan")
    else:
        L_max = int(diag_lengths.max())
        kept = diag_lengths[diag_lengths >= L_MIN]
        # DET numerator: sum of l for diagonal lines with l >= l_min
        # DET denominator: total recurrence points in off-diagonal mask, i.e.
        # sum of all diagonal-line lengths
        det_num = int(kept.sum())
        det_den = int(diag_lengths.sum())
        DET = (det_num / det_den) if det_den > 0 else float("nan")
        # ENTR over the histogram of kept (l >= l_min)
        if kept.size > 0:
            counts = np.bincount(kept)
            counts = counts[counts > 0]
            p = counts / counts.sum()
            ENTR = float(-(p * np.log(p)).sum())
        else:
            ENTR = float("nan")

    # --- Vertical-line statistics ------------------------------------------
    vert_lengths = _vert_line_lengths(R_bin)
    if vert_lengths.size == 0:
        LAM = 0.0
        TT = float("nan")
    else:
        kept_v = vert_lengths[vert_lengths >= V_MIN]
        lam_num = int(kept_v.sum())
        lam_den = int(vert_lengths.sum())
        LAM = (lam_num / lam_den) if lam_den > 0 else float("nan")
        TT = float(kept_v.mean()) if kept_v.size > 0 else float("nan")

    return RqaScalars(
        n=n, RR=float(RR), DET=float(DET), LAM=float(LAM),
        L_max=L_max, TT=float(TT), ENTR=float(ENTR),
    )
