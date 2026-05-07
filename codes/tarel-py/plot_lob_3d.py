"""
plot_lob_3d.py — Render a Limit Order Book parquet file as a 3D isometric
bar plot using plotly.

Expected parquet schema (long format, one row per (snapshot, side, level)):

    timestamp_ns : uint64    Snapshot timestamp in nanoseconds (UTC).
    symbol       : string    Instrument symbol (e.g. "BTC/USDC").
    exchange     : string    Venue identifier (e.g. "binance").
    side         : string    "bid" or "ask".
    level        : uint64    Depth level, 0 = top of book.
    price        : float64   Price at that level.
    size         : float64   Resting size at that level (volume).

Output:
    A figure with x = time, y = signed depth level (bids negative, asks positive,
    leaving y = 0 as the mid-price gap), z = volume. Each (snapshot, level) cell
    is rendered as a 3D bar. Bars are colored by depth level using two
    sequential color ramps (one per side). The camera uses an orthographic
    projection so the geometry reads as an isometric stack of bars, matching
    the reference style.

Usage:
    python plot_lob_3d.py path/to/file.parquet \
        --out-prefix outputs/lob_3d \
        --snapshots 60 \
        --levels 15

The script writes:
    <out-prefix>.html      Interactive Plotly figure (CDN-loaded plotly.js).
    <out-prefix>.png       Static high-resolution raster via kaleido.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Bar geometry: build a single Mesh3d trace from a batch of axis-aligned bars.
# ---------------------------------------------------------------------------

# Unit cuboid: 8 vertices in [0,1]^3.
_CUBE_VERTS = np.array(
    [
        [0, 0, 0],  # 0
        [1, 0, 0],  # 1
        [1, 1, 0],  # 2
        [0, 1, 0],  # 3
        [0, 0, 1],  # 4
        [1, 0, 1],  # 5
        [1, 1, 1],  # 6
        [0, 1, 1],  # 7
    ],
    dtype=np.float64,
)

# 12 triangle faces (two per cube side).
_CUBE_FACES = np.array(
    [
        [0, 1, 2], [0, 2, 3],   # bottom
        [4, 5, 6], [4, 6, 7],   # top [0, 1, 5], [0, 5, 4],   # -y
        [1, 2, 6], [1, 6, 5],   # +x
        [2, 3, 7], [2, 7, 6],   # +y
        [3, 0, 4], [3, 4, 7],   # -x
    ],
    dtype=np.int64,
)

# 12 edges (vertex index pairs) used to draw the contour outline around a bar.
_CUBE_EDGES = np.array(
    [
        [0, 1], [1, 2], [2, 3], [3, 0],   # bottom rectangle
        [4, 5], [5, 6], [6, 7], [7, 4],   # top rectangle
        [0, 4], [1, 5], [2, 6], [3, 7],   # vertical struts
    ],
    dtype=np.int64,
)


def make_bars_mesh(
    centers: np.ndarray,
    heights: np.ndarray,
    dx: float,
    dy: float,
    face_colors: Iterable[str],
    name: str,
    opacity: float = 1.0,
) -> go.Mesh3d | None:
    """Vectorized construction of a Mesh3d trace containing many bars.

    Parameters
    ----------
    centers : (N, 2) array of (x, y) bar centers.
    heights : (N,) array of bar heights (z extents). Bars start at z=0.
    dx, dy  : Half-extents of each bar along x and y.
    face_colors : per-bar color (12 faces of each bar share the same color).
    name    : Legend entry.
    opacity : Mesh3d opacity.
    """
    n = int(heights.shape[0])
    if n == 0:
        return None

    # Vectorized vertex construction:
    # base[k, :, :] is the 8x3 vertex matrix for bar k.
    base = np.broadcast_to(_CUBE_VERTS, (n, 8, 3)).copy()
    base[:, :, 0] = (base[:, :, 0] - 0.5) * (2.0 * dx) + centers[:, 0:1]
    base[:, :, 1] = (base[:, :, 1] - 0.5) * (2.0 * dy) + centers[:, 1:2]
    base[:, :, 2] = base[:, :, 2] * heights[:, None]
    verts = base.reshape(n * 8, 3)

    # Faces: shift each bar's face indices by 8*k.
    offsets = (np.arange(n) * 8).reshape(n, 1, 1)
    faces = (_CUBE_FACES[None, :, :] + offsets).reshape(n * 12, 3)

    # Each bar contributes 12 faces, all the same color.
    face_colors = list(face_colors)
    if len(face_colors) != n:
        raise ValueError(f"face_colors length {len(face_colors)} != bars {n}")
    fcolor = np.repeat(face_colors, 12).tolist()

    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        facecolor=fcolor,
        name=name,
        opacity=opacity,
        flatshading=True,
        hoverinfo="skip",
        showlegend=True,
        lighting=dict(
            ambient=0.55,
            diffuse=0.75,
            specular=0.15,
            roughness=0.5,
            fresnel=0.05,
        ),
        lightposition=dict(x=2000, y=2000, z=4000),
    )


def make_bars_edges(
    centers: np.ndarray,
    heights: np.ndarray,
    dx: float,
    dy: float,
    color: str,
    width: float,
    name: str,
) -> go.Scatter3d | None:
    """Build a single Scatter3d line trace containing the contour edges of all
    bars. Each cuboid contributes 12 line segments separated by NaN breaks so
    a single trace can render every bar's outline.
    """
    n = int(heights.shape[0])
    if n == 0:
        return None

    # Vertex coordinates for every bar (same construction as make_bars_mesh).
    base = np.broadcast_to(_CUBE_VERTS, (n, 8, 3)).copy()
    base[:, :, 0] = (base[:, :, 0] - 0.5) * (2.0 * dx) + centers[:, 0:1]
    base[:, :, 1] = (base[:, :, 1] - 0.5) * (2.0 * dy) + centers[:, 1:2]
    base[:, :, 2] = base[:, :, 2] * heights[:, None]

    # For each bar, lay down the 12 edges as: [v_a, v_b, NaN] triplets.
    # Resulting flat arrays have shape (n * 12 * 3, 3).
    edge_pairs = base[:, _CUBE_EDGES, :]                         # (n, 12, 2, 3)
    nans = np.full((n, 12, 1, 3), np.nan, dtype=np.float64)      # separator
    triplets = np.concatenate([edge_pairs, nans], axis=2)        # (n, 12, 3, 3)
    flat = triplets.reshape(-1, 3)                               # (n*12*3, 3)

    return go.Scatter3d(
        x=flat[:, 0],
        y=flat[:, 1],
        z=flat[:, 2],
        mode="lines",
        line=dict(color=color, width=width),
        name=name,
        hoverinfo="skip",
        showlegend=False,
    )


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    h = hex_str.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _rgba_str(rgb: np.ndarray, alpha: float) -> str:
    r, g, b = (int(round(v)) for v in np.clip(rgb, 0, 255))
    a = max(0.0, min(1.0, float(alpha)))
    return f"rgba({r},{g},{b},{a:.4f})"


def gradient(
    n: int,
    start_hex: str,
    end_hex: str,
    *,
    alpha_start: float = 1.0,
    alpha_end: float = 1.0,
) -> list[str]:
    """Linear interpolation between two colors, returning n rgba(...) strings.

    Both color and alpha are interpolated linearly across the n samples. When
    alpha_start == alpha_end == 1.0 the result is opaque; pass a smaller
    alpha_end (e.g. 0.3) to dim deeper levels.
    """
    if n <= 0:
        return []
    a = np.array(_hex_to_rgb(start_hex), dtype=float)
    b = np.array(_hex_to_rgb(end_hex), dtype=float)
    if n == 1:
        return [_rgba_str(a, alpha_start)]
    ts = np.linspace(0.0, 1.0, n)
    out = []
    for t in ts:
        rgb = a + (b - a) * t
        alpha = alpha_start + (alpha_end - alpha_start) * t
        out.append(_rgba_str(rgb, alpha))
    return out


# ---------------------------------------------------------------------------
# Data loading and sub-sampling
# ---------------------------------------------------------------------------

@dataclass
class PreparedLOB:
    df: pd.DataFrame              # columns: snap_idx, side, level, price, size, mid
    timestamps_ns: np.ndarray     # selected timestamps (uint64)
    symbol: str
    exchange: str
    n_levels: int


def load_lob(parquet_path: Path) -> pd.DataFrame:
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    expected = {"timestamp_ns", "side", "level", "price", "size"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"parquet missing required columns: {missing}")
    return df


def prepare_lob(
    df: pd.DataFrame,
    n_snapshots: int = 60,
    n_levels: int = 15,
) -> PreparedLOB:
    """Subsample to evenly-spaced snapshots, restrict to top n_levels per side,
    and attach mid-price for every (timestamp, side, level) row."""
    ts_all = np.array(sorted(df["timestamp_ns"].unique()), dtype=np.uint64)
    if len(ts_all) > n_snapshots:
        idx = np.linspace(0, len(ts_all) - 1, n_snapshots).round().astype(int)
        ts = ts_all[idx]
    else:
        ts = ts_all

    sub = df[df["timestamp_ns"].isin(ts) & (df["level"] < n_levels)].copy()

    bid0 = (
        sub[(sub["side"] == "bid") & (sub["level"] == 0)]
        .set_index("timestamp_ns")["price"]
        .rename("bid0")
    )
    ask0 = (
        sub[(sub["side"] == "ask") & (sub["level"] == 0)]
        .set_index("timestamp_ns")["price"]
        .rename("ask0")
    )
    mid = ((bid0 + ask0) / 2.0).rename("mid")
    sub = sub.join(mid, on="timestamp_ns")

    snap_map = {int(t): i for i, t in enumerate(ts)}
    sub["snap_idx"] = sub["timestamp_ns"].map(snap_map).astype(int)

    sub = sub[["snap_idx", "side", "level", "price", "size", "mid"]].reset_index(drop=True)

    return PreparedLOB(
        df=sub,
        timestamps_ns=ts,
        symbol=str(df["symbol"].iloc[0]) if "symbol" in df.columns else "",
        exchange=str(df["exchange"].iloc[0]) if "exchange" in df.columns else "",
        n_levels=int(n_levels),
    )


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def build_figure(
    prep: PreparedLOB,
    *,
    bid_start: str = "#0d3b66",
    bid_end: str = "#7fb3d5",
    ask_start: str = "#a83232",
    ask_end: str = "#f5b58a",
    bar_width_x: float = 0.42,
    bar_width_y: float = 0.42,
    z_clip_quantile: float = 0.99,
    title: str | None = None,
    width: int = 1800,
    height: int = 1100,
    opacity: float = 1.0,
    dim_deeper: float = 0.0,
    edge_color: str | None = "rgba(20,20,20,0.55)",
    edge_width: float = 1.2,
) -> go.Figure:
    """
    opacity     : Global trace-level opacity multiplier in [0, 1]. Lower values
                  make every bar uniformly more see-through.
    dim_deeper  : Per-level alpha falloff in [0, 1]. 0 keeps every level fully
                  opaque (modulo `opacity`); 1 fades the deepest level all the
                  way to alpha=0 so only the top of book is fully visible.
    edge_color  : Color of the contour around each bar. Pass None (or empty
                  string) to disable edges entirely.
    edge_width  : Stroke width of the bar contours, in pixels.
    bid_start/bid_end, ask_start/ask_end :
                  Hex color stops for the bid (negative-y) and ask (positive-y)
                  per-level color ramps. Level 0 is `*_start`, deepest plotted
                  level is `*_end`.
    """
    df = prep.df
    if df.empty:
        raise ValueError("prepared LOB frame is empty.")

    # Map (side, level) -> y position. y = 0 is the mid; bids occupy negative y,
    # asks occupy positive y. Best bid sits at y = -1, best ask at y = +1.
    # Cast level to signed first — `level` may arrive as uint64, which would
    # silently underflow when negated.
    levels_i = df["level"].values.astype(np.int64)
    y = np.where(df["side"].values == "bid",
                 -(levels_i + 1),
                 +(levels_i + 1)).astype(float)
    df = df.assign(y=y)

    # Volume cap so a few outsized bars don't crush the rest of the surface.
    z_cap = float(np.quantile(df["size"].values, z_clip_quantile))
    z_cap = max(z_cap, float(df["size"].median()) * 8.0)  # avoid degenerate cap
    sizes = np.minimum(df["size"].values, z_cap)

    # Color ramps: deepest level color at level 0 (best), faded at deepest level.
    # When dim_deeper > 0, the per-level alpha decays linearly from 1 (top of
    # book) to (1 - dim_deeper) at the deepest plotted level.
    dim_deeper = float(np.clip(dim_deeper, 0.0, 1.0))
    alpha_end = max(0.0, 1.0 - dim_deeper)
    bid_palette = gradient(prep.n_levels, bid_start, bid_end,
                           alpha_start=1.0, alpha_end=alpha_end)
    ask_palette = gradient(prep.n_levels, ask_start, ask_end,
                           alpha_start=1.0, alpha_end=alpha_end)
    side_palette = {"bid": bid_palette, "ask": ask_palette}
    colors = [side_palette[s][int(l)] for s, l in zip(df["side"].values, df["level"].values)]

    # Centers + heights.
    centers = np.column_stack([df["snap_idx"].values, df["y"].values])
    bid_mask = (df["side"].values == "bid")
    ask_mask = ~bid_mask

    opacity = float(np.clip(opacity, 0.0, 1.0))
    bid_trace = make_bars_mesh(
        centers=centers[bid_mask],
        heights=sizes[bid_mask],
        dx=bar_width_x,
        dy=bar_width_y,
        face_colors=[c for c, m in zip(colors, bid_mask) if m],
        name="Bid depth",
        opacity=opacity,
    )
    ask_trace = make_bars_mesh(
        centers=centers[ask_mask],
        heights=sizes[ask_mask],
        dx=bar_width_x,
        dy=bar_width_y,
        face_colors=[c for c, m in zip(colors, ask_mask) if m],
        name="Ask depth",
        opacity=opacity,
    )

    traces = [t for t in (bid_trace, ask_trace) if t is not None]

    # Bar contours (optional). Drawn on top of the meshes so they read as
    # outlines even when bars are translucent.
    if edge_color:
        bid_edges = make_bars_edges(
            centers=centers[bid_mask],
            heights=sizes[bid_mask],
            dx=bar_width_x,
            dy=bar_width_y,
            color=edge_color,
            width=edge_width,
            name="Bid edges",
        )
        ask_edges = make_bars_edges(
            centers=centers[ask_mask],
            heights=sizes[ask_mask],
            dx=bar_width_x,
            dy=bar_width_y,
            color=edge_color,
            width=edge_width,
            name="Ask edges",
        )
        traces.extend([t for t in (bid_edges, ask_edges) if t is not None])

    # Mid-price reference line (z=0 plane) drawn as a thin Scatter3d.
    mid_line = go.Scatter3d(
        x=np.arange(len(prep.timestamps_ns)),
        y=np.zeros(len(prep.timestamps_ns)),
        z=np.zeros(len(prep.timestamps_ns)),
        mode="lines",
        line=dict(color="rgba(40,40,40,0.55)", width=4),
        name="Mid",
        hoverinfo="skip",
        showlegend=True,
    )
    traces.append(mid_line)

    # X-axis tick labels: snapshot index -> wall-clock time.
    times = pd.to_datetime(prep.timestamps_ns, unit="ns", utc=True)
    n = len(prep.timestamps_ns)
    n_ticks = min(8, n)
    tick_idx = np.linspace(0, n - 1, n_ticks).round().astype(int)
    tick_text = [times[i].strftime("%H:%M:%S") for i in tick_idx]

    if title is None:
        sym = prep.symbol or "?"
        ex = prep.exchange or "?"
        t0 = times[0].strftime("%Y-%m-%d %H:%M:%S")
        t1 = times[-1].strftime("%H:%M:%S UTC")
        title = (
            f"Limit Order Book — {sym} @ {ex} · "
            f"{n} snapshots · {prep.n_levels} levels/side · "
            f"{t0} → {t1}"
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center",
                   font=dict(size=18, family="Inter, Helvetica, Arial, sans-serif")),
        scene=dict(
            xaxis=dict(
                title="Time (UTC)",
                tickmode="array",
                tickvals=tick_idx,
                ticktext=tick_text,
                backgroundcolor="rgba(247,248,251,1)",
                gridcolor="rgba(0,0,0,0.10)",
                zerolinecolor="rgba(0,0,0,0.15)",
                showspikes=False,
            ),
            yaxis=dict(
                title="Depth level (bid ← 0 → ask)",
                backgroundcolor="rgba(247,248,251,1)",
                gridcolor="rgba(0,0,0,0.10)",
                zerolinecolor="rgba(0,0,0,0.30)",
                showspikes=False,
            ),
            zaxis=dict(
                title="Volume (capped at q{:.0f}%)".format(z_clip_quantile * 100),
                backgroundcolor="rgba(247,248,251,1)",
                gridcolor="rgba(0,0,0,0.10)",
                zerolinecolor="rgba(0,0,0,0.15)",
                showspikes=False,
            ),
            aspectmode="manual",
            aspectratio=dict(x=2.4, y=1.4, z=0.95),
            camera=dict(
                eye=dict(x=2.0, y=-2.0, z=1.25),
                up=dict(x=0, y=0, z=1),
                projection=dict(type="orthographic"),  # true isometric feel
                center=dict(x=0, y=0, z=-0.05),
            ),
        ),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=70, b=0),
        paper_bgcolor="white",
        width=width,
        height=height,
    )
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Render a 3D Limit Order Book plot from parquet.")
    ap.add_argument("parquet", type=Path, help="Path to LOB parquet file.")
    ap.add_argument("-o", "--out-prefix", type=Path, default=Path("lob_3d"),
                    help="Output prefix (without extension). Writes <prefix>.html and <prefix>.png.")
    ap.add_argument("--snapshots", type=int, default=60,
                    help="Number of evenly-spaced snapshots to plot (default: 60).")
    ap.add_argument("--levels", type=int, default=15,
                    help="Top N levels per side to plot (default: 15).")
    ap.add_argument("--width", type=int, default=1800, help="Image width in px.")
    ap.add_argument("--height", type=int, default=1100, help="Image height in px.")
    ap.add_argument("--scale", type=float, default=2.0,
                    help="kaleido scale factor (>1 = sharper raster).")
    ap.add_argument("--no-png", action="store_true", help="Skip PNG export (HTML only).")
    ap.add_argument("--opacity", type=float, default=1.0,
                    help="Global bar opacity in [0, 1] (default: 1.0 = fully opaque). "
                         "Lower values make every bar uniformly more see-through.")
    ap.add_argument("--dim-deeper", type=float, default=0.0,
                    help="Per-level alpha falloff in [0, 1] (default: 0). "
                         "0 = every level fully opaque; 1 = deepest level invisible. "
                         "Useful when foreground bars obscure the back of the surface.")
    # Face color ramps.
    ap.add_argument("--bid-start", default="#0d3b66",
                    help="Hex color for the best bid (level 0). Default: #0d3b66.")
    ap.add_argument("--bid-end", default="#7fb3d5",
                    help="Hex color for the deepest plotted bid level. Default: #7fb3d5.")
    ap.add_argument("--ask-start", default="#a83232",
                    help="Hex color for the best ask (level 0). Default: #a83232.")
    ap.add_argument("--ask-end", default="#f5b58a",
                    help="Hex color for the deepest plotted ask level. Default: #f5b58a.")
    # Bar contour outlines.
    ap.add_argument("--edge-color", default="rgba(20,20,20,0.55)",
                    help="Color of the contour drawn around each bar. CSS color "
                         "string (hex or rgba(...)). Default: rgba(20,20,20,0.55).")
    ap.add_argument("--edge-width", type=float, default=1.2,
                    help="Stroke width of the bar contours in px. Default: 1.2.")
    ap.add_argument("--no-edges", action="store_true",
                    help="Disable the contour outlines entirely.")
    args = ap.parse_args()

    df_raw = load_lob(args.parquet)
    prep = prepare_lob(df_raw, n_snapshots=args.snapshots, n_levels=args.levels)
    fig = build_figure(
        prep,
        width=args.width,
        height=args.height,
        opacity=args.opacity,
        dim_deeper=args.dim_deeper,
        bid_start=args.bid_start,
        bid_end=args.bid_end,
        ask_start=args.ask_start,
        ask_end=args.ask_end,
        edge_color=None if args.no_edges else args.edge_color,
        edge_width=args.edge_width,
    )

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    html_out = args.out_prefix.with_suffix(".html")
    fig.write_html(html_out, include_plotlyjs="cdn", full_html=True)
    print(f"wrote {html_out}")

    if not args.no_png:
        png_out = args.out_prefix.with_suffix(".png")
        fig.write_image(png_out, width=args.width, height=args.height, scale=args.scale)
        print(f"wrote {png_out}")


if __name__ == "__main__":
    main()
