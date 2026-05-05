
#!/usr/bin/env python3
"""Generate training plots from logan-rs metrics CSV.

Usage:
    python3 plot_metrics.py metrics.csv              # saves PNGs to docs/
    python3 plot_metrics.py metrics.csv --out plots/  # custom output dir

Produces:
    - loss_curves.png       Critic & generator loss over epochs
    - wasserstein_dist.png  Wasserstein distance estimate over epochs
    - gradient_penalty.png  Gradient penalty magnitude over epochs
    - critic_scores.png     Critic scores on real vs fake over epochs
    - epoch_time.png        Wall-clock time per epoch
"""

import argparse
import csv
import os
import sys

def read_csv(path: str) -> dict[str, list[float]]:
    with open(path) as f:
        reader = csv.DictReader(f)
        cols: dict[str, list[float]] = {k: [] for k in reader.fieldnames}
        for row in reader:
            for k, v in row.items():
                cols[k].append(float(v))
    return cols

def plot_all(data: dict[str, list[float]], out_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    epochs = data["epoch"]

    # --- Loss curves ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, data["critic_loss"], label="Critic loss", alpha=0.85)
    ax.plot(epochs, data["generator_loss"], label="Generator loss", alpha=0.85)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("WGAN-GP Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)

    # --- Wasserstein distance ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, data["wasserstein_dist"], color="tab:green", alpha=0.85)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("W distance")
    ax.set_title("Wasserstein Distance Estimate (E[C(real)] − E[C(fake)])")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "wasserstein_dist.png"), dpi=150)
    plt.close(fig)

    # --- Gradient penalty ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, data["gradient_penalty"], color="tab:orange", alpha=0.85)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("GP (before λ scaling)")
    ax.set_title("Gradient Penalty")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gradient_penalty.png"), dpi=150)
    plt.close(fig)

    # --- Critic scores (real vs fake) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, data["critic_real"], label="C(real)", alpha=0.85)
    ax.plot(epochs, data["critic_fake"], label="C(fake)", alpha=0.85)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Critic score")
    ax.set_title("Critic Scores: Real vs Fake")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "critic_scores.png"), dpi=150)
    plt.close(fig)

    # --- Epoch time ---
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(epochs, data["epoch_secs"], color="tab:purple", alpha=0.7, width=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds")
    ax.set_title("Wall-Clock Time per Epoch")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "epoch_time.png"), dpi=150)
    plt.close(fig)

    print(f"plots saved to {out_dir}/")

def main():
    parser = argparse.ArgumentParser(description="Plot logan-rs training metrics")
    parser.add_argument("csv", help="Path to metrics CSV file")
    parser.add_argument("--out", default="docs", help="Output directory for PNG plots (default: docs)")
    args = parser.parse_args()

    data = read_csv(args.csv)
    plot_all(data, args.out)

if __name__ == "__main__":
    main()
