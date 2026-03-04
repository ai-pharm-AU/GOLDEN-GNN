#!/usr/bin/env python3
"""Plot mean±std link prediction training curves across seeds.

Expects per-seed curve CSVs produced by `run_linkpred_multiseed.py`:
  curves/{tag}_seed{S}.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot mean±std linkpred curves across seeds.")
    parser.add_argument("--curves_dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out_png", required=True)
    args = parser.parse_args()

    curves_dir = Path(args.curves_dir).resolve()
    files = sorted(curves_dir.glob(f"{args.tag}_seed*.csv"))
    if not files:
        raise FileNotFoundError(f"No curve CSVs found for tag={args.tag} under {curves_dir}")

    frames = []
    for p in files:
        df = pd.read_csv(p)
        df["seed_file"] = p.name
        frames.append(df)
    all_df = pd.concat(frames, axis=0, ignore_index=True)

    metric_cols = [c for c in ["train_loss", "val_loss", "val_auc", "val_ap", "test_auc", "test_ap"] if c in all_df.columns]
    if "epoch" not in all_df.columns or not metric_cols:
        raise ValueError("Curve CSVs must include 'epoch' and at least one metric column")

    agg = all_df.groupby("epoch", as_index=False)[metric_cols].agg(["mean", "std"])
    agg.columns = ["_".join([c for c in col if c]) for col in agg.columns.to_flat_index()]
    agg = agg.rename(columns={"epoch_": "epoch"})

    epochs = agg["epoch"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.2))
    axes = axes.ravel()

    def plot_with_band(ax: plt.Axes, y_col: str, title: str) -> None:
        mean = agg[f"{y_col}_mean"].to_numpy(dtype=float)
        std = agg.get(f"{y_col}_std", pd.Series([np.nan] * len(agg))).to_numpy(dtype=float)
        ax.plot(epochs, mean, color="#2C3E50", linewidth=2.0)
        if np.any(np.isfinite(std)):
            ax.fill_between(epochs, mean - std, mean + std, color="#2C3E50", alpha=0.18, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)

    # Losses
    if "train_loss" in metric_cols:
        plot_with_band(axes[0], "train_loss", "Train loss (mean±std)")
    if "val_loss" in metric_cols:
        plot_with_band(axes[1], "val_loss", "Val loss (mean±std)")
    # Metrics
    if "val_auc" in metric_cols:
        plot_with_band(axes[2], "val_auc", "Val AUC (mean±std)")
    if "val_ap" in metric_cols:
        plot_with_band(axes[3], "val_ap", "Val AP (mean±std)")

    fig.suptitle(f"Link prediction curves: {args.tag} (n={len(files)} seeds)")
    fig.tight_layout()
    out_png = Path(args.out_png).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close(fig)

    print(f"[ok] wrote: {out_png}", flush=True)


if __name__ == "__main__":
    main()

