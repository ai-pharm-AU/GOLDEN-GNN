#!/usr/bin/env python3
"""Run multi-seed link prediction training and summarize results.

This script repeatedly calls `work_alpha_gnn_20260212/scripts/gnn_linkpred_holdout.py`
for a list of training seeds, aggregates mean±std metrics, and selects a
representative seed (closest to median test AUC).

Outputs under `--out_dir`:
  curves/{tag}_seed{S}.csv (+ .summary.json sidecar from gnn_linkpred_holdout.py)
  embeddings/{tag}_seed{S}.npz
  linkpred_multiseed_{tag}.csv                 (one row per seed)
  linkpred_multiseed_{tag}_summary.csv         (mean±std)
  selected_seed_{tag}.txt
  selected_curve_{tag}.csv
  selected_embedding_{tag}.npz
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd), check=False, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_mean(values: list[float]) -> float:
    x = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(x))


def safe_std(values: list[float]) -> float:
    x = np.asarray(values, dtype=np.float64)
    return float(np.nanstd(x, ddof=1)) if np.sum(~np.isnan(x)) >= 2 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed GNN link prediction training.")
    parser.add_argument("--python_bin", default="python")
    parser.add_argument(
        "--gnn_script",
        default="/home/zzz0054/GoldenF/work_alpha_gnn_20260212/scripts/gnn_linkpred_holdout.py",
    )
    parser.add_argument("--cwd", default="/home/zzz0054/GoldenF/work_alpha_gnn_20260212")
    parser.add_argument("--edge_file", required=True)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--weight_column", default="NLOGPMF")
    parser.add_argument(
        "--edge_weight_transform", default="log1p", choices=["none", "log1p", "sqrt"]
    )
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--pos_samples_per_epoch", type=int, default=50000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seeds", default="11,12,13,14,15,16,17,18,19,20")
    parser.add_argument(
        "--split_seed_mode",
        default="same_as_train_seed",
        choices=["same_as_train_seed", "fixed"],
    )
    parser.add_argument("--split_seed", type=int, default=-1)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    curves_dir = out_dir / "curves"
    emb_dir = out_dir / "embeddings"
    curves_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_list(args.seeds)
    if not seeds:
        raise ValueError("No seeds provided")

    cwd = Path(args.cwd).resolve()
    gnn_script = Path(args.gnn_script).resolve()
    if not gnn_script.exists():
        raise FileNotFoundError(f"gnn_script not found: {gnn_script}")

    rows: list[dict[str, object]] = []
    for seed in seeds:
        split_seed = seed if args.split_seed_mode == "same_as_train_seed" else args.split_seed
        if split_seed < 0:
            raise ValueError("split_seed must be set when split_seed_mode=fixed")

        out_csv = curves_dir / f"{args.tag}_seed{seed}.csv"
        out_npz = emb_dir / f"{args.tag}_seed{seed}.npz"

        cmd = [
            args.python_bin,
            str(gnn_script),
            "--edge_file",
            args.edge_file,
            "--feature_npz",
            args.feature_npz,
            "--weight_column",
            args.weight_column,
            "--edge_weight_transform",
            args.edge_weight_transform,
            "--hidden_dim",
            str(args.hidden_dim),
            "--num_layers",
            str(args.num_layers),
            "--dropout",
            str(args.dropout),
            "--lr",
            str(args.lr),
            "--weight_decay",
            str(args.weight_decay),
            "--epochs",
            str(args.epochs),
            "--eval_every",
            str(args.eval_every),
            "--patience",
            str(args.patience),
            "--val_ratio",
            str(args.val_ratio),
            "--test_ratio",
            str(args.test_ratio),
            "--pos_samples_per_epoch",
            str(args.pos_samples_per_epoch),
            "--seed",
            str(seed),
            "--split_seed",
            str(split_seed),
            "--device",
            args.device,
            "--output_csv",
            str(out_csv),
            "--output_npz",
            str(out_npz),
        ]
        run_cmd(cmd, cwd=cwd)

        summary_path = out_csv.with_suffix(".summary.json")
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary json: {summary_path}")
        s = load_json(summary_path)
        best = s.get("best", {})
        final_at_best = s.get("final_at_best", {})
        rows.append(
            {
                "seed": seed,
                "split_seed": split_seed,
                "best_epoch": best.get("epoch", 0),
                "val_auc": final_at_best.get("val_auc", float("nan")),
                "val_ap": final_at_best.get("val_ap", float("nan")),
                "val_loss": final_at_best.get("val_loss", float("nan")),
                "test_auc": final_at_best.get("test_auc", float("nan")),
                "test_ap": final_at_best.get("test_ap", float("nan")),
                "test_loss": final_at_best.get("test_loss", float("nan")),
                "elapsed_sec": s.get("runtime", {}).get("elapsed_sec", float("nan")),
                "n_params_trainable": s.get("model", {}).get("n_params_trainable", float("nan")),
                "metrics_csv": str(out_csv),
                "embedding_npz": str(out_npz),
                "summary_json": str(summary_path),
            }
        )

    df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    for col in [
        "best_epoch",
        "val_auc",
        "val_ap",
        "val_loss",
        "test_auc",
        "test_ap",
        "test_loss",
        "elapsed_sec",
        "n_params_trainable",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    per_seed_csv = out_dir / f"linkpred_multiseed_{args.tag}.csv"
    df.to_csv(per_seed_csv, index=False)

    test_auc_values = [float(x) for x in df["test_auc"].tolist()]
    median_test_auc = float(np.nanmedian(np.asarray(test_auc_values, dtype=np.float64)))
    df["_dist"] = (df["test_auc"] - median_test_auc).abs()
    selected_row = df.sort_values(["_dist", "seed"]).iloc[0]
    selected_seed = int(selected_row["seed"])

    summary = {
        "tag": args.tag,
        "n_seeds": int(len(seeds)),
        "seeds": ",".join(str(s) for s in seeds),
        "split_seed_mode": args.split_seed_mode,
        "median_test_auc": median_test_auc,
        "selected_seed": selected_seed,
        "val_auc_mean": safe_mean(df["val_auc"].astype(float).tolist()),
        "val_auc_std": safe_std(df["val_auc"].astype(float).tolist()),
        "val_ap_mean": safe_mean(df["val_ap"].astype(float).tolist()),
        "val_ap_std": safe_std(df["val_ap"].astype(float).tolist()),
        "test_auc_mean": safe_mean(df["test_auc"].astype(float).tolist()),
        "test_auc_std": safe_std(df["test_auc"].astype(float).tolist()),
        "test_ap_mean": safe_mean(df["test_ap"].astype(float).tolist()),
        "test_ap_std": safe_std(df["test_ap"].astype(float).tolist()),
        "best_epoch_mean": safe_mean(df["best_epoch"].astype(float).tolist()),
        "best_epoch_std": safe_std(df["best_epoch"].astype(float).tolist()),
    }

    summary_csv = out_dir / f"linkpred_multiseed_{args.tag}_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    selected_seed_path = out_dir / f"selected_seed_{args.tag}.txt"
    selected_seed_path.write_text(str(selected_seed) + "\n", encoding="utf-8")

    # Copy representative artifacts.
    selected_curve_src = curves_dir / f"{args.tag}_seed{selected_seed}.csv"
    selected_curve_dst = out_dir / f"selected_curve_{args.tag}.csv"
    shutil.copyfile(selected_curve_src, selected_curve_dst)

    selected_emb_src = emb_dir / f"{args.tag}_seed{selected_seed}.npz"
    selected_emb_dst = out_dir / f"selected_embedding_{args.tag}.npz"
    shutil.copyfile(selected_emb_src, selected_emb_dst)

    print(f"[ok] wrote: {per_seed_csv}", flush=True)
    print(f"[ok] wrote: {summary_csv}", flush=True)
    print(f"[ok] selected_seed={selected_seed} (median_test_auc={median_test_auc:.6g})", flush=True)
    print(f"[ok] wrote: {selected_seed_path}", flush=True)
    print(f"[ok] wrote: {selected_curve_dst}", flush=True)
    print(f"[ok] wrote: {selected_emb_dst}", flush=True)


if __name__ == "__main__":
    main()
