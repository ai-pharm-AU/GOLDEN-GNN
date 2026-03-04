#!/usr/bin/env python3
"""Summarize feature ablation outputs into leaderboard/report."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_"
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if math.isnan(v):
                    vals.append("")
                else:
                    vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def safe_mean(series: pd.Series) -> float:
    if series.empty:
        return math.nan
    return float(series.mean())


def load_eval_metrics(eval_csv: Path, k: int) -> dict[str, float]:
    if not eval_csv.exists():
        return {}
    df = pd.read_csv(eval_csv)
    if df.empty:
        return {}
    d = df[df["k"] == k]
    if d.empty:
        return {}
    return {
        "silhouette_cosine_k": safe_mean(d["silhouette_cosine"]),
        "silhouette_euclidean_k": safe_mean(d["silhouette_euclidean"]),
        "davies_bouldin_k": safe_mean(d["davies_bouldin"]),
        "calinski_harabasz_k": safe_mean(d["calinski_harabasz"]),
    }


def load_linkpred_metrics(linkpred_csv: Path) -> dict[str, float]:
    if not linkpred_csv.exists():
        return {}
    df = pd.read_csv(linkpred_csv)
    if df.empty or "val_auc" not in df.columns:
        return {}
    out: dict[str, float] = {
        "val_auc_max": float(df["val_auc"].max()),
        "val_ap_max": float(df["val_ap"].max()) if "val_ap" in df.columns else math.nan,
        "train_loss_last": float(df["train_loss"].iloc[-1]) if "train_loss" in df.columns else math.nan,
    }

    best_idx = int(df["val_auc"].astype(float).idxmax())
    best = df.loc[best_idx]
    if "epoch" in df.columns:
        out["best_epoch_val_auc"] = float(best["epoch"])
    if "val_ap" in df.columns:
        out["val_ap_at_val_auc_max"] = float(best["val_ap"])
    if "train_loss" in df.columns:
        out["train_loss_at_val_auc_max"] = float(best["train_loss"])
    if "val_loss" in df.columns:
        out["val_loss_at_val_auc_max"] = float(best["val_loss"])
    if "test_auc" in df.columns:
        out["test_auc_at_val_auc_max"] = float(best["test_auc"])
    if "test_ap" in df.columns:
        out["test_ap_at_val_auc_max"] = float(best["test_ap"])
    return out


def load_linkpred_multiseed_summary(summary_csv: Path) -> dict[str, float]:
    if not summary_csv.exists():
        return {}
    df = pd.read_csv(summary_csv)
    if df.empty:
        return {}
    row = df.iloc[0]
    out: dict[str, float] = {}
    for k in [
        "n_seeds",
        "val_auc_mean",
        "val_auc_std",
        "val_ap_mean",
        "val_ap_std",
        "test_auc_mean",
        "test_auc_std",
        "test_ap_mean",
        "test_ap_std",
        "best_epoch_mean",
        "best_epoch_std",
        "selected_seed",
        "median_test_auc",
    ]:
        if k in df.columns:
            try:
                out[k if k != "n_seeds" else "n_train_seeds"] = float(row[k])
            except Exception:
                out[k if k != "n_seeds" else "n_train_seeds"] = math.nan
    return out


def load_validation_metrics(validation_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    text_csv = validation_dir / "text_coherence_by_seed.csv"
    ari_csv = validation_dir / "pairwise_ari.csv"
    enrich_csv = validation_dir / "meta_enrichment_seed_summary.csv"

    if text_csv.exists():
        d = pd.read_csv(text_csv)
        if not d.empty:
            if "text_coherence_clean" in d.columns:
                out["text_coherence_clean_mean"] = safe_mean(d["text_coherence_clean"])
            if "text_null_clean_p" in d.columns:
                out["text_null_clean_p_mean"] = safe_mean(d["text_null_clean_p"])

    if ari_csv.exists():
        d = pd.read_csv(ari_csv)
        if not d.empty and "ari" in d.columns:
            out["pairwise_ari_mean"] = safe_mean(d["ari"])
            out["pairwise_ari_std"] = float(d["ari"].std())

    if enrich_csv.exists():
        d = pd.read_csv(enrich_csv)
        if not d.empty and "attr_type" in d.columns:
            for attr in ["gse_id", "mesh"]:
                sub = d[d["attr_type"] == attr]
                if not sub.empty:
                    out[f"{attr}_significant_mean"] = safe_mean(sub["significant"])
                    if "null_sig_p" in sub.columns:
                        out[f"{attr}_null_sig_p_mean"] = safe_mean(sub["null_sig_p"])
                    if "max_log10q" in sub.columns:
                        out[f"{attr}_max_log10q_mean"] = safe_mean(sub["max_log10q"])
    return out


def load_confound_metrics(confound_md: Path) -> dict[str, float]:
    if not confound_md.exists():
        return {}
    text = confound_md.read_text(encoding="utf-8")

    pat_gse = re.search(r"primary GSE:.*?NMI=([0-9.]+|nan) \| ARI=([0-9.]+|nan)", text)
    pat_plat = re.search(
        r"primary platform:.*?NMI=([0-9.]+|nan) \| ARI=([0-9.]+|nan)", text
    )

    out: dict[str, float] = {}
    if pat_gse:
        out["primary_gse_nmi"] = float(pat_gse.group(1))
        out["primary_gse_ari"] = float(pat_gse.group(2))
    if pat_plat:
        out["primary_platform_nmi"] = float(pat_plat.group(1))
        out["primary_platform_ari"] = float(pat_plat.group(2))
    return out


def eligible_recipe(row: pd.Series, baseline: pd.Series) -> bool:
    keys = [
        "text_coherence_clean_mean",
        "mesh_significant_mean",
        "gse_id_significant_mean",
        "primary_gse_nmi",
        "primary_platform_nmi",
    ]
    if any(pd.isna(row.get(k, np.nan)) for k in keys):
        return False
    if any(pd.isna(baseline.get(k, np.nan)) for k in keys):
        return False

    if row["text_coherence_clean_mean"] < baseline["text_coherence_clean_mean"] * 0.98:
        return False
    if row["mesh_significant_mean"] < baseline["mesh_significant_mean"] * 0.95:
        return False
    if row["gse_id_significant_mean"] < baseline["gse_id_significant_mean"] * 0.95:
        return False
    if row["primary_gse_nmi"] > baseline["primary_gse_nmi"] * 1.05:
        return False
    if row["primary_platform_nmi"] > baseline["primary_platform_nmi"] * 1.05:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize feature ablation outputs.")
    parser.add_argument("--subset", required=True, choices=["085", "090"])
    parser.add_argument(
        "--task_dir",
        default="/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation",
    )
    parser.add_argument("--recipes", required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    task_dir = Path(args.task_dir).resolve()
    recipes = parse_csv_list(args.recipes)

    csv_dir = task_dir / "csv" / f"subset{args.subset}"
    report_dir = task_dir / "reports" / f"subset{args.subset}"
    if not csv_dir.exists():
        # Support alternate task roots, e.g. human_only/subset{SUBSET}/...
        csv_dir = task_dir / "csv"
        report_dir = task_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for recipe in recipes:
        row: dict[str, object] = {"recipe": recipe, "subset": args.subset}

        eval_csv = csv_dir / f"{recipe}_eval_subset{args.subset}.csv"
        linkpred_csv = csv_dir / f"{recipe}_linkpred_subset{args.subset}.csv"
        multiseed_summary_csv = csv_dir / f"linkpred_multiseed_{recipe}_subset{args.subset}_summary.csv"
        val_dir = report_dir / f"{recipe}_validation_subset{args.subset}"
        confound_md = report_dir / f"{recipe}_confound_subset{args.subset}.md"

        row.update(load_eval_metrics(eval_csv, args.k))
        row.update(load_linkpred_metrics(linkpred_csv))
        row.update(load_linkpred_multiseed_summary(multiseed_summary_csv))
        row.update(load_validation_metrics(val_dir))
        row.update(load_confound_metrics(confound_md))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No rows collected for summary")

    df = df.sort_values("recipe").reset_index(drop=True)
    if "text_only" in set(df["recipe"]):
        baseline = df[df["recipe"] == "text_only"].iloc[0]
    else:
        baseline = df.iloc[0]

    df["eligible"] = df.apply(lambda r: eligible_recipe(r, baseline), axis=1)

    # ranking preference: eligible first, then silhouette, then coherence, then ARI.
    for col in ["silhouette_cosine_k", "text_coherence_clean_mean", "pairwise_ari_mean"]:
        if col not in df.columns:
            df[col] = np.nan

    df = df.sort_values(
        by=["eligible", "silhouette_cosine_k", "text_coherence_clean_mean", "pairwise_ari_mean"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    recommended = df.iloc[0]["recipe"] if not df.empty else ""

    out_csv = report_dir / f"leaderboard_feature_ablation_subset{args.subset}.csv"
    out_md = report_dir / f"feature_ablation_report_subset{args.subset}.md"
    df.to_csv(out_csv, index=False)

    lines: list[str] = []
    lines.append(f"# Feature Ablation Summary (subset {args.subset})")
    lines.append("")
    lines.append(f"- recipes: {', '.join(recipes)}")
    lines.append(f"- k: {args.k}")
    lines.append(f"- recommended_recipe: `{recommended}`")
    lines.append("")
    lines.append("## Decision rule")
    lines.append("")
    lines.append("- Keep recipes with acceptable coherence/enrichment/confound versus `text_only` baseline.")
    lines.append("- Among eligible, rank by higher silhouette_cosine@k then coherence then ARI.")
    lines.append("")
    lines.append("## Leaderboard (top 8)")
    lines.append("")

    show_cols = [
        "recipe",
        "eligible",
        "silhouette_cosine_k",
        "text_coherence_clean_mean",
        "pairwise_ari_mean",
        "mesh_significant_mean",
        "gse_id_significant_mean",
        "primary_gse_nmi",
        "primary_platform_nmi",
        "test_auc_mean",
        "test_auc_std",
        "test_auc_at_val_auc_max",
        "val_auc_max",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    preview = df[show_cols].head(8)
    lines.append(markdown_table(preview))
    lines.append("")
    lines.append(f"- full_csv: `{out_csv}`")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
