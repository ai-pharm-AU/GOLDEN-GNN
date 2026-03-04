#!/usr/bin/env python3
"""Generate plots for GNN feature-engineering comparison.

Outputs:
- UMAP side-by-side (text_only vs text_meta_structlite) for subset090/085
- Key metric bar plots from metrics_reference_summary*.csv
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_cmd(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )


def resolve_python_with_umap(project_root: Path) -> str:
    candidates = [
        project_root / ".venv/bin/python",
        Path(sys.executable),
    ]
    seen: set[str] = set()
    for py in candidates:
        py_s = str(py)
        if py_s in seen:
            continue
        seen.add(py_s)
        if not Path(py_s).exists():
            continue
        probe = subprocess.run(
            [py_s, "-c", "import umap"],
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            return py_s
    raise RuntimeError(
        "umap-learn is not available in current python or project .venv. "
        "Please install it or run with /home/zzz0054/GoldenF/.venv/bin/python."
    )


def read_metric(df: pd.DataFrame, metric: str, attr_type: str = "") -> tuple[float, float]:
    if attr_type:
        row = df[(df["metric"] == metric) & (df["attr_type"] == attr_type)]
    else:
        row = df[(df["metric"] == metric) & (df["attr_type"].fillna("") == "")]
        if row.empty:
            row = df[df["metric"] == metric]
    if row.empty:
        return (float("nan"), float("nan"))
    r = row.iloc[0]
    return float(r["gnn_a_value"]), float(r["gnn_b_value"])


def plot_metric_bar(summary_csv: Path, out_png: Path, subset_label: str) -> None:
    df = pd.read_csv(summary_csv)

    compact_specs = [
        ("silhouette_cosine_mean_k10", "", "Silhouette(cos)"),
        ("text_coherence_clean_mean", "", "Text coh(clean)"),
        ("pairwise_ari_mean", "", "Stability ARI"),
        ("primary_gse_nmi", "", "Confound GSE NMI (lower)"),
        ("primary_platform_nmi", "", "Confound Platform NMI (lower)"),
    ]
    enrichment_specs = [
        ("significant_mean", "mesh", "MeSH sig"),
        ("significant_mean", "gse_id", "GSE sig"),
    ]

    compact_a: list[float] = []
    compact_b: list[float] = []
    compact_labels: list[str] = []
    for metric, attr_type, label in compact_specs:
        a, b = read_metric(df, metric, attr_type)
        compact_a.append(a)
        compact_b.append(b)
        compact_labels.append(label)

    enrich_a: list[float] = []
    enrich_b: list[float] = []
    enrich_labels: list[str] = []
    for metric, attr_type, label in enrichment_specs:
        a, b = read_metric(df, metric, attr_type)
        enrich_a.append(a)
        enrich_b.append(b)
        enrich_labels.append(label)

    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))

    x0 = np.arange(len(compact_labels))
    bars0_a = axes[0].bar(
        x0 - width / 2,
        compact_a,
        width=width,
        label="GNN text_only",
        color="#34699A",
    )
    bars0_b = axes[0].bar(
        x0 + width / 2,
        compact_b,
        width=width,
        label="GNN text_meta_structlite",
        color="#F39C12",
    )
    axes[0].set_xticks(x0, compact_labels, rotation=15, ha="right")
    axes[0].set_ylabel("Metric value")
    axes[0].set_title("Compact metrics (0-1 scale)")
    axes[0].grid(axis="y", alpha=0.25)

    x1 = np.arange(len(enrich_labels))
    bars1_a = axes[1].bar(
        x1 - width / 2,
        enrich_a,
        width=width,
        label="GNN text_only",
        color="#34699A",
    )
    bars1_b = axes[1].bar(
        x1 + width / 2,
        enrich_b,
        width=width,
        label="GNN text_meta_structlite",
        color="#F39C12",
    )
    axes[1].set_xticks(x1, enrich_labels)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Significant term count (log scale)")
    axes[1].set_title("Metadata enrichment strength")
    axes[1].grid(axis="y", alpha=0.25, which="both")

    for bars, fmt, ax in [
        (bars0_a, "{:.3f}", axes[0]),
        (bars0_b, "{:.3f}", axes[0]),
        (bars1_a, "{:.1f}", axes[1]),
        (bars1_b, "{:.1f}", axes[1]),
    ]:
        for bar in bars:
            h = float(bar.get_height())
            if not np.isfinite(h):
                continue
            ax.text(
                bar.get_x() + bar.get_width() * 0.5,
                h * (1.02 if h > 1 else 1.01),
                fmt.format(h),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle(f"GNN vs GNN (subset {subset_label}, k=10)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=240)
    plt.close(fig)


def main() -> None:
    project_root = Path("/home/zzz0054/GoldenF")
    base = Path("/home/zzz0054/GoldenF/work_alpha_gnn_20260212")
    task = base / "task_20260219_gnn_feature_ablation"
    plots = task / "reports/plots"
    plots.mkdir(parents=True, exist_ok=True)
    plot_python = resolve_python_with_umap(project_root)

    umap_script = base / "scripts/plot_embedding_umap_scatter.py"

    emb_090_a = task / "features/subset090/text_only_gnn_holdout_subset090.npz"
    emb_090_b = task / "features/subset090/text_meta_structlite_gnn_holdout_subset090.npz"
    emb_085_a = task / "features/subset085/text_only_gnn_holdout_subset085.npz"
    emb_085_b = task / "features/subset085/text_meta_structlite_gnn_holdout_subset085.npz"

    # UMAP plots
    run_cmd(
        [
            plot_python,
            str(umap_script),
            "--embedding_npz",
            str(emb_090_a),
            "--compare_npz",
            str(emb_090_b),
            "--output_png",
            str(plots / "umap_gnn_textonly_vs_new_subset090.png"),
            "--k",
            "10",
            "--seed",
            "11",
            "--compare_titles",
            "GNN text_only|GNN text_meta_structlite",
            "--color_mode",
            "label",
        ],
        cwd=base,
    )

    run_cmd(
        [
            plot_python,
            str(umap_script),
            "--embedding_npz",
            str(emb_085_a),
            "--compare_npz",
            str(emb_085_b),
            "--output_png",
            str(plots / "umap_gnn_textonly_vs_new_subset085.png"),
            "--k",
            "10",
            "--seed",
            "11",
            "--compare_titles",
            "GNN text_only|GNN text_meta_structlite",
            "--color_mode",
            "label",
        ],
        cwd=base,
    )

    # Metric bar plots
    comp = task / "reports/metrics_reference/comparisons"
    plot_metric_bar(
        comp / "metrics_reference_summary_feature_subset090.csv",
        plots / "metrics_bar_gnn_textonly_vs_new_subset090.png",
        "0.90",
    )
    plot_metric_bar(
        comp / "metrics_reference_summary_feature_subset085.csv",
        plots / "metrics_bar_gnn_textonly_vs_new_subset085.png",
        "0.85",
    )

    readme = plots / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Plots: GNN text_only vs text_meta_structlite",
                "",
                "- `umap_gnn_textonly_vs_new_subset090.png`",
                "- `umap_gnn_textonly_vs_new_subset085.png`",
                "- `metrics_bar_gnn_textonly_vs_new_subset090.png`",
                "- `metrics_bar_gnn_textonly_vs_new_subset085.png`",
            ]
        ),
        encoding="utf-8",
    )

    print(plots)


if __name__ == "__main__":
    main()
