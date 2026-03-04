#!/usr/bin/env python3
"""Build a single-page poster for subset 0.85/0.90 fusion-vs-gnn results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _get_value(
    df: pd.DataFrame,
    metric: str,
    model_col: str,
    *,
    category: str | None = None,
    attr_type: str | None = None,
) -> float:
    mask = df["metric"].eq(metric)
    if category is not None:
        mask &= df["category"].eq(category)
    if attr_type is not None:
        mask &= df["attr_type"].fillna("").eq(attr_type)
    row = df.loc[mask]
    if row.empty:
        raise KeyError(
            f"Missing metric '{metric}'"
            f"{' in category ' + category if category else ''}"
            f"{' with attr_type ' + attr_type if attr_type else ''}"
        )
    return float(row.iloc[0][model_col])


def _plot_image(ax: plt.Axes, path: Path, title: str) -> None:
    img = mpimg.imread(path)
    ax.imshow(img)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")


def _silhouette_curve_085(base_dir: Path) -> tuple[pd.Series, pd.Series]:
    fusion_csv = base_dir / "task_20260217_metrics_080_085/csv/internal_metrics_fusion_085.csv"
    gnn_csv = (
        base_dir / "task_20260217_metrics_080_085/csv/internal_metrics_gnn_holdout_085.csv"
    )
    fusion = pd.read_csv(fusion_csv).groupby("k")["silhouette_cosine"].mean().sort_index()
    gnn = pd.read_csv(gnn_csv).groupby("k")["silhouette_cosine"].mean().sort_index()
    return fusion, gnn


def build_poster(base_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    subset090 = base_dir / "task_20260217_alpha05_subset090"
    subset085 = base_dir / "task_20260217_metrics_080_085"

    summary_090 = pd.read_csv(
        subset090 / "reports/metrics_reference/comparisons/metrics_reference_summary.csv"
    )
    summary_085 = pd.read_csv(
        subset085 / "reports/metrics_reference/comparisons/metrics_reference_summary_085.csv"
    )

    compare_090 = pd.read_csv(subset090 / "csv/compare_fusion_vs_gnn_subset090.csv")
    fusion085_curve, gnn085_curve = _silhouette_curve_085(base_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(24, 14), dpi=220, facecolor="#F5F7FA")
    gs = fig.add_gridspec(
        nrows=4,
        ncols=4,
        height_ratios=[0.25, 1.0, 1.0, 0.85],
        hspace=0.32,
        wspace=0.26,
    )

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.01,
        0.78,
        "Alpha=0.5 Fusion vs GNN Holdout: Poster Summary (Subset 0.85 + 0.90)",
        fontsize=23,
        fontweight="bold",
        color="#102A43",
    )
    ax_title.text(
        0.01,
        0.34,
        "Data source: task_20260217_metrics_080_085 + task_20260217_alpha05_subset090 | k=10 primary comparison",
        fontsize=12,
        color="#334E68",
    )

    ax_umap_090 = fig.add_subplot(gs[1, 0:2])
    _plot_image(
        ax_umap_090,
        subset090
        / "reports/embedding_scatter/umap_fusion_vs_gnn_k10_subset090_superpag_overlap_gradient_pointer_text.png",
        "Subset 0.90 | UMAP + super-PAG overlap",
    )

    ax_umap_085 = fig.add_subplot(gs[1, 2:4])
    _plot_image(
        ax_umap_085,
        subset085
        / "reports/embedding_scatter/umap_fusion_vs_gnn_k10_subset085_superpag_overlap_gradient_pointer_text.png",
        "Subset 0.85 | UMAP + super-PAG overlap",
    )

    ax_curve = fig.add_subplot(gs[2, 0:2])
    styles = [
        ("fusion_alpha0.5_concat_postl2", "subset0.90 fusion", "#B83B5E"),
        ("gnn_holdout_l2", "subset0.90 gnn holdout", "#0F4C75"),
    ]
    for model, label, color in styles:
        dfm = compare_090.loc[compare_090["model"].eq(model)].sort_values("k")
        ax_curve.plot(dfm["k"], dfm["sil_cos"], marker="o", linewidth=2.2, label=label, color=color)
    ax_curve.plot(
        fusion085_curve.index,
        fusion085_curve.values,
        marker="s",
        linewidth=2.2,
        color="#F08A5D",
        label="subset0.85 fusion",
    )
    ax_curve.plot(
        gnn085_curve.index,
        gnn085_curve.values,
        marker="s",
        linewidth=2.2,
        color="#3282B8",
        label="subset0.85 gnn holdout",
    )
    ax_curve.set_title("Cosine silhouette vs k", fontsize=13, fontweight="bold")
    ax_curve.set_xlabel("k")
    ax_curve.set_ylabel("silhouette (cosine)")
    ax_curve.grid(alpha=0.25, linestyle="--")
    ax_curve.legend(loc="best", fontsize=9)

    ax_quality = fig.add_subplot(gs[2, 2])
    quality_metrics = [
        ("silhouette_cosine_mean_k10", "SilCos"),
        ("text_coherence_clean_mean", "TxtCoh"),
        ("pairwise_ari_mean", "StabARI"),
    ]
    labels = ["090-Fusion", "090-GNN", "085-Fusion", "085-GNN"]
    colors = ["#D1495B", "#00798C", "#EDAe49", "#30638E"]
    metric_names = [m[1] for m in quality_metrics]
    x = np.arange(len(metric_names))
    width = 0.18

    vals = []
    for metric_key, _ in quality_metrics:
        vals.append(
            [
                _get_value(summary_090, metric_key, "fusion_value"),
                _get_value(summary_090, metric_key, "gnn_value"),
                _get_value(summary_085, metric_key, "fusion_value"),
                _get_value(summary_085, metric_key, "gnn_value"),
            ]
        )
    arr = np.array(vals).T
    for i in range(arr.shape[0]):
        ax_quality.bar(x + (i - 1.5) * width, arr[i], width=width, label=labels[i], color=colors[i])
    ax_quality.set_xticks(x)
    ax_quality.set_xticklabels(metric_names)
    ax_quality.set_ylim(0, 0.82)
    ax_quality.grid(alpha=0.2, axis="y")
    ax_quality.set_title("Quality metrics @k=10", fontsize=13, fontweight="bold")
    ax_quality.legend(fontsize=8, loc="upper left")

    ax_enrich = fig.add_subplot(gs[2, 3])
    enrich_metrics = [
        ("gse_id", "GSE sig"),
        ("mesh", "MeSH sig"),
    ]
    ex = np.arange(len(enrich_metrics))
    enrich_vals = np.array(
        [
            [
                _get_value(
                    summary_090,
                    "significant_mean",
                    "fusion_value",
                    category="meta_enrichment",
                    attr_type=attr,
                ),
                _get_value(
                    summary_090,
                    "significant_mean",
                    "gnn_value",
                    category="meta_enrichment",
                    attr_type=attr,
                ),
                _get_value(
                    summary_085,
                    "significant_mean",
                    "fusion_value",
                    category="meta_enrichment",
                    attr_type=attr,
                ),
                _get_value(
                    summary_085,
                    "significant_mean",
                    "gnn_value",
                    category="meta_enrichment",
                    attr_type=attr,
                ),
            ]
            for attr, _ in enrich_metrics
        ]
    ).T
    for i in range(enrich_vals.shape[0]):
        ax_enrich.bar(ex + (i - 1.5) * width, enrich_vals[i], width=width, color=colors[i], label=labels[i])
    ax_enrich.set_xticks(ex)
    ax_enrich.set_xticklabels([m[1] for m in enrich_metrics])
    ax_enrich.set_title("Meta enrichment counts @k=10", fontsize=13, fontweight="bold")
    ax_enrich.grid(alpha=0.2, axis="y")

    ax_confound = fig.add_subplot(gs[3, 0])
    conf_metrics = [("primary_gse_nmi", "GSE NMI"), ("primary_platform_nmi", "Platform NMI")]
    cx = np.arange(len(conf_metrics))
    conf_vals = np.array(
        [
            [
                _get_value(summary_090, metric, "fusion_value", category="confound"),
                _get_value(summary_090, metric, "gnn_value", category="confound"),
                _get_value(summary_085, metric, "fusion_value", category="confound"),
                _get_value(summary_085, metric, "gnn_value", category="confound"),
            ]
            for metric, _ in conf_metrics
        ]
    ).T
    for i in range(conf_vals.shape[0]):
        ax_confound.bar(cx + (i - 1.5) * width, conf_vals[i], width=width, color=colors[i], label=labels[i])
    ax_confound.set_xticks(cx)
    ax_confound.set_xticklabels([m[1] for m in conf_metrics])
    ax_confound.set_title("Confound similarity (lower is better)", fontsize=12, fontweight="bold")
    ax_confound.grid(alpha=0.2, axis="y")
    ax_confound.legend(fontsize=7, loc="upper right")

    ax_text = fig.add_subplot(gs[3, 1:4])
    ax_text.axis("off")
    summary_lines = [
        "Key takeaways",
        (
            f"1) Internal separation: GNN dominates on silhouette_cosine @k=10 "
            f"(subset0.90: { _get_value(summary_090, 'silhouette_cosine_mean_k10', 'gnn_value'):.3f} "
            f"vs fusion { _get_value(summary_090, 'silhouette_cosine_mean_k10', 'fusion_value'):.3f}; "
            f"subset0.85: { _get_value(summary_085, 'silhouette_cosine_mean_k10', 'gnn_value'):.3f} "
            f"vs fusion { _get_value(summary_085, 'silhouette_cosine_mean_k10', 'fusion_value'):.3f})."
        ),
        (
            f"2) Text coherence: fusion slightly higher "
            f"(subset0.90 clean={ _get_value(summary_090, 'text_coherence_clean_mean', 'fusion_value'):.3f} "
            f"vs gnn={ _get_value(summary_090, 'text_coherence_clean_mean', 'gnn_value'):.3f}; "
            f"subset0.85 clean={ _get_value(summary_085, 'text_coherence_clean_mean', 'fusion_value'):.3f} "
            f"vs gnn={ _get_value(summary_085, 'text_coherence_clean_mean', 'gnn_value'):.3f})."
        ),
        (
            f"3) Stability: GNN pairwise ARI is higher "
            f"(subset0.90 { _get_value(summary_090, 'pairwise_ari_mean', 'gnn_value'):.3f} "
            f"vs fusion { _get_value(summary_090, 'pairwise_ari_mean', 'fusion_value'):.3f}; "
            f"subset0.85 { _get_value(summary_085, 'pairwise_ari_mean', 'gnn_value'):.3f} "
            f"vs fusion { _get_value(summary_085, 'pairwise_ari_mean', 'fusion_value'):.3f})."
        ),
        (
            f"4) Meta enrichment: fusion has larger significant-hit counts for GSE/MeSH in both subsets "
            f"(e.g., subset0.85 MeSH { _get_value(summary_085, 'significant_mean', 'fusion_value', category='meta_enrichment', attr_type='mesh'):.1f} "
            f"vs gnn { _get_value(summary_085, 'significant_mean', 'gnn_value', category='meta_enrichment', attr_type='mesh'):.1f})."
        ),
        (
            f"5) Confound metrics are close: primary_gse_nmi "
            f"(subset0.90 fusion { _get_value(summary_090, 'primary_gse_nmi', 'fusion_value', category='confound'):.3f}, "
            f"gnn { _get_value(summary_090, 'primary_gse_nmi', 'gnn_value', category='confound'):.3f}); "
            f"no up/down conflict records detected (both CSVs contain only header)."
        ),
    ]
    ax_text.text(
        0.0,
        0.96,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=11.2,
        linespacing=1.5,
        color="#102A43",
    )

    png_path = output_dir / "poster_alpha05_fusion_vs_gnn_subset085_090.png"
    pdf_path = output_dir / "poster_alpha05_fusion_vs_gnn_subset085_090.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build combined subset085/090 poster.")
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Base folder: work_alpha_gnn_20260212",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for poster artifacts",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (base_dir / "reports/poster_20260219_085_090").resolve()
    )

    png_path, pdf_path = build_poster(base_dir, output_dir)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
