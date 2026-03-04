#!/usr/bin/env python3
"""Create a shareable bundle for human-only subset085/subset090 GNN results.

Default ("paper") mode copies compact, paper-facing artifacts (tables/reports +
selected embeddings). "full" mode copies the entire human_only subset folders.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path


def parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)


def collect_manifest(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        rows.append(
            {
                "path": rel,
                "bytes": int(p.stat().st_size),
                "sha256": sha256_file(p),
            }
        )
    return rows


def paper_copy_subset(src_subset: Path, dst_subset: Path, subset: str, recipes: list[str]) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {"files": [], "dirs": []}

    # Inputs: keep only compact provenance.
    for name in [
        "inputs/inputs_manifest.json",
        f"inputs/filter_stats_subset{subset}.md",
        f"inputs/human_ids_subset{subset}.txt",
    ]:
        src = src_subset / name
        if src.exists():
            copy_file(src, dst_subset / name)
        else:
            missing["files"].append(str(src))

    # Config + manifests.
    for path in [
        "reports/run_manifest.md",
        f"features/feature_manifest_subset{subset}.json",
    ]:
        src = src_subset / path
        if src.exists():
            copy_file(src, dst_subset / path)
        else:
            missing["files"].append(str(src))

    copy_tree(src_subset / "configs", dst_subset / "configs")

    # High-level outputs.
    for path in [
        f"reports/leaderboard_feature_ablation_subset{subset}.csv",
        f"reports/feature_ablation_report_subset{subset}.md",
    ]:
        src = src_subset / path
        if src.exists():
            copy_file(src, dst_subset / path)
        else:
            missing["files"].append(str(src))

    for recipe in recipes:
        # Link prediction: representative curve + multi-seed summaries.
        for path in [
            f"csv/{recipe}_linkpred_subset{subset}.csv",
            f"csv/linkpred_multiseed_{recipe}_subset{subset}.csv",
            f"csv/linkpred_multiseed_{recipe}_subset{subset}_summary.csv",
            f"csv/selected_seed_{recipe}_subset{subset}.txt",
        ]:
            src = src_subset / path
            if src.exists():
                copy_file(src, dst_subset / path)
            else:
                missing["files"].append(str(src))

        # Selected embedding for downstream inspection.
        for path in [
            f"csv/selected_embedding_{recipe}_subset{subset}.npz",
            f"csv/selected_curve_{recipe}_subset{subset}.csv",
        ]:
            src = src_subset / path
            if src.exists():
                copy_file(src, dst_subset / path)
            else:
                missing["files"].append(str(src))

        # Clustering eval (kmeans metrics table).
        eval_path = src_subset / f"csv/{recipe}_eval_subset{subset}.csv"
        if eval_path.exists():
            copy_file(eval_path, dst_subset / f"csv/{recipe}_eval_subset{subset}.csv")
        else:
            missing["files"].append(str(eval_path))

        # Validation + confound reports.
        val_dir = src_subset / f"reports/{recipe}_validation_subset{subset}"
        if val_dir.exists():
            copy_tree(val_dir, dst_subset / f"reports/{recipe}_validation_subset{subset}")
        else:
            missing["dirs"].append(str(val_dir))

        conf_md = src_subset / f"reports/{recipe}_confound_subset{subset}.md"
        if conf_md.exists():
            copy_file(conf_md, dst_subset / f"reports/{recipe}_confound_subset{subset}.md")
        else:
            missing["files"].append(str(conf_md))

    return missing


def _mpl_setup() -> None:
    # Headless + consistent, paper-facing defaults.
    import matplotlib  # type: ignore

    matplotlib.use("Agg", force=True)
    import matplotlib as mpl  # type: ignore

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
            "lines.linewidth": 2.0,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _plot_linkpred_meanstd(curve_files: list[Path], out_png: Path, title: str) -> None:
    _mpl_setup()

    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.lines import Line2D  # type: ignore
    from matplotlib.patches import Patch  # type: ignore
    from matplotlib.ticker import MaxNLocator  # type: ignore
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    frames = []
    for p in curve_files:
        df = pd.read_csv(p)
        df["seed_file"] = p.name
        frames.append(df)
    all_df = pd.concat(frames, axis=0, ignore_index=True)

    if "epoch" not in all_df.columns:
        raise ValueError("curve CSVs must include 'epoch'")

    metric_candidates = [
        "train_loss",
        "val_loss",
        "val_auc",
        "val_ap",
        "test_auc",
        "test_ap",
    ]
    metrics = [m for m in metric_candidates if m in all_df.columns]
    if not metrics:
        raise ValueError(f"curve CSVs missing expected metric columns: {metric_candidates}")

    agg = all_df.groupby("epoch", as_index=False)[metrics].agg(["mean", "std"])
    agg.columns = ["_".join([c for c in col if c]) for col in agg.columns.to_flat_index()]
    agg = agg.rename(columns={"epoch_": "epoch"})
    epochs = agg["epoch"].to_numpy(dtype=float)

    # 3x2 grid (up to 6 metrics)
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 9.2), constrained_layout=False)
    axes = axes.ravel()

    line_color = "#1f77b4"
    band_alpha = 0.18

    def _apply_limits(ax, metric: str) -> None:
        ax.set_xlim(float(np.nanmin(epochs)), float(np.nanmax(epochs)))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if metric.endswith("_auc") or metric.endswith("_ap"):
            ax.set_ylim(0.0, 1.0)
        elif metric.endswith("_loss"):
            ax.set_ylim(bottom=0.0)

    def plot_band(ax, base: str) -> None:
        mean = agg[f"{base}_mean"].to_numpy(dtype=float)
        std = agg.get(f"{base}_std", pd.Series([np.nan] * len(agg))).to_numpy(dtype=float)
        ax.plot(epochs, mean, color=line_color)
        if np.any(np.isfinite(std)):
            ax.fill_between(epochs, mean - std, mean + std, color=line_color, alpha=band_alpha, linewidth=0)
        ax.set_title(base)
        ax.set_xlabel("epoch")
        _apply_limits(ax, base)

    for i, metric in enumerate(metric_candidates):
        ax = axes[i]
        if metric in metrics:
            plot_band(ax, metric)
        else:
            ax.axis("off")

    fig.suptitle(title, y=0.995)
    fig.legend(
        handles=[
            Line2D([0], [0], color=line_color, linewidth=2.0, label="mean"),
            Patch(facecolor=line_color, alpha=band_alpha, label="±1 std"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.93])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def _plot_linkpred_single(curve_csv: Path, out_png: Path, title: str) -> None:
    _mpl_setup()

    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.lines import Line2D  # type: ignore
    from matplotlib.ticker import MaxNLocator  # type: ignore
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    df = pd.read_csv(curve_csv)
    if df.empty:
        raise ValueError(f"Empty curve CSV: {curve_csv}")
    if "epoch" not in df.columns:
        raise ValueError("curve CSV must include 'epoch'")

    metric_candidates = [
        "train_loss",
        "val_loss",
        "val_auc",
        "val_ap",
        "test_auc",
        "test_ap",
    ]
    metrics = [m for m in metric_candidates if m in df.columns]
    epochs = df["epoch"].to_numpy(dtype=float)

    fig, axes = plt.subplots(3, 2, figsize=(12.5, 9.2), constrained_layout=False)
    axes = axes.ravel()

    line_color = "#1f77b4"

    def _apply_limits(ax, metric: str) -> None:
        ax.set_xlim(float(np.nanmin(epochs)), float(np.nanmax(epochs)))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if metric.endswith("_auc") or metric.endswith("_ap"):
            ax.set_ylim(0.0, 1.0)
        elif metric.endswith("_loss"):
            ax.set_ylim(bottom=0.0)

    for i, metric in enumerate(metric_candidates):
        ax = axes[i]
        if metric not in metrics:
            ax.axis("off")
            continue
        y = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
        ax.plot(epochs, y, color=line_color)
        ax.set_title(metric)
        ax.set_xlabel("epoch")
        _apply_limits(ax, metric)

    fig.suptitle(title, y=0.995)
    fig.legend(
        handles=[Line2D([0], [0], color=line_color, linewidth=2.0, label="selected seed")],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=1,
        frameon=False,
    )
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.93])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def _plot_linkpred_compare(meanstd_png_a: Path, meanstd_png_b: Path, out_png: Path) -> None:
    # placeholder helper for future (kept for backward compatibility if we expand plots later)
    _ = (meanstd_png_a, meanstd_png_b)
    out_png.parent.mkdir(parents=True, exist_ok=True)


def write_subset_plots(
    src_subset: Path, dst_subset: Path, subset: str, recipes: list[str]
) -> list[str]:
    """Generate plot PNGs inside dst_subset/plots. Returns warning strings."""
    warnings: list[str] = []
    plots_dir = dst_subset / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    src_curves_dir = src_subset / "csv" / "curves"
    for recipe in recipes:
        tag = f"{recipe}_subset{subset}"
        meanstd_png = plots_dir / f"linkpred_meanstd_{tag}.png"
        selected_png = plots_dir / f"linkpred_selected_{tag}.png"

        # Mean±std across seeds (requires per-seed curves in source run directory).
        curve_files = sorted(src_curves_dir.glob(f"{tag}_seed*.csv"))
        if curve_files:
            try:
                _plot_linkpred_meanstd(
                    curve_files=curve_files,
                    out_png=meanstd_png,
                    title=f"Link prediction (mean±std) {tag} (n={len(curve_files)} seeds)",
                )
            except Exception as e:
                warnings.append(f"[plot] failed mean±std for {tag}: {e}")
        else:
            warnings.append(f"[plot] missing per-seed curves in source: {src_curves_dir} (tag={tag})")

        # Representative curve (already copied into bundle csv/)
        curve_csv = dst_subset / "csv" / f"{recipe}_linkpred_subset{subset}.csv"
        if curve_csv.exists():
            try:
                _plot_linkpred_single(
                    curve_csv=curve_csv,
                    out_png=selected_png,
                    title=f"Link prediction (selected seed) {tag}",
                )
            except Exception as e:
                warnings.append(f"[plot] failed selected curve for {tag}: {e}")
        else:
            warnings.append(f"[plot] missing curve CSV in bundle: {curve_csv}")

    readme_lines = [
        f"# Plots (subset{subset})",
        "",
        "Generated at bundle creation time.",
        "",
    ]
    for recipe in recipes:
        tag = f"{recipe}_subset{subset}"
        readme_lines.extend(
            [
                f"## {recipe}",
                "",
                f"- mean±std curve: `linkpred_meanstd_{tag}.png`",
                f"- selected-seed curve: `linkpred_selected_{tag}.png`",
                "",
            ]
        )
    if warnings:
        readme_lines.extend(["## Warnings", ""] + [f"- {w}" for w in warnings] + [""])
    (plots_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    return warnings


def write_subset_readme(dst_subset: Path, subset: str, recipes: list[str]) -> None:
    title = f"# subset{subset} (human-only)"

    def p(rel: str) -> str:
        return f"`{rel}`"

    lines = [
        title,
        "",
        "## Start here",
        "",
        f"- leaderboard: {p(f'reports/leaderboard_feature_ablation_subset{subset}.csv')}",
        f"- summary report: {p(f'reports/feature_ablation_report_subset{subset}.md')}",
        "",
        "## Inputs (provenance)",
        "",
        f"- human IDs: {p(f'inputs/human_ids_subset{subset}.txt')}",
        f"- filter stats: {p(f'inputs/filter_stats_subset{subset}.md')}",
        f"- inputs manifest: {p('inputs/inputs_manifest.json')}",
        "",
        "## Recipes",
        "",
    ]

    for recipe in recipes:
        lines.extend(
            [
                f"### {recipe}",
                "",
                f"- linkpred curve (representative seed): {p(f'csv/{recipe}_linkpred_subset{subset}.csv')}",
                f"- linkpred multi-seed (per-seed): {p(f'csv/linkpred_multiseed_{recipe}_subset{subset}.csv')}",
                f"- linkpred multi-seed (mean±std): {p(f'csv/linkpred_multiseed_{recipe}_subset{subset}_summary.csv')}",
                f"- selected seed: {p(f'csv/selected_seed_{recipe}_subset{subset}.txt')}",
                f"- selected curve copy: {p(f'csv/selected_curve_{recipe}_subset{subset}.csv')}",
                f"- selected embedding: {p(f'csv/selected_embedding_{recipe}_subset{subset}.npz')}",
                f"- clustering eval: {p(f'csv/{recipe}_eval_subset{subset}.csv')}",
                f"- validation report: {p(f'reports/{recipe}_validation_subset{subset}/validation_report.md')}",
                f"- confound report: {p(f'reports/{recipe}_confound_subset{subset}.md')}",
                "",
            ]
        )

    lines.extend(
        [
            "## Configs + feature specs",
            "",
            f"- feature manifest: {p(f'features/feature_manifest_subset{subset}.json')}",
            f"- run configs: {p('configs/')}",
            "",
            "## Plots",
            "",
            f"- plots index: {p('plots/README.md')}",
            "",
        ]
    )

    (dst_subset / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_top_readme(out_dir: Path, subsets: list[str], recipes: list[str], mode: str) -> None:
    def p(rel: str) -> str:
        return f"`{rel}`"

    lines = [
        "# Human-only GNN share bundle",
        "",
        f"- created_utc: {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}",
        f"- mode: {mode}",
        f"- subsets: {', '.join(subsets)}",
        f"- recipes: {', '.join(recipes)}",
        "",
        "## Index",
        "",
    ]
    for subset in subsets:
        lines.append(f"- subset{subset}: {p(f'subset{subset}/README.md')}")
    lines.extend(
        [
            "",
            "## Start here (most important)",
            "",
            "- model spec: `model/gnn_architecture_config.md`",
            "- pipeline/task README: `task_README.md`",
        ]
    )
    for subset in subsets:
        lines.extend(
            [
                f"- subset{subset} leaderboard: {p(f'subset{subset}/reports/leaderboard_feature_ablation_subset{subset}.csv')}",
                f"- subset{subset} report: {p(f'subset{subset}/reports/feature_ablation_report_subset{subset}.md')}",
            ]
        )

    lines.extend(
        [
            "",
            "## What’s included",
            "",
            "- link prediction: train/val/test curves + multi-seed mean±std summaries",
            "- embedding evaluation: clustering metrics (silhouette/DB/CH) over KMeans seeds",
            "- validation: text coherence + stability (pairwise ARI) + metadata enrichment",
            "- confounds: primary GSE/platform association (NMI/ARI)",
            "",
            "Note: in `paper` mode we copy compact provenance files (IDs + manifests) rather than full filtered metadata/edge tables.",
            "",
            "## Plots",
            "",
            "- link prediction curves are pre-rendered under each `subset*/plots/` folder.",
            "",
            "## Bundle integrity",
            "",
            f"- file manifest (sha256): {p('MANIFEST.csv')}",
            f"- missing-files report: {p('missing_files.json')}",
            "",
            "## How to reproduce",
            "",
            "Run (human-only):",
            "```bash",
            "python work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/scripts/run_feature_ablation_subset.py \\",
            "  --subset 090 --human_only --recipes text_only,text_structlite \\",
            "  --stages build,train,eval,validate,confound,summary",
            "python work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/scripts/run_feature_ablation_subset.py \\",
            "  --subset 085 --human_only --recipes text_only,text_structlite \\",
            "  --stages build,train,eval,validate,confound,summary",
            "```",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a share bundle for human-only GNN results.")
    parser.add_argument(
        "--task_dir",
        default="/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation",
    )
    parser.add_argument("--subsets", default="085,090")
    parser.add_argument("--recipes", default="text_only,text_structlite")
    parser.add_argument("--mode", default="paper", choices=["paper", "full"])
    parser.add_argument("--out_dir", default="")
    parser.add_argument(
        "--with_plots",
        action="store_true",
        help="Generate PNG plots into the bundle (recommended).",
    )
    args = parser.parse_args()

    task_dir = Path(args.task_dir).resolve()
    human_root = task_dir / "human_only"
    subsets = parse_csv_list(args.subsets)
    recipes = parse_csv_list(args.recipes)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (task_dir / "share" / f"gnn_human_only_bundle_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Top-level docs.
    copy_file(Path("/home/zzz0054/GoldenF/docs/gnn_architecture_config.md"), out_dir / "model/gnn_architecture_config.md")
    copy_file(task_dir / "README.md", out_dir / "task_README.md")

    missing_all: dict[str, dict[str, list[str]]] = {}
    for subset in subsets:
        src_subset = human_root / f"subset{subset}"
        dst_subset = out_dir / f"subset{subset}"
        if not src_subset.exists():
            missing_all[subset] = {"files": [f"missing subset dir: {src_subset}"], "dirs": []}
            continue

        if args.mode == "full":
            copy_tree(src_subset, dst_subset)
            missing_all[subset] = {"files": [], "dirs": []}
        else:
            missing_all[subset] = paper_copy_subset(src_subset, dst_subset, subset=subset, recipes=recipes)

    write_top_readme(out_dir, subsets=subsets, recipes=recipes, mode=args.mode)
    for subset in subsets:
        dst_subset = out_dir / f"subset{subset}"
        if dst_subset.exists():
            write_subset_readme(dst_subset, subset=subset, recipes=recipes)
            if args.with_plots:
                src_subset = human_root / f"subset{subset}"
                write_subset_plots(src_subset, dst_subset, subset=subset, recipes=recipes)

    (out_dir / "missing_files.json").write_text(
        json.dumps(missing_all, indent=2) + "\n", encoding="utf-8"
    )

    manifest_rows = collect_manifest(out_dir)
    manifest_path = out_dir / "MANIFEST.csv"
    import csv  # local import to keep top-level minimal

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256"])
        w.writeheader()
        for row in manifest_rows:
            w.writerow(row)

    print(f"[ok] wrote bundle: {out_dir}", flush=True)
    print(f"[ok] wrote manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
