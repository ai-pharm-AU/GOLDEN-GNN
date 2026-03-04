#!/usr/bin/env python3
"""Generate metrics_reference-style comparison report for feature ablation outputs."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def gini(values: np.ndarray) -> float:
    x = np.sort(values.astype(np.float64))
    n = len(x)
    if n == 0:
        return 0.0
    total = x.sum()
    if total == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * x) / (n * total)) - (n + 1) / n)


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return x / norms


def load_npz_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    data = np.load(path, allow_pickle=True)
    ids = data["ID"] if "ID" in data else data["GOID"]
    emb = data["embeddings"] if "embeddings" in data else data["emb"]
    return [str(x) for x in ids], np.asarray(emb, dtype=np.float32)


def load_desc(desc_tsv: Path) -> dict[str, str]:
    df = pd.read_csv(desc_tsv, sep="\t")
    return {str(i): str(d) for i, d in zip(df["GOID"], df["DESCRIPTION"])}


def text_coherence(ids: list[str], labels: np.ndarray, id_to_desc: dict[str, str]) -> float:
    docs = [id_to_desc.get(i, "") for i in ids]
    vec = TfidfVectorizer(stop_words="english", min_df=2)
    x = vec.fit_transform(docs)
    if x.shape[1] == 0:
        return float("nan")

    weighted_sum = 0.0
    total = 0
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        x_c = x[idx]
        centroid = np.asarray(x_c.mean(axis=0))
        sims = cosine_similarity(x_c, centroid).ravel()
        weighted_sum += float(sims.mean()) * len(idx)
        total += len(idx)
    return weighted_sum / total if total > 0 else float("nan")


def compute_cluster_balance_csv(
    embedding_files: list[Path],
    desc_tsv: Path,
    seeds: list[int],
    k: int,
) -> pd.DataFrame:
    id_to_desc = load_desc(desc_tsv)
    rows = []
    for emb_path in embedding_files:
        ids, emb = load_npz_embeddings(emb_path)
        keep = [i for i, gid in enumerate(ids) if gid in id_to_desc]
        ids2 = [ids[i] for i in keep]
        x = emb[keep]
        x = l2_normalize_rows(x)
        for seed in seeds:
            labels = KMeans(n_clusters=k, random_state=seed, n_init="auto").fit_predict(x)
            sizes = np.bincount(labels, minlength=k)
            rows.append(
                {
                    "embedding_file": str(emb_path),
                    "n_samples": len(ids2),
                    "dim": int(x.shape[1]),
                    "k": int(k),
                    "seed": int(seed),
                    "l2_norm": 1,
                    "largest_cluster_frac": float(sizes.max() / sizes.sum()),
                    "smallest_cluster_frac": float(sizes[sizes > 0].min() / sizes.sum()),
                    "size_gini": gini(sizes[sizes > 0]),
                    "text_coherence_weighted": text_coherence(ids2, labels, id_to_desc),
                }
            )
    return pd.DataFrame(rows)


def mean_internal_metrics(eval_csv: Path, k: int) -> dict[str, float]:
    d = pd.read_csv(eval_csv)
    d = d[d["k"] == k]
    return {
        "silhouette_euclidean_mean_k10": float(d["silhouette_euclidean"].mean()),
        "silhouette_cosine_mean_k10": float(d["silhouette_cosine"].mean()),
        "davies_bouldin_mean_k10": float(d["davies_bouldin"].mean()),
        "calinski_harabasz_mean_k10": float(d["calinski_harabasz"].mean()),
    }


def mean_text_stability(validation_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    t = pd.read_csv(validation_dir / "text_coherence_by_seed.csv")
    out["text_coherence_raw_mean"] = float(t["text_coherence_raw"].mean())
    out["text_null_raw_p_mean"] = float(t["text_null_raw_p"].mean())
    out["text_coherence_clean_mean"] = float(t["text_coherence_clean"].mean())
    out["text_null_clean_p_mean"] = float(t["text_null_clean_p"].mean())

    a = pd.read_csv(validation_dir / "pairwise_ari.csv")
    out["pairwise_ari_mean"] = float(a["ari"].mean())
    out["pairwise_ari_std"] = float(a["ari"].std())
    return out


def mean_enrichment(validation_dir: Path) -> dict[tuple[str, str], float]:
    d = pd.read_csv(validation_dir / "meta_enrichment_seed_summary.csv")
    out: dict[tuple[str, str], float] = {}
    for attr in sorted(d["attr_type"].unique()):
        s = d[d["attr_type"] == attr]
        out[("significant_mean", attr)] = float(s["significant"].mean())
        out[("null_sig_mean", attr)] = float(s["null_sig_mean"].mean())
        out[("p_sig_mean", attr)] = float(s["null_sig_p"].mean())
        out[("max_log10q_mean", attr)] = float(s["max_log10q"].mean())
        key_pmax = "null_maxlog10q_p" if "null_maxlog10q_p" in s.columns else None
        out[("p_maxlog10q_mean", attr)] = float(s[key_pmax].mean()) if key_pmax else math.nan
    return out


def parse_confound(md_path: Path) -> dict[str, float]:
    txt = md_path.read_text(encoding="utf-8")
    gse = re.search(r"primary GSE:.*?NMI=([0-9.]+|nan) \| ARI=([0-9.]+|nan)", txt)
    pla = re.search(r"primary platform:.*?NMI=([0-9.]+|nan) \| ARI=([0-9.]+|nan)", txt)
    out = {}
    if gse:
        out["primary_gse_nmi"] = float(gse.group(1))
        out["primary_gse_ari"] = float(gse.group(2))
    if pla:
        out["primary_platform_nmi"] = float(pla.group(1))
        out["primary_platform_ari"] = float(pla.group(2))
    return out


def pick_compare_recipe(leaderboard: pd.DataFrame, baseline: str) -> str:
    candidates = leaderboard[
        (~leaderboard["recipe"].isin([baseline, "shuffle_control", "random_control"]))
    ].copy()
    if candidates.empty:
        return baseline
    candidates = candidates.sort_values("silhouette_cosine_k", ascending=False)
    return str(candidates.iloc[0]["recipe"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate metrics_reference-style comparison for ablation.")
    parser.add_argument("--subset", required=True, choices=["085", "090"])
    parser.add_argument(
        "--task_dir",
        default="/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation",
    )
    parser.add_argument(
        "--base_dir",
        default="/home/zzz0054/GoldenF/work_alpha_gnn_20260212",
    )
    parser.add_argument("--baseline_recipe", default="text_only")
    parser.add_argument("--compare_recipe", default="")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seeds", default="11,12")
    args = parser.parse_args()

    task_dir = Path(args.task_dir).resolve()
    base_dir = Path(args.base_dir).resolve()
    subset = args.subset
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    leaderboard_path = (
        task_dir
        / f"reports/subset{subset}/leaderboard_feature_ablation_subset{subset}.csv"
    )
    leaderboard = pd.read_csv(leaderboard_path) if leaderboard_path.exists() else pd.DataFrame()

    baseline = args.baseline_recipe
    if args.compare_recipe.strip():
        compare = args.compare_recipe.strip()
    else:
        if leaderboard.empty:
            raise FileNotFoundError(
                f"Leaderboard not found for subset{subset}: {leaderboard_path}. "
                "Provide --compare_recipe explicitly."
            )
        compare = pick_compare_recipe(leaderboard, baseline)

    csv_dir = task_dir / f"csv/subset{subset}"
    report_subset_dir = task_dir / f"reports/subset{subset}"

    eval_a = csv_dir / f"{baseline}_eval_subset{subset}.csv"
    eval_b = csv_dir / f"{compare}_eval_subset{subset}.csv"

    emb_a = Path(pd.read_csv(eval_a).iloc[0]["embedding_file"])
    emb_b = Path(pd.read_csv(eval_b).iloc[0]["embedding_file"])

    val_a = report_subset_dir / f"{baseline}_validation_subset{subset}"
    val_b = report_subset_dir / f"{compare}_validation_subset{subset}"
    conf_a = report_subset_dir / f"{baseline}_confound_subset{subset}.md"
    conf_b = report_subset_dir / f"{compare}_confound_subset{subset}.md"

    desc_tsv = Path("/home/zzz0054/GoldenF/data/RummaGEO/rummageo_descriptions.tsv")
    cluster_df = compute_cluster_balance_csv([emb_a, emb_b], desc_tsv, seeds, args.k)

    comp_dir = task_dir / "reports/metrics_reference/comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)

    cluster_csv = comp_dir / f"cluster_balance_text_coherence_feature_subset{subset}.csv"
    cluster_df.to_csv(cluster_csv, index=False)

    def mean_cluster_for(emb: Path) -> dict[str, float]:
        d = cluster_df[cluster_df["embedding_file"] == str(emb)]
        return {
            "largest_cluster_frac_mean_k10": float(d["largest_cluster_frac"].mean()),
            "smallest_cluster_frac_mean_k10": float(d["smallest_cluster_frac"].mean()),
            "size_gini_mean_k10": float(d["size_gini"].mean()),
            "text_coherence_weighted_mean_k10": float(d["text_coherence_weighted"].mean()),
        }

    internal_a = mean_internal_metrics(eval_a, args.k)
    internal_b = mean_internal_metrics(eval_b, args.k)
    cb_a = mean_cluster_for(emb_a)
    cb_b = mean_cluster_for(emb_b)
    txt_stab_a = mean_text_stability(val_a)
    txt_stab_b = mean_text_stability(val_b)
    enr_a = mean_enrichment(val_a)
    enr_b = mean_enrichment(val_b)
    confm_a = parse_confound(conf_a)
    confm_b = parse_confound(conf_b)

    rows = []
    for metric in [
        "silhouette_euclidean_mean_k10",
        "silhouette_cosine_mean_k10",
        "davies_bouldin_mean_k10",
        "calinski_harabasz_mean_k10",
    ]:
        rows.append(
            {
                "category": "internal",
                "metric": metric,
                "attr_type": "",
                "gnn_a_value": internal_a[metric],
                "gnn_b_value": internal_b[metric],
            }
        )

    for metric in [
        "largest_cluster_frac_mean_k10",
        "smallest_cluster_frac_mean_k10",
        "size_gini_mean_k10",
        "text_coherence_weighted_mean_k10",
    ]:
        rows.append(
            {
                "category": "cluster_balance",
                "metric": metric,
                "attr_type": "",
                "gnn_a_value": cb_a[metric],
                "gnn_b_value": cb_b[metric],
            }
        )

    for metric in [
        "text_coherence_raw_mean",
        "text_null_raw_p_mean",
        "text_coherence_clean_mean",
        "text_null_clean_p_mean",
    ]:
        rows.append(
            {
                "category": "text_coherence_meta",
                "metric": metric,
                "attr_type": "",
                "gnn_a_value": txt_stab_a[metric],
                "gnn_b_value": txt_stab_b[metric],
            }
        )

    for metric in ["pairwise_ari_mean", "pairwise_ari_std"]:
        rows.append(
            {
                "category": "stability",
                "metric": metric,
                "attr_type": "",
                "gnn_a_value": txt_stab_a[metric],
                "gnn_b_value": txt_stab_b[metric],
            }
        )

    attr_types = sorted({k[1] for k in enr_a.keys()} | {k[1] for k in enr_b.keys()})
    enrich_metrics = [
        "significant_mean",
        "null_sig_mean",
        "p_sig_mean",
        "max_log10q_mean",
        "p_maxlog10q_mean",
    ]
    for attr in attr_types:
        for metric in enrich_metrics:
            rows.append(
                {
                    "category": "meta_enrichment",
                    "metric": metric,
                    "attr_type": attr,
                    "gnn_a_value": enr_a.get((metric, attr), math.nan),
                    "gnn_b_value": enr_b.get((metric, attr), math.nan),
                }
            )

    for metric in [
        "primary_gse_nmi",
        "primary_gse_ari",
        "primary_platform_nmi",
        "primary_platform_ari",
    ]:
        rows.append(
            {
                "category": "confound",
                "metric": metric,
                "attr_type": "",
                "gnn_a_value": confm_a.get(metric, math.nan),
                "gnn_b_value": confm_b.get(metric, math.nan),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_csv = comp_dir / f"metrics_reference_summary_feature_subset{subset}.csv"
    summary_df.to_csv(summary_csv, index=False)

    md_path = comp_dir / f"metrics_reference_comparison_feature_subset{subset}.md"
    lines = []
    lines.append(f"# Metrics Reference: {baseline} vs {compare} (subset 0.{subset}, k={args.k})")
    lines.append("")
    lines.append("This file summarizes key metrics in the same style as previous metrics_reference reports.")
    lines.append(f"- model A (`gnn_a_value`): `{baseline}`")
    lines.append(f"- model B (`gnn_b_value`): `{compare}`")
    lines.append(f"- summary CSV: `{summary_csv}`")
    lines.append(f"- cluster balance CSV: `{cluster_csv}`")
    lines.append("")

    sec = summary_df[summary_df["category"] == "internal"]
    lines.append("## Internal Metrics (mean across seeds, k=10)")
    for _, r in sec.iterrows():
        lines.append(
            f"- {r['metric']}: modelA={r['gnn_a_value']:.6f} | modelB={r['gnn_b_value']:.6f}"
        )
    lines.append("")

    sec = summary_df[summary_df["category"] == "cluster_balance"]
    lines.append("## Cluster Balance + TF-IDF Coherence (mean across seeds, k=10)")
    for _, r in sec.iterrows():
        lines.append(
            f"- {r['metric']}: modelA={r['gnn_a_value']:.6f} | modelB={r['gnn_b_value']:.6f}"
        )
    lines.append("")

    sec = summary_df[summary_df["category"] == "text_coherence_meta"]
    lines.append("## Text Coherence (metadata-enriched TF-IDF) + Null p-values")
    for _, r in sec.iterrows():
        lines.append(
            f"- {r['metric']}: modelA={r['gnn_a_value']:.6f} | modelB={r['gnn_b_value']:.6f}"
        )
    lines.append("")

    sec = summary_df[summary_df["category"] == "stability"]
    lines.append("## Seed Stability (ARI across seeds)")
    for _, r in sec.iterrows():
        lines.append(
            f"- {r['metric']}: modelA={r['gnn_a_value']:.6f} | modelB={r['gnn_b_value']:.6f}"
        )
    lines.append("")

    lines.append("## Metadata Enrichment (mean across seeds, k=10)")
    for attr in attr_types:
        lines.append(f"- {attr}:")
        sub = summary_df[
            (summary_df["category"] == "meta_enrichment")
            & (summary_df["attr_type"] == attr)
        ]
        for _, r in sub.iterrows():
            lines.append(
                f"  - {r['metric']}: modelA={r['gnn_a_value']:.6f} | modelB={r['gnn_b_value']:.6f}"
            )
    lines.append("")

    sec = summary_df[summary_df["category"] == "confound"]
    lines.append("## Confound NMI/ARI (primary labels, seed=11)")
    for _, r in sec.iterrows():
        lines.append(
            f"- {r['metric']}: modelA={r['gnn_a_value']:.6f} | modelB={r['gnn_b_value']:.6f}"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"baseline={baseline}")
    print(f"compare={compare}")
    print(summary_csv)
    print(cluster_csv)
    print(md_path)


if __name__ == "__main__":
    main()
