import argparse
import os
import pathlib
import sys
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return x / norms


def load_embedding_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "ID" in data:
        ids = data["ID"]
    elif "GOID" in data:
        ids = data["GOID"]
    else:
        raise KeyError(f"{path} must contain 'ID' or 'GOID'.")

    if "embeddings" in data:
        emb = data["embeddings"]
    elif "emb" in data:
        emb = data["emb"]
    else:
        raise KeyError(f"{path} must contain 'embeddings' or 'emb'.")

    return ids, emb


def align_embeddings(
    go_embedding_file: str, pager_embedding_file: str, cut_dim: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    go_ids_raw, go_emb = load_embedding_npz(go_embedding_file)
    pager_ids_raw, pager_emb = load_embedding_npz(pager_embedding_file)

    go_ids = np.asarray([str(x) for x in go_ids_raw])
    pager_ids = np.asarray([str(x) for x in pager_ids_raw])

    common_ids = np.intersect1d(go_ids, pager_ids)
    if common_ids.size == 0:
        raise ValueError("No overlapping IDs between GO and PAGER embeddings.")

    go_index = {gid: i for i, gid in enumerate(go_ids)}
    pager_index = {gid: i for i, gid in enumerate(pager_ids)}

    go_aligned = np.stack([go_emb[go_index[gid]] for gid in common_ids]).astype(
        np.float32
    )
    pager_aligned = np.stack(
        [pager_emb[pager_index[gid]] for gid in common_ids]
    ).astype(np.float32)

    if cut_dim > go_aligned.shape[1] or cut_dim > pager_aligned.shape[1]:
        raise ValueError(
            f"cut_dim={cut_dim} exceeds available dimensions "
            f"(go={go_aligned.shape[1]}, pager={pager_aligned.shape[1]})"
        )

    go_cut = go_aligned[:, :cut_dim]
    pager_cut = pager_aligned[:, :cut_dim]
    return common_ids, l2_normalize_rows(go_cut), l2_normalize_rows(pager_cut)


def compute_pairwise_geometry_stats(
    go_norm: np.ndarray,
    pager_norm: np.ndarray,
    sample_size: int,
    seeds: list[int],
    scatter_points: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows: list[dict[str, float | int]] = []
    scatter_x = np.array([])
    scatter_y = np.array([])

    n = go_norm.shape[0]
    m = min(sample_size, n)

    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=m, replace=False)
        go_s = go_norm[idx]
        pager_s = pager_norm[idx]

        sim_go = cosine_similarity(go_s)
        sim_pager = cosine_similarity(pager_s)
        tri = np.triu_indices(m, k=1)

        vals_go = sim_go[tri]
        vals_pager = sim_pager[tri]

        sp = spearmanr(vals_go, vals_pager).correlation
        pe = pearsonr(vals_go, vals_pager).statistic

        rows.append(
            {
                "seed": seed,
                "sample_size": m,
                "spearman_pairwise_cosine": float(sp),
                "pearson_pairwise_cosine": float(pe),
            }
        )

        if i == 0:
            pair_count = vals_go.shape[0]
            keep = min(scatter_points, pair_count)
            keep_idx = rng.choice(pair_count, size=keep, replace=False)
            scatter_x = vals_go[keep_idx]
            scatter_y = vals_pager[keep_idx]

    return pd.DataFrame(rows), scatter_x, scatter_y


def compute_neighbor_overlap(
    go_norm: np.ndarray,
    pager_norm: np.ndarray,
    sample_size: int,
    seeds: list[int],
    k_list: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    n = go_norm.shape[0]
    m = min(sample_size, n)

    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=m, replace=False)
        go_s = go_norm[idx]
        pager_s = pager_norm[idx]

        sim_go = cosine_similarity(go_s)
        sim_pager = cosine_similarity(pager_s)
        np.fill_diagonal(sim_go, -np.inf)
        np.fill_diagonal(sim_pager, -np.inf)

        for k in k_list:
            if k >= m:
                continue
            nn_go = np.argpartition(-sim_go, kth=k, axis=1)[:, :k]
            nn_pager = np.argpartition(-sim_pager, kth=k, axis=1)[:, :k]

            overlaps = [
                len(set(a.tolist()) & set(b.tolist())) / k
                for a, b in zip(nn_go, nn_pager, strict=False)
            ]
            observed = float(np.mean(overlaps))
            expected = k / max((m - 1), 1)

            rows.append(
                {
                    "seed": seed,
                    "sample_size": m,
                    "k": int(k),
                    "overlap_mean": observed,
                    "random_expectation": expected,
                    "overlap_vs_random_ratio": observed / max(expected, 1e-12),
                }
            )

    return pd.DataFrame(rows)


def maybe_subsample_for_clustering(
    go_norm: np.ndarray,
    pager_norm: np.ndarray,
    cluster_sample_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    n = go_norm.shape[0]
    if cluster_sample_size <= 0 or n <= cluster_sample_size:
        return go_norm, pager_norm, n

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=cluster_sample_size, replace=False)
    return go_norm[idx], pager_norm[idx], cluster_sample_size


def compute_cluster_agreement(
    go_norm: np.ndarray,
    pager_norm: np.ndarray,
    k_list: list[int],
    seeds: list[int],
    cluster_sample_size: int,
) -> pd.DataFrame:
    go_x, pager_x, used_n = maybe_subsample_for_clustering(
        go_norm, pager_norm, cluster_sample_size, seed=1234
    )

    rows: list[dict[str, float | int]] = []
    for k in k_list:
        for seed in seeds:
            labels_go = KMeans(
                n_clusters=k, random_state=seed, n_init="auto"
            ).fit_predict(go_x)
            labels_pager = KMeans(
                n_clusters=k, random_state=seed, n_init="auto"
            ).fit_predict(pager_x)
            ari = adjusted_rand_score(labels_go, labels_pager)
            rows.append(
                {
                    "k": int(k),
                    "seed": int(seed),
                    "n_samples": int(used_n),
                    "ari": float(ari),
                }
            )

    return pd.DataFrame(rows)


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def plot_row_cosine_hist(row_cos: np.ndarray, out_png: str) -> None:
    ensure_parent(out_png)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 5.2))
    sns.histplot(row_cos, bins=80, stat="density", color="#4C72B0", alpha=0.8)
    plt.axvline(float(np.mean(row_cos)), color="#C44E52", linestyle="--", linewidth=2)
    plt.title("GO vs PAGER Per-Entity Cosine Similarity")
    plt.xlabel("cosine(GO_i, PAGER_i)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def plot_pairwise_scatter(
    scatter_x: np.ndarray,
    scatter_y: np.ndarray,
    pairwise_stats: pd.DataFrame,
    out_png: str,
) -> None:
    ensure_parent(out_png)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7.5, 6.2))
    plt.hexbin(scatter_x, scatter_y, gridsize=65, cmap="viridis", mincnt=1)
    plt.colorbar(label="Pair count")
    rho_mean = pairwise_stats["spearman_pairwise_cosine"].mean()
    plt.title(f"Pairwise Geometry Agreement (mean Spearman={rho_mean:.3f})")
    plt.xlabel("GO pairwise cosine")
    plt.ylabel("PAGER pairwise cosine")
    plt.tight_layout()
    plt.savefig(out_png, dpi=230, bbox_inches="tight")
    plt.close()


def plot_neighbor_overlap(overlap_df: pd.DataFrame, out_png: str) -> None:
    ensure_parent(out_png)
    agg = (
        overlap_df.groupby("k", as_index=False)
        .agg(
            overlap_mean=("overlap_mean", "mean"),
            overlap_std=("overlap_mean", "std"),
            random_expectation=("random_expectation", "mean"),
        )
        .sort_values("k")
    )
    agg["overlap_std"] = agg["overlap_std"].fillna(0.0)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8.8, 5.5))
    x = agg["k"].to_numpy()
    y = agg["overlap_mean"].to_numpy()
    std = agg["overlap_std"].to_numpy()
    rand = agg["random_expectation"].to_numpy()

    plt.plot(x, y, marker="o", linewidth=2.2, color="#55A868", label="Observed overlap")
    plt.fill_between(x, y - std, y + std, alpha=0.2, color="#55A868")
    plt.plot(
        x,
        rand,
        marker="s",
        linewidth=2,
        linestyle="--",
        color="#C44E52",
        label="Random expectation",
    )
    plt.title("Cross-Modal Neighbor Overlap")
    plt.xlabel("Top-k neighbors")
    plt.ylabel("Overlap fraction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def plot_cluster_ari(cluster_df: pd.DataFrame, out_png: str) -> None:
    ensure_parent(out_png)
    agg = (
        cluster_df.groupby("k", as_index=False)
        .agg(ari_mean=("ari", "mean"), ari_std=("ari", "std"))
        .sort_values("k")
    )
    agg["ari_std"] = agg["ari_std"].fillna(0.0)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8.8, 5.5))
    x = agg["k"].to_numpy()
    y = agg["ari_mean"].to_numpy()
    std = agg["ari_std"].to_numpy()
    plt.plot(x, y, marker="o", linewidth=2.2, color="#8172B2", label="ARI")
    plt.fill_between(x, y - std, y + std, alpha=0.2, color="#8172B2")
    plt.axhline(0.0, color="#444444", linewidth=1.2, linestyle=":")
    plt.title("GO vs PAGER Cluster Agreement (kmeans ARI)")
    plt.xlabel("k")
    plt.ylabel("Adjusted Rand Index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def build_summary_row(
    n_common: int,
    dim: int,
    row_cos: np.ndarray,
    pairwise_stats: pd.DataFrame,
    overlap_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
) -> pd.DataFrame:
    overlap_agg = (
        overlap_df.groupby("k", as_index=False)
        .agg(
            overlap_mean=("overlap_mean", "mean"),
            random_expectation=("random_expectation", "mean"),
        )
        .sort_values("k")
    )
    overlap10 = overlap_agg[overlap_agg["k"] == 10]

    ari_agg = (
        cluster_df.groupby("k", as_index=False)
        .agg(ari_mean=("ari", "mean"))
        .sort_values("k")
    )
    ari10 = ari_agg[ari_agg["k"] == 10]

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_common_ids": int(n_common),
        "dim_used": int(dim),
        "row_cos_mean": float(np.mean(row_cos)),
        "row_cos_std": float(np.std(row_cos)),
        "row_cos_p05": float(np.percentile(row_cos, 5)),
        "row_cos_p50": float(np.percentile(row_cos, 50)),
        "row_cos_p95": float(np.percentile(row_cos, 95)),
        "pairwise_spearman_mean": float(
            pairwise_stats["spearman_pairwise_cosine"].mean()
        ),
        "pairwise_spearman_std": float(
            pairwise_stats["spearman_pairwise_cosine"].std()
        ),
        "pairwise_spearman_min": float(
            pairwise_stats["spearman_pairwise_cosine"].min()
        ),
        "pairwise_spearman_max": float(
            pairwise_stats["spearman_pairwise_cosine"].max()
        ),
        "pairwise_pearson_mean": float(
            pairwise_stats["pearson_pairwise_cosine"].mean()
        ),
        "pairwise_pearson_std": float(pairwise_stats["pearson_pairwise_cosine"].std()),
        "nn_overlap_overall_mean": float(overlap_df["overlap_mean"].mean()),
        "nn_overlap_at10_mean": float(overlap10["overlap_mean"].iloc[0])
        if not overlap10.empty
        else np.nan,
        "nn_overlap_at10_random": float(overlap10["random_expectation"].iloc[0])
        if not overlap10.empty
        else np.nan,
        "nn_overlap_at10_ratio": float(
            overlap10["overlap_mean"].iloc[0]
            / max(overlap10["random_expectation"].iloc[0], 1e-12)
        )
        if not overlap10.empty
        else np.nan,
        "ari_overall_mean": float(cluster_df["ari"].mean()),
        "ari_overall_std": float(cluster_df["ari"].std()),
        "ari_at10_mean": float(ari10["ari_mean"].iloc[0])
        if not ari10.empty
        else np.nan,
    }
    return pd.DataFrame([summary])


def write_readout(
    readout_md: str,
    summary_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
) -> None:
    ensure_parent(readout_md)
    s = summary_df.iloc[0]

    overlap_agg = (
        overlap_df.groupby("k", as_index=False)
        .agg(
            overlap_mean=("overlap_mean", "mean"),
            overlap_std=("overlap_mean", "std"),
            random_expectation=("random_expectation", "mean"),
        )
        .sort_values("k")
    )
    overlap_agg["overlap_std"] = overlap_agg["overlap_std"].fillna(0.0)

    cluster_agg = (
        cluster_df.groupby("k", as_index=False)
        .agg(ari_mean=("ari", "mean"), ari_std=("ari", "std"))
        .sort_values("k")
    )
    cluster_agg["ari_std"] = cluster_agg["ari_std"].fillna(0.0)

    overlap_lines = [
        f"- k={int(r.k)}: overlap={r.overlap_mean:.4f} +/- {r.overlap_std:.4f}, random={r.random_expectation:.4f}"
        for r in overlap_agg.itertuples(index=False)
    ]
    ari_lines = [
        f"- k={int(r.k)}: ARI={r.ari_mean:.6f} +/- {r.ari_std:.6f}"
        for r in cluster_agg.itertuples(index=False)
    ]

    text = "\n".join(
        [
            "# Modality Consistency Readout",
            "",
            "## Headline",
            "",
            "GO and PAGER embeddings show weak geometric agreement, consistent with a modality-conflict explanation for the midpoint fusion dip.",
            "",
            "## Key numbers",
            "",
            f"- Shared IDs: {int(s['n_common_ids'])}",
            f"- Per-entity cosine mean: {s['row_cos_mean']:.4f} (std {s['row_cos_std']:.4f})",
            f"- Pairwise geometry Spearman: {s['pairwise_spearman_mean']:.4f} (std {s['pairwise_spearman_std']:.4f})",
            f"- Neighbor overlap@10: {s['nn_overlap_at10_mean']:.4f} vs random {s['nn_overlap_at10_random']:.4f}",
            f"- kmeans ARI at k=10: {s['ari_at10_mean']:.6f}",
            "",
            "## Neighbor overlap by k",
            "",
            *overlap_lines,
            "",
            "## Cluster agreement by k",
            "",
            *ari_lines,
            "",
            "## Interpretation",
            "",
            "- The two modalities preserve different local/global neighborhoods.",
            "- Equal linear fusion can blend conflicting neighborhoods, reducing separability.",
            "- This supports using endpoint-dominant or learned fusion instead of naive 50/50 mixing.",
            "",
        ]
    )
    with open(readout_md, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute GO-vs-PAGER modality consistency metrics and figures."
    )
    parser.add_argument(
        "--go_embedding_file",
        default="data/RummaGEO/rummageo_go_embeddings.npz",
    )
    parser.add_argument(
        "--pager_embedding_file",
        default="data/RummaGEO/rummageo_pager_embeddings.npz",
    )
    parser.add_argument("--cut_dim", type=int, default=384)
    parser.add_argument("--pairwise_sample_size", type=int, default=1800)
    parser.add_argument("--pairwise_seeds", default="7,11,19,23,31")
    parser.add_argument("--neighbor_k_list", default="5,10,20,50")
    parser.add_argument("--cluster_k_list", default="10,20,30,40,50")
    parser.add_argument("--cluster_seeds", default="11,22,33")
    parser.add_argument(
        "--cluster_sample_size",
        type=int,
        default=0,
        help="0 means use full aligned dataset",
    )
    parser.add_argument("--scatter_points", type=int, default=45000)

    parser.add_argument(
        "--summary_csv",
        default="work_alpha_gnn_20260212/reports/baseline/modality_consistency_summary.csv",
    )
    parser.add_argument(
        "--pairwise_stats_csv",
        default="work_alpha_gnn_20260212/reports/baseline/modality_consistency_pairwise_stats.csv",
    )
    parser.add_argument(
        "--nn_overlap_csv",
        default="work_alpha_gnn_20260212/reports/baseline/modality_consistency_nn_overlap_by_k.csv",
    )
    parser.add_argument(
        "--cluster_agreement_csv",
        default="work_alpha_gnn_20260212/reports/baseline/modality_consistency_cluster_agreement.csv",
    )
    parser.add_argument(
        "--readout_md",
        default="work_alpha_gnn_20260212/reports/baseline/modality_consistency_readout.md",
    )

    parser.add_argument(
        "--row_hist_png",
        default="work_alpha_gnn_20260212/artifacts/plots/modality_row_cosine_hist.png",
    )
    parser.add_argument(
        "--pairwise_scatter_png",
        default="work_alpha_gnn_20260212/artifacts/plots/modality_pairwise_similarity_scatter.png",
    )
    parser.add_argument(
        "--nn_overlap_png",
        default="work_alpha_gnn_20260212/artifacts/plots/modality_nn_overlap_vs_k.png",
    )
    parser.add_argument(
        "--cluster_ari_png",
        default="work_alpha_gnn_20260212/artifacts/plots/modality_cluster_ari_vs_k.png",
    )
    args = parser.parse_args()

    pairwise_seeds = parse_int_list(args.pairwise_seeds)
    neighbor_k_list = parse_int_list(args.neighbor_k_list)
    cluster_k_list = parse_int_list(args.cluster_k_list)
    cluster_seeds = parse_int_list(args.cluster_seeds)

    common_ids, go_norm, pager_norm = align_embeddings(
        args.go_embedding_file, args.pager_embedding_file, args.cut_dim
    )
    row_cos = np.sum(go_norm * pager_norm, axis=1)

    pairwise_stats, scatter_x, scatter_y = compute_pairwise_geometry_stats(
        go_norm,
        pager_norm,
        args.pairwise_sample_size,
        pairwise_seeds,
        args.scatter_points,
    )
    overlap_df = compute_neighbor_overlap(
        go_norm,
        pager_norm,
        args.pairwise_sample_size,
        pairwise_seeds,
        neighbor_k_list,
    )
    cluster_df = compute_cluster_agreement(
        go_norm,
        pager_norm,
        cluster_k_list,
        cluster_seeds,
        args.cluster_sample_size,
    )
    summary_df = build_summary_row(
        n_common=common_ids.shape[0],
        dim=args.cut_dim,
        row_cos=row_cos,
        pairwise_stats=pairwise_stats,
        overlap_df=overlap_df,
        cluster_df=cluster_df,
    )

    for out in [
        args.summary_csv,
        args.pairwise_stats_csv,
        args.nn_overlap_csv,
        args.cluster_agreement_csv,
    ]:
        ensure_parent(out)

    summary_df.to_csv(args.summary_csv, index=False)
    pairwise_stats.to_csv(args.pairwise_stats_csv, index=False)
    overlap_df.to_csv(args.nn_overlap_csv, index=False)
    cluster_df.to_csv(args.cluster_agreement_csv, index=False)

    plot_row_cosine_hist(row_cos, args.row_hist_png)
    plot_pairwise_scatter(
        scatter_x, scatter_y, pairwise_stats, args.pairwise_scatter_png
    )
    plot_neighbor_overlap(overlap_df, args.nn_overlap_png)
    plot_cluster_ari(cluster_df, args.cluster_ari_png)
    write_readout(args.readout_md, summary_df, overlap_df, cluster_df)

    print(f"Wrote summary: {args.summary_csv}")
    print(f"Wrote pairwise stats: {args.pairwise_stats_csv}")
    print(f"Wrote overlap by k: {args.nn_overlap_csv}")
    print(f"Wrote cluster agreement: {args.cluster_agreement_csv}")
    print(f"Wrote readout: {args.readout_md}")
    print(f"Wrote plot: {args.row_hist_png}")
    print(f"Wrote plot: {args.pairwise_scatter_png}")
    print(f"Wrote plot: {args.nn_overlap_png}")
    print(f"Wrote plot: {args.cluster_ari_png}")


if __name__ == "__main__":
    main()
