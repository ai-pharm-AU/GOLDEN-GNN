import argparse
import colorsys
import json
import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.data_loader import DataLoader

try:
    import umap  # type: ignore
except ImportError:  # pragma: no cover
    umap = None


def load_embeddings(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    ids, emb = DataLoader.load_embeddings(npz_path)
    return np.asarray(ids).astype(str), np.asarray(emb)


def compute_labels(emb: np.ndarray, k: int, seed: int) -> np.ndarray:
    return KMeans(n_clusters=k, random_state=seed, n_init="auto").fit_predict(emb)


def run_umap(
    emb: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    seed: int,
) -> np.ndarray:
    if umap is not None:
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=seed,
            low_memory=True,
        )
        return reducer.fit_transform(emb)
    # Fallback keeps plots available when umap-learn is not installed.
    return PCA(n_components=2, random_state=seed).fit_transform(emb)


def build_cmap(k: int) -> ListedColormap:
    base = plt.get_cmap("tab10")
    colors = [base(i % 10) for i in range(k)]
    return ListedColormap(colors, name="cluster_tab10")


def jaccard(a: set[str], b: set[str]) -> float:
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def overlap_coeff(a: set[str], b: set[str]) -> float:
    denom = min(len(a), len(b))
    if denom == 0:
        return 0.0
    return len(a & b) / denom


def load_cluster_gene_sets(cluster_json_path: str) -> tuple[dict[int, set[str]], dict[str, int]]:
    with open(cluster_json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    clusters_raw = payload.get("clusters", {})
    cluster_gene_sets: dict[int, set[str]] = {}
    gs_to_cluster: dict[str, int] = {}
    for cluster_key, items in clusters_raw.items():
        cid = int(cluster_key)
        gene_set: set[str] = set()
        for item in items:
            gs_id = str(item.get("gs_id", ""))
            if gs_id:
                gs_to_cluster[gs_id] = cid
            for gene in item.get("genes", []):
                gene_set.add(str(gene))
        cluster_gene_sets[cid] = gene_set
    return cluster_gene_sets, gs_to_cluster


def build_cluster_similarity(
    cluster_gene_sets: dict[int, set[str]],
    metric: str,
) -> tuple[list[int], np.ndarray]:
    metric_fn = jaccard if metric == "jaccard" else overlap_coeff
    cluster_ids = sorted(cluster_gene_sets)
    n = len(cluster_ids)
    sim = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            a = cluster_gene_sets[cluster_ids[i]]
            b = cluster_gene_sets[cluster_ids[j]]
            val = metric_fn(a, b)
            sim[i, j] = val
            sim[j, i] = val
    return cluster_ids, sim


def embed_similarity_to_colors(
    cluster_ids: list[int],
    sim: np.ndarray,
    method: str,
    seed: int,
    color_style: str,
    gradient_cmap: str,
) -> dict[int, tuple[float, float, float, float]]:
    if method != "mds":
        raise ValueError(f"Unsupported color embedding method: {method}")
    if not cluster_ids:
        return {}
    if len(cluster_ids) == 1:
        return {cluster_ids[0]: (0.121, 0.466, 0.705, 1.0)}

    dist = np.clip(1.0 - sim, 0.0, 1.0)
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=seed,
        n_init=8,
    )
    coords = mds.fit_transform(dist)
    centered = coords - coords.mean(axis=0, keepdims=True)
    angles = np.arctan2(centered[:, 1], centered[:, 0])
    hues = (angles + np.pi) / (2.0 * np.pi)

    color_map: dict[int, tuple[float, float, float, float]] = {}
    if color_style == "gradient":
        x = coords[:, 0]
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        if x_max - x_min < 1e-12:
            x_norm = np.full_like(x, 0.5, dtype=np.float64)
        else:
            x_norm = (x - x_min) / (x_max - x_min)
        cmap = plt.get_cmap(gradient_cmap)
        for idx, cid in enumerate(cluster_ids):
            color_map[cid] = tuple(float(v) for v in cmap(float(x_norm[idx])))
    else:
        for idx, cid in enumerate(cluster_ids):
            r, g, b = colorsys.hsv_to_rgb(float(hues[idx]), 0.70, 0.92)
            color_map[cid] = (r, g, b, 1.0)
    return color_map


def majority_map_labels_to_clusters(
    ids: np.ndarray,
    labels: np.ndarray,
    gs_to_cluster: dict[str, int],
) -> dict[int, int]:
    mapping: dict[int, int] = {}
    unique_labels = np.unique(labels).tolist()
    for raw_label in unique_labels:
        lbl = int(raw_label)
        mask = labels == raw_label
        counts: dict[int, int] = {}
        for gs_id in ids[mask]:
            cid = gs_to_cluster.get(str(gs_id))
            if cid is None:
                continue
            counts[cid] = counts.get(cid, 0) + 1
        if counts:
            best_cluster = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
            mapping[lbl] = best_cluster
    return mapping


def build_overlap_point_colors(
    ids: np.ndarray,
    labels: np.ndarray,
    k: int,
    cluster_json_path: str,
    overlap_metric: str,
    color_embed_method: str,
    color_seed: int,
    overlap_color_style: str,
    overlap_gradient_cmap: str,
) -> tuple[np.ndarray, dict]:
    cluster_gene_sets, gs_to_cluster = load_cluster_gene_sets(cluster_json_path)
    cluster_ids, sim = build_cluster_similarity(cluster_gene_sets, overlap_metric)
    cluster_color_map = embed_similarity_to_colors(
        cluster_ids,
        sim,
        color_embed_method,
        color_seed,
        overlap_color_style,
        overlap_gradient_cmap,
    )
    label_to_cluster = majority_map_labels_to_clusters(ids, labels, gs_to_cluster)

    base_colors = list(plt.get_cmap("tab10").colors)
    point_colors = np.zeros((labels.shape[0], 4), dtype=np.float64)
    for i, lbl in enumerate(labels):
        label_i = int(lbl)
        mapped_cluster = label_to_cluster.get(label_i)
        if mapped_cluster is not None and mapped_cluster in cluster_color_map:
            point_colors[i] = cluster_color_map[mapped_cluster]
        else:
            rgb = base_colors[label_i % len(base_colors)]
            point_colors[i] = (rgb[0], rgb[1], rgb[2], 1.0)
    info = {
        "cluster_ids": cluster_ids,
        "similarity": sim,
        "label_to_cluster": label_to_cluster,
    }
    return point_colors, info


def annotate_overlap_text(
    ax: plt.Axes,
    coords: np.ndarray,
    labels: np.ndarray,
    info: dict | None,
    top_n_pairs: int,
) -> None:
    if not info:
        return
    cluster_ids = info.get("cluster_ids")
    sim = info.get("similarity")
    label_to_cluster = info.get("label_to_cluster")
    if cluster_ids is None or sim is None or label_to_cluster is None:
        return
    if len(cluster_ids) == 0:
        return

    cid_to_idx = {int(cid): idx for idx, cid in enumerate(cluster_ids)}
    centroids: dict[int, np.ndarray] = {}
    for raw_label, raw_cluster in label_to_cluster.items():
        lbl = int(raw_label)
        cid = int(raw_cluster)
        mask = labels == lbl
        if not np.any(mask):
            continue
        centroid = coords[mask].mean(axis=0)
        centroids[cid] = centroid
        ax.text(
            float(centroid[0]),
            float(centroid[1]),
            f"C{cid}",
            fontsize=8,
            ha="center",
            va="center",
            color="black",
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.80),
            zorder=6,
        )

    pair_rows: list[tuple[float, int, int]] = []
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            pair_rows.append((float(sim[i, j]), int(cluster_ids[i]), int(cluster_ids[j])))
    pair_rows.sort(key=lambda x: x[0], reverse=True)
    top_rows = pair_rows[: max(0, top_n_pairs)]
    if not top_rows:
        return
    x = coords[:, 0]
    y = coords[:, 1]
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    x_span = max(1e-6, x_max - x_min)
    y_span = max(1e-6, y_max - y_min)

    text_x = x_max + 0.22 * x_span
    y_positions = np.linspace(y_max - 0.07 * y_span, y_min + 0.07 * y_span, num=len(top_rows))
    ax.set_xlim(x_min - 0.04 * x_span, x_max + 0.40 * x_span)

    for row_idx, (ov, a, b) in enumerate(top_rows):
        ca = centroids.get(a)
        cb = centroids.get(b)
        if ca is None or cb is None:
            continue
        target = (ca + cb) / 2.0
        text_y = float(y_positions[row_idx])
        ax.annotate(
            f"C{a}-C{b}: {ov:.2f}",
            xy=(float(target[0]), float(target[1])),
            xytext=(float(text_x), text_y),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=7,
            color="black",
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.85),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=0.7, shrinkA=2, shrinkB=2),
            zorder=7,
        )


def scatter_axis(
    ax: plt.Axes,
    coords: np.ndarray,
    labels: np.ndarray,
    k: int,
    title: str,
    point_size: float,
    alpha: float,
    point_colors: np.ndarray | None = None,
) -> None:
    if point_colors is None:
        cmap = build_cmap(k)
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=labels,
            s=point_size,
            cmap=cmap,
            alpha=alpha,
            linewidths=0,
        )
    else:
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=point_colors,
            s=point_size,
            alpha=alpha,
            linewidths=0,
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def write_readme(
    path: str,
    emb_paths: list[str],
    k: int,
    seed: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    color_mode: str,
    overlap_metric: str,
    color_embed_method: str,
    overlap_color_style: str,
    overlap_gradient_cmap: str,
    show_overlap_text: bool,
    overlap_text_top_n: int,
    cluster_json_paths: list[str],
    projection_method: str,
) -> None:
    lines = [
        "# Embedding Scatter Plot Parameters",
        "",
        f"- embeddings: {', '.join(emb_paths)}",
        f"- kmeans k: {k}",
        f"- kmeans seed: {seed}",
        f"- projection method: {projection_method}",
        f"- umap n_neighbors: {n_neighbors}",
        f"- umap min_dist: {min_dist}",
        f"- umap metric: {metric}",
        f"- color mode: {color_mode}",
    ]
    if color_mode == "superpag_overlap":
        lines.extend(
            [
                f"- overlap metric: {overlap_metric}",
                f"- color embedding: {color_embed_method}",
                f"- overlap color style: {overlap_color_style}",
                f"- overlap gradient cmap: {overlap_gradient_cmap}",
                f"- show overlap text: {show_overlap_text}",
                f"- top overlap text lines: {overlap_text_top_n}",
                f"- cluster gene json: {', '.join(cluster_json_paths)}",
                "",
                "Note: colors encode super-PAG similarity from cluster-level gene overlap.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Note: colors reflect per-embedding kmeans labels and are not aligned across embeddings.",
            ]
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def parse_compare_titles(raw: str) -> list[str]:
    if not raw:
        return []
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UMAP scatter plot for embeddings (single or side-by-side)."
    )
    parser.add_argument("--embedding_npz", required=True)
    parser.add_argument("--compare_npz", default="")
    parser.add_argument("--output_png", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--umap_neighbors", type=int, default=20)
    parser.add_argument("--umap_min_dist", type=float, default=0.15)
    parser.add_argument("--umap_metric", default="cosine")
    parser.add_argument("--title", default="")
    parser.add_argument("--compare_titles", default="")
    parser.add_argument("--point_size", type=float, default=6.0)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--color_mode", default="label", choices=["label", "superpag_overlap"])
    parser.add_argument(
        "--overlap_metric",
        default="jaccard",
        choices=["jaccard", "overlap_coeff"],
    )
    parser.add_argument("--color_embed_method", default="mds", choices=["mds"])
    parser.add_argument("--color_seed", type=int, default=11)
    parser.add_argument(
        "--overlap_color_style",
        default="hue",
        choices=["hue", "gradient"],
    )
    parser.add_argument("--overlap_gradient_cmap", default="turbo")
    parser.add_argument("--show_overlap_text", action="store_true")
    parser.add_argument("--overlap_text_top_n", type=int, default=5)
    parser.add_argument("--cluster_genes_json_a", default="")
    parser.add_argument("--cluster_genes_json_b", default="")
    parser.add_argument("--write_readme", action="store_true")
    parser.add_argument("--readme_path", default="")
    args = parser.parse_args()
    projection_method = "umap"
    if umap is None:
        projection_method = "pca_fallback"
        print(
            "Warning: umap-learn not found; using PCA fallback for 2D projection.",
            file=sys.stderr,
        )

    out_dir = os.path.dirname(args.output_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ids_a, emb_a = load_embeddings(args.embedding_npz)
    labels_a = compute_labels(emb_a, args.k, args.seed)
    coords_a = run_umap(
        emb_a,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        seed=args.seed,
    )
    point_colors_a = None
    overlap_info_a = None
    if args.color_mode == "superpag_overlap":
        if not args.cluster_genes_json_a:
            raise SystemExit("--cluster_genes_json_a is required when --color_mode=superpag_overlap")
        point_colors_a, overlap_info_a = build_overlap_point_colors(
            ids=ids_a,
            labels=labels_a,
            k=args.k,
            cluster_json_path=args.cluster_genes_json_a,
            overlap_metric=args.overlap_metric,
            color_embed_method=args.color_embed_method,
            color_seed=args.color_seed,
            overlap_color_style=args.overlap_color_style,
            overlap_gradient_cmap=args.overlap_gradient_cmap,
        )

    cluster_json_paths: list[str] = [args.cluster_genes_json_a] if args.cluster_genes_json_a else []
    if args.compare_npz:
        ids_b, emb_b = load_embeddings(args.compare_npz)
        labels_b = compute_labels(emb_b, args.k, args.seed)
        coords_b = run_umap(
            emb_b,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            seed=args.seed,
        )
        point_colors_b = None
        overlap_info_b = None
        if args.color_mode == "superpag_overlap":
            if not args.cluster_genes_json_b:
                raise SystemExit(
                    "--cluster_genes_json_b is required in compare mode when --color_mode=superpag_overlap"
                )
            point_colors_b, overlap_info_b = build_overlap_point_colors(
                ids=ids_b,
                labels=labels_b,
                k=args.k,
                cluster_json_path=args.cluster_genes_json_b,
                overlap_metric=args.overlap_metric,
                color_embed_method=args.color_embed_method,
                color_seed=args.color_seed,
                overlap_color_style=args.overlap_color_style,
                overlap_gradient_cmap=args.overlap_gradient_cmap,
            )
            cluster_json_paths.append(args.cluster_genes_json_b)

        titles = parse_compare_titles(args.compare_titles)
        if len(titles) != 2:
            titles = [os.path.basename(args.embedding_npz), os.path.basename(args.compare_npz)]
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
        scatter_axis(
            axes[0],
            coords_a,
            labels_a,
            args.k,
            titles[0],
            args.point_size,
            args.alpha,
            point_colors_a,
        )
        if args.show_overlap_text and args.color_mode == "superpag_overlap":
            annotate_overlap_text(
                axes[0],
                coords_a,
                labels_a,
                overlap_info_a,
                args.overlap_text_top_n,
            )
        scatter_axis(
            axes[1],
            coords_b,
            labels_b,
            args.k,
            titles[1],
            args.point_size,
            args.alpha,
            point_colors_b,
        )
        if args.show_overlap_text and args.color_mode == "superpag_overlap":
            annotate_overlap_text(
                axes[1],
                coords_b,
                labels_b,
                overlap_info_b,
                args.overlap_text_top_n,
            )
        fig.tight_layout()
        fig.savefig(args.output_png, dpi=200)
        plt.close(fig)
        emb_paths = [args.embedding_npz, args.compare_npz]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.8))
        title = args.title or os.path.basename(args.embedding_npz)
        scatter_axis(
            ax,
            coords_a,
            labels_a,
            args.k,
            title,
            args.point_size,
            args.alpha,
            point_colors_a,
        )
        if args.show_overlap_text and args.color_mode == "superpag_overlap":
            annotate_overlap_text(
                ax,
                coords_a,
                labels_a,
                overlap_info_a,
                args.overlap_text_top_n,
            )
        fig.tight_layout()
        fig.savefig(args.output_png, dpi=200)
        plt.close(fig)
        emb_paths = [args.embedding_npz]

    if args.write_readme and args.readme_path:
        write_readme(
            args.readme_path,
            emb_paths,
            args.k,
            args.seed,
            args.umap_neighbors,
            args.umap_min_dist,
            args.umap_metric,
            args.color_mode,
            args.overlap_metric,
            args.color_embed_method,
            args.overlap_color_style,
            args.overlap_gradient_cmap,
            args.show_overlap_text,
            args.overlap_text_top_n,
            cluster_json_paths,
            projection_method,
        )

    print(f"Wrote: {args.output_png}")


if __name__ == "__main__":
    main()
