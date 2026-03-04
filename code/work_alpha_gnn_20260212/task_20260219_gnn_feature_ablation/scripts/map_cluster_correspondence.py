"""
map_cluster_correspondence.py

Side-by-side matched-color UMAP for two GNN embedding variants.

For each subset, clusters in both embeddings are matched via the Hungarian
algorithm (maximising overlap count) so that corresponding clusters share the
same tab10 color across panels.
"""

import argparse
import os
import pathlib
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
# parents[3] from __file__ = project root (script → scripts/ → task_…/ → work_alpha…/ → GoldenF/)
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
_WORK_SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"  # work_alpha…/scripts/

for _p in (_WORK_SCRIPTS, _PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from plot_embedding_umap_scatter import load_embeddings, run_umap  # noqa: E402


# ---------------------------------------------------------------------------
# K-means helpers
# ---------------------------------------------------------------------------

def best_kmeans(emb: np.ndarray, k: int, seeds: list[int]) -> tuple[np.ndarray, float, int]:
    """Run k-means with multiple seeds; return labels from run with lowest inertia."""
    from sklearn.cluster import KMeans

    best_labels: np.ndarray | None = None
    best_inertia = float("inf")
    best_seed = seeds[0]
    for seed in seeds:
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        km.fit(emb)
        if km.inertia_ < best_inertia:
            best_inertia = km.inertia_
            best_labels = km.labels_.copy()
            best_seed = seed
    assert best_labels is not None
    return best_labels, best_inertia, best_seed


# ---------------------------------------------------------------------------
# Inner-join alignment
# ---------------------------------------------------------------------------

def inner_join_embeddings(
    ids_a: np.ndarray,
    emb_a: np.ndarray,
    ids_b: np.ndarray,
    emb_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (common_ids, emb_a_aligned, emb_b_aligned) in id-sorted order."""
    set_a = set(ids_a.tolist())
    set_b = set(ids_b.tolist())
    common = sorted(set_a & set_b)
    if not common:
        raise ValueError("No common IDs between the two embeddings.")

    idx_a = {v: i for i, v in enumerate(ids_a.tolist())}
    idx_b = {v: i for i, v in enumerate(ids_b.tolist())}
    rows_a = [idx_a[c] for c in common]
    rows_b = [idx_b[c] for c in common]

    common_ids = np.asarray(common)
    return common_ids, emb_a[rows_a], emb_b[rows_b]


# ---------------------------------------------------------------------------
# Overlap matrix and Hungarian matching
# ---------------------------------------------------------------------------

def build_overlap_matrix(labels_a: np.ndarray, labels_b: np.ndarray, k: int) -> np.ndarray:
    """M[i][j] = number of gene sets in cluster i (A) and cluster j (B)."""
    M = np.zeros((k, k), dtype=np.int32)
    for la, lb in zip(labels_a, labels_b):
        M[int(la), int(lb)] += 1
    return M


def hungarian_match(M: np.ndarray) -> dict[int, int]:
    """Return match_dict: B-cluster → A-cluster (maximise overlap)."""
    row_ind, col_ind = linear_sum_assignment(-M)
    # row_ind are A indices, col_ind are B indices
    return {int(b): int(a) for a, b in zip(row_ind, col_ind)}


def remap_labels(labels_b: np.ndarray, match_dict: dict[int, int]) -> np.ndarray:
    """Remap labels_b so matched clusters share label integers with labels_a."""
    remapped = np.empty_like(labels_b)
    for i, lb in enumerate(labels_b):
        remapped[i] = match_dict.get(int(lb), int(lb))
    return remapped


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _get_cmap(k: int):
    if k <= 10:
        return plt.get_cmap("tab10")
    return plt.get_cmap("tab20")


def render_panels(
    coords_a: np.ndarray,
    labels_a: np.ndarray,
    coords_b: np.ndarray,
    remapped_b: np.ndarray,
    k: int,
    subset: str,
    mean_overlap_pct: float,
    output_path: str,
) -> None:
    cmap = _get_cmap(k)
    colors_a = cmap(labels_a % cmap.N)
    colors_b = cmap(remapped_b % cmap.N)

    fig = plt.figure(figsize=(13, 6.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[5.5, 0.7], hspace=0.08)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_leg = fig.add_subplot(gs[1, :])

    scatter_kw = dict(s=6, alpha=0.7, linewidths=0, rasterized=True)

    ax_a.scatter(coords_a[:, 0], coords_a[:, 1], c=colors_a, **scatter_kw)
    ax_a.set_title(f"Text-only GNN (subset{subset})  k={k}")
    ax_a.set_xticks([])
    ax_a.set_yticks([])

    ax_b.scatter(coords_b[:, 0], coords_b[:, 1], c=colors_b, **scatter_kw)
    ax_b.set_title(f"Text+structlite GNN (subset{subset})  k={k}")
    ax_b.set_xticks([])
    ax_b.set_yticks([])

    patches = [
        mpatches.Patch(color=cmap(i % cmap.N), label=f"Cluster {i}")
        for i in range(k)
    ]
    ax_leg.legend(
        handles=patches, ncol=k, loc="center",
        frameon=False, fontsize=8, handlelength=1.2,
    )
    ax_leg.axis("off")

    fig.suptitle(
        f"Cluster correspondence  |  subset {subset}  |  mean overlap = {mean_overlap_pct:.1f}%",
        fontsize=11, y=0.99,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {output_path}")


# ---------------------------------------------------------------------------
# Correspondence table
# ---------------------------------------------------------------------------

def _print_correspondence(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    match_dict: dict[int, int],
    M: np.ndarray,
    subset: str,
) -> float:
    """Print per-cluster table; return mean overlap %."""
    n = len(labels_a)
    print(f"\n=== Subset {subset} cluster correspondence ===")
    print(f"{'B-cluster':>10}  {'→ A-cluster':>12}  {'count_A':>8}  {'count_B':>8}  {'overlap':>8}  {'overlap%':>9}")
    print("-" * 65)
    overlap_pcts = []
    for b_cl, a_cl in sorted(match_dict.items(), key=lambda kv: kv[1]):
        cnt_a = int(np.sum(labels_a == a_cl))
        cnt_b = int(np.sum(labels_b == b_cl))
        overlap = int(M[a_cl, b_cl])
        denom = max(cnt_a, cnt_b, 1)
        pct = 100.0 * overlap / denom
        overlap_pcts.append(pct)
        print(f"{b_cl:>10}  {a_cl:>12}  {cnt_a:>8}  {cnt_b:>8}  {overlap:>8}  {pct:>8.1f}%")
    mean_pct = float(np.mean(overlap_pcts)) if overlap_pcts else 0.0
    print(f"\n  Mean overlap: {mean_pct:.1f}%  (n={n} gene sets)")
    return mean_pct


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_npz_paths(run_dir: str, subset: str) -> tuple[str, str]:
    """Return (text_only_path, structlite_path) for a given subset string."""
    base = pathlib.Path(run_dir)
    path_a = base / f"subset{subset}" / "inputs" / f"gnn_old_holdout_subset_{subset}_human.npz"
    path_b = base / f"subset{subset}" / "features" / f"text_structlite_gnn_holdout_subset{subset}.npz"
    for p in (path_a, path_b):
        if not p.exists():
            raise FileNotFoundError(f"NPZ not found: {p}")
    return str(path_a), str(path_b)


# ---------------------------------------------------------------------------
# Per-subset processing
# ---------------------------------------------------------------------------

def process_subset(run_dir: str, subset: str, args) -> None:
    path_a, path_b = resolve_npz_paths(run_dir, subset)

    ids_a, emb_a = load_embeddings(path_a)
    ids_b, emb_b = load_embeddings(path_b)

    common_ids, emb_a_al, emb_b_al = inner_join_embeddings(ids_a, emb_a, ids_b, emb_b)
    print(f"\nSubset {subset}: {len(common_ids)} common gene sets (A={len(ids_a)}, B={len(ids_b)})")

    seeds = list(range(11, 16))
    labels_a, inertia_a, seed_a = best_kmeans(emb_a_al, args.k, seeds)
    labels_b, inertia_b, seed_b = best_kmeans(emb_b_al, args.k, seeds)
    print(f"  K-means A: best_seed={seed_a}, inertia={inertia_a:.4f}")
    print(f"  K-means B: best_seed={seed_b}, inertia={inertia_b:.4f}")

    M = build_overlap_matrix(labels_a, labels_b, args.k)
    match_dict = hungarian_match(M)
    remapped_b = remap_labels(labels_b, match_dict)

    umap_kw = dict(n_neighbors=15, min_dist=0.1, metric="cosine", seed=args.seed)
    coords_a = run_umap(emb_a_al, **umap_kw)
    coords_b = run_umap(emb_b_al, **umap_kw)

    mean_pct = _print_correspondence(labels_a, labels_b, match_dict, M, subset)

    output_path = os.path.join(
        args.output_dir, f"cluster_correspondence_subset{subset}.png"
    )
    render_panels(
        coords_a, labels_a,
        coords_b, remapped_b,
        args.k, subset, mean_pct, output_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matched-color side-by-side UMAP for GNN cluster correspondence."
    )
    parser.add_argument("--run_dir", required=True, help="Path to the run directory")
    parser.add_argument("--subsets", nargs="+", default=["085", "090"], help="Subset IDs to process")
    parser.add_argument("--k", type=int, default=10, help="Number of k-means clusters")
    parser.add_argument("--output_dir", required=True, help="Directory for output PNGs")
    parser.add_argument("--seed", type=int, default=42, help="UMAP random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for subset in args.subsets:
        process_subset(args.run_dir, subset, args)


if __name__ == "__main__":
    main()
