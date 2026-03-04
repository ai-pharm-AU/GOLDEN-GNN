# Cluster Correspondence Report & Legend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a shared cluster legend to the side-by-side UMAP PNGs and produce a Markdown report summarising the cluster correspondence findings for subsets 085 and 090.

**Architecture:** Modify `render_panels()` in the existing script to append a legend strip, then re-run to regenerate PNGs. Write a standalone Markdown report at `reports/cluster_correspondence_report.md` that embeds the updated images and the correspondence tables.

**Tech Stack:** Python 3.11, matplotlib, existing `.venv`. No new dependencies.

---

### Task 1: Add shared legend to render_panels()

**Files:**
- Modify: `work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/scripts/map_cluster_correspondence.py`

**Step 1: Open and read the current render_panels() function**

Read the file to locate the function (around line 105–140).

**Step 2: Replace render_panels() with the legend-aware version**

Replace the body of `render_panels()` so it:
- Creates `fig` with `figsize=(13, 6.5)` and `gridspec_kw={"height_ratios": [5, 0.5]}` (3-row: 2 scatter panels on top, legend strip on bottom spanning full width)
- Draws the two scatter panels as before (axes[0][0], axes[0][1])
- In the bottom axis (axes[1][0] spanning both cols via `subplot2grid` or just `fig.add_subplot`): draw k colored patches with labels "Cluster 0" … "Cluster k-1" using `matplotlib.patches.Patch`; call `ax_legend.legend(handles=patches, ncol=k, loc="center", frameon=False, fontsize=8)`; hide axis spines/ticks

Concrete replacement for `render_panels()`:

```python
import matplotlib.patches as mpatches

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
    ax_a.set_xticks([]); ax_a.set_yticks([])

    ax_b.scatter(coords_b[:, 0], coords_b[:, 1], c=colors_b, **scatter_kw)
    ax_b.set_title(f"Text+structlite GNN (subset{subset})  k={k}")
    ax_b.set_xticks([]); ax_b.set_yticks([])

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
```

**Step 3: Regenerate PNGs**

```bash
cd /home/zzz0054/GoldenF
.venv/bin/python \
  work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/scripts/map_cluster_correspondence.py \
  --run_dir work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260220_233515 \
  --subsets 085 090 --k 10 \
  --output_dir work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/reports/plots \
  --seed 42
```

Expected stdout: two "Wrote: …" lines, no errors.

**Step 4: Verify PNGs exist and are reasonable size**

```bash
ls -lh work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/reports/plots/cluster_correspondence_subset*.png
```

Expected: two PNGs, each ≥ 200 KB.

---

### Task 2: Write the Markdown report

**Files:**
- Create: `work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/reports/cluster_correspondence_report.md`

**Step 1: Write the report**

The report must contain:

```markdown
# GNN Cluster Correspondence: Text-only vs Text+structlite

**Date:** 2026-02-26
**Run:** `single_085_090_human_only_20260220_233515`
**Script:** `scripts/map_cluster_correspondence.py`

## Background

Two GNN embeddings trained on the same gene sets are compared:

| Embedding | Node features | NPZ path (subset N) |
|-----------|--------------|---------------------|
| **Text-only GNN** (`gnn_old_holdout`) | GO text embeddings | `inputs/gnn_old_holdout_subset_{N}_human.npz` |
| **Text+structlite GNN** | GO text + degree, clustering coeff, PageRank | `features/text_structlite_gnn_holdout_subset{N}.npz` |

The question: do the same gene sets cluster together across both spaces?

## Method

For each subset:

1. Load both embeddings; inner-join on GS ID.
2. K-means independently (k=10, seeds 11–15, best inertia selected).
3. Build k×k overlap matrix M where `M[i][j]` = gene sets in A-cluster i AND B-cluster j.
4. Hungarian algorithm (`scipy.optimize.linear_sum_assignment(-M)`) finds optimal 1-to-1 correspondence.
5. Remap B labels so matched clusters share the same integer (and color) as their A counterpart.
6. UMAP independently on each embedding (n_neighbors=15, min_dist=0.1, cosine, seed=42).
7. Render side-by-side scatter with tab10 — same color = same matched cluster.

## Results

### Subset 085 (n = 4 298 gene sets)

![Cluster correspondence subset 085](plots/cluster_correspondence_subset085.png)

K-means best seeds: A=12 (inertia 999.45), B=14 (inertia 1113.80).

| B-cluster | → A-cluster | count A | count B | overlap | overlap % |
|----------:|------------:|--------:|--------:|--------:|----------:|
| 0 | 0 | 464 | 484 | 137 | 28.3 % |
| 1 | 1 | 499 | 582 | 118 | 20.3 % |
| 5 | 2 | 410 | 229 | 167 | 40.7 % |
| 4 | 3 | 423 | 592 | 184 | 31.1 % |
| 9 | 4 | 344 | 347 |  58 | 16.7 % |
| 8 | 5 | 407 |  42 |   4 |  1.0 % |
| 2 | 6 | 498 | 442 |  73 | 14.7 % |
| 6 | 7 | 474 | 817 | 296 | 36.2 % |
| 7 | 8 | 261 | 268 | 181 | 67.5 % |
| 3 | 9 | 518 | 495 | 116 | 22.4 % |

**Mean overlap: 27.9 %**

**Notable:** B8→A5 has only 1.0 % overlap. B8 has just 42 members while A5 has 407 — the structlite
embedding compresses what was a large A cluster into a tiny isolated group, suggesting those gene
sets gain a distinctive structural signature with the added graph features.
Cluster 7 in A (474 members) absorbs B6 (817 members): the structlite space merges two adjacent
A-regions into one large blob.

---

### Subset 090 (n = 1 631 gene sets)

![Cluster correspondence subset 090](plots/cluster_correspondence_subset090.png)

K-means best seeds: A=13 (inertia 302.99), B=11 (inertia 395.43).

| B-cluster | → A-cluster | count A | count B | overlap | overlap % |
|----------:|------------:|--------:|--------:|--------:|----------:|
| 7 | 0 | 268 | 305 | 144 | 47.2 % |
| 1 | 1 | 238 | 228 | 125 | 52.5 % |
| 2 | 2 | 171 |  95 |  94 | 55.0 % |
| 8 | 3 | 139 | 101 |  22 | 15.8 % |
| 0 | 4 | 152 | 246 |  42 | 17.1 % |
| 6 | 5 | 121 | 111 |  69 | 57.0 % |
| 9 | 6 |  93 |  22 |   0 |  0.0 % |
| 5 | 7 | 145 |  69 |   8 |  5.5 % |
| 4 | 8 | 127 | 223 |  65 | 29.1 % |
| 3 | 9 | 177 | 231 |  83 | 35.9 % |

**Mean overlap: 31.5 %**

**Notable:** B9→A6 has 0 % overlap. A6 has only 93 members (the smallest cluster in A); the
Hungarian match assigned B9 (22 members) to it purely by elimination. Both are likely the
"isolated island" visible in the text-only UMAP — a small, structurally distinct pocket of gene
sets that the structlite embedding disperses into other clusters entirely.
Clusters 1, 2, 5, and 6 in A show 52–57 % overlap — among the strongest correspondence found,
suggesting those gene sets have coherent structure that survives the addition of graph features.

---

## Interpretation

Mean pairwise overlap across both subsets is ≈ 30 %, well below the 100 % that would indicate
identical cluster structure. This confirms that structural graph features (degree, clustering
coefficient, PageRank) materially reshape the embedding geometry: the GNN does not simply reproduce
the text-only embedding when structural features are added.

Key patterns:

- **Moderate-to-strong correspondence** (≥ 40 % overlap): clusters with coherent text content
  that also form cohesive graph neighbourhoods. These gene sets are stable under feature addition.
- **Near-zero correspondence** (≤ 5 % overlap): gene sets that form a compact text-only cluster
  but are structurally heterogeneous — their graph-feature signatures scatter them across the
  structlite space.
- **Size asymmetry** (e.g. A-cluster 407 ↔ B-cluster 42 in subset 085): structlite features can
  either compress or expand clusters relative to their text-only counterparts, implying that
  graph topology reshapes local density.

The colour-matched side-by-side UMAPs make drift and fragmentation immediately visible: clusters
that look the same colour across panels are well-preserved; colour changes flag structural
reorganisation.

## Files

| Artifact | Path |
|----------|------|
| Script | `scripts/map_cluster_correspondence.py` |
| PNG subset 085 | `reports/plots/cluster_correspondence_subset085.png` |
| PNG subset 090 | `reports/plots/cluster_correspondence_subset090.png` |
| This report | `reports/cluster_correspondence_report.md` |
```

**Step 2: Confirm the file was written**

```bash
wc -l work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/reports/cluster_correspondence_report.md
```

Expected: ≥ 80 lines.
