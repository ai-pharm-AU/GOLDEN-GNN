# Poster: Fusion vs GNN (Subset 0.85 + 0.90)

## Output files
- `poster_alpha05_fusion_vs_gnn_subset085_090.png`
- `poster_alpha05_fusion_vs_gnn_subset085_090.pdf`

## Main message on poster
- GNN holdout has much stronger internal separation (silhouette cosine) at `k=10`.
- Fusion (alpha=0.5, concat_postl2) is slightly better on text coherence and shows higher meta-enrichment hit counts (notably `gse_id` and `mesh`).
- GNN is more stable across seeds by pairwise ARI.
- Confound alignment (`primary_gse_nmi`, `primary_platform_nmi`) is close between fusion and GNN.
- `up_down_conflicts` files for both subsets contain header only (no conflict records).

## Data sources used
- `task_20260217_alpha05_subset090/reports/metrics_reference/comparisons/metrics_reference_summary.csv`
- `task_20260217_metrics_080_085/reports/metrics_reference/comparisons/metrics_reference_summary_085.csv`
- `task_20260217_alpha05_subset090/csv/compare_fusion_vs_gnn_subset090.csv`
- `task_20260217_metrics_080_085/csv/internal_metrics_fusion_085.csv`
- `task_20260217_metrics_080_085/csv/internal_metrics_gnn_holdout_085.csv`
- `task_20260217_alpha05_subset090/reports/embedding_scatter/umap_fusion_vs_gnn_k10_subset090_superpag_overlap_gradient_pointer_text.png`
- `task_20260217_metrics_080_085/reports/embedding_scatter/umap_fusion_vs_gnn_k10_subset085_superpag_overlap_gradient_pointer_text.png`

## Rebuild command
```bash
python work_alpha_gnn_20260212/scripts/build_combined_poster_085_090.py
```
