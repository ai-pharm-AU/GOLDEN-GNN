# Stage 06 — Knowledge Graph

## Summary
KG construction, cluster-level graph metrics, matched comparisons, sparse visual dashboards, and case-study outputs.

## Payload
### Code
- [`KnowledgeGraph/build_kg.py`](code/KnowledgeGraph/build_kg.py) — KG construction pipeline and core shared utilities.
- [`KnowledgeGraph/kg_cluster_metrics.py`](code/KnowledgeGraph/kg_cluster_metrics.py) — Build KGs and compute metrics for all A/B cluster groups.
- [`KnowledgeGraph/kg_cluster_summarize.py`](code/KnowledgeGraph/kg_cluster_summarize.py) — LLM-driven cluster summary generation from KG evidence.
- [`KnowledgeGraph/kg_cluster_vis_sparse.py`](code/KnowledgeGraph/kg_cluster_vis_sparse.py) — Sparse visualization renderer for cluster KG pages.
- [`KnowledgeGraph/kg_matched_comparison.py`](code/KnowledgeGraph/kg_matched_comparison.py) — Matched cluster-pair KG comparison report generation.
- [`KnowledgeGraph/vis_case_study_A8_B7.py`](code/KnowledgeGraph/vis_case_study_A8_B7.py) — Case-study visual comparison script for A8 vs B7 clusters.

### Docs
- [`KnowledgeGraph/findings_cluster8_kg_visualization.md`](doc/KnowledgeGraph/findings_cluster8_kg_visualization.md) — Narrative findings from cluster-8 KG visualization.
- [`docs/gene_overlap_edge_methodology.md`](doc/docs/gene_overlap_edge_methodology.md) — Methodology note linking mtype construction and KG assembly.

### Results
- [`KnowledgeGraph/cluster_metrics_out/casestudy_A8B7_centrality.csv`](result/KnowledgeGraph/cluster_metrics_out/casestudy_A8B7_centrality.csv) — Case-study centrality table for A8/B7.
- [`KnowledgeGraph/cluster_metrics_out/casestudy_A8B7_report.html`](result/KnowledgeGraph/cluster_metrics_out/casestudy_A8B7_report.html) — Case-study narrative HTML report for A8/B7.
- [`KnowledgeGraph/cluster_metrics_out/casestudy_A8B7_stats.csv`](result/KnowledgeGraph/cluster_metrics_out/casestudy_A8B7_stats.csv) — Case-study edge-type statistics for A8/B7.
- [`KnowledgeGraph/cluster_metrics_out/cluster_matched_comparison.csv`](result/KnowledgeGraph/cluster_metrics_out/cluster_matched_comparison.csv) — Matched A/B cluster comparison metrics.
- [`KnowledgeGraph/cluster_metrics_out/cluster_summaries.csv`](result/KnowledgeGraph/cluster_metrics_out/cluster_summaries.csv) — Cluster-level biological summary CSV.
- [`KnowledgeGraph/cluster_metrics_out/cluster_summaries.html`](result/KnowledgeGraph/cluster_metrics_out/cluster_summaries.html) — Cluster-level biological summary dashboard.
- [`KnowledgeGraph/cluster_metrics_out/comparison.html`](result/KnowledgeGraph/cluster_metrics_out/comparison.html) — Matched A/B cluster comparison HTML report.
- [`KnowledgeGraph/cluster_metrics_out/graph_stats.csv`](result/KnowledgeGraph/cluster_metrics_out/graph_stats.csv) — KG graph-level statistics summary table.
- [`KnowledgeGraph/cluster_metrics_out/index.html`](result/KnowledgeGraph/cluster_metrics_out/index.html) — Main KG cluster metrics dashboard index.
- [`KnowledgeGraph/cluster_metrics_out/vis_sparse/case_study_A8_B7.html`](result/KnowledgeGraph/cluster_metrics_out/vis_sparse/case_study_A8_B7.html) — Case-study sparse HTML visualization for A8 vs B7.
- [`KnowledgeGraph/cluster_metrics_out/vis_sparse/index_sparse.html`](result/KnowledgeGraph/cluster_metrics_out/vis_sparse/index_sparse.html) — Sparse KG visual index across groups.
- [`KnowledgeGraph/kg_adjacency_heatmaps.png`](result/KnowledgeGraph/kg_adjacency_heatmaps.png) — KG edge-type adjacency heatmap panel.
- [`KnowledgeGraph/kg_edge_summary.png`](result/KnowledgeGraph/kg_edge_summary.png) — Edge-type summary figure.
- [`KnowledgeGraph/kg_sparse_network.html`](result/KnowledgeGraph/kg_sparse_network.html) — Interactive sparse-network KG visualization.

## Notes
- Oversized files skipped to HF queue: 0
- Missing source items: 0
