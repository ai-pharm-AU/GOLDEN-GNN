# Stage 04 — Internal, Consistency, Confound, Enrichment Metrics

## Summary
Unified metrics layer covering internal clustering quality, modality consistency, confounds, and enrichment validation.

## Payload
### Code
- [`work_alpha_gnn_20260212/scripts/analyze_mesh_vs_gse_confound.py`](code/work_alpha_gnn_20260212/scripts/analyze_mesh_vs_gse_confound.py) — Confound analysis (MeSH vs GSE vs platform).
- [`work_alpha_gnn_20260212/scripts/modality_consistency_report.py`](code/work_alpha_gnn_20260212/scripts/modality_consistency_report.py) — GO-vs-PAGER modality consistency computation and plots.
- [`work_alpha_gnn_20260212/scripts/run_gnn_enrichment.py`](code/work_alpha_gnn_20260212/scripts/run_gnn_enrichment.py) — Cluster-level ORA enrichment pipeline.
- [`work_alpha_gnn_20260212/scripts/validate_cluster_meta_enrichment.py`](code/work_alpha_gnn_20260212/scripts/validate_cluster_meta_enrichment.py) — Metadata enrichment and text-coherence validation.

### Docs
- [`docs/metrics_reference.md`](doc/docs/metrics_reference.md) — Metric definitions and interpretation reference.
- [`work_alpha_gnn_20260212/reports/bio_validation_090_k10/mesh_vs_gse_confound_seed11.md`](doc/work_alpha_gnn_20260212/reports/bio_validation_090_k10/mesh_vs_gse_confound_seed11.md) — Confound check readout for seed 11.
- [`work_alpha_gnn_20260212/reports/bio_validation_090_k10/validation_report.md`](doc/work_alpha_gnn_20260212/reports/bio_validation_090_k10/validation_report.md) — Main bio-validation report (subset 0.90, k=10).

### Results
- [`outputs/human_only_combined_clean/all_analyses_combined.csv`](result/outputs/human_only_combined_clean/all_analyses_combined.csv) — Merged table of internal, stability, confound, and enrichment analyses.
- [`outputs/human_only_combined_clean/internal_metrics_combined.csv`](result/outputs/human_only_combined_clean/internal_metrics_combined.csv) — Combined internal clustering metrics across methods.
- [`outputs/human_only_combined_clean/meta_enrichment_combined.csv`](result/outputs/human_only_combined_clean/meta_enrichment_combined.csv) — Combined metadata enrichment metrics across methods.
- [`work_alpha_gnn_20260212/reports/baseline/modality_consistency_summary.csv`](result/work_alpha_gnn_20260212/reports/baseline/modality_consistency_summary.csv) — Modality consistency summary statistics.
- [`work_alpha_gnn_20260212/reports/enrichment/20260213_k10_seed11/gnn_enrichment_k10_seed11_cluster_assignments.csv`](result/work_alpha_gnn_20260212/reports/enrichment/20260213_k10_seed11/gnn_enrichment_k10_seed11_cluster_assignments.csv) — Seeded enrichment run outputs (summary, report, term tables).
- [`work_alpha_gnn_20260212/reports/enrichment/20260213_k10_seed11/gnn_enrichment_k10_seed11_consensus_themes.csv`](result/work_alpha_gnn_20260212/reports/enrichment/20260213_k10_seed11/gnn_enrichment_k10_seed11_consensus_themes.csv) — Seeded enrichment run outputs (summary, report, term tables).
- [`work_alpha_gnn_20260212/reports/enrichment/20260213_k10_seed11/gnn_enrichment_k10_seed11_consensus_themes.md`](result/work_alpha_gnn_20260212/reports/enrichment/20260213_k10_seed11/gnn_enrichment_k10_seed11_consensus_themes.md) — Seeded enrichment run outputs (summary, report, term tables).
- [`work_alpha_gnn_20260212/reports/enrichment/20260213_k10_seed11/gnn_enrichment_k10_seed11_dedup_map.csv`](result/work_alpha_gnn_20260212/reports/enrichment/20260213_k10_seed11/gnn_enrichment_k10_seed11_dedup_map.csv) — Seeded enrichment run outputs (summary, report, term tables).
- [`work_alpha_gnn_20260212/reports/enrichment/20260213_k10_seed11/gnn_enrichment_k10_seed11_dedup_terms.csv`](result/work_alpha_gnn_20260212/reports/enrichment/20260213_k10_seed11/gnn_enrichment_k10_seed11_dedup_terms.csv) — Seeded enrichment run outputs (summary, report, term tables).
- [`work_alpha_gnn_20260212/reports/enrichment/20260213_k10_seed11/gnn_enrichment_k10_seed11_report.md`](result/work_alpha_gnn_20260212/reports/enrichment/20260213_k10_seed11/gnn_enrichment_k10_seed11_report.md) — Seeded enrichment run outputs (summary, report, term tables).
- [`work_alpha_gnn_20260212/reports/enrichment/20260213_k10_seed11/gnn_enrichment_k10_seed11_results.csv`](result/work_alpha_gnn_20260212/reports/enrichment/20260213_k10_seed11/gnn_enrichment_k10_seed11_results.csv) — Seeded enrichment run outputs (summary, report, term tables).
- [`work_alpha_gnn_20260212/reports/enrichment/20260213_k10_seed11/gnn_enrichment_k10_seed11_summary.csv`](result/work_alpha_gnn_20260212/reports/enrichment/20260213_k10_seed11/gnn_enrichment_k10_seed11_summary.csv) — Seeded enrichment run outputs (summary, report, term tables).

## Notes
- Oversized files skipped to HF queue: 0
- Missing source items: 0
