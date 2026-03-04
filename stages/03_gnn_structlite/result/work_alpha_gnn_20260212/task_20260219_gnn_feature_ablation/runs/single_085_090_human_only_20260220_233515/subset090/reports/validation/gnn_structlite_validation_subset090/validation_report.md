# Cluster Validation Report

- embedding: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260220_233515/subset090/features/text_structlite_gnn_holdout_subset090.npz`
- enriched metadata: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260220_233515/subset090/inputs/enriched_gs_metadata_subset090_human.csv`
- k: 10
- seeds: 11, 12, 13, 14, 15
- n_perm: 200
- min_support: 5
- fdr_q: 0.05

## Text Coherence

| seed | coh_raw | p_raw | coh_clean | p_clean |
|---:|---:|---:|---:|---:|
| 11 | 0.2849 | 0.0050 | 0.3006 | 0.0050 |
| 12 | 0.2836 | 0.0050 | 0.2995 | 0.0050 |
| 13 | 0.2816 | 0.0050 | 0.2976 | 0.0050 |
| 14 | 0.2847 | 0.0050 | 0.3006 | 0.0050 |
| 15 | 0.2840 | 0.0050 | 0.2998 | 0.0050 |

## Seed Stability (ARI)

- pairs: 10 | ARI mean=0.688 std=0.086

## Metadata Enrichment Summary

| seed | attr_type | significant | null_sig_mean | p_sig | max_log10q | p_maxlog10q |
|---:|---|---:|---:|---:|---:|---:|
| 11 | mesh | 229 | 0.05 | 0.0050 | 16.02 | 0.0050 |
| 11 | geo_platform | 17 | 0.01 | 0.0050 | 13.85 | 0.0050 |
| 11 | gse_id | 45 | 0.01 | 0.0050 | 18.43 | 0.0050 |
| 11 | direction | 1 | 0.03 | 0.0348 | 1.74 | 0.0149 |
| 11 | pubmed_present | 1 | 0.03 | 0.0348 | 5.95 | 0.0050 |
| 11 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 12 | mesh | 207 | 0.01 | 0.0050 | 16.02 | 0.0050 |
| 12 | geo_platform | 17 | 0.03 | 0.0050 | 13.85 | 0.0050 |
| 12 | gse_id | 44 | 0.04 | 0.0050 | 17.09 | 0.0050 |
| 12 | direction | 0 | 0.01 | 1.0000 | 0.35 | 0.1642 |
| 12 | pubmed_present | 1 | 0.06 | 0.0597 | 4.05 | 0.0050 |
| 12 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 13 | mesh | 192 | 0.01 | 0.0050 | 15.61 | 0.0050 |
| 13 | geo_platform | 17 | 0.00 | 0.0050 | 13.44 | 0.0050 |
| 13 | gse_id | 40 | 0.04 | 0.0050 | 16.68 | 0.0050 |
| 13 | direction | 0 | 0.01 | 1.0000 | -0.00 | 1.0000 |
| 13 | pubmed_present | 1 | 0.04 | 0.0348 | 4.05 | 0.0050 |
| 13 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 14 | mesh | 215 | 0.03 | 0.0050 | 16.26 | 0.0050 |
| 14 | geo_platform | 14 | 0.03 | 0.0050 | 14.10 | 0.0050 |
| 14 | gse_id | 30 | 0.03 | 0.0050 | 17.95 | 0.0050 |
| 14 | direction | 0 | 0.03 | 1.0000 | 0.40 | 0.2189 |
| 14 | pubmed_present | 1 | 0.01 | 0.0199 | 4.19 | 0.0050 |
| 14 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 15 | mesh | 192 | 0.03 | 0.0050 | 14.87 | 0.0050 |
| 15 | geo_platform | 14 | 0.04 | 0.0050 | 13.37 | 0.0050 |
| 15 | gse_id | 32 | 0.01 | 0.0050 | 17.93 | 0.0050 |
| 15 | direction | 0 | 0.01 | 1.0000 | 0.11 | 0.4328 |
| 15 | pubmed_present | 1 | 0.02 | 0.0249 | 5.04 | 0.0050 |
| 15 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
