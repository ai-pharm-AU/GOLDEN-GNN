# Cluster Validation Report

- embedding: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260220_233515/subset090/inputs/gnn_old_holdout_subset_090_human.npz`
- enriched metadata: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260220_233515/subset090/inputs/enriched_gs_metadata_subset090_human.csv`
- k: 10
- seeds: 11, 12, 13, 14, 15
- n_perm: 200
- min_support: 5
- fdr_q: 0.05

## Text Coherence

| seed | coh_raw | p_raw | coh_clean | p_clean |
|---:|---:|---:|---:|---:|
| 11 | 0.2772 | 0.0050 | 0.2933 | 0.0050 |
| 12 | 0.2779 | 0.0050 | 0.2939 | 0.0050 |
| 13 | 0.2790 | 0.0050 | 0.2949 | 0.0050 |
| 14 | 0.2788 | 0.0050 | 0.2947 | 0.0050 |
| 15 | 0.2780 | 0.0050 | 0.2940 | 0.0050 |

## Seed Stability (ARI)

- pairs: 10 | ARI mean=0.631 std=0.064

## Metadata Enrichment Summary

| seed | attr_type | significant | null_sig_mean | p_sig | max_log10q | p_maxlog10q |
|---:|---|---:|---:|---:|---:|---:|
| 11 | mesh | 86 | 0.00 | 0.0050 | 9.24 | 0.0050 |
| 11 | geo_platform | 10 | 0.04 | 0.0050 | 7.43 | 0.0050 |
| 11 | gse_id | 22 | 0.01 | 0.0050 | 16.20 | 0.0050 |
| 11 | direction | 5 | 0.03 | 0.0050 | 3.32 | 0.0100 |
| 11 | pubmed_present | 1 | 0.01 | 0.0199 | 2.92 | 0.0050 |
| 11 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 12 | mesh | 62 | 0.01 | 0.0050 | 10.49 | 0.0050 |
| 12 | geo_platform | 9 | 0.05 | 0.0050 | 9.14 | 0.0050 |
| 12 | gse_id | 24 | 0.03 | 0.0050 | 17.31 | 0.0050 |
| 12 | direction | 3 | 0.03 | 0.0050 | 2.30 | 0.0050 |
| 12 | pubmed_present | 1 | 0.02 | 0.0249 | 3.32 | 0.0050 |
| 12 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 13 | mesh | 104 | 0.07 | 0.0050 | 11.59 | 0.0050 |
| 13 | geo_platform | 13 | 0.02 | 0.0050 | 10.30 | 0.0050 |
| 13 | gse_id | 25 | 0.02 | 0.0050 | 16.14 | 0.0050 |
| 13 | direction | 1 | 0.02 | 0.0249 | 1.80 | 0.0149 |
| 13 | pubmed_present | 1 | 0.03 | 0.0299 | 2.84 | 0.0050 |
| 13 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 14 | mesh | 94 | 0.08 | 0.0050 | 12.61 | 0.0050 |
| 14 | geo_platform | 11 | 0.03 | 0.0050 | 8.25 | 0.0050 |
| 14 | gse_id | 22 | 0.03 | 0.0050 | 17.19 | 0.0050 |
| 14 | direction | 1 | 0.06 | 0.0547 | 1.37 | 0.0547 |
| 14 | pubmed_present | 1 | 0.03 | 0.0348 | 3.13 | 0.0050 |
| 14 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 15 | mesh | 89 | 0.04 | 0.0050 | 10.64 | 0.0050 |
| 15 | geo_platform | 7 | 0.01 | 0.0050 | 8.00 | 0.0050 |
| 15 | gse_id | 22 | 0.02 | 0.0050 | 17.25 | 0.0050 |
| 15 | direction | 2 | 0.01 | 0.0050 | 3.22 | 0.0050 |
| 15 | pubmed_present | 1 | 0.03 | 0.0299 | 3.23 | 0.0050 |
| 15 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
