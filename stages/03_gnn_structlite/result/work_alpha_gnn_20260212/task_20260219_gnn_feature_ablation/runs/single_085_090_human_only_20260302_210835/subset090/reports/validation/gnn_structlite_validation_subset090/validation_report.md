# Cluster Validation Report

- embedding: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260302_210835/subset090/features/text_structlite_gnn_holdout_subset090.npz`
- enriched metadata: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260302_210835/subset090/inputs/enriched_gs_metadata_subset090_human.csv`
- k: 10
- seeds: 11, 12, 13, 14, 15
- n_perm: 200
- min_support: 5
- fdr_q: 0.05

## Text Coherence

| seed | coh_raw | p_raw | coh_clean | p_clean |
|---:|---:|---:|---:|---:|
| 11 | 0.2816 | 0.0050 | 0.2976 | 0.0050 |
| 12 | 0.2812 | 0.0050 | 0.2971 | 0.0050 |
| 13 | 0.2824 | 0.0050 | 0.2982 | 0.0050 |
| 14 | 0.2810 | 0.0050 | 0.2970 | 0.0050 |
| 15 | 0.2817 | 0.0050 | 0.2976 | 0.0050 |

## Seed Stability (ARI)

- pairs: 10 | ARI mean=0.661 std=0.115

## Metadata Enrichment Summary

| seed | attr_type | significant | null_sig_mean | p_sig | max_log10q | p_maxlog10q |
|---:|---|---:|---:|---:|---:|---:|
| 11 | mesh | 203 | 0.01 | 0.0050 | 7.30 | 0.0050 |
| 11 | geo_platform | 9 | 0.03 | 0.0050 | 5.52 | 0.0050 |
| 11 | gse_id | 42 | 0.04 | 0.0050 | 11.04 | 0.0050 |
| 11 | direction | 0 | 0.01 | 1.0000 | 0.22 | 0.3433 |
| 11 | pubmed_present | 3 | 0.04 | 0.0050 | 3.99 | 0.0050 |
| 11 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 12 | mesh | 195 | 0.04 | 0.0050 | 6.64 | 0.0050 |
| 12 | geo_platform | 14 | 0.01 | 0.0050 | 6.73 | 0.0050 |
| 12 | gse_id | 42 | 0.03 | 0.0050 | 9.58 | 0.0050 |
| 12 | direction | 0 | 0.04 | 1.0000 | 0.53 | 0.1493 |
| 12 | pubmed_present | 2 | 0.02 | 0.0050 | 3.93 | 0.0050 |
| 12 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 13 | mesh | 170 | 0.01 | 0.0050 | 4.90 | 0.0050 |
| 13 | geo_platform | 10 | 0.04 | 0.0050 | 8.72 | 0.0050 |
| 13 | gse_id | 36 | 0.01 | 0.0050 | 19.56 | 0.0050 |
| 13 | direction | 0 | 0.01 | 1.0000 | 0.30 | 0.2935 |
| 13 | pubmed_present | 1 | 0.03 | 0.0348 | 6.34 | 0.0050 |
| 13 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 14 | mesh | 180 | 0.01 | 0.0050 | 6.14 | 0.0050 |
| 14 | geo_platform | 15 | 0.03 | 0.0050 | 5.11 | 0.0050 |
| 14 | gse_id | 42 | 0.01 | 0.0050 | 9.58 | 0.0050 |
| 14 | direction | 0 | 0.03 | 1.0000 | 0.02 | 0.5174 |
| 14 | pubmed_present | 2 | 0.03 | 0.0050 | 3.93 | 0.0050 |
| 14 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 15 | mesh | 176 | 0.03 | 0.0050 | 7.60 | 0.0050 |
| 15 | geo_platform | 12 | 0.04 | 0.0050 | 7.35 | 0.0050 |
| 15 | gse_id | 38 | 0.01 | 0.0050 | 9.58 | 0.0050 |
| 15 | direction | 0 | 0.01 | 1.0000 | 0.42 | 0.1741 |
| 15 | pubmed_present | 2 | 0.02 | 0.0050 | 3.93 | 0.0050 |
| 15 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
