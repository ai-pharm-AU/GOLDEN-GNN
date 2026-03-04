# Cluster Validation Report

- embedding: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260220_233515/subset085/inputs/fusion_subset_085_human.npz`
- enriched metadata: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260220_233515/subset085/inputs/enriched_gs_metadata_subset085_human.csv`
- k: 10
- seeds: 11, 12, 13, 14, 15
- n_perm: 200
- min_support: 5
- fdr_q: 0.05

## Text Coherence

| seed | coh_raw | p_raw | coh_clean | p_clean |
|---:|---:|---:|---:|---:|
| 11 | 0.2450 | 0.0050 | 0.2649 | 0.0050 |
| 12 | 0.2439 | 0.0050 | 0.2638 | 0.0050 |
| 13 | 0.2455 | 0.0050 | 0.2654 | 0.0050 |
| 14 | 0.2463 | 0.0050 | 0.2662 | 0.0050 |
| 15 | 0.2450 | 0.0050 | 0.2649 | 0.0050 |

## Seed Stability (ARI)

- pairs: 10 | ARI mean=0.641 std=0.104

## Metadata Enrichment Summary

| seed | attr_type | significant | null_sig_mean | p_sig | max_log10q | p_maxlog10q |
|---:|---|---:|---:|---:|---:|---:|
| 11 | mesh | 356 | 0.02 | 0.0050 | 29.51 | 0.0050 |
| 11 | geo_platform | 15 | 0.03 | 0.0050 | 7.51 | 0.0050 |
| 11 | gse_id | 106 | 0.03 | 0.0050 | 22.84 | 0.0050 |
| 11 | direction | 0 | 0.01 | 1.0000 | 0.94 | 0.0149 |
| 11 | pubmed_present | 1 | 0.01 | 0.0149 | 5.25 | 0.0050 |
| 11 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 12 | mesh | 393 | 0.02 | 0.0050 | 22.31 | 0.0050 |
| 12 | geo_platform | 13 | 0.02 | 0.0050 | 9.11 | 0.0050 |
| 12 | gse_id | 100 | 0.02 | 0.0050 | 23.73 | 0.0050 |
| 12 | direction | 1 | 0.00 | 0.0050 | 13.65 | 0.0050 |
| 12 | pubmed_present | 1 | 0.02 | 0.0249 | 6.80 | 0.0050 |
| 12 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 13 | mesh | 433 | 0.07 | 0.0050 | 31.57 | 0.0050 |
| 13 | geo_platform | 14 | 0.04 | 0.0050 | 12.69 | 0.0050 |
| 13 | gse_id | 109 | 0.01 | 0.0050 | 24.42 | 0.0050 |
| 13 | direction | 1 | 0.01 | 0.0149 | 6.64 | 0.0050 |
| 13 | pubmed_present | 1 | 0.03 | 0.0299 | 6.85 | 0.0050 |
| 13 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 14 | mesh | 422 | 0.02 | 0.0050 | 30.40 | 0.0050 |
| 14 | geo_platform | 22 | 0.01 | 0.0050 | 6.42 | 0.0050 |
| 14 | gse_id | 121 | 0.01 | 0.0050 | 21.65 | 0.0050 |
| 14 | direction | 1 | 0.01 | 0.0100 | 6.04 | 0.0050 |
| 14 | pubmed_present | 1 | 0.02 | 0.0249 | 6.41 | 0.0050 |
| 14 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 15 | mesh | 374 | 0.06 | 0.0050 | 29.82 | 0.0050 |
| 15 | geo_platform | 16 | 0.03 | 0.0050 | 6.25 | 0.0050 |
| 15 | gse_id | 100 | 0.01 | 0.0050 | 21.61 | 0.0050 |
| 15 | direction | 1 | 0.01 | 0.0149 | 5.61 | 0.0050 |
| 15 | pubmed_present | 1 | 0.03 | 0.0348 | 6.69 | 0.0050 |
| 15 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
