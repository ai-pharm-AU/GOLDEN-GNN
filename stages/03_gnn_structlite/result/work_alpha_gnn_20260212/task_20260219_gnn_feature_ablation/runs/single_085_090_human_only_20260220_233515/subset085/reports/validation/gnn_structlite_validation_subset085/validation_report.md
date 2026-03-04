# Cluster Validation Report

- embedding: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260220_233515/subset085/features/text_structlite_gnn_holdout_subset085.npz`
- enriched metadata: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260220_233515/subset085/inputs/enriched_gs_metadata_subset085_human.csv`
- k: 10
- seeds: 11, 12, 13, 14, 15
- n_perm: 200
- min_support: 5
- fdr_q: 0.05

## Text Coherence

| seed | coh_raw | p_raw | coh_clean | p_clean |
|---:|---:|---:|---:|---:|
| 11 | 0.2457 | 0.0050 | 0.2656 | 0.0050 |
| 12 | 0.2457 | 0.0050 | 0.2656 | 0.0050 |
| 13 | 0.2482 | 0.0050 | 0.2680 | 0.0050 |
| 14 | 0.2482 | 0.0050 | 0.2681 | 0.0050 |
| 15 | 0.2455 | 0.0050 | 0.2654 | 0.0050 |

## Seed Stability (ARI)

- pairs: 10 | ARI mean=0.713 std=0.147

## Metadata Enrichment Summary

| seed | attr_type | significant | null_sig_mean | p_sig | max_log10q | p_maxlog10q |
|---:|---|---:|---:|---:|---:|---:|
| 11 | mesh | 425 | 0.01 | 0.0050 | 22.73 | 0.0050 |
| 11 | geo_platform | 14 | 0.01 | 0.0050 | 13.80 | 0.0050 |
| 11 | gse_id | 115 | 0.00 | 0.0050 | 23.61 | 0.0050 |
| 11 | direction | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 11 | pubmed_present | 2 | 0.03 | 0.0050 | 2.77 | 0.0050 |
| 11 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 12 | mesh | 426 | 0.04 | 0.0050 | 22.73 | 0.0050 |
| 12 | geo_platform | 14 | 0.04 | 0.0050 | 13.80 | 0.0050 |
| 12 | gse_id | 115 | 0.01 | 0.0050 | 23.61 | 0.0050 |
| 12 | direction | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 12 | pubmed_present | 2 | 0.03 | 0.0050 | 2.71 | 0.0050 |
| 12 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 13 | mesh | 437 | 0.03 | 0.0050 | 39.21 | 0.0050 |
| 13 | geo_platform | 21 | 0.03 | 0.0050 | 26.65 | 0.0050 |
| 13 | gse_id | 125 | 0.04 | 0.0050 | 40.09 | 0.0050 |
| 13 | direction | 0 | 0.01 | 1.0000 | -0.00 | 1.0000 |
| 13 | pubmed_present | 2 | 0.01 | 0.0050 | 3.06 | 0.0050 |
| 13 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 14 | mesh | 447 | 0.06 | 0.0050 | 41.36 | 0.0050 |
| 14 | geo_platform | 21 | 0.02 | 0.0050 | 28.24 | 0.0050 |
| 14 | gse_id | 132 | 0.03 | 0.0050 | 42.24 | 0.0050 |
| 14 | direction | 0 | 0.01 | 1.0000 | -0.00 | 1.0000 |
| 14 | pubmed_present | 2 | 0.03 | 0.0050 | 3.27 | 0.0050 |
| 14 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 15 | mesh | 413 | 0.02 | 0.0050 | 22.73 | 0.0050 |
| 15 | geo_platform | 11 | 0.01 | 0.0050 | 13.80 | 0.0050 |
| 15 | gse_id | 115 | 0.02 | 0.0050 | 23.61 | 0.0050 |
| 15 | direction | 0 | 0.01 | 1.0000 | -0.00 | 1.0000 |
| 15 | pubmed_present | 2 | 0.01 | 0.0050 | 3.84 | 0.0050 |
| 15 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
