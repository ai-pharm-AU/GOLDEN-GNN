# Cluster Validation Report

- embedding: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260302_210835/subset085/inputs/gnn_old_holdout_subset_085_human.npz`
- enriched metadata: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260302_210835/subset085/inputs/enriched_gs_metadata_subset085_human.csv`
- k: 10
- seeds: 11, 12, 13, 14, 15
- n_perm: 200
- min_support: 5
- fdr_q: 0.05

## Text Coherence

| seed | coh_raw | p_raw | coh_clean | p_clean |
|---:|---:|---:|---:|---:|
| 11 | 0.2428 | 0.0050 | 0.2628 | 0.0050 |
| 12 | 0.2434 | 0.0050 | 0.2634 | 0.0050 |
| 13 | 0.2421 | 0.0050 | 0.2622 | 0.0050 |
| 14 | 0.2426 | 0.0050 | 0.2626 | 0.0050 |
| 15 | 0.2427 | 0.0050 | 0.2627 | 0.0050 |

## Seed Stability (ARI)

- pairs: 10 | ARI mean=0.603 std=0.167

## Metadata Enrichment Summary

| seed | attr_type | significant | null_sig_mean | p_sig | max_log10q | p_maxlog10q |
|---:|---|---:|---:|---:|---:|---:|
| 11 | mesh | 321 | 0.03 | 0.0050 | 28.49 | 0.0050 |
| 11 | geo_platform | 20 | 0.03 | 0.0050 | 9.61 | 0.0050 |
| 11 | gse_id | 95 | 0.04 | 0.0050 | 19.46 | 0.0050 |
| 11 | direction | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 11 | pubmed_present | 1 | 0.01 | 0.0199 | 3.47 | 0.0050 |
| 11 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 12 | mesh | 271 | 0.01 | 0.0050 | 31.25 | 0.0050 |
| 12 | geo_platform | 16 | 0.03 | 0.0050 | 16.20 | 0.0050 |
| 12 | gse_id | 93 | 0.01 | 0.0050 | 19.59 | 0.0050 |
| 12 | direction | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 12 | pubmed_present | 1 | 0.03 | 0.0299 | 4.13 | 0.0050 |
| 12 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 13 | mesh | 277 | 0.03 | 0.0050 | 28.99 | 0.0050 |
| 13 | geo_platform | 11 | 0.06 | 0.0050 | 7.30 | 0.0050 |
| 13 | gse_id | 80 | 0.04 | 0.0050 | 17.16 | 0.0050 |
| 13 | direction | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 13 | pubmed_present | 0 | 0.03 | 1.0000 | -0.00 | 1.0000 |
| 13 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 14 | mesh | 280 | 0.01 | 0.0050 | 29.84 | 0.0050 |
| 14 | geo_platform | 13 | 0.00 | 0.0050 | 8.92 | 0.0050 |
| 14 | gse_id | 77 | 0.01 | 0.0050 | 19.54 | 0.0050 |
| 14 | direction | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 14 | pubmed_present | 1 | 0.03 | 0.0299 | 3.15 | 0.0050 |
| 14 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 15 | mesh | 306 | 0.02 | 0.0050 | 28.69 | 0.0050 |
| 15 | geo_platform | 18 | 0.04 | 0.0050 | 9.64 | 0.0050 |
| 15 | gse_id | 92 | 0.01 | 0.0050 | 19.46 | 0.0050 |
| 15 | direction | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 15 | pubmed_present | 1 | 0.01 | 0.0199 | 3.47 | 0.0050 |
| 15 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
