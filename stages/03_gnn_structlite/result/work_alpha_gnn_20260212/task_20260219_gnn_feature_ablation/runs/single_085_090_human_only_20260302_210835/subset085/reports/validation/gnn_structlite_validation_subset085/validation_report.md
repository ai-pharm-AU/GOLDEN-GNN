# Cluster Validation Report

- embedding: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260302_210835/subset085/features/text_structlite_gnn_holdout_subset085.npz`
- enriched metadata: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260302_210835/subset085/inputs/enriched_gs_metadata_subset085_human.csv`
- k: 10
- seeds: 11, 12, 13, 14, 15
- n_perm: 200
- min_support: 5
- fdr_q: 0.05

## Text Coherence

| seed | coh_raw | p_raw | coh_clean | p_clean |
|---:|---:|---:|---:|---:|
| 11 | 0.2452 | 0.0050 | 0.2652 | 0.0050 |
| 12 | 0.2452 | 0.0050 | 0.2651 | 0.0050 |
| 13 | 0.2456 | 0.0050 | 0.2656 | 0.0050 |
| 14 | 0.2449 | 0.0050 | 0.2648 | 0.0050 |
| 15 | 0.2454 | 0.0050 | 0.2653 | 0.0050 |

## Seed Stability (ARI)

- pairs: 10 | ARI mean=0.713 std=0.123

## Metadata Enrichment Summary

| seed | attr_type | significant | null_sig_mean | p_sig | max_log10q | p_maxlog10q |
|---:|---|---:|---:|---:|---:|---:|
| 11 | mesh | 386 | 0.03 | 0.0050 | 22.92 | 0.0050 |
| 11 | geo_platform | 9 | 0.02 | 0.0050 | 13.99 | 0.0050 |
| 11 | gse_id | 113 | 0.01 | 0.0050 | 23.80 | 0.0050 |
| 11 | direction | 0 | 0.01 | 1.0000 | -0.00 | 1.0000 |
| 11 | pubmed_present | 2 | 0.02 | 0.0050 | 4.59 | 0.0050 |
| 11 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 12 | mesh | 378 | 0.04 | 0.0050 | 22.92 | 0.0050 |
| 12 | geo_platform | 9 | 0.01 | 0.0050 | 13.99 | 0.0050 |
| 12 | gse_id | 111 | 0.01 | 0.0050 | 23.80 | 0.0050 |
| 12 | direction | 0 | 0.01 | 1.0000 | -0.00 | 1.0000 |
| 12 | pubmed_present | 2 | 0.04 | 0.0050 | 3.67 | 0.0050 |
| 12 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 13 | mesh | 436 | 0.09 | 0.0050 | 22.84 | 0.0050 |
| 13 | geo_platform | 11 | 0.01 | 0.0050 | 13.91 | 0.0050 |
| 13 | gse_id | 115 | 0.01 | 0.0050 | 23.72 | 0.0050 |
| 13 | direction | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 13 | pubmed_present | 2 | 0.02 | 0.0050 | 2.35 | 0.0050 |
| 13 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 14 | mesh | 400 | 0.04 | 0.0050 | 22.92 | 0.0050 |
| 14 | geo_platform | 10 | 0.03 | 0.0050 | 13.99 | 0.0050 |
| 14 | gse_id | 104 | 0.03 | 0.0050 | 23.80 | 0.0050 |
| 14 | direction | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 14 | pubmed_present | 2 | 0.03 | 0.0050 | 2.23 | 0.0149 |
| 14 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 15 | mesh | 414 | 0.03 | 0.0050 | 22.88 | 0.0050 |
| 15 | geo_platform | 10 | 0.03 | 0.0050 | 13.95 | 0.0050 |
| 15 | gse_id | 102 | 0.02 | 0.0050 | 23.76 | 0.0050 |
| 15 | direction | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 15 | pubmed_present | 2 | 0.01 | 0.0050 | 4.01 | 0.0050 |
| 15 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
