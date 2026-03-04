# Cluster Validation Report

- embedding: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260220_233515/subset090/inputs/fusion_subset_090_human.npz`
- enriched metadata: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260220_233515/subset090/inputs/enriched_gs_metadata_subset090_human.csv`
- k: 10
- seeds: 11, 12, 13, 14, 15
- n_perm: 200
- min_support: 5
- fdr_q: 0.05

## Text Coherence

| seed | coh_raw | p_raw | coh_clean | p_clean |
|---:|---:|---:|---:|---:|
| 11 | 0.2799 | 0.0050 | 0.2959 | 0.0050 |
| 12 | 0.2857 | 0.0050 | 0.3019 | 0.0050 |
| 13 | 0.2826 | 0.0050 | 0.2984 | 0.0050 |
| 14 | 0.2830 | 0.0050 | 0.2990 | 0.0050 |
| 15 | 0.2809 | 0.0050 | 0.2971 | 0.0050 |

## Seed Stability (ARI)

- pairs: 10 | ARI mean=0.586 std=0.096

## Metadata Enrichment Summary

| seed | attr_type | significant | null_sig_mean | p_sig | max_log10q | p_maxlog10q |
|---:|---|---:|---:|---:|---:|---:|
| 11 | mesh | 114 | 0.03 | 0.0050 | 10.75 | 0.0050 |
| 11 | geo_platform | 12 | 0.01 | 0.0050 | 6.48 | 0.0050 |
| 11 | gse_id | 30 | 0.04 | 0.0050 | 11.66 | 0.0050 |
| 11 | direction | 3 | 0.03 | 0.0050 | 5.20 | 0.0050 |
| 11 | pubmed_present | 1 | 0.01 | 0.0199 | 2.59 | 0.0050 |
| 11 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 12 | mesh | 106 | 0.03 | 0.0050 | 35.85 | 0.0050 |
| 12 | geo_platform | 12 | 0.01 | 0.0050 | 33.39 | 0.0050 |
| 12 | gse_id | 27 | 0.01 | 0.0050 | 36.92 | 0.0050 |
| 12 | direction | 2 | 0.01 | 0.0050 | 3.82 | 0.0050 |
| 12 | pubmed_present | 1 | 0.03 | 0.0299 | 2.15 | 0.0199 |
| 12 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 13 | mesh | 117 | 0.04 | 0.0050 | 19.75 | 0.0050 |
| 13 | geo_platform | 13 | 0.03 | 0.0050 | 15.17 | 0.0050 |
| 13 | gse_id | 28 | 0.04 | 0.0050 | 20.82 | 0.0050 |
| 13 | direction | 3 | 0.04 | 0.0050 | 3.50 | 0.0050 |
| 13 | pubmed_present | 1 | 0.02 | 0.0249 | 2.73 | 0.0050 |
| 13 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 14 | mesh | 116 | 0.04 | 0.0050 | 17.09 | 0.0050 |
| 14 | geo_platform | 13 | 0.03 | 0.0050 | 5.62 | 0.0050 |
| 14 | gse_id | 30 | 0.04 | 0.0050 | 17.16 | 0.0050 |
| 14 | direction | 3 | 0.01 | 0.0050 | 8.90 | 0.0050 |
| 14 | pubmed_present | 2 | 0.06 | 0.0100 | 2.87 | 0.0050 |
| 14 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
| 15 | mesh | 153 | 0.01 | 0.0050 | 12.88 | 0.0050 |
| 15 | geo_platform | 12 | 0.01 | 0.0050 | 8.13 | 0.0050 |
| 15 | gse_id | 28 | 0.01 | 0.0050 | 16.84 | 0.0050 |
| 15 | direction | 2 | 0.02 | 0.0050 | 3.21 | 0.0050 |
| 15 | pubmed_present | 1 | 0.04 | 0.0398 | 3.64 | 0.0050 |
| 15 | organism | 0 | 0.00 | 1.0000 | -0.00 | 1.0000 |
