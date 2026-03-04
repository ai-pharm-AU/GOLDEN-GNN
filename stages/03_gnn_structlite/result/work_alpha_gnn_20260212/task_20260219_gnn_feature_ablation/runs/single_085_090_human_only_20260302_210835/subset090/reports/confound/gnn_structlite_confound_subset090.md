# Mesh vs GSE Confound Check

- embedding: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260302_210835/subset090/features/text_structlite_gnn_holdout_subset090.npz`
- enriched metadata: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260302_210835/subset090/inputs/enriched_gs_metadata_subset090_human.csv`
- seed: 11 | k: 10 | N: 1631
- min_support: 5 | fdr_q: 0.05

## Confound Association (Primary Labels)

- primary GSE: items=1631 | unique=409 | NMI=0.354 | ARI=0.015
- primary platform: items=1631 | unique=23 | NMI=0.069 | ARI=0.017

## Cluster Summary

| cluster | size | top_gse | gse_purity | mesh_sig | best_mesh (q,ER) | gse_sig | best_gse (q,ER) | plat_sig | best_plat (q,ER) |
|---:|---:|---|---:|---:|---|---:|---|---:|---|
| 0 | 95 | GSE213142 | 0.200 | 19 | Albinism (2.0e-04,3.33) | 4 | GSE147240 (4.5e-06,17.17) | 0 |  |
| 1 | 84 | GSE213142 | 0.095 | 30 | T-Lymphocytes (3.1e-06,4.99) | 6 | GSE229066 (3.7e-04,8.49) | 1 | "GPL16791" (1.3e-04,2.16) |
| 2 | 91 | GSE219208 | 0.154 | 6 | Bone (5.5e-03,8.96) | 3 | GSE219208 (9.1e-12,13.21) | 2 | "GPL21697" (9.4e-06,4.80) |
| 3 | 463 | GSE216249 | 0.063 | 60 | p53 (2.3e-07,2.30) | 11 | GSE216249 (7.7e-08,2.76) | 2 | GPL15520 (3.0e-06,2.37) |
| 4 | 232 | GSE213142 | 0.112 | 50 | Physiological (3.7e-06,4.69) | 4 | GSE218585 (3.0e-05,5.16) | 0 |  |
| 5 | 67 | GSE184029 | 0.104 | 57 | CX3C (1.5e-05,20.87) | 8 | GSE159759 (9.1e-07,24.34) | 1 | "GPL15520" (3.9e-06,20.87) |
| 6 | 24 | GSE219208 | 0.208 | 1 | Prognosis (2.2e-02,16.99) | 1 | GSE219208 (3.3e-04,17.88) | 1 | "GPL21697" (4.1e-03,7.28) |
| 7 | 103 | GSE214921 | 0.058 | 24 | Male (9.8e-07,2.74) | 2 | GSE214921 (5.8e-06,15.83) | 1 | GPL24247 (4.1e-03,6.79) |
| 8 | 252 | GSE213142 | 0.067 | 9 | Sequences (4.6e-03,5.55) | 2 | GSE134459 (8.6e-03,4.85) | 0 |  |
| 9 | 220 | GSE213142 | 0.041 | 17 | Cell (3.2e-03,1.39) | 1 | GSE139292 (4.5e-03,5.56) | 1 | "GPL18573" (7.6e-05,1.67) |

## Interpretation Guide

- High `mesh_sig` with moderate/low `gse_purity` suggests biology-like themes not tied to one study.
- Very high `gse_purity` and strong `best_gse` indicates a study/batch-driven cluster (still real structure, but a confound for cross-study biology).
- Platform enrichment is a common technical confound; treat strong `best_plat` similarly.
