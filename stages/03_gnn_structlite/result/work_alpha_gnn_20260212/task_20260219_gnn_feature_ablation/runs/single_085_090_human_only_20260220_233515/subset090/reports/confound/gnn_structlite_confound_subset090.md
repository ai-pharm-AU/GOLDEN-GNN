# Mesh vs GSE Confound Check

- embedding: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260220_233515/subset090/features/text_structlite_gnn_holdout_subset090.npz`
- enriched metadata: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260220_233515/subset090/inputs/enriched_gs_metadata_subset090_human.csv`
- seed: 11 | k: 10 | N: 1631
- min_support: 5 | fdr_q: 0.05

## Confound Association (Primary Labels)

- primary GSE: items=1631 | unique=409 | NMI=0.379 | ARI=0.022
- primary platform: items=1631 | unique=23 | NMI=0.074 | ARI=0.015

## Cluster Summary

| cluster | size | top_gse | gse_purity | mesh_sig | best_mesh (q,ER) | gse_sig | best_gse (q,ER) | plat_sig | best_plat (q,ER) |
|---:|---:|---|---:|---:|---|---:|---|---:|---|
| 0 | 120 | GSE213142 | 0.100 | 4 | Down (2.0e-03,8.16) | 1 | GSE147240 (1.3e-05,13.59) | 0 |  |
| 1 | 243 | GSE213142 | 0.058 | 22 | Prospective (8.8e-04,4.47) | 4 | GSE136711 (3.0e-03,4.47) | 1 | GPL21290 (3.8e-02,3.36) |
| 2 | 98 | GSE219208 | 0.184 | 9 | Hematopoiesis (4.9e-03,9.25) | 4 | GSE219208 (3.7e-19,15.77) | 2 | "GPL21697" (1.2e-08,5.65) |
| 3 | 117 | GSE213142 | 0.137 | 44 | STAT1 (3.1e-05,8.36) | 8 | GSE218585 (1.3e-05,8.36) | 2 | "GPL20301" (1.7e-03,2.56) |
| 4 | 232 | GSE213142 | 0.060 | 13 | Cultured (3.9e-04,2.34) | 1 | GSE139292 (6.7e-03,5.27) | 1 | "GPL18573" (9.5e-04,1.56) |
| 5 | 69 | GSE184029 | 0.101 | 57 | CX3C (9.8e-06,20.26) | 8 | GSE159759 (1.1e-06,23.64) | 4 | "GPL15520" (3.1e-06,20.26) |
| 6 | 209 | GSE213142 | 0.115 | 37 | Medicine (6.3e-06,5.85) | 9 | GSE229066 (2.3e-06,5.85) | 2 | "GPL23227" (3.9e-06,6.50) |
| 7 | 125 | GSE214921 | 0.048 | 28 | Adult (4.5e-06,3.70) | 2 | GSE214921 (1.4e-05,13.05) | 1 | GPL24247 (1.0e-02,5.59) |
| 8 | 190 | GSE73109 | 0.047 | 59 | c-bcl-6 (2.5e-06,4.54) | 4 | GSE73109 (1.7e-03,4.54) | 0 |  |
| 9 | 228 | GSE216249 | 0.132 | 18 | Liposomes (1.3e-16,5.80) | 4 | GSE216249 (5.4e-18,5.80) | 4 | GPL15520 (1.4e-14,4.67) |

## Interpretation Guide

- High `mesh_sig` with moderate/low `gse_purity` suggests biology-like themes not tied to one study.
- Very high `gse_purity` and strong `best_gse` indicates a study/batch-driven cluster (still real structure, but a confound for cross-study biology).
- Platform enrichment is a common technical confound; treat strong `best_plat` similarly.
