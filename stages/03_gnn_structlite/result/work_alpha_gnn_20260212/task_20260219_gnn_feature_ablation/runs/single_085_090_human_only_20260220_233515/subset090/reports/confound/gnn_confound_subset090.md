# Mesh vs GSE Confound Check

- embedding: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260220_233515/subset090/inputs/gnn_old_holdout_subset_090_human.npz`
- enriched metadata: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260220_233515/subset090/inputs/enriched_gs_metadata_subset090_human.csv`
- seed: 11 | k: 10 | N: 1631
- min_support: 5 | fdr_q: 0.05

## Confound Association (Primary Labels)

- primary GSE: items=1631 | unique=409 | NMI=0.344 | ARI=0.015
- primary platform: items=1631 | unique=23 | NMI=0.056 | ARI=0.008

## Cluster Summary

| cluster | size | top_gse | gse_purity | mesh_sig | best_mesh (q,ER) | gse_sig | best_gse (q,ER) | plat_sig | best_plat (q,ER) |
|---:|---:|---|---:|---:|---|---:|---|---:|---|
| 0 | 183 | GSE213142 | 0.087 | 13 | Transposable (2.8e-03,6.24) | 3 | GSE242259 (2.2e-04,8.91) | 1 | "GPL24676" (9.0e-03,1.49) |
| 1 | 341 | GSE213142 | 0.062 | 23 | T-Lymphocytes (2.5e-05,2.39) | 4 | GSE159759 (5.8e-03,4.78) | 1 | "GPL15520" (2.3e-02,4.10) |
| 2 | 130 | GSE218585 | 0.077 | 15 | STAT1 (1.5e-05,8.36) | 1 | GSE218585 (3.7e-06,8.36) | 0 |  |
| 3 | 62 | GSE174745 | 0.048 | 5 | Abnormalities (1.3e-02,13.15) | 0 |  | 1 | GPL21290 (4.3e-02,7.52) |
| 4 | 168 | GSE219208 | 0.113 | 8 | Herpesvirus (2.8e-03,5.55) | 2 | GSE219208 (6.4e-17,9.71) | 2 | "GPL21697" (4.7e-06,3.64) |
| 5 | 159 | GSE213142 | 0.082 | 15 | Adrenocorticotropic (1.9e-04,10.26) | 2 | GSE181158 (1.1e-04,10.26) | 0 |  |
| 6 | 237 | GSE216249 | 0.093 | 22 | Liposomes (3.5e-07,4.09) | 7 | GSE216249 (5.9e-08,4.09) | 2 | GPL15520 (3.7e-08,3.65) |
| 7 | 113 | GSE213142 | 0.106 | 7 | RNA-Seq (1.7e-04,4.25) | 0 |  | 0 |  |
| 8 | 129 | GSE213142 | 0.124 | 8 | Dose-Response (2.9e-05,10.11) | 1 | GSE85443 (5.6e-04,9.48) | 1 | "GPL11154" (4.1e-02,1.74) |
| 9 | 109 | GSE214921 | 0.055 | 23 | Male (7.8e-10,3.08) | 2 | GSE214921 (1.2e-05,14.96) | 2 | GPL24247 (9.0e-03,6.41) |

## Interpretation Guide

- High `mesh_sig` with moderate/low `gse_purity` suggests biology-like themes not tied to one study.
- Very high `gse_purity` and strong `best_gse` indicates a study/batch-driven cluster (still real structure, but a confound for cross-study biology).
- Platform enrichment is a common technical confound; treat strong `best_plat` similarly.
