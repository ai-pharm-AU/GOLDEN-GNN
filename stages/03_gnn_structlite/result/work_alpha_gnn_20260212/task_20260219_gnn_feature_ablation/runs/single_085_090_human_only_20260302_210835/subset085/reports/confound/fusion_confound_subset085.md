# Mesh vs GSE Confound Check

- embedding: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260302_210835/subset085/inputs/fusion_subset_085_human.npz`
- enriched metadata: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260302_210835/subset085/inputs/enriched_gs_metadata_subset085_human.csv`
- seed: 11 | k: 10 | N: 4298
- min_support: 5 | fdr_q: 0.05

## Confound Association (Primary Labels)

- primary GSE: items=4298 | unique=954 | NMI=0.309 | ARI=0.009
- primary platform: items=4298 | unique=29 | NMI=0.035 | ARI=0.011

## Cluster Summary

| cluster | size | top_gse | gse_purity | mesh_sig | best_mesh (q,ER) | gse_sig | best_gse (q,ER) | plat_sig | best_plat (q,ER) |
|---:|---:|---|---:|---:|---|---:|---|---:|---|
| 0 | 344 | GSE184029 | 0.049 | 63 | HSP70 (3.6e-15,11.80) | 17 | GSE184029 (2.6e-15,11.80) | 1 | "GPL21290" (1.7e-04,3.05) |
| 1 | 994 | GSE216249 | 0.046 | 41 | Liposomes (5.4e-15,3.26) | 12 | GSE216249 (2.6e-15,3.26) | 1 | GPL15520 (3.5e-06,2.04) |
| 2 | 444 | GSE213142 | 0.032 | 41 | Cardiorespiratory (1.5e-04,8.47) | 9 | GSE164873 (1.0e-04,8.47) | 1 | "GPL23227" (3.4e-03,5.65) |
| 3 | 630 | GSE213142 | 0.052 | 34 | Alleles (3.7e-07,2.37) | 7 | GSE213142 (1.6e-04,2.30) | 0 |  |
| 4 | 220 | GSE174745 | 0.036 | 15 | HEK293 (1.9e-03,2.23) | 6 | GSE102469 (2.6e-03,6.84) | 1 | "GPL18573" (3.6e-03,1.58) |
| 5 | 217 | GSE219208 | 0.097 | 28 | Aldehyde (2.6e-04,16.51) | 12 | GSE219208 (1.4e-23,18.91) | 2 | "GPL21697" (3.1e-08,4.36) |
| 6 | 414 | GSE213142 | 0.046 | 35 | U2AF (3.7e-07,5.05) | 9 | GSE80045 (2.1e-07,5.19) | 1 | GPL11154 (3.3e-04,1.98) |
| 7 | 253 | GSE231324 | 0.036 | 91 | Male (3.7e-30,3.24) | 11 | GSE231324 (3.2e-09,16.99) | 4 | GPL21145 (8.3e-04,12.13) |
| 8 | 190 | GSE218585 | 0.121 | 36 | STAT1 (4.7e-18,11.31) | 8 | GSE218585 (1.3e-18,13.01) | 0 |  |
| 9 | 592 | GSE228991 | 0.030 | 58 | Artificial (8.3e-05,5.58) | 15 | GSE208116 (7.2e-10,6.17) | 4 | GPL19057 (1.7e-04,2.72) |

## Interpretation Guide

- High `mesh_sig` with moderate/low `gse_purity` suggests biology-like themes not tied to one study.
- Very high `gse_purity` and strong `best_gse` indicates a study/batch-driven cluster (still real structure, but a confound for cross-study biology).
- Platform enrichment is a common technical confound; treat strong `best_plat` similarly.
