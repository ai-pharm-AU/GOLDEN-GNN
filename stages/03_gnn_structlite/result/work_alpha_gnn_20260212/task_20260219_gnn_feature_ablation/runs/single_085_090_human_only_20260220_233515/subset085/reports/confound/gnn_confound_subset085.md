# Mesh vs GSE Confound Check

- embedding: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260220_233515/subset085/inputs/gnn_old_holdout_subset_085_human.npz`
- enriched metadata: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260220_233515/subset085/inputs/enriched_gs_metadata_subset085_human.csv`
- seed: 11 | k: 10 | N: 4298
- min_support: 5 | fdr_q: 0.05

## Confound Association (Primary Labels)

- primary GSE: items=4298 | unique=954 | NMI=0.314 | ARI=0.008
- primary platform: items=4298 | unique=29 | NMI=0.028 | ARI=0.003

## Cluster Summary

| cluster | size | top_gse | gse_purity | mesh_sig | best_mesh (q,ER) | gse_sig | best_gse (q,ER) | plat_sig | best_plat (q,ER) |
|---:|---:|---|---:|---:|---|---:|---|---:|---|
| 0 | 430 | GSE213142 | 0.030 | 14 | Nerve (1.1e-03,3.33) | 9 | GSE118946 (2.2e-05,10.00) | 2 | "GPL23227" (2.3e-02,5.00) |
| 1 | 607 | GSE216249 | 0.054 | 47 | Liposomes (2.4e-10,3.83) | 9 | GSE216249 (9.4e-11,3.83) | 3 | GPL15520 (2.4e-10,2.96) |
| 2 | 442 | GSE218585 | 0.050 | 47 | Antiviral (8.0e-10,4.23) | 9 | GSE218585 (1.1e-09,5.35) | 2 | GPL18460 (2.3e-02,4.86) |
| 3 | 400 | GSE162699 | 0.043 | 41 | Polypyrimidine (1.6e-04,3.58) | 9 | GSE162699 (2.2e-04,3.65) | 1 | GPL30065 (3.5e-02,5.37) |
| 4 | 445 | GSE220620 | 0.034 | 29 | Domains (3.2e-04,3.03) | 8 | GSE220620 (6.7e-04,3.62) | 0 |  |
| 5 | 399 | GSE219208 | 0.055 | 34 | 58 (8.6e-04,10.77) | 13 | GSE219208 (3.4e-20,10.77) | 3 | "GPL21697" (1.4e-03,2.46) |
| 6 | 328 | GSE213142 | 0.049 | 29 | Heterocyclic (1.9e-05,6.29) | 6 | GSE119424 (7.2e-04,5.90) | 0 |  |
| 7 | 413 | GSE213142 | 0.034 | 27 | Liquid (6.4e-10,9.11) | 9 | GSE255774 (6.4e-11,9.71) | 3 | GPL23976 (3.5e-02,6.94) |
| 8 | 267 | GSE231324 | 0.034 | 82 | Female (3.9e-29,2.87) | 10 | GSE231324 (5.2e-09,16.10) | 5 | "GPL20795" (1.2e-02,3.17) |
| 9 | 567 | GSE216249 | 0.032 | 27 | Structure (2.2e-05,5.35) | 13 | GSE86577 (1.0e-03,5.51) | 1 | GPL21697 (1.5e-02,2.21) |

## Interpretation Guide

- High `mesh_sig` with moderate/low `gse_purity` suggests biology-like themes not tied to one study.
- Very high `gse_purity` and strong `best_gse` indicates a study/batch-driven cluster (still real structure, but a confound for cross-study biology).
- Platform enrichment is a common technical confound; treat strong `best_plat` similarly.
