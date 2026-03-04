# Mesh vs GSE Confound Check

- embedding: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260302_210835/subset085/features/text_structlite_gnn_holdout_subset085.npz`
- enriched metadata: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260302_210835/subset085/inputs/enriched_gs_metadata_subset085_human.csv`
- seed: 11 | k: 10 | N: 4298
- min_support: 5 | fdr_q: 0.05

## Confound Association (Primary Labels)

- primary GSE: items=4298 | unique=954 | NMI=0.323 | ARI=0.009
- primary platform: items=4298 | unique=29 | NMI=0.031 | ARI=0.005

## Cluster Summary

| cluster | size | top_gse | gse_purity | mesh_sig | best_mesh (q,ER) | gse_sig | best_gse (q,ER) | plat_sig | best_plat (q,ER) |
|---:|---:|---|---:|---:|---|---:|---|---:|---|
| 0 | 421 | GSE213142 | 0.033 | 33 | Craniofacial (2.0e-05,6.24) | 12 | GSE143145 (9.9e-05,10.21) | 0 |  |
| 1 | 806 | GSE162699 | 0.021 | 73 | Immunotherapy (2.8e-05,2.78) | 14 | GSE150924 (1.3e-04,5.33) | 0 |  |
| 2 | 474 | GSE213142 | 0.036 | 26 | Asthma (9.5e-04,7.77) | 5 | GSE241874 (1.1e-03,6.35) | 0 |  |
| 3 | 289 | GSE218585 | 0.062 | 76 | Female (5.6e-17,2.35) | 9 | GSE218585 (4.6e-09,6.69) | 2 | "GPL20795" (3.4e-04,3.61) |
| 4 | 491 | GSE216249 | 0.088 | 78 | Liposomes (1.4e-23,6.17) | 21 | GSE216249 (1.6e-24,6.17) | 3 | GPL15520 (1.0e-14,3.74) |
| 5 | 35 | GSE219208 | 0.114 | 4 | Prognosis (1.9e-03,13.64) | 3 | GSE219208 (1.4e-03,22.33) | 0 |  |
| 6 | 195 | GSE219208 | 0.077 | 26 | Guide (5.6e-04,11.02) | 9 | GSE219208 (4.1e-13,15.03) | 1 | "GPL21697" (3.4e-04,3.36) |
| 7 | 675 | GSE213142 | 0.022 | 29 | Elongation (5.9e-05,5.31) | 2 | GSE244112 (7.8e-03,4.46) | 0 |  |
| 8 | 500 | GSE162699 | 0.036 | 67 | Codon (1.2e-04,2.91) | 15 | GSE232518 (1.6e-04,7.52) | 1 | GPL15520 (3.4e-04,2.34) |
| 9 | 412 | GSE213142 | 0.085 | 95 | Alleles (6.0e-13,3.55) | 23 | GSE213142 (4.2e-10,3.73) | 2 | "GPL23227" (7.4e-07,8.69) |

## Interpretation Guide

- High `mesh_sig` with moderate/low `gse_purity` suggests biology-like themes not tied to one study.
- Very high `gse_purity` and strong `best_gse` indicates a study/batch-driven cluster (still real structure, but a confound for cross-study biology).
- Platform enrichment is a common technical confound; treat strong `best_plat` similarly.
