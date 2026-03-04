# Mesh vs GSE Confound Check

- embedding: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260220_233515/subset085/features/text_structlite_gnn_holdout_subset085.npz`
- enriched metadata: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260220_233515/subset085/inputs/enriched_gs_metadata_subset085_human.csv`
- seed: 11 | k: 10 | N: 4298
- min_support: 5 | fdr_q: 0.05

## Confound Association (Primary Labels)

- primary GSE: items=4298 | unique=954 | NMI=0.334 | ARI=0.011
- primary platform: items=4298 | unique=29 | NMI=0.032 | ARI=0.005

## Cluster Summary

| cluster | size | top_gse | gse_purity | mesh_sig | best_mesh (q,ER) | gse_sig | best_gse (q,ER) | plat_sig | best_plat (q,ER) |
|---:|---:|---|---:|---:|---|---:|---|---:|---|
| 0 | 419 | GSE213142 | 0.029 | 44 | Lymphoid (2.1e-07,7.69) | 5 | GSE136711 (6.5e-07,8.55) | 0 |  |
| 1 | 645 | GSE162699 | 0.037 | 69 | Codon (1.4e-06,2.94) | 16 | GSE162699 (3.6e-06,3.20) | 1 | "GPL11154" (4.3e-03,1.39) |
| 2 | 496 | GSE216249 | 0.087 | 79 | Liposomes (2.2e-23,6.11) | 21 | GSE216249 (2.4e-24,6.11) | 4 | GPL15520 (1.6e-14,3.70) |
| 3 | 229 | GSE219208 | 0.061 | 31 | Guide (8.1e-05,10.95) | 11 | GSE219208 (1.3e-10,11.94) | 1 | "GPL21697" (9.9e-04,3.02) |
| 4 | 459 | GSE213142 | 0.087 | 115 | Alleles (1.1e-14,3.52) | 24 | GSE213142 (3.0e-12,3.82) | 2 | "GPL23227" (2.1e-06,7.80) |
| 5 | 42 | GSE219208 | 0.119 | 4 | Prognosis (3.4e-03,11.37) | 3 | GSE219208 (1.2e-04,23.26) | 1 | "GPL21697" (3.1e-02,5.20) |
| 6 | 573 | GSE152497 | 0.031 | 25 | Semen (7.7e-05,3.19) | 9 | GSE152497 (1.9e-05,3.65) | 3 | GPL23976 (6.2e-04,7.50) |
| 7 | 556 | GSE213142 | 0.018 | 20 | Asthma (1.7e-03,6.63) | 6 | GSE173707 (1.4e-03,6.63) | 0 |  |
| 8 | 613 | GSE249723 | 0.020 | 78 | Adoptive (1.5e-06,3.90) | 14 | GSE150924 (1.6e-05,7.01) | 0 |  |
| 9 | 266 | GSE218585 | 0.053 | 71 | Female (5.2e-18,2.46) | 6 | GSE231324 (4.4e-09,16.16) | 2 | "GPL20795" (8.6e-03,3.18) |

## Interpretation Guide

- High `mesh_sig` with moderate/low `gse_purity` suggests biology-like themes not tied to one study.
- Very high `gse_purity` and strong `best_gse` indicates a study/batch-driven cluster (still real structure, but a confound for cross-study biology).
- Platform enrichment is a common technical confound; treat strong `best_plat` similarly.
