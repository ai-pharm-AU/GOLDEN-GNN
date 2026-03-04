# Mesh vs GSE Confound Check

- embedding: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260220_233515/subset090/inputs/fusion_subset_090_human.npz`
- enriched metadata: `/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/runs/single_085_090_human_only_20260220_233515/subset090/inputs/enriched_gs_metadata_subset090_human.csv`
- seed: 11 | k: 10 | N: 1631
- min_support: 5 | fdr_q: 0.05

## Confound Association (Primary Labels)

- primary GSE: items=1631 | unique=409 | NMI=0.341 | ARI=0.018
- primary platform: items=1631 | unique=23 | NMI=0.057 | ARI=0.019

## Cluster Summary

| cluster | size | top_gse | gse_purity | mesh_sig | best_mesh (q,ER) | gse_sig | best_gse (q,ER) | plat_sig | best_plat (q,ER) |
|---:|---:|---|---:|---:|---|---:|---|---:|---|
| 0 | 68 | GSE218585 | 0.147 | 17 | Janus (2.8e-09,15.52) | 1 | GSE218585 (3.9e-09,15.99) | 0 |  |
| 1 | 210 | GSE213142 | 0.129 | 12 | Genomics (1.2e-03,2.17) | 1 | GSE213142 (2.7e-03,2.14) | 0 |  |
| 2 | 91 | GSE219208 | 0.099 | 7 | Brain (1.6e-03,6.52) | 1 | GSE219208 (2.8e-05,8.49) | 1 | "GPL21697" (8.1e-03,3.52) |
| 3 | 223 | GSE213142 | 0.067 | 14 | T-Lymphocytes (9.6e-04,2.51) | 2 | GSE159759 (5.8e-04,7.31) | 1 | "GPL15520" (5.5e-03,6.27) |
| 4 | 57 | GSE152421 | 0.070 | 5 | Male (2.2e-06,3.34) | 3 | GSE115377 (2.4e-02,14.31) | 2 | GPL21290 (2.8e-02,8.18) |
| 5 | 67 | GSE219208 | 0.149 | 1 | Bone (2.8e-03,12.17) | 5 | GSE219208 (7.5e-08,12.81) | 2 | "GPL21697" (5.5e-03,4.35) |
| 6 | 456 | GSE216249 | 0.072 | 24 | Liposomes (2.9e-11,3.19) | 6 | GSE216249 (2.2e-12,3.19) | 2 | GPL15520 (3.3e-07,2.48) |
| 7 | 203 | GSE213142 | 0.103 | 9 | Carcinoma (4.3e-04,2.91) | 1 | GSE164873 (8.5e-03,6.70) | 0 |  |
| 8 | 51 | GSE206762 | 0.078 | 17 | Adult (3.2e-09,7.11) | 2 | GSE214921 (2.1e-02,15.99) | 2 | "GPL20301" (5.7e-03,3.52) |
| 9 | 205 | GSE220620 | 0.093 | 48 | Endopeptidase (2.9e-11,7.96) | 8 | GSE184029 (6.1e-12,7.96) | 2 | "GPL18573" (8.1e-03,1.53) |

## Interpretation Guide

- High `mesh_sig` with moderate/low `gse_purity` suggests biology-like themes not tied to one study.
- Very high `gse_purity` and strong `best_gse` indicates a study/batch-driven cluster (still real structure, but a confound for cross-study biology).
- Platform enrichment is a common technical confound; treat strong `best_plat` similarly.
