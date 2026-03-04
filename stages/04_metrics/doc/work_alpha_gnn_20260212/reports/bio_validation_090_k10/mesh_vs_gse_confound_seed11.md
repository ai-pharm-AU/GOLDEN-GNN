# Mesh vs GSE Confound Check

- embedding: `work_alpha_gnn_20260212/subset_85_90_20260213/embeddings/gnn_holdout_subset_090_l2.npz`
- enriched metadata: `work_alpha_gnn_20260212/reports/bio_meta_cache_090/enriched_gs_metadata.csv`
- seed: 11 | k: 10 | N: 1643
- min_support: 5 | fdr_q: 0.05

## Confound Association (Primary Labels)

- primary GSE: items=1643 | unique=414 | NMI=0.347 | ARI=0.015
- primary platform: items=1643 | unique=23 | NMI=0.053 | ARI=0.009

## Cluster Summary

| cluster | size | top_gse | gse_purity | mesh_sig | best_mesh (q,ER) | gse_sig | best_gse (q,ER) | plat_sig | best_plat (q,ER) |
|---:|---:|---|---:|---:|---|---:|---|---:|---|
| 0 | 118 | GSE213142 | 0.102 | 8 | Carcinoma (1.1e-03,3.60) | 0 |  | 0 |  |
| 1 | 228 | GSE216249 | 0.118 | 27 | Liposomes (1.3e-12,5.26) | 8 | GSE216249 (5.3e-14,5.26) | 4 | GPL15520 (7.8e-11,4.10) |
| 2 | 144 | GSE218585 | 0.076 | 22 | STAT1 (8.9e-07,8.37) | 2 | GSE218585 (3.9e-07,8.37) | 0 |  |
| 3 | 61 | GSE179832 | 0.033 | 5 | Biomarkers (5.3e-03,4.95) | 0 |  | 1 | GPL21290 (3.7e-02,7.70) |
| 4 | 373 | GSE213142 | 0.062 | 21 | T-Lymphocytes (9.0e-05,2.20) | 4 | GSE229066 (7.5e-03,3.03) | 0 |  |
| 5 | 172 | GSE219208 | 0.110 | 8 | Herpesvirus (2.9e-03,5.46) | 2 | GSE219208 (8.9e-17,9.55) | 2 | "GPL21697" (6.6e-06,3.58) |
| 6 | 120 | GSE213142 | 0.067 | 1 | Subunit (2.8e-02,5.48) | 0 |  | 0 |  |
| 7 | 140 | GSE213142 | 0.100 | 8 | Proteins (4.6e-03,1.51) | 0 |  | 0 |  |
| 8 | 176 | GSE213142 | 0.091 | 13 | Transposable (2.0e-03,6.53) | 3 | GSE242259 (1.9e-04,9.34) | 1 | "GPL24676" (5.2e-03,1.52) |
| 9 | 111 | GSE214921 | 0.054 | 28 | Adult (1.8e-12,5.22) | 2 | GSE214921 (1.3e-05,14.80) | 3 | GPL16809 (1.3e-04,14.80) |

## Interpretation Guide

- High `mesh_sig` with moderate/low `gse_purity` suggests biology-like themes not tied to one study.
- Very high `gse_purity` and strong `best_gse` indicates a study/batch-driven cluster (still real structure, but a confound for cross-study biology).
- Platform enrichment is a common technical confound; treat strong `best_plat` similarly.
