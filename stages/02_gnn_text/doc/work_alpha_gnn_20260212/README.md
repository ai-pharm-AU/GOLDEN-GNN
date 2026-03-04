# Alpha + GNN Workbench (2026-02-12)

This directory contains reproducible experiments for:

- validating the mid-alpha dip hypothesis for concatenated fusion,
- and running an unsupervised graph-neural embedding pilot.

All commands below assume the project venv:

```bash
source /home/zzz0054/GoldenF/.venv/bin/activate
```

## Layout

- `scripts/`: experiment and plotting scripts
- `configs/`: optional command snippets/configs
- `artifacts/logs/`: tmux/log output
- `artifacts/csv/`: tabular experiment results
- `artifacts/plots/`: generated figures
- `embeddings/`: produced embedding files
- `reports/`: summarized findings

## Recommended runs (tmux)

### 1) Fusion hypothesis sweep (kmeans, fast-ish)

```bash
tmux new-session -d -s alpha_fusion_hypothesis \
  "bash -lc 'source /home/zzz0054/GoldenF/.venv/bin/activate && \
   cd /home/zzz0054/GoldenF && \
   python work_alpha_gnn_20260212/scripts/fusion_sweep_experiment.py \
     --gs_file data/RummaGEO/gs_ids.txt \
     --go_embedding_file data/RummaGEO/rummageo_go_embeddings.npz \
     --pager_embedding_file data/RummaGEO/rummageo_pager_embeddings.npz \
     --merge_methods concat_linear,concat_sqrt,concat_postl2,weighted_norm_both \
     --pre_normalize both \
     --algos kmeans \
     --k_list 10,20,30,40,50 \
     --seeds 13,42,77 \
     --alpha_start 0.0 --alpha_stop 1.0 --alpha_step 0.05 \
     --cut_dim 384 \
     --output_csv work_alpha_gnn_20260212/artifacts/csv/fusion_hypothesis_kmeans.csv \
     2>&1 | tee work_alpha_gnn_20260212/artifacts/logs/fusion_hypothesis_kmeans.log'"
```

### 2) Plot + endpoint-vs-middle summary

```bash
python work_alpha_gnn_20260212/scripts/plot_fusion_metrics.py \
  --input_csv work_alpha_gnn_20260212/artifacts/csv/fusion_hypothesis_kmeans.csv \
  --output_png work_alpha_gnn_20260212/artifacts/plots/fusion_hypothesis_kmeans.png \
  --summary_csv work_alpha_gnn_20260212/reports/baseline/fusion_hypothesis_summary.csv
```

### 3) Unsupervised GNN pilot (GraphSAGE-style edge reconstruction)

```bash
tmux new-session -d -s gnn_pilot \
  "bash -lc 'source /home/zzz0054/GoldenF/.venv/bin/activate && \
   cd /home/zzz0054/GoldenF && \
   python work_alpha_gnn_20260212/scripts/gnn_unsupervised_pilot.py \
     --edge_file data/geneset_extraction_080/edges_with_mtype.txt \
     --feature_npz data/RummaGEO/rummageo_go_embeddings.npz \
     --weight_column NLOGPMF \
     --hidden_dim 256 \
     --num_layers 2 \
     --epochs 200 \
     --lr 1e-3 \
     --dropout 0.1 \
     --pos_samples_per_epoch 50000 \
     --output_npz work_alpha_gnn_20260212/embeddings/gnn_pilot_embeddings.npz \
     --metrics_csv work_alpha_gnn_20260212/artifacts/csv/gnn_pilot_training_curve.csv \
     2>&1 | tee work_alpha_gnn_20260212/artifacts/logs/gnn_pilot.log'"
```

### 4) Evaluate a single embedding file (GO/PAGER/GNN)

```bash
python work_alpha_gnn_20260212/scripts/evaluate_single_embedding.py \
  --embedding_file work_alpha_gnn_20260212/embeddings/gnn_pilot_embeddings.npz \
  --gs_file data/RummaGEO/gs_ids.txt \
  --algos kmeans \
  --k_list 10,20,30,40,50 \
  --seeds 13,42,77 \
  --output_csv work_alpha_gnn_20260212/artifacts/csv/gnn_pilot_eval.csv
```

### 5) Holdout link prediction (GNN quality check)

```bash
tmux new-session -d -s gnn_holdout \
  "bash -lc 'source /home/zzz0054/GoldenF/.venv/bin/activate && \
   cd /home/zzz0054/GoldenF && \
   python work_alpha_gnn_20260212/scripts/gnn_linkpred_holdout.py \
     --edge_file data/geneset_extraction_080/edges_with_mtype.txt \
     --feature_npz data/RummaGEO/rummageo_go_embeddings.npz \
     --output_csv work_alpha_gnn_20260212/artifacts/csv/gnn_holdout_linkpred.csv \
     --output_npz work_alpha_gnn_20260212/embeddings/gnn_holdout_embeddings.npz \
     2>&1 | tee work_alpha_gnn_20260212/artifacts/logs/gnn_holdout.log'"
```

### 6) Graph change adaptation (1%-5% perturbation)

```bash
tmux new-session -d -s gnn_adapt \
  "bash -lc 'source /home/zzz0054/GoldenF/.venv/bin/activate && \
   cd /home/zzz0054/GoldenF && \
   python work_alpha_gnn_20260212/scripts/gnn_graph_change_adaptation.py \
     --edge_file data/geneset_extraction_080/edges_with_mtype.txt \
     --feature_npz data/RummaGEO/rummageo_go_embeddings.npz \
     --output_csv work_alpha_gnn_20260212/artifacts/csv/gnn_graph_change_adaptation.csv \
     2>&1 | tee work_alpha_gnn_20260212/artifacts/logs/gnn_graph_change_adaptation.log'"
```

### 7) Text coherence + cluster balance check

```bash
python work_alpha_gnn_20260212/scripts/eval_cluster_coherence.py \
  --desc_tsv data/RummaGEO/rummageo_descriptions.tsv \
  --embedding_files data/RummaGEO/rummageo_go_embeddings.npz,data/RummaGEO/rummageo_pager_embeddings.npz,work_alpha_gnn_20260212/embeddings/gnn_pilot_embeddings_l2.npz \
  --k 10 \
  --seeds 13,42 \
  --l2_norm 1 \
  --output_csv work_alpha_gnn_20260212/artifacts/csv/coherence_k10_go_pager_gnnl2.csv
```

### 8) Visual comparison: GNN vs golden fusion

```bash
python work_alpha_gnn_20260212/scripts/plot_gnn_vs_golden_fusion.py \
  --fusion_csv work_alpha_gnn_20260212/artifacts/csv/fusion_robust_kmeans_reduced.csv \
  --gnn_pilot_csv work_alpha_gnn_20260212/artifacts/csv/eval_gnn_pilot_l2_robust.csv \
  --gnn_holdout_csv work_alpha_gnn_20260212/artifacts/csv/eval_gnn_holdout_l2_robust.csv \
  --go_csv work_alpha_gnn_20260212/artifacts/csv/eval_go_robust.csv \
  --pager_csv work_alpha_gnn_20260212/artifacts/csv/eval_pager_l2_robust.csv \
  --golden_alpha 0.25 \
  --summary_csv work_alpha_gnn_20260212/reports/baseline/gnn_vs_golden_summary_by_k.csv \
  --lineplot_png work_alpha_gnn_20260212/artifacts/plots/gnn_vs_golden_metrics_by_k.png \
  --gain_heatmap_png work_alpha_gnn_20260212/artifacts/plots/gnn_vs_golden_gain_heatmap.png
```

### 9) Direct GO-vs-PAGER modality consistency report

```bash
python work_alpha_gnn_20260212/scripts/modality_consistency_report.py \
  --go_embedding_file data/RummaGEO/rummageo_go_embeddings.npz \
  --pager_embedding_file data/RummaGEO/rummageo_pager_embeddings.npz \
  --cut_dim 384 \
  --pairwise_sample_size 1800 \
  --pairwise_seeds 7,11,19,23,31 \
  --neighbor_k_list 5,10,20,50 \
  --cluster_k_list 10,20,30,40,50 \
  --cluster_seeds 11,22,33
```

Outputs:

- `work_alpha_gnn_20260212/reports/baseline/modality_consistency_summary.csv`
- `work_alpha_gnn_20260212/reports/baseline/modality_consistency_readout.md`
- `work_alpha_gnn_20260212/reports/baseline/modality_consistency_pairwise_stats.csv`
- `work_alpha_gnn_20260212/reports/baseline/modality_consistency_nn_overlap_by_k.csv`
- `work_alpha_gnn_20260212/reports/baseline/modality_consistency_cluster_agreement.csv`
- `work_alpha_gnn_20260212/artifacts/plots/modality_row_cosine_hist.png`
- `work_alpha_gnn_20260212/artifacts/plots/modality_pairwise_similarity_scatter.png`
- `work_alpha_gnn_20260212/artifacts/plots/modality_nn_overlap_vs_k.png`
- `work_alpha_gnn_20260212/artifacts/plots/modality_cluster_ari_vs_k.png`

## tmux helpers

```bash
tmux ls
tmux attach -t alpha_fusion_hypothesis
tmux attach -t gnn_pilot
```

Detach: `Ctrl+b` then `d`

## Bio validation: metadata enrichment + description coherence (GEO + PubMed)

This validates whether unsupervised clusters are meaningful using *independent* annotations:

- text coherence on `NAME` + `DESCRIPTION` (+ optional GEO series text),
- metadata enrichment on PubMed MeSH terms / GEO platforms / study IDs,
- stability across multiple clustering seeds,
- permutation null via label shuffling.

All commands assume the project venv is active:

```bash
source /home/zzz0054/GoldenF/.venv/bin/activate
```

### 1) Build enriched metadata table (with caching)

Offline-only (no network fetch; still parses `GSE...` and PMIDs):

```bash
python work_alpha_gnn_20260212/scripts/fetch_geo_pubmed_metadata.py \
  --embedding_npz work_alpha_gnn_20260212/subset_85_90_20260213/embeddings/gnn_holdout_subset_090_l2.npz \
  --metadata_csv data/mcxqrhgm61ztvegwj8ltzbcvv1h06vni.csv \
  --output_dir work_alpha_gnn_20260212/reports/bio_meta_cache_090 \
  --skip_geo --skip_pubmed
```

Full (fetch GEO + PubMed; writes caches to `--output_dir`):

```bash
python work_alpha_gnn_20260212/scripts/fetch_geo_pubmed_metadata.py \
  --embedding_npz work_alpha_gnn_20260212/subset_85_90_20260213/embeddings/gnn_holdout_subset_090_l2.npz \
  --metadata_csv data/mcxqrhgm61ztvegwj8ltzbcvv1h06vni.csv \
  --output_dir work_alpha_gnn_20260212/reports/bio_meta_cache_090
```

Output: `work_alpha_gnn_20260212/reports/bio_meta_cache_090/enriched_gs_metadata.csv`

### 2) Run validation on the clustered embedding

```bash
python work_alpha_gnn_20260212/scripts/validate_cluster_meta_enrichment.py \
  --embedding_npz work_alpha_gnn_20260212/subset_85_90_20260213/embeddings/gnn_holdout_subset_090_l2.npz \
  --enriched_metadata_csv work_alpha_gnn_20260212/reports/bio_meta_cache_090/enriched_gs_metadata.csv \
  --k 10 \
  --seeds 11,12,13,14,15 \
  --n_perm 200 \
  --output_dir work_alpha_gnn_20260212/reports/bio_validation_090_k10
```

Key outputs:

- `.../text_coherence_by_seed.csv`
- `.../meta_enrichment_results.csv`
- `.../meta_enrichment_seed_summary.csv`
- `.../pairwise_ari.csv`
- `.../validation_report.md`
