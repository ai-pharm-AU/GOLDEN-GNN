# Repro Commands by Stage

## Quick Start (recommended)

Run the entire pipeline with a single command from within `deliverables/`:

```bash
cd /path/to/deliverables
bash run_pipeline.sh             # CPU (default)
bash run_pipeline.sh --device cuda  # GPU
```

This runs Stage 03 (GNN Structlite + metrics + validation + plots) and Stage 06
(Knowledge Graph), using the pre-filtered human-only inputs in `data/human_only_085_090/`.

Outputs are written to:
- `outputs/runs/single_085_090_human_only_<timestamp>/` — per-subset run artifacts
- `outputs/plots/single_085_090_human_only_<timestamp>/` — UMAP plots + tables
- `KnowledgeGraph/cluster_metrics_out/` — KG metrics and visualizations

---

## Stage-by-stage commands

All commands assume you are in the `deliverables/` directory with the venv active.

```bash
cd /path/to/deliverables
source /home/zzz0054/GoldenF/.venv/bin/activate   # or: export VENV_PATH=... && source $VENV_PATH/bin/activate
```

### Stage 01 — mtype Build *(skipped — done on another machine)*

Pre-filtered outputs already available in `data/human_only_085_090/`.

### Stage 02 — GNN (Text) *(reference commands)*

```bash
python code/work_alpha_gnn_20260212/scripts/gnn_unsupervised_pilot.py --help
python code/work_alpha_gnn_20260212/scripts/gnn_linkpred_holdout.py --help
python code/work_alpha_gnn_20260212/scripts/evaluate_single_embedding.py --help
```

Pre-computed GNN text embeddings (`gnn_old_holdout_subset_*_human.npz`) are included
in `data/human_only_085_090/` and used as inputs to Stage 03.

### Stage 03 — GNN Structlite + Metrics

```bash
python code/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/scripts/run_single_085_090_human_only.py \
  --repo_root "$(pwd)" \
  --prefiltered_data_root "$(pwd)/data/human_only_085_090" \
  --device cpu
```

### Stage 04 — Internal, Consistency, Confound, Enrichment Metrics *(reference commands)*

```bash
python code/work_alpha_gnn_20260212/scripts/modality_consistency_report.py --help
python code/work_alpha_gnn_20260212/scripts/validate_cluster_meta_enrichment.py --help
python code/work_alpha_gnn_20260212/scripts/analyze_mesh_vs_gse_confound.py --help
python code/work_alpha_gnn_20260212/scripts/run_gnn_enrichment.py --help
```

### Stage 05 — Visualizations *(reference commands)*

```bash
python code/work_alpha_gnn_20260212/scripts/plot_embedding_umap_scatter.py --help
python code/work_alpha_gnn_20260212/scripts/build_combined_poster_085_090.py --help
```

### Stage 06 — Knowledge Graph

```bash
python code/KnowledgeGraph/kg_cluster_metrics.py
python code/KnowledgeGraph/kg_matched_comparison.py
python code/KnowledgeGraph/kg_cluster_summarize.py
python code/KnowledgeGraph/vis_case_study_A8_B7.py
```

The KG scripts auto-detect the latest run directory from `outputs/runs/` or `rerun/`.
