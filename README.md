# GOLDEN-GNN: Advancing Interpretable Cross-Domain Data Harmonization in the CFDE
Zeru Zhang1, Qi Li2, Cody Nichols3, Robert S Welner4, Hao Chen*, Jake Y. Chen5*, and Zongliang Yue1*
1. Health Outcomes Research and Policy, Harrison College of Pharmacy, Auburn University, AL, USA
2. Mathematics and Computer Science Department, School of Natural Sciences Mathematics & Business, Fisk University, TN, USA
3. Computer Science and Software Engineering Department, Samuel Ginn College of Engineering, Auburn University, Auburn, AL, USA
4. Hematology & Oncology, School of Medicine, University of Alabama at Birmingham, AL, USA
5. Biomedical Informatics and Data Science, School of Medicine, University of Alabama at Birmingham, AL, USA
Contact: hzc0134@auburn.edu, jakechen@uab.edu, and zzy0065@auburn.edu


## Deliverables Index

- Upload-ready pipeline deliverables package: [`README.md`](README.md)
- Run full deliverables pipeline: `bash run_pipeline.sh --device cpu|cuda`
- Stage-by-stage commands: [`REPRO_COMMANDS.md`](REPRO_COMMANDS.md)
- Consistency verification script: `python verify_consistency.py`

# Human-Only GNN Structlite and Knowledge Graph Deliverables Tool

This deliverables package reproduces the human-only subset `0.85/0.90` workflow from local code in this repository. It runs Stage 03 (GNN Structlite + metrics/plots) and Stage 06 (knowledge graph analysis), then writes run artifacts, tables, and interactive HTML outputs.

Large files over `95MB` are listed in [`HF_UPLOAD_QUEUE.csv`](HF_UPLOAD_QUEUE.csv), and package inventory is tracked in [`MANIFEST.csv`](MANIFEST.csv).

---

## Features

* **One-command orchestration** using `run_pipeline.sh`:

  * Stage 03 human-only regeneration (`subset085`, `subset090`)
  * Stage 06 KG metrics, matched comparison, cluster summaries, case-study visualization
* Uses shipped human-only inputs in [`data/human_only_085_090/`](data/human_only_085_090/README.md)
* Produces UMAP plots, internal metrics tables, and additional comparison metrics
* Produces KG CSV/HTML outputs under `KnowledgeGraph/cluster_metrics_out/`
* Includes reproducibility/support files: `REPRO_COMMANDS.md`, `MANIFEST.csv`, `HF_UPLOAD_QUEUE.csv`, `_meta/source_map.json`

---

## Installation

Install dependencies:

```bash
cd /home/zzz0054/deliverables
pip install -r requirement.txt
```

Set and activate virtual environment (default expected by `run_pipeline.sh`):

```bash
export VENV_PATH=/home/zzz0054/GoldenF/.venv
source "$VENV_PATH/bin/activate"
```

Ensure your project has:

* `data/human_only_085_090/subset085/*` and `data/human_only_085_090/subset090/*`
* `data/RummaGEO/rummageo_descriptions.tsv`
* Write access to `outputs/` and `KnowledgeGraph/cluster_metrics_out/`

---

## Usage

General command:

```bash
cd /home/zzz0054/deliverables
bash run_pipeline.sh \
    --device cpu
```

Direct Stage 03 command used by the wrapper:

```bash
python code/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/scripts/run_single_085_090_human_only.py \
    --repo_root "$(pwd)" \
    --prefiltered_data_root "$(pwd)/data/human_only_085_090" \
    --device cpu
```

---

## Command-Line Arguments

| Argument | Type | Default | Description |
| ------------------------------ | ------ | ---------------------------------- | ------------------------------------------------------------------------------- |
| `run_pipeline.sh --device` | str | `cpu` | Device passed to Stage 03 runner (`cpu` or `cuda`) |
| `run_single ... --python_bin` | str | current interpreter | Python executable used to launch downstream scripts |
| `run_single ... --repo_root` | str | auto-detected | Deliverables repo root |
| `run_single ... --prefiltered_data_root` | str | `None` | Prefiltered human-only data directory; skips organism filtering path |
| `run_single ... --master_meta_csv` | str | `None` | Required only when `--prefiltered_data_root` is not provided |
| `run_single ... --seeds` | str | `11,12,13,14,15` | Comma-separated seeds for k-means evaluations |
| `run_single ... --epochs` | int | `120` | Link prediction training epochs for `gnn_linkpred_holdout.py` |
| `run_single ... --eval_every` | int | `10` | Evaluation frequency during link prediction training |
| `run_single ... --n_perm` | int | `200` | Number of permutations in meta enrichment validation |
| `run_single ... --device` | str | `cpu` | Device for GNN/link prediction subprocesses |
| `VENV_PATH` (env var) | str | `/home/zzz0054/GoldenF/.venv` | Virtual environment path sourced by `run_pipeline.sh` |

---

## Output Files

If `run_pipeline.sh` is used:

### `outputs/runs/single_085_090_human_only_<timestamp>/`

Contains run artifacts per subset:

```text
subset085/
subset090/
run_manifest_human_only.md
```

### `outputs/plots/single_085_090_human_only_<timestamp>/`

Contains:

```text
umap_subset085_fusion_alpha05.png
umap_subset085_gnn_old_holdout.png
umap_subset085_gnn_structlite.png
internal_metrics_table_subset085.csv
additional_metrics_comparison_subset085_090_combined.csv
README.md
```

### `KnowledgeGraph/cluster_metrics_out/`

Contains:

```text
cluster_assignments_k10.csv
knowledge_coverage.csv
centrality.csv
graph_stats.csv
cluster_matched_comparison.csv
cluster_summaries.csv
comparison.html
index.html
vis_sparse/case_study_A8_B7.html
```

---

## How It Works (Overview)

1. **Activate env and parse runtime options** from `run_pipeline.sh`.
2. **Run Stage 03 regeneration** with `run_single_085_090_human_only.py` on `subset085/090`.
3. **Train + evaluate embeddings** (`fusion`, `gnn`, `gnn_structlite`) and export UMAP/metrics tables.
4. **Run KG metrics pipeline** across A0-A9 and B0-B9 groups.
5. **Run KG matched comparison + LLM summarization + A8/B7 case study**.
6. **Write outputs** to `outputs/runs`, `outputs/plots`, and `KnowledgeGraph/cluster_metrics_out`.

---

## Supported Clustering Methods

### Embedding variants evaluated:

* **fusion** (`fusion_subset_*_human.npz`)
* **gnn** (`gnn_old_holdout_subset_*_human.npz`)
* **gnn_structlite** (`text_structlite_gnn_holdout_subset*.npz`)

### Clustering and matching strategy:

* **KMeans** at `k=10` with seeds `11,12,13,14,15` for embedding clustering/evaluation
* **Hungarian matching** for one-to-one A-cluster/B-cluster correspondence in KG comparison

---

## Example Run Manifest (`run_manifest_human_only.md`)

```text
# Human-only single_085_090 run manifest
- seeds: 11,12,13,14,15
- epochs: 120
- eval_every: 10
- n_perm: 200
- device: cpu
- output_plot_dir: /home/zzz0054/deliverables/outputs/plots/single_085_090_human_only_<timestamp>
```

---

## Internal Module Dependencies

This pipeline depends on:

```python
# Stage 03 driver
code/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/scripts/run_single_085_090_human_only.py

# Stage 03 subprocess scripts called by the driver
code/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/scripts/build_node_features_from_meta.py
code/work_alpha_gnn_20260212/scripts/gnn_linkpred_holdout.py
code/work_alpha_gnn_20260212/scripts/evaluate_single_embedding.py
code/work_alpha_gnn_20260212/scripts/validate_cluster_meta_enrichment.py
code/work_alpha_gnn_20260212/scripts/analyze_mesh_vs_gse_confound.py
code/work_alpha_gnn_20260212/scripts/plot_embedding_umap_scatter.py

# Stage 06 KG scripts called by run_pipeline.sh
code/KnowledgeGraph/kg_cluster_metrics.py
code/KnowledgeGraph/kg_matched_comparison.py
code/KnowledgeGraph/kg_cluster_summarize.py
code/KnowledgeGraph/vis_case_study_A8_B7.py
```

---

## Example Output in Terminal

```text
==============================================
 GoldenF Pipeline - deliverables/
 repo_root : /home/zzz0054/deliverables
 device    : cpu
==============================================
--- Stage 03: GNN Structlite + downstream metrics ---
[done] run_root=/home/zzz0054/deliverables/outputs/runs/single_085_090_human_only_20260302_235146
[done] plot_dir=/home/zzz0054/deliverables/outputs/plots/single_085_090_human_only_20260302_235146
--- Stage 06: Knowledge Graph ---
Done. All 20 groups processed.
Open: /home/zzz0054/deliverables/KnowledgeGraph/cluster_metrics_out/index.html
==============================================
 Pipeline complete.
==============================================
```
