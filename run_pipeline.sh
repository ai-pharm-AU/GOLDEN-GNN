#!/usr/bin/env bash
# run_pipeline.sh — Reproduce the full GoldenF pipeline from within deliverables/
#
# Usage:
#   cd /path/to/deliverables
#   bash run_pipeline.sh [--device cpu|cuda]
#
# Prerequisites:
#   - Python venv at /home/zzz0054/GoldenF/.venv (or set VENV_PATH env var)
#   - All data in data/human_only_085_090/ (pre-filtered human-only inputs)
#   - data/RummaGEO/rummageo_descriptions.tsv
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"   # deliverables/ root

# Allow overriding the venv via environment variable
VENV_PATH="${VENV_PATH:-/home/zzz0054/GoldenF/.venv}"
source "${VENV_PATH}/bin/activate"

# Parse optional --device argument
DEVICE="cpu"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

echo "=============================================="
echo " GoldenF Pipeline — deliverables/"
echo " repo_root : ${REPO}"
echo " device    : ${DEVICE}"
echo " python    : $(which python)"
echo "=============================================="

# ---------------------------------------------------------------------------
# Stage 03+: GNN Structlite + Metrics + Validation + Plots
# ---------------------------------------------------------------------------
echo ""
echo "--- Stage 03: GNN Structlite + downstream metrics ---"
python "${REPO}/code/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/scripts/run_single_085_090_human_only.py" \
  --repo_root "${REPO}" \
  --prefiltered_data_root "${REPO}/data/human_only_085_090" \
  --device "${DEVICE}"

# ---------------------------------------------------------------------------
# Stage 06: Knowledge Graph (uses the latest run from outputs/runs/ or rerun/)
# ---------------------------------------------------------------------------
echo ""
echo "--- Stage 06: Knowledge Graph ---"
cd "${REPO}"
python code/KnowledgeGraph/kg_cluster_metrics.py
python code/KnowledgeGraph/kg_matched_comparison.py
python code/KnowledgeGraph/kg_cluster_summarize.py
python code/KnowledgeGraph/vis_case_study_A8_B7.py

echo ""
echo "=============================================="
echo " Pipeline complete."
echo " Outputs: ${REPO}/outputs/"
echo " KG:      ${REPO}/KnowledgeGraph/cluster_metrics_out/"
echo "=============================================="
