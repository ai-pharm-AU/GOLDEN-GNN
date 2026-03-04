# Stage 02 — GNN (Text)

## Summary
GraphSAGE-style unsupervised embedding training and holdout evaluation using text/GO-derived features.

## Payload
### Code
- [`work_alpha_gnn_20260212/scripts/evaluate_single_embedding.py`](code/work_alpha_gnn_20260212/scripts/evaluate_single_embedding.py) — Internal clustering metric evaluation for a single embedding.
- [`work_alpha_gnn_20260212/scripts/gnn_linkpred_holdout.py`](code/work_alpha_gnn_20260212/scripts/gnn_linkpred_holdout.py) — Holdout link-prediction training and embedding export.
- [`work_alpha_gnn_20260212/scripts/gnn_unsupervised_pilot.py`](code/work_alpha_gnn_20260212/scripts/gnn_unsupervised_pilot.py) — Pilot unsupervised GNN training script.

### Docs
- [`docs/gnn_architecture_config.md`](doc/docs/gnn_architecture_config.md) — Architecture/training/evaluation configuration summary.
- [`docs/revised_gnn_representation_learning.md`](doc/docs/revised_gnn_representation_learning.md) — Code-aligned GNN representation-learning description.
- [`work_alpha_gnn_20260212/README.md`](doc/work_alpha_gnn_20260212/README.md) — Main reproducibility guide for alpha + GNN workbench.

### Results
- [`work_alpha_gnn_20260212/artifacts/csv/eval_gnn_holdout_l2_robust.csv`](result/work_alpha_gnn_20260212/artifacts/csv/eval_gnn_holdout_l2_robust.csv) — Robust internal-metric evaluation for holdout GNN.
- [`work_alpha_gnn_20260212/embeddings/gnn_holdout_embeddings_l2.npz`](result/work_alpha_gnn_20260212/embeddings/gnn_holdout_embeddings_l2.npz) — Holdout GNN embedding (L2-normalized).
- [`work_alpha_gnn_20260212/embeddings/gnn_pilot_embeddings_l2.npz`](result/work_alpha_gnn_20260212/embeddings/gnn_pilot_embeddings_l2.npz) — Pilot GNN embedding (L2-normalized).
- [`work_alpha_gnn_20260212/reports/baseline/gnn_vs_golden_summary_by_k.csv`](result/work_alpha_gnn_20260212/reports/baseline/gnn_vs_golden_summary_by_k.csv) — Summary comparison between GNN and golden fusion baselines.

## Notes
- Oversized files skipped to HF queue: 0
- Missing source items: 0
