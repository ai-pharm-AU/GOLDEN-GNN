# GNN Architecture and Training Configuration

**Project**: Graph-Based Gene-Set Representation Learning  
**Date**: 2026-02-20  
**Model**: GraphSAGE-style Unsupervised Encoder

---

## Model Architecture

### Network Structure
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Model Type** | GraphSAGE | Neighborhood aggregation with concatenation |
| **Number of Layers** | `2` | Two-layer GNN encoder |
| **Receptive Field** | **2-hop** | Each layer aggregates 1-hop neighbors |
| **Hidden Dimension** | `256` | Output embedding dimension |
| **Input Dimension** | `384` | GO baseline feature dimension |
| **Dropout Rate** | `0.1` | Applied after each layer |

### Layer Operation
```
For each layer l:
  neighbor_agg = SparseMatMul(A_norm, h_l)
  h_concat = Concat([h_l, neighbor_agg])  # 2x dimension
  h_l+1 = ReLU(Linear(h_concat))
  h_l+1 = Dropout(h_l+1, p=0.1)
```

### Decoder
- **Type**: Dot-product edge decoder
- **Formula**: `score(u,v) = z_u^T · z_v`
- **Activation**: Sigmoid for probability estimation
- **Loss**: Binary Cross-Entropy with Logits

---

## Training Configuration

### Unsupervised Pilot Training
| Parameter | Value | Notes |
|-----------|-------|-------|
| **Epochs** | `200` | Full unsupervised reconstruction |
| **Learning Rate** | `1e-3` (0.001) | Adam optimizer |
| **Optimizer** | Adam | Default β₁=0.9, β₂=0.999 |
| **Positive Samples/Epoch** | `50,000` | Observed edges sampled per epoch |
| **Negative Sampling** | 1:1 ratio | Equal number of non-edges |
| **Objective** | Edge reconstruction | BCE(pos) + BCE(neg) |

### Holdout Link Prediction Training
| Parameter | Value | Notes |
|-----------|-------|-------|
| **Epochs** | `120` | With validation monitoring |
| **Learning Rate** | `1e-3` | Same as pilot |
| **Validation Ratio** | `0.1` (10%) | Held-out edges for evaluation |
| **Eval Frequency** | Every `10` epochs | AUC/AP computed on validation set |
| **Positive Samples/Epoch** | `50,000` | Training edges only |

---

## Graph Construction

### Edge Processing Pipeline
| Step | Configuration | Purpose |
|------|--------------|---------|
| **Weight Column** | `NLOGPMF` | mtype significance score |
| **Weight Transform** | `log1p` | Stabilize edge weights: log(1+w) |
| **Self-loops** | Added | Unit weight self-connections |
| **Normalization** | Symmetric | D^(-1/2) · A · D^(-1/2) |
| **Duplicate Handling** | Max weight | Merge parallel edges by max |

### Graph Thresholds (Three Variants)

| Threshold | Nodes | Edges | Coverage | Use Case |
|-----------|------:|------:|----------|----------|
| **080** | 28,437 | 176,163 | Broadest | Full dataset baseline |
| **085** | 4,343 | 6,630 | Intermediate | Balanced specificity |
| **090** | 1,643 | 2,387 | Strictest | High-confidence subset |

**Edge Filtering**: Only includes gene-set pairs where `ID` exists in both graph and feature tables.

---

## Data Preprocessing

### Feature Initialization
- **Source**: GO embeddings (RummaGEO)
- **Raw Dimension**: 384
- **Normalization**: L2 row-normalization before training
- **Formula**: `x_i ← x_i / ||x_i||₂`

### Training/Validation Split (Holdout Mode)
- **Method**: Random edge split
- **Training Set**: 90% of edges
- **Validation Set**: 10% of edges
- **Negative Sampling**: Uniform random non-edges (avoiding observed edges)

---

## Performance Benchmarks

### Link Prediction Quality (Holdout Validation)

| Graph | Best AUC | Best AP | Epoch |
|-------|----------|---------|-------|
| **080 (full)** | 0.9829 | 0.9806 | 10 |
| **085 (subset)** | 0.8754 | 0.9079 | 90 |
| **090 (subset)** | 0.9177 | 0.9374 | 110 |

### Clustering Metrics (k=10, 30, 50 averaged)

**Full Graph (080)**:
| Embedding | Sil-Euclidean | Sil-Cosine | Davies-Bouldin | Calinski-Harabasz |
|-----------|---------------|------------|----------------|-------------------|
| GO baseline | 0.0586 | 0.0946 | 3.5234 | 131.75 |
| PAGER (L2) | 0.0542 | 0.0921 | 3.5631 | 130.69 |
| **GNN pilot (L2)** | **0.2230** | **0.3380** | **1.4542** | **1827.79** |
| **GNN holdout (L2)** | **0.1958** | **0.2973** | **1.5988** | **2412.81** |

**Subset 090 (strictest)**:
| Embedding | Sil-Euclidean | Sil-Cosine | Davies-Bouldin |
|-----------|---------------|------------|----------------|
| GO baseline | 0.0532 | 0.0828 | 3.8066 |
| **GNN pilot** | **0.2351** | **0.3587** | **1.4171** |
| **GNN holdout** | **0.2789** | **0.4228** | **1.1521** |

**Key Insight**: Stricter graphs (090) yield better geometric separation.

---

## Experimental Variants

### Feature Ablation Study (task_20260219)
Multiple feature initialization strategies tested with same GNN architecture:
- `text_only`: Pure text embeddings
- `node2vec_only`: Pure structural embeddings
- `text_node2vec`: Combined text + structure
- `text_meta_structlite`: Text + metadata + lightweight structure

**All variants use**:
- `hidden_dim=256`, `num_layers=2`, `epochs=120`
- `lr=1e-3`, `dropout=0.1`

---

## Computational Details

### Hardware Configuration
- **Device**: Auto-select (CUDA if available, else CPU)
- **Precision**: Float32
- **Sparse Operations**: PyTorch sparse tensors for adjacency matrix

### Random Seeds
Multiple seeds used for robustness:
- **Common seeds**: 11, 13, 42, 77
- **Purpose**: Cluster stability analysis, cross-validation

### Output Files
- **Embeddings**: `.npz` format (ID array + embedding matrix)
- **Metrics**: `.csv` format (epoch, loss, AUC, AP)
- **Logs**: Training curves with pos/neg logit separation

---

## References

### Implementation Files
- **Model Definition**: `work_alpha_gnn_20260212/scripts/gnn_unsupervised_pilot.py`
- **Holdout Training**: `work_alpha_gnn_20260212/scripts/gnn_linkpred_holdout.py`
- **Evaluation**: `work_alpha_gnn_20260212/scripts/evaluate_single_embedding.py`

### Documentation
- **Full Pipeline**: `work_alpha_gnn_20260212/reports/paper_style_pipeline_080_085_090_20260214.md`
- **README**: `work_alpha_gnn_20260212/README.md`
- **Metrics Reference**: `docs/metrics_reference.md`

### Data Locations
- **Edge Files**: `data/geneset_extraction_{080,085,090}/edges_with_mtype.txt`
- **GO Features**: `data/RummaGEO/rummageo_go_embeddings.npz`
- **Embeddings**: `work_alpha_gnn_20260212/embeddings/gnn_{pilot,holdout}_embeddings.npz`

---

## Summary

This GNN implementation uses a **2-layer GraphSAGE encoder** to learn **256-dimensional** embeddings via unsupervised edge reconstruction. The model aggregates **2-hop neighborhood** information and consistently outperforms baseline GO/PAGER embeddings on clustering quality metrics. Training uses **120-200 epochs** with **Adam optimizer** (lr=1e-3) on graphs of varying strictness (080/085/090 thresholds), achieving **>0.87 AUC** on link prediction and **3-5x improvement** in silhouette scores over baselines.
