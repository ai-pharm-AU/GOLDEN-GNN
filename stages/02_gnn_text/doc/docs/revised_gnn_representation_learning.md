# GOLDEN-GNN Representation Learning (Code-Aligned Description)

## Drop-In Revised Paragraph (Manuscript-Ready)

We trained an unsupervised GraphSAGE-style graph neural network on the m-type weighted gene-set graph, where nodes are gene sets and undirected edges represent significant m-type associations weighted by `NLOGPMF` (with duplicate edges merged by maximum weight and weights stabilized via a `log1p` transform). Node attributes were initialized from L2-normalized embedding vectors (GO baseline; 384D by default), and in “hybrid” configurations we concatenated these text/ontology embeddings with standardized metadata covariates (e.g., gene-set size, direction, organism, PubMed presence/year) and lightweight structural descriptors computed from the same graph (e.g., log-degree/strength, weighted clustering coefficient, PageRank; optionally core/triangle and approximate centrality features). The encoder aggregates normalized weighted neighborhoods using sparse full-batch propagation, concatenating self and neighbor summaries at each layer, followed by a linear transform, ReLU, and dropout (default: 2 layers, 256D latent, dropout 0.1). Training used an edge-reconstruction objective with a dot-product decoder and binary cross-entropy on logits, sampling up to 50,000 observed edges per epoch as positives and an equal number of uniformly sampled non-edges as negatives (Adam, lr=1e-3). For holdout link prediction, edges were split by seeded random permutation into train/validation/test (0.8/0.1/0.1); message passing used the training graph only, while model selection used validation AUC (evaluated every 10 epochs) with optional early stopping via a patience parameter. The best-checkpoint node embeddings were exported and clustered downstream (primarily KMeans; with Agglomerative/KMedoids and optional HDBSCAN in exploratory analyses), with cluster quality assessed via internal separation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz) and stability via ARI across seeds/holdouts. Clusters were interpreted as “super-PAGs” by mapping member gene sets to genes and taking the union of genes per cluster to form super–gene sets for downstream enrichment and overlap analyses.

We learn unsupervised representations for gene sets (nodes) on the *m-type* weighted gene-set network, where each undirected edge links two gene sets and carries an m-type significance weight (default column: `NLOGPMF`). Node features are initialized from precomputed embedding vectors and, in the “hybrid” setting, augmented with lightweight metadata and structural descriptors. The primary architecture implemented in this codebase is a **GraphSAGE-style encoder** with a dot-product edge decoder (scripts: `work_alpha_gnn_20260212/scripts/gnn_unsupervised_pilot.py`, `work_alpha_gnn_20260212/scripts/gnn_linkpred_holdout.py`).

## Graph Construction and Preprocessing

Edges are loaded from a tab-delimited file with columns `GS_A_ID`, `GS_B_ID`, and a configurable weight column (default: `NLOGPMF`). The pipeline:

1. Aligns graph endpoints to the set of node IDs present in the feature table (IDs not present in the features are dropped).
2. Treats the graph as **undirected** by canonicalizing each pair to `(min(u,v), max(u,v))`, removes self-edges, and merges duplicate pairs by taking the **maximum** weight.
3. Transforms edge weights for numerical stability (default: `log1p`; alternatives: `none`, `sqrt`).
4. Builds a sparse adjacency matrix with **explicit self-loops** (diagonal weight = 1) and applies **symmetric normalization**:
   \[
   \hat{A} = D^{-1/2} (A + I) D^{-1/2}.
   \]

In the holdout protocol, the normalized adjacency is constructed **from training edges only** (validation/test edges are excluded from message passing), so evaluation reflects generalization rather than leakage through the graph operator.

## Node Feature Initialization (Including “Hybrid” Features)

All feature matrices are treated as node attributes and are **L2-normalized per node** prior to GNN training.

- **Baseline**: GO embedding vectors (e.g., RummaGEO; 384D in the default configuration).
- **Hybrid feature recipes (ablation workbench)**: concatenations of (i) GO embeddings (“text”), (ii) standardized metadata features, and/or (iii) “structlite” graph descriptors computed from the same edge file. Concretely, the feature builder (`work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/scripts/build_node_features_from_meta.py`) constructs:
  - Metadata block (z-scored): `log1p(GS_SIZE)`, one-hot direction (`up`/`dn`), `pubmed_present`, a scaled publication-year summary with missingness indicator, and one-hot organism indicators (top organisms).
  - Structural block (normalized by `zscore`/`robust`/`rankgauss`): `log1p(degree)`, `log1p(weighted strength)`, weighted clustering coefficient, PageRank; optionally core number and triangle counts (`basic_plus`) and approximate centralities (`centrality_plus`).
  - Optional node2vec/PAGER block can be PCA-reduced and concatenated for additional structural signal.

Each recipe is saved as an NPZ with keys `ID` and `embeddings` and can be passed directly into the GNN training scripts.

## Unsupervised Objective and Training Protocols

### Encoder–Decoder Model (GraphSAGE-style)

The encoder is a GraphSAGE-style message passing network implemented with sparse matrix multiplication:

- At each layer, neighbor aggregation is computed as `neigh = A_hat @ h`.
- The model concatenates self and neighbor representations: `[h || neigh]`.
- A linear transform + ReLU is applied, followed by dropout (default `p=0.1`).
- The final layer outputs a latent vector `z_i` (default dimensionality: 256).

The decoder scores candidate edges via dot product,
\[
s(u,v)= z_u^\top z_v,
\]
and is trained with binary cross-entropy **on logits** (positive edges labeled 1, negatives labeled 0).

### (A) Pilot: Edge Reconstruction (No Holdout)

In the pilot setting (`gnn_unsupervised_pilot.py`), all observed edges are used as the positive pool. Each epoch:

1. Computes node embeddings with a full-batch forward pass over the sparse normalized adjacency.
2. Samples up to `pos_samples_per_epoch` positive edges uniformly without replacement from the observed edge list (default: 50,000).
3. Samples an equal number of negative pairs by uniform random node-pair sampling, rejecting self-pairs and rejecting any pair present in the observed edge set.
4. Optimizes the sum of BCE losses for positive and negative logits using Adam (default `lr=1e-3`).

The script writes a training curve (epoch, loss, mean positive logit, mean negative logit) and saves the final latent matrix to NPZ (`ID`, `embeddings`).

### (B) Holdout: Link Prediction With Model Selection

For a stricter quality-control and selection signal, the holdout protocol (`gnn_linkpred_holdout.py`) performs a seeded **random edge split** into train/validation/test (defaults: 0.8/0.1/0.1). Key details:

- The message-passing adjacency is built from **train edges only** (including transformed weights and self-loops).
- Validation/test negatives are sampled once using a fixed RNG and are constrained to be non-edges with respect to the **full observed edge set**, ensuring negatives are not accidentally “positives” elsewhere in the data.
- During training, each epoch samples up to 50,000 positive train edges and a 1:1 number of negatives sampled as non-edges with respect to the **train edge set**.
- Every `eval_every` epochs (default: 10), the model is scored on validation and test sets using AUC and average precision (AP).
- The “best” checkpoint is selected by **maximum validation AUC**; optional early stopping can be enabled via `--patience` (stop if validation AUC does not improve for `patience` evaluation steps).

Embeddings can be exported from the best checkpoint for downstream clustering and biological validation.

## Cluster Discovery and Super-Gene Set (“Super-PAG”) Formation

Latent embeddings are clustered downstream using standard unsupervised clustering algorithms. In this repository:

- Internal clustering evaluation is implemented for **KMeans** and **Agglomerative clustering** (with optional KMedoids) across a grid of `k` values and random seeds, reporting Silhouette (Euclidean and cosine), Davies-Bouldin, and Calinski-Harabasz indices (`work_alpha_gnn_20260212/scripts/evaluate_single_embedding.py`).
- Exploratory utilities also include **HDBSCAN** (e.g., `demo.py`, `clustering/clustering_evaluator.py`) when a variable number of clusters is desired.
- Stability is assessed via **pairwise ARI across seeds** and via perturbation/holdout analyses (e.g., leave-one-GSE-out ARI: `work_alpha_gnn_20260212/scripts/leave_one_gse_out_stability.py`).

To construct “super-PAGs” as *super–gene sets*, each embedding-derived cluster (a set of gene-set nodes) is converted to a gene set by taking the **union of member genes** using the gene-set membership table (`GS_ID`, `GENE_SYMBOL`). This union-of-genes representation is then used for downstream analyses (e.g., cluster-level over-representation testing) and for comparing cluster similarity via gene overlap metrics (Jaccard/overlap coefficient).
