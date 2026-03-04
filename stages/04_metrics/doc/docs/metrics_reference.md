# Metrics Reference
This document details the mathematical definitions, inputs, outputs, and statistical/biological interpretations of the metrics used in this project to evaluate clustering quality and biological relevance.

## 1. Clustering Quality Metrics (Internal)

These metrics evaluate the quality of clusters based solely on the embedding space, without external truth labels.

### Silhouette Score
* **Formula**: For a sample $i$, let $a(i)$ be the mean distance to other points in the same cluster, and $b(i)$ be the mean distance to points in the nearest other cluster.
  $$ s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} $$
  $$ \text{Silhouette} = \frac{1}{N} \sum_{i=1}^{N} s(i) $$
* **Input**: Feature matrix $X$, Cluster labels $L$.
* **Output**: Scalar value between -1 and 1.
* **Statistical Meaning**: Measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation). High values indicate well-separated, dense clusters.
* **Biological Meaning**: A high silhouette score suggests that the gene sets within a cluster are distinctly different from those in other clusters in the embedding space.

### Davies-Bouldin Index
* **Formula**: Let $R_{ij}$ be the ratio of the sum of within-cluster dispersions ($s_i, s_j$) to the distance between cluster centroids ($d_{ij}$).
  $$ R_{ij} = \frac{s_i + s_j}{d_{ij}} $$
  $$ DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} R_{ij} $$
* **Input**: Feature matrix $X$, Cluster labels $L$.
* **Output**: Scalar value $\ge 0$.
* **Statistical Meaning**: Represents the average "similarity" between clusters, where similarity is a function that compares the distance between clusters with the size of the clusters themselves. Lower values are better.
* **Biological Meaning**: A low DB index implies that the derived biological modules are compact and distinct from one another.

### Calinski-Harabasz Index
* **Formula**: Ratio of difference between clusters dispersion ($B_k$) and within-cluster dispersion ($W_k$).
  $$ CH = \frac{Tr(B_k)}{Tr(W_k)} \times \frac{N - k}{k - 1} $$
* **Input**: Feature matrix $X$, Cluster labels $L$.
* **Output**: Scalar value $\ge 0$.
* **Statistical Meaning**: A higher score indicates clusters that are dense and well-separated. It penalizes higher $k$ less aggressively than other metrics.
* **Biological Meaning**: Used to validate that the chosen number of clusters ($k$) provides a distinct partitioning of the biological space.

**Project Notes**: We record both `silhouette_euclidean` and `silhouette_cosine` in `work_alpha_gnn_20260212/scripts/evaluate_single_embedding.py` (cosine uses row-wise L2 normalization). Alpha sweeps in `alpha_sweep.py` and `run_alpha_subset.py` compute Euclidean silhouette/DB/CH on the raw embedding and may subsample via `sample_size`.

## 1b. Cluster Balance Metrics (Sanity Checks)

These metrics flag degenerate clusterings (e.g., one giant cluster).

### Largest Cluster Fraction
* **Definition**: $\max_c |C_c| / N$.
* **Input**: Cluster labels.
* **Output**: Fraction in [0, 1].
* **Interpretation**: High values indicate a dominant cluster and potential collapse.

### Smallest Cluster Fraction
* **Definition**: $\min_{c:|C_c|>0} |C_c| / N$.
* **Input**: Cluster labels.
* **Output**: Fraction in [0, 1].
* **Interpretation**: Very small values indicate tiny clusters or fragmentation.

### Gini Coefficient (Cluster Size Gini)
* **Definition**: Gini coefficient of the non-empty cluster sizes.
* **Input**: Cluster labels.
* **Output**: Scalar in [0, 1] (0 = perfectly balanced, higher = more imbalanced).
* **Interpretation**: Captures overall size inequality across clusters.

**Project Notes**: Computed in `work_alpha_gnn_20260212/scripts/eval_cluster_coherence.py` and reported alongside text coherence in `coherence_*` CSVs.

---

## 2. Clustering Stability & External Validation

These metrics compare two labelings of the same items (e.g., cluster-vs-cluster for seed stability, or cluster-vs-metadata for confound checks).

### Adjusted Rand Index (ARI)
* **Formula**: Measures the similarity between two data clusterings. Adjusted for chance (expected value is 0).
  $$ ARI = \frac{\text{Index} - \text{Expected Index}}{\text{Max Index} - \text{Expected Index}} $$
* **Input**: Two labelings over the same items (e.g., two clustering runs, or clusters vs metadata labels).
* **Output**: Scalar value between -1 and 1.
* **Statistical Meaning**: A value of 1.0 indicates perfect labeling match. 0.0 indicates random labeling.
* **Biological Meaning**: Used to check "Seed Stability". If ARI is high between different random seeds of the same algorithm, the biological modules found are robust and not artifacts of initialization.

### Normalized Mutual Information (NMI)
* **Formula**: Mutual Information normalized by entropy to scale between 0 and 1.
  $$ NMI(Y, C) = \frac{2 \times I(Y; C)}{H(Y) + H(C)} $$
* **Input**: Two labelings over the same items (e.g., two clustering runs, or clusters vs metadata labels).
* **Output**: Scalar value between 0 and 1.
* **Statistical Meaning**: Measures the reduction in uncertainty about one clustering given the other.
* **Biological Meaning**: Often used to check if the clusters simply recapitulate known metadata (e.g., if NMI with "Study ID" is high, the clusters might be batch effects rather than biology).

---

## 3. Biological Enrichment Metrics

These metrics validate if clusters correspond to known metadata or gene-level signatures (MeSH terms, GEO study IDs/platforms, direction, organism, PubMed presence, and gene-level ORA terms).

### Hypergeometric P-value
* **Formula**: Calculates the probability of observing $k$ or more overlaps between a cluster and a biological term by chance.
  $$ P(X \ge k) = \sum_{i=k}^{\min(n, m)} \frac{\binom{m}{i} \binom{M-m}{n-i}}{\binom{M}{n}} $$
  * $M$: Total items in the universe (gene sets for metadata enrichment; genes for gene-level ORA).
  * $n$: Items in the cluster.
  * $m$: Items annotated with the term.
  * $k$: Items in cluster AND annotated with the term.
* **Input**: Cluster set, Term set, Universe set.
* **Output**: Probability value ($0 \le p \le 1$).
* **Statistical Meaning**: The probability that the observed overlap occurred purely by random sampling.
* **Biological Meaning**: A low p-value indicates faithful biological capture. For example, if a cluster of 100 gene sets contains 50 tagged with "Liposomes", and "Liposomes" is rare in the full dataset, the p-value will be extremely low.

### False Discovery Rate (Q-value)
* **Formula**: Benjamini-Hochberg correction applied to the set of P-values.
* **Statistical Meaning**: Controls the expected proportion of false positives among rejected hypotheses.
* **Biological Meaning**: Used to filter enrichment results. A q-value < 0.05 gives high confidence that the biological association is real.

### Enrichment Ratio (ER)
* **Formula**: The observed proportion of the term in the cluster divided by the expected proportion in the background.
  $$ ER = \frac{k / n}{m / M} = \frac{\text{Observed Fraction}}{\text{Background Fraction}} $$
* **Input**: $k, n, m, M$ (same as Hypergeometric).
* **Output**: Scalar factor ($\ge 0$).
* **Statistical Meaning**: Effect size. An ER of 5.0 means the term appears 5 times more frequently in the cluster than expected by random chance.
* **Biological Meaning**: Indicates the strength of the signal. A significant p-value with low ER might just mean a large cluster; a high ER confirms specificity (the cluster is *about* that term).

**Project Notes**: In `validate_cluster_meta_enrichment.py`, terms are filtered by `min_support` (default 5) and BH-FDR is applied across all cluster-by-term tests (k x V). In `run_gnn_enrichment.py`, the same hypergeometric + BH-FDR logic is applied to gene-level ORA with term-size filters.

---

## 4. Text Coherence Metrics

Metrics evaluating the semantic consistency of the textual descriptions of gene sets within a cluster.

### TF-IDF Centroid Coherence
* **Formula**:
  1. Represent all gene set descriptions as TF-IDF vectors.
  2. For each cluster $C$, compute the centroid vector $\mu_C = \frac{1}{|C|} \sum_{d \in C} \vec{v}_d$.
  3. Compute average cosine similarity of items to their centroid: $S_C = \frac{1}{|C|} \sum_{d \in C} \cos(\vec{v}_d, \mu_C)$.
  4. Final score is the weighted average across all clusters:
  $$ \text{Coherence} = \frac{\sum_{C} |C| \cdot S_C}{\sum_{C} |C|} $$
* **Input**: List of text descriptions, Cluster labels.
* **Output**: Scalar value (typically 0.0 to 1.0).
* **Statistical Meaning**: Measures the tightness of the text distribution in vector space.
* **Biological Meaning**: High text coherence implies that the gene sets in the cluster are described using similar language (e.g., all mention "immune response"), providing independent validation that the numeric embeddings captured semantic meaning.

**Project Notes**: In `validate_cluster_meta_enrichment.py`, TF-IDF is built from `NAME`, `DESCRIPTION`, `geo_series_title`, `geo_series_summary`, and `geo_overall_design`, using word n-grams (1-2) plus char n-grams (3-5) with English stop words. We report both raw text and a cleaned variant (GSE IDs, numbers, underscores removed) along with permutation p-values (`text_null_*_p`).

---

## 5. Applied Walkthrough: From Data to Metric

To ground the definitions above, we trace one concrete cluster from the GNN validation run on the 0.90 subset (N=1643) and contrast it with a baseline embedding on the same subset.

### Data Point A: The "Liposomes" Cluster (GNN, subset_090)
* **Source**: `work_alpha_gnn_20260212/reports/bio_validation_090_k10/meta_enrichment_results.csv`
* **Embedding**: `work_alpha_gnn_20260212/subset_85_90_20260213/embeddings/gnn_holdout_subset_090_l2.npz`
* **Cluster ID**: 1 (Seed 11)
* **Algorithm**: KMeans with $k=10$
* **Biological Signal**: "Liposomes" (MeSH)
* **File Reference**: [GNN Enrichment Results](../work_alpha_gnn_20260212/reports/bio_validation_090_k10/meta_enrichment_results.csv)

#### 1. Clustering Quality (Silhouette)
* **Observed Value**: `silhouette_cosine = 0.488` (seed 11, k=10) from `work_alpha_gnn_20260212/subset_85_90_20260213/csv/eval_gnn_holdout_subset_090_l2.csv` (Euclidean silhouette for the same run is 0.330).
* **Interpretation**: A cosine silhouette near 0.49 indicates that the gene sets in Cluster 1 are well-grouped in embedding space and distinct from other clusters.

#### 2. Cross-Validation with External Metadata (Liposomes)
We validate this structure using MeSH terms, which the GNN never saw during training.
* **Counts**:
  * Total Universe ($M$): 1643 gene sets.
  * Cluster Size ($n$): 228 gene sets.
  * Global "Liposomes" sets ($m$): 37.
  * Overlap ($k$): 27.
* **Metric Calculation**:
  * **Expected Overlap**: $m \times (n/M) = 37 \times (228/1643) \approx 5.1$ gene sets.
  * **Enrichment Ratio (ER)**: $\text{Observed} / \text{Expected} = 27 / 5.1 \approx \mathbf{5.26}$.
  * **Hypergeometric P-value**: $P \approx 1.68 \times 10^{-16}$.
  * **Q-value (FDR)**: $q \approx 9.34 \times 10^{-13}$.
* **Conclusion**: The cluster is ~5x enriched for Liposomes. This is strong evidence that the GNN embedding captures liposome-related biology without supervision.

#### 3. Confound Check (Study ID vs Biology)
* **Source**: `work_alpha_gnn_20260212/reports/bio_validation_090_k10/mesh_vs_gse_confound_seed11.md`
* **Variable**: `gse_id` (GSE216249)
* **Metric**: `gse_purity` = 0.118 (11.8%).
* **Interpretation**: Only ~12% of the cluster comes from the top study, suggesting the "Liposomes" signal is not driven by a single experiment.

---

### Data Point B: GO-only Baseline (Same Subset)
* **Source**: `work_alpha_gnn_20260212/subset_85_90_20260213/csv/eval_go_subset_090_l2.csv`
* **Embedding**: `work_alpha_gnn_20260212/subset_85_90_20260213/embeddings/go_embeddings_subset_090_l2.npz`
* **Algorithm**: KMeans with $k=10$ (Seed 11)

#### 1. Clustering Quality (Silhouette)
* **Observed Value**: `silhouette_cosine = 0.090` (seed 11). This is ~82% lower than the GNN's 0.488 on the same subset.
* **Interpretation**: The GO-only embedding produces much fuzzier clusters under the same evaluation setup.

#### 2. Impact on Enrichment (Expected)
We have not run `validate_cluster_meta_enrichment.py` on the GO-only embedding for this subset yet. Given the much lower silhouette, we expect fewer and weaker enriched metadata terms; running the same validation would confirm the gap quantitatively.
