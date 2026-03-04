# LLM-Based Super–Gene Set Summarization and Knowledge Graph Construction

## 1. Gene Overlap Edge Construction

### 1.1 Similarity Metric: Composite Jaccard–Simpson Score

For each gene-set pair (G_i, G_j), the gene overlap is determined as the set intersection of their constituent gene symbols:

> OLAP = |G_i ∩ G_j|

The similarity score is a weighted composite of the Jaccard and Simpson coefficients:

> S(G_i, G_j) = α · J(G_i, G_j) + (1 − α) · C(G_i, G_j)

where:

> J(G_i, G_j) = |G_i ∩ G_j| / |G_i ∪ G_j| = OLAP / (|G_i| + |G_j| − OLAP)
>
> C(G_i, G_j) = |G_i ∩ G_j| / min(|G_i|, |G_j|)

The Jaccard index *J* penalizes pairs of very different sizes, while the Simpson coefficient *C* (the overlap coefficient) captures containment relationships where a smaller gene set is largely subsumed by a larger one. We set α = 0.8, weighting Jaccard dominantly to favor balanced overlap over containment, while retaining a 20% contribution from Simpson to preserve sensitivity to biologically meaningful subset relationships (e.g., a focused signaling pathway contained within a broader regulatory program).

### 1.2 Statistical Significance: Hypergeometric Test

To distinguish genuine functional overlap from chance co-occurrence, we employ a hypergeometric distribution test. Given a gene universe of size *N* (the total number of unique gene symbols across all gene sets; *N* = 62,088 in our dataset), only gene-set pairs whose observed overlap exceeds random expectation are retained:

> Retain pair only if: N · OLAP > n₁ · n₂

where n₁ and n₂ are the sizes of the two gene sets. Pairs failing this filter are discarded outright, ensuring that only statistically enriched overlaps enter downstream analysis. For qualifying pairs, we compute:

| Statistic | Definition | Purpose |
|-----------|-----------|---------|
| **NLOGPMF** | −log P(X = k) | Point significance of the exact overlap count |
| **NLOGCDF** | −log F(min(n₁, n₂) − k; N, n₁, n₂) | Tail significance; penalizes near-complete containment |
| **CDF** | F(k − 1; N, n₁, n₂) | Cumulative probability below observed overlap |

where *F* is the hypergeometric CDF. This filter acts as a conservative pre-screen: if two gene sets overlap no more than expected under the null hypothesis of random gene assignment, the pair is excluded regardless of its raw Jaccard score.

### 1.3 Similarity Threshold Filtering

From the full pairwise computation, we apply a composite similarity threshold to retain only high-confidence edges. Two graph variants were constructed at different thresholds:

| Threshold | Nodes | Edges | Rationale |
|-----------|------:|------:|-----------|
| 0.80 | 28,437 | 176,163 | Broadest; baseline |
| **0.85** | **4,343** | **6,630** | Intermediate; balanced specificity |
| **0.90** | **1,643** | **2,387** | Strictest; highest confidence |

Knowledge graphs were constructed for both the 0.85 and 0.90 thresholds, with further restriction to *Homo sapiens*-only gene sets by cross-referencing organism annotations from the master metadata.

The computation is implemented in `mtype.py` using SciPy's `hypergeom` distribution functions and parallelized across gene-set pairs via Python's `multiprocessing.Process`.

---

## 2. Knowledge Graph Construction

### 2.1 Cluster Derivation

Gene sets were clustered using k-means (k = 10, seeds 11–15, best-inertia run selected) applied independently to two GNN embedding spaces:

- **A-series (A0–A9):** GNN text-only embeddings
- **B-series (B0–B9):** GNN text + structural-lite embeddings

Both embedding matrices were inner-joined on common gene set IDs before clustering. This yields 20 cluster groups (10 per GNN variant), each serving as the node set for an independent knowledge graph.

### 2.2 Edge Types

Each knowledge graph represents gene sets as nodes and defines edges using six types of relationships:

**Type 1: Gene Overlap** (`gene_overlap`)
Derived from the pre-computed PAGER pairwise statistics described in Section 1. Only intra-cluster pairs with similarity exceeding the threshold (0.85 or 0.90) are retained. Each edge carries `overlap_count`, `similarity` (composite Jaccard–Simpson), and `neg_log_pval` (−log CDF from hypergeometric test).

**Type 2: Shared PubMed References** (`pubmed_costudy`)
Gene-set pairs whose source GEO series are linked to the same PubMed article(s). PMIDs are extracted from two sources: (i) the `pmids` column in the enriched metadata CSV (direct PMID annotations), and (ii) NCBI E-utilities esearch API queries for GEO series accession numbers in PubMed, enriching gene sets lacking direct PMID annotations. Each edge records the number of shared PMIDs and the specific PMID identifiers.

**Type 3: Shared GO Biological Process Terms** (`shared_go_bp`)
Gene member lists for each gene set are submitted to the Enrichr API against the `GO_Biological_Process_2023` library. Only terms with adjusted p ≤ 0.05 are retained. To remove non-discriminative terms, GO terms appearing in more than 80% of gene sets within a cluster are excluded (e.g., highly generic terms like "metabolic process" that would connect nearly all nodes). Each edge records the shared GO terms and the minimum adjusted p-value across both gene sets.

**Type 4: Shared Reactome Pathways** (`shared_pathway`)
Gene member lists are submitted to the Reactome AnalysisService (identifiers/projection endpoint). Pathways returned for each gene set are indexed, and pairs sharing at least one pathway receive an edge. Each edge records the shared pathway IDs and names.

**Type 5: Shared Disease Associations** (`shared_disease`)
Gene member lists are submitted to Enrichr against the `DisGeNET` library. The same quality filters apply: adjusted p ≤ 0.05 and exclusion of terms appearing in more than 80% of gene sets. This provides disease-specific signal beyond demographic MeSH annotations.

**Type 6: Shared MeSH Descriptors** (`shared_mesh_term`)
MeSH descriptors are extracted from the metadata CSV's `mesh_descriptors` column (semicolon-delimited). Gene-set pairs sharing at least one MeSH term receive an edge. This captures shared biological context and disease annotations from the underlying GEO studies.

### 2.3 Gene Member Retrieval

Gene members for each gene set were retrieved via the RummaGEO GraphQL API in batches of 20 gene sets per request. Results were cached locally (`gene_members_cache.json`) to avoid redundant API calls. Gene symbols starting with "ENSG" or "ENST" were filtered out prior to Enrichr submission to ensure compatibility with the Enrichr gene symbol vocabulary. Full gene member coverage (100% of gene sets) was achieved across all clusters.

### 2.4 Graph Assembly

The resulting multi-relational knowledge graph is represented as a NetworkX MultiGraph, allowing multiple edge types between any node pair. Edges from all six sources are merged into a single graph. Per-cluster outputs include:

- GraphML and JSON (node-link format) exports
- Per-group edge list CSVs
- Interactive vis.js HTML visualizations for each of the 20 cluster groups
- A summary dashboard (`index.html`) linking all groups with graph-level statistics

---

## 3. LLM-Based Cluster Summarization

### 3.1 Model

To enhance interpretability, we applied large language model–based summarization to each cluster group. We used **OpenBioLLM-8B** (aaditya/Llama3-OpenBioLLM-8B), a Llama-3-8B model fine-tuned on biomedical corpora, loaded with 4-bit quantization (BitsAndBytes, `bnb_4bit_compute_dtype=float16`) distributed across three GPUs (~18 GB total VRAM).

Generation parameters: `max_new_tokens=200`, `temperature=0.3`, `do_sample=True`.

### 3.2 Input Construction

For each of the 20 cluster groups, the LLM prompt was constructed from three data sources:

1. **Top 5 gene sets by network centrality:** Gene set names and GEO series summaries (truncated to 300 characters) for the five highest degree-centrality nodes in the cluster's knowledge graph.

2. **Top 10 shared Reactome pathways:** The most frequently appearing pathway names across `shared_pathway` edges within the cluster, ranked by edge frequency.

3. **Top 10 shared GO Biological Process terms:** The most frequently appearing GO BP terms across `shared_go_bp` edges, with GO ID suffixes stripped for readability.

### 3.3 Prompt Design

The prompt follows the Llama-3 instruction template with a system role establishing the bioinformatics analysis context:

> "You are an expert bioinformatician analyzing gene set clusters from GNN-based embedding analysis of human transcriptomics data."

The user turn provides the cluster size, embedding variant label (GNN text-only or GNN-beta structlite), top gene sets with summaries, top pathways, and top GO terms. The model is instructed to produce a concise 2–3 sentence biological summary focusing on cell biology, tissue type, disease context, or molecular mechanisms.

### 3.4 Output

Summaries were generated for all 20 clusters (A0–A9 and B0–B9). Each summary captures dominant biological themes while reducing redundancy across the constituent gene sets. Outputs are stored as a structured CSV (`cluster_summaries.csv`) and a formatted HTML dashboard (`cluster_summaries.html`) with per-cluster cards showing the summary alongside the input gene sets, pathways, and GO terms.

These summaries facilitate construction of an interpretable gene–function knowledge graph by providing human-readable functional annotations at the cluster level, enabling rapid identification of biological themes across the GNN-derived embedding space.

---

*Generated from the GoldenF codebase (`mtype.py`, `build_kg.py`, `kg_cluster_metrics.py`, `kg_cluster_summarize.py`). March 2026.*
