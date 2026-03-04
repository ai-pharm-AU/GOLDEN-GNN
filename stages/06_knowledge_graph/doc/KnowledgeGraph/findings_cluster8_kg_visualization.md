# Cluster 8 Super-PAG Knowledge Graph — Visualization Findings

**Graph:** `KnowledgeGraph/cluster8_subset085_kg.graphml`
**Nodes:** 181 gene sets (95 up-regulated, 86 down-regulated)
**Edges:** 34,124 across 6 evidence types
**Origin:** Intersection of text-only GNN cluster 8 ∩ text+structural GNN cluster 7 (subset threshold 0.85)

Outputs: `kg_sparse_network.html`, `kg_adjacency_heatmaps.png`, `kg_edge_summary.png`

---

## 1. Gene Overlap (8 edges)

**What it measures:** Pairs of gene sets that share actual gene members, scored by Jaccard similarity and negative log p-value (from PAGER).

**Structure:** Extremely sparse — only 8 edges connecting 13 of the 181 nodes. The other 168 nodes have completely disjoint gene membership. In the heatmap, these appear as isolated bright dots against a white background.

**Key observations:**
- Similarity ranges from 0.40 to 1.00 (mean 0.74); all edges are statistically meaningful overlaps.
- The perfect-similarity pair (sim=1.00) is GSE61476 dn ↔ GSE61476 up — the same study's opposing differential expression contrasts, sharing 15 genes.
- Most overlapping pairs come from the **same GSE series** (e.g., GSE112509, GSE244307, GSE164471), suggesting the super-PAG captures recurrent transcriptional programs within individual studies rather than cross-study gene reuse.
- One cross-study overlap exists: GSE114007 up ↔ GSE145118/GSE145119 dn (sim=0.40, 5 genes) — a weak but notable signal.

**Interpretation:** Gene-level content is largely non-redundant across the 181 gene sets. The super-PAG's coherence is driven by semantic/contextual similarity (pathway, MeSH, publication) rather than shared gene membership.

---

## 2. PubMed Co-study (128 edges)

**What it measures:** Pairs of gene sets whose source GEO series are linked to the same PubMed article(s).

**Structure:** Sparse but structured. The heatmap shows a clear **diagonal band pattern** — nodes cluster into study-level blocks along the diagonal after hierarchical reordering. Studies with many contrasts (e.g., GSE231324 with multiple cell types) form dense local cliques.

**Key observations:**
- 128 edges span 40 unique PMIDs; most pairs share exactly 1 PMID (109/128 edges), but some share up to 4.
- The top-connected nodes all belong to **GSE231324** (estrogen/testosterone effects on muscle primary cultures), each with degree 8 — this study contributed the most contrasts to the super-PAG.
- The diagonal band confirms that co-study connectivity is **within-GSE**, not cross-study. Gene sets from different publications are almost never co-cited.

**Interpretation:** The super-PAG aggregates gene sets from ~40 independent publications. Publication-level clustering is tight but shallow — no hub publication dominates the whole network, confirming the cluster represents a genuine biological theme rather than an artifact of any single study.

---

## 3. Shared MeSH Term (9,591 edges)

**What it measures:** Pairs of gene sets whose associated GEO studies share at least one MeSH descriptor (pulled from PubMed metadata as a proxy for shared biological context/disease annotation).

**Structure:** Dense but bimodal. The heatmap shows a large solid-blue block alongside white rows/columns — a clean partition between annotated and unannotated nodes.

**Key observations:**
- 42 of 181 nodes (23%) have **zero MeSH edges** — their source publications lack MeSH annotations in the metadata. These appear as blank rows/columns in the heatmap.
- The remaining 139 nodes are densely interconnected: most share the universal terms **"Humans"** (9,591 edges — present on every annotated pair), **"Female"** (4,278), and **"Male"** (3,403).
- More specific terms appear at lower frequency: **"Transcriptome"** (1,225), **"Adult"** (1,035), **"Gene Expression Profiling"** (820), **"Middle Aged"** (595), **"Aged"** (253).
- Top MeSH-connected nodes are again from **GSE231324** (degree 138 = connected to every other annotated node).

**Interpretation:** The dominant MeSH signal is demographic/methodological (human subjects, mixed sex, adult age range, RNA-seq) rather than disease-specific. This is consistent with the super-PAG representing a broad transcriptional response program in human tissue studies. The 42 unannotated nodes represent a gap that could be filled by manual curation or fallback to DisGeNET.

---

## 4. Shared Pathway (16,290 edges)

**What it measures:** Pairs of gene sets whose gene members both map to at least one common Reactome pathway.

**Structure:** Near-complete graph. The heatmap is almost entirely solid blue — every pair of nodes shares pathway membership. The diagonal (self-loops, excluded) is the only white line visible.

**Key observations:**
- All 181 nodes participate; no isolated nodes.
- Every edge shares between 2 and 138 pathways (mean 52.8 per pair) — gene sets are not just barely connected but deeply co-embedded in pathway space.
- The most universally shared pathways are **"Translation"** and **"Metabolism of proteins"** (both present in all 16,290 edges), followed by ribosomal/translation-initiation sub-pathways (rRNA processing, 43S complex formation, etc.).
- This is a strong signal: the super-PAG is dominated by gene sets with heavy **ribosomal/translational activity**, which is consistent with a stress-response or growth/atrophy transcriptional program.

**Interpretation:** Pathway overlap is so dense it provides weak discriminative power between individual gene set pairs — it essentially confirms all 181 gene sets belong to the same broad biological domain (protein synthesis, translational regulation). To extract sub-structure, differential pathway analysis (which pathways are *unique* to a subset of nodes rather than universal) would be the next step.

---

---

## 5. Shared Disease (7,669 edges)

**Source:** Enrichr API / DisGeNET library (free, no API key required). Gene sets are submitted to EnrichrAPI; pairs with at least one shared significant disease term (adj-p < 0.05) receive an edge.

**What it measures:** Pairs of gene sets whose member genes are enriched for the same disease associations in the DisGeNET database. The filter excludes terms present in ≥80% of gene sets to remove non-discriminative hits; 190 of 196 DisGeNET terms pass this filter.

**Structure:** Medium-dense (~42% of max possible edges; 7,669 edges). The heatmap shows a block structure similar to MeSH but with finer-grain disease specificity — the annotated block is large but not uniform, unlike MeSH's near-complete annotated sub-block.

**Key observations:**
- 7,669 edges across 181 nodes.
- Provides disease-specific rather than demographic signal, in contrast to MeSH (which is dominated by "Humans", "Female", "Male").
- No catch-all universal term equivalent to MeSH's "Humans" — every shared term reflects an actual disease biology connection.
- Block structure mirrors MeSH but discriminates sub-groups within the annotated set more finely.

**Interpretation:** More discriminative than MeSH for identifying disease-coherent sub-clusters within the super-PAG. Captures actual disease biology (e.g., shared muscle-disease enrichment) rather than demographic metadata.

---

## 6. Shared GO Biological Process (438 edges)

**Source:** Enrichr API / GO_Biological_Process_2023. Same pipeline as shared_disease: adj-p < 0.05, terms appearing in ≥80% of gene sets excluded.

**What it measures:** Pairs of gene sets co-enriched for the same specific biological process GO terms, after filtering near-universal terms (e.g., "Translation" would be excluded if it appears in ≥80% of sets).

**Structure:** Sparse — 438 edges, approximately 2.4 edges per node on average. The sparsest of the dense evidence types, making it the most discriminative.

**Key observations:**
- 438 edges — sharply fewer than shared_pathway (16,290) despite both being process-level annotations.
- GO BP filtering successfully removes the "Translation" saturation problem seen in Reactome: pathway co-membership edges are near-complete because all gene sets hit the same top-level ribosomal pathways.
- Selective GO term co-enrichment identifies specific sub-processes shared only among a subset of nodes, providing sub-cluster resolution.

**Interpretation:** Higher-resolution biological process annotation than Reactome pathways. While shared_pathway saturates at "Translation" and "Metabolism of proteins" (present in all pairs), shared_go_bp captures specific downstream processes that partition the super-PAG into meaningful sub-groups.

---

## Note on `shared_tf_regulator` (attempted, 0 edges)

**Tried:** Enrichr / ChEA_2022 TF enrichment using the same adj-p < 0.05 pipeline.

**Result:** 0 significant edges (all adj-p ≈ 1.0 across all gene set submissions).

**Reason:** Gene sets in this super-PAG are small (median ~25 genes, max ~45 HGNC symbols after ENSG→symbol conversion). TF ChIP-seq enrichment databases like ChEA require larger gene lists to achieve statistical power — small gene sets produce inflated p-values and no significant TF associations. This is a limitation of gene set size, not of the approach. The shared_tf_regulator edge type was not added to the graph.

---

## Visualization Updates

The HTML visualization (`kg_sparse_network.html`) was rebuilt using custom vis.js rather than pyvis:
- **Static layout after stabilization:** physics runs briefly to position nodes, then freezes. Subsequent rendering is fast and deterministic.
- **Draggable nodes:** users can manually reposition nodes after layout.
- **Click info panel:** clicking a node or edge opens a detail panel on the right showing ID, attributes, and connected edge types.
- **Edge labels:** sparse edges display their type label directly on the edge.

The adjacency heatmap (`kg_adjacency_heatmaps.png`) is now a **2×3 panel** (6 edge types).

The sparse network visualization shows **574 edges** (gene_overlap + pubmed_costudy + shared_go_bp — the three lowest-density edge types).

---

## Summary Table

| Edge type | Edges | Nodes covered | Key signal | Discriminative power |
|-----------|------:|:-------------:|-----------|---------------------|
| `gene_overlap` | 8 | 13 / 181 (7%) | Same-study opposing contrasts share genes | High — very selective |
| `pubmed_costudy` | 128 | ~100 / 181 | GSE-level study blocks along diagonal | Medium — study-level clusters |
| `shared_mesh_term` | 9,591 | 139 / 181 (77%) | Bimodal: annotated vs unannotated; dominant terms are demographic | Low-medium — mostly universal demographics |
| `shared_pathway` | 16,290 | 181 / 181 (100%) | Universal ribosomal/translational pathway co-membership | Low — near-complete graph |
| `shared_disease` | 7,669 | 181 / 181 (100%) | Disease-specific block structure; finer-grain than MeSH | Medium — no universal catch-all term |
| `shared_go_bp` | 438 | ~180 / 181 | Sparse selective GO term co-enrichment; avoids Translation saturation | High — most discriminative of dense types |

**Total edges:** 34,124

**Overall:** The super-PAG is a semantically coherent cluster held together primarily by shared pathway biology (translation/ribosome) and contextual metadata (human tissue, mixed sex, adult). Actual gene-content sharing is minimal and almost entirely within-study, suggesting these 181 gene sets capture the same biological theme through independent experimental lenses rather than measuring identical gene programs. The two new evidence types (shared_disease, shared_go_bp) add discriminative sub-structure: disease associations provide block-level partitioning above MeSH granularity, while GO BP co-enrichment identifies specific biological process sub-clusters at high precision.
