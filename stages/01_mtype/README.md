# Stage 01 — mtype Build

## Summary
Gene-overlap edge construction with composite Jaccard-Simpson similarity and hypergeometric filtering.

## Payload
### Code
- [`mtype.py`](code/mtype.py) — Core mtype similarity and hypergeometric computation script.

### Docs
- [`docs/gene_overlap_edge_methodology.md`](doc/docs/gene_overlap_edge_methodology.md) — Method write-up for overlap edges and significance filtering.

### Results
- [`data/geneset_extraction_080/edges_with_mtype.txt`](result/data/geneset_extraction_080/edges_with_mtype.txt) — Threshold-0.80 edge file with mtype-derived fields.
- [`data/geneset_extraction_080/extraction_summary.json`](result/data/geneset_extraction_080/extraction_summary.json) — Extraction summary statistics for threshold 0.80.
- [`data/geneset_extraction_085/edges.csv`](result/data/geneset_extraction_085/edges.csv) — Threshold-0.85 extracted edge table.
- [`data/geneset_extraction_085/extraction_summary.json`](result/data/geneset_extraction_085/extraction_summary.json) — Extraction summary statistics for threshold 0.85.
- [`data/geneset_extraction_090/edges.csv`](result/data/geneset_extraction_090/edges.csv) — Threshold-0.90 extracted edge table.
- [`data/geneset_extraction_090/extraction_summary.json`](result/data/geneset_extraction_090/extraction_summary.json) — Extraction summary statistics for threshold 0.90.

## Notes
- Oversized files skipped to HF queue: 0
- Missing source items: 0
