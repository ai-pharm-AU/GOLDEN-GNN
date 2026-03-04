#!/usr/bin/env python3
"""
kg_matched_comparison.py — Side-by-side KG metrics for matched cluster pairs.

Uses the Hungarian algorithm on the cluster_assignments_k10.csv overlap matrix to
find the optimal 1-to-1 correspondence between GNN (A) and GNN-beta (B) clusters,
then compares KG metrics for each matched pair.

Usage:
    cd /home/zzz0054/GoldenF
    .venv/bin/python KnowledgeGraph/kg_matched_comparison.py

Outputs (KnowledgeGraph/cluster_metrics_out/):
    cluster_matched_comparison.csv  — 10 rows, all metrics side-by-side + delta/ratio
    comparison.html                 — dark-theme table, color-coded deltas, links to vis HTMLs
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent / "cluster_metrics_out"
ASSIGNMENTS_CSV = OUT_DIR / "cluster_assignments_k10.csv"
COVERAGE_CSV    = OUT_DIR / "knowledge_coverage.csv"
STATS_CSV       = OUT_DIR / "graph_stats.csv"

EDGE_TYPES = [
    "gene_overlap", "pubmed_costudy", "shared_mesh_term",
    "shared_pathway", "shared_disease", "shared_go_bp",
]

# Metrics from graph_stats.csv to compare (excludes per-edge-type counts — those are large)
STAT_METRICS = [
    "n_nodes", "n_edges", "density", "mean_degree",
    "average_clustering", "n_connected_components",
] + [f"n_edges_{et}" for et in EDGE_TYPES]

COVERAGE_METRICS = EDGE_TYPES  # coverage columns are named by edge type


# ---------------------------------------------------------------------------
# Step 1 — Build overlap matrix + derive matched pairs
# ---------------------------------------------------------------------------

def derive_matched_pairs(assignments_path: Path) -> pd.DataFrame:
    """
    Build 10×10 overlap matrix from cluster_assignments_k10.csv and
    apply the Hungarian algorithm to find optimal 1-to-1 matching.

    Returns DataFrame with columns:
        a_cluster (int), b_cluster (int), overlap_count (int),
        a_group (str), b_group (str), overlap_pct (float)
    """
    df = pd.read_csv(assignments_path)
    k = 10

    # M[i][j] = number of gene sets in GNN cluster i AND structlite cluster j
    M = np.zeros((k, k), dtype=int)
    for _, row in df.iterrows():
        i = int(row["gnn_cluster"])
        j = int(row["structlite_cluster"])
        if 0 <= i < k and 0 <= j < k:
            M[i, j] += 1

    # Hungarian: maximise overlap (negate for minimisation)
    row_ind, col_ind = linear_sum_assignment(-M)

    # Build cluster size lookups
    size_a = df.groupby("gnn_cluster").size().to_dict()
    size_b = df.groupby("structlite_cluster").size().to_dict()

    pairs = []
    for i, j in zip(row_ind, col_ind):
        count = int(M[i, j])
        denom = max(size_a.get(i, 1), size_b.get(j, 1))
        pct = round(100.0 * count / denom, 1)
        pairs.append({
            "a_cluster": i,
            "b_cluster": j,
            "a_group": f"A{i}",
            "b_group": f"B{j}",
            "overlap_count": count,
            "size_A": size_a.get(i, 0),
            "size_B": size_b.get(j, 0),
            "overlap_pct": pct,
        })

    return pd.DataFrame(pairs).sort_values("a_cluster").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2 — Join with KG metrics
# ---------------------------------------------------------------------------

def build_comparison(
    pairs_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    stats_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each matched pair, fetch all KG metrics for A and B groups,
    compute delta (B − A) and ratio (B / A) for each numeric metric.
    """
    cov = coverage_df.set_index("group_id")
    stats = stats_df.set_index("group_id")

    rows = []
    for _, pair in pairs_df.iterrows():
        ag, bg = pair["a_group"], pair["b_group"]
        row: dict = {
            "a_group": ag,
            "b_group": bg,
            "overlap_count": pair["overlap_count"],
            "overlap_pct": pair["overlap_pct"],
            "size_A": pair["size_A"],
            "size_B": pair["size_B"],
        }

        # Graph stats
        for m in STAT_METRICS:
            va = float(stats.loc[ag, m]) if ag in stats.index and m in stats.columns else float("nan")
            vb = float(stats.loc[bg, m]) if bg in stats.index and m in stats.columns else float("nan")
            row[f"{m}_A"] = va
            row[f"{m}_B"] = vb
            row[f"{m}_delta"] = (vb - va) if not (np.isnan(va) or np.isnan(vb)) else float("nan")
            row[f"{m}_ratio"] = (vb / va) if (not (np.isnan(va) or np.isnan(vb)) and va != 0) else float("nan")

        # Knowledge coverage
        for m in COVERAGE_METRICS:
            col = m  # coverage CSV columns are named by edge type directly
            va = float(cov.loc[ag, col]) if ag in cov.index and col in cov.columns else float("nan")
            vb = float(cov.loc[bg, col]) if bg in cov.index and col in cov.columns else float("nan")
            row[f"cov_{m}_A"] = va
            row[f"cov_{m}_B"] = vb
            row[f"cov_{m}_delta"] = (vb - va) if not (np.isnan(va) or np.isnan(vb)) else float("nan")

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3 — Generate comparison.html
# ---------------------------------------------------------------------------

_DELTA_METRICS = (
    [f"{m}_delta" for m in STAT_METRICS] +
    [f"cov_{m}_delta" for m in COVERAGE_METRICS]
)


def _fmt(val, col: str) -> str:
    """Format a cell value for HTML display."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '<span style="color:#555">—</span>'
    if isinstance(val, float):
        if "pct" in col or col.startswith("cov_"):
            return f"{val:.3f}"
        if "density" in col or "clustering" in col:
            return f"{val:.4f}"
        if "ratio" in col:
            return f"{val:.2f}×"
        return f"{val:,.1f}"
    return str(val)


def _delta_style(val, col: str) -> str:
    """Return inline style for delta cells (green=positive, red=negative)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    if val > 0:
        return ' style="color:#2ecc71;font-weight:bold"'
    if val < 0:
        return ' style="color:#e74c3c;font-weight:bold"'
    return ' style="color:#888"'


def make_comparison_html(comp_df: pd.DataFrame) -> str:
    """Generate a self-contained dark-theme HTML comparison table."""

    # Section 1: Overview (overlap + node/edge counts)
    overview_cols = [
        ("a_group", "GNN Cluster"),
        ("b_group", "GNN-beta Cluster"),
        ("overlap_count", "Overlap"),
        ("overlap_pct", "Overlap %"),
        ("size_A", "Size A"),
        ("size_B", "Size B"),
        ("n_nodes_A", "Nodes A"),
        ("n_nodes_B", "Nodes B"),
        ("n_nodes_delta", "Δ Nodes"),
        ("n_edges_A", "Edges A"),
        ("n_edges_B", "Edges B"),
        ("density_A", "Density A"),
        ("density_B", "Density B"),
        ("mean_degree_A", "Deg A"),
        ("mean_degree_B", "Deg B"),
        ("mean_degree_delta", "Δ Deg"),
        ("average_clustering_A", "Clust A"),
        ("average_clustering_B", "Clust B"),
    ]

    # Section 2: Coverage
    cov_cols = (
        [("a_group", "GNN"), ("b_group", "GNN-beta"), ("overlap_pct", "Overlap %")]
        + [(f"cov_{et}_A", f"{et[:8]} A") for et in EDGE_TYPES]
        + [(f"cov_{et}_B", f"{et[:8]} B") for et in EDGE_TYPES]
        + [(f"cov_{et}_delta", f"Δ {et[:8]}") for et in EDGE_TYPES]
    )

    def build_table(cols, df):
        headers = "".join(f"<th>{label}</th>" for _, label in cols)
        body = ""
        for _, r in df.iterrows():
            ag, bg = r["a_group"], r["b_group"]
            cells = ""
            for col, _ in cols:
                val = r.get(col)
                fmt = _fmt(val, col)
                if "delta" in col:
                    sty = _delta_style(val, col)
                    cells += f"<td{sty}>{fmt}</td>"
                elif col == "a_group":
                    cells += f'<td><a href="vis_{ag}.html" style="color:#3498db">{ag}</a></td>'
                elif col == "b_group":
                    cells += f'<td><a href="vis_{bg}.html" style="color:#9b59b6">{bg}</a></td>'
                else:
                    cells += f"<td>{fmt}</td>"
            pct = float(r.get("overlap_pct", 0))
            row_style = ' style="background:#0f2a1a"' if pct >= 40 else (
                        ' style="background:#2a0f0f"' if pct < 5 else "")
            body += f"<tr{row_style}>{cells}</tr>\n"
        return f"<thead><tr>{headers}</tr></thead><tbody>{body}</tbody>"

    t1 = build_table(overview_cols, comp_df)
    t2 = build_table(cov_cols, comp_df)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>KG Matched Cluster Comparison — GNN vs GNN-beta</title>
<style>
body{{background:#1a1a2e;color:#ccc;font-family:'Segoe UI',sans-serif;padding:24px;overflow-x:auto;}}
h1{{color:#e0e0e0;font-size:20px;margin-bottom:4px;}}
h2{{color:#3498db;font-size:15px;margin:24px 0 8px;}}
p{{color:#888;font-size:12px;margin-bottom:14px;}}
table{{border-collapse:collapse;font-size:12px;white-space:nowrap;margin-bottom:8px;}}
th{{background:#0f3460;color:#e0e0e0;padding:6px 10px;text-align:right;
   border-bottom:2px solid #3498db;}}
th:first-child,th:nth-child(2){{text-align:left;}}
td{{padding:5px 10px;border-bottom:1px solid #0f3460;text-align:right;}}
td:first-child,td:nth-child(2){{text-align:left;}}
tr:hover td{{background:#16213e!important;}}
a{{color:#3498db;text-decoration:none;}}
a:hover{{text-decoration:underline;}}
.legend{{font-size:11px;color:#888;margin-top:6px;}}
.legend span{{display:inline-block;padding:2px 8px;border-radius:3px;margin-right:8px;}}
</style>
</head>
<body>
<h1>KG Matched Cluster Comparison &mdash; GNN vs GNN-beta (k=10, subset 085)</h1>
<p>
  Each row is an optimally matched cluster pair (Hungarian algorithm on the 10×10 overlap matrix).<br>
  <b>Δ columns</b>: B &minus; A &nbsp;
  <span style="color:#2ecc71">■ green = B &gt; A</span> &nbsp;
  <span style="color:#e74c3c">■ red = B &lt; A</span><br>
  Row shading: <span style="background:#0f2a1a;padding:1px 6px">high overlap ≥40%</span> &nbsp;
  <span style="background:#2a0f0f;padding:1px 6px">low overlap &lt;5%</span>
</p>
<p><a href="index.html">← Back to full summary</a></p>

<h2>Graph Structure</h2>
<table>{t1}</table>

<h2>Knowledge Coverage (fraction of nodes with ≥1 edge of each type)</h2>
<table>{t2}</table>

<p style="margin-top:20px;color:#555;font-size:11px">
  Generated by KnowledgeGraph/kg_matched_comparison.py
</p>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("KG Matched Cluster Comparison")
    print("=" * 50)

    # Step 1
    print("Step 1: Deriving matched pairs via Hungarian algorithm …")
    pairs_df = derive_matched_pairs(ASSIGNMENTS_CSV)
    print(f"  {len(pairs_df)} matched pairs found")
    print()
    print(f"  {'A':>4}  {'B':>4}  {'Overlap':>8}  {'Pct':>7}")
    for _, r in pairs_df.iterrows():
        print(f"  {r['a_group']:>4}  {r['b_group']:>4}  {r['overlap_count']:>8,}  {r['overlap_pct']:>6.1f}%")

    # Step 2
    print("\nStep 2: Loading KG metrics …")
    coverage_df = pd.read_csv(COVERAGE_CSV)
    stats_df    = pd.read_csv(STATS_CSV)
    print(f"  coverage: {len(coverage_df)} rows, stats: {len(stats_df)} rows")

    comp_df = build_comparison(pairs_df, coverage_df, stats_df)
    print(f"  comparison: {len(comp_df)} rows, {len(comp_df.columns)} columns")

    # Step 3
    print("\nStep 3: Saving outputs …")
    comp_csv = OUT_DIR / "cluster_matched_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)
    print(f"  {comp_csv}  ({len(comp_df)} rows × {len(comp_df.columns)} cols)")

    html = make_comparison_html(comp_df)
    html_path = OUT_DIR / "comparison.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"  {html_path}  ({len(html):,} bytes)")

    # Verify
    print("\nVerification:")
    reload = pd.read_csv(comp_csv)
    assert len(reload) == 10, f"Expected 10 rows, got {len(reload)}"

    a8b7 = reload[(reload["a_group"] == "A8") & (reload["b_group"] == "B7")]
    a5b8 = reload[(reload["a_group"] == "A5") & (reload["b_group"] == "B8")]
    if not a8b7.empty:
        pct = float(a8b7["overlap_pct"].iloc[0])
        print(f"  A8↔B7 overlap: {pct:.1f}%  (expected ~67.5%)")
        assert 60 <= pct <= 75, f"A8↔B7 overlap out of expected range: {pct}"
    if not a5b8.empty:
        pct = float(a5b8["overlap_pct"].iloc[0])
        print(f"  A5↔B8 overlap: {pct:.1f}%  (expected ~1.0%)")
        assert pct <= 5, f"A5↔B8 overlap unexpectedly high: {pct}"

    print("\nAll checks passed.")
    print(f"\nOpen: {html_path}")


if __name__ == "__main__":
    main()
