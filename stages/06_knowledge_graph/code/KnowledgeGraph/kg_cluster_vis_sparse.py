#!/usr/bin/env python3
"""
kg_cluster_vis_sparse.py — Regenerate compact vis.js HTMLs for all 20 cluster groups.

Reads pre-built edge CSVs from cluster_metrics_out/ and rebuilds the HTML
visualizations using only the 3 "sparse" edge types:
  - gene_overlap
  - pubmed_costudy
  - shared_mesh_term

This keeps each HTML under ~1 MB (vs 40–460 MB when all 6 edge types are embedded).
The full edge data (including pathway, disease, GO BP) remains available in the
per-group CSVs.

Outputs saved to KnowledgeGraph/cluster_metrics_out/vis_sparse/:
  index_sparse.html         — summary dashboard
  vis_sparse_<GROUP>.html   — compact interactive KG per group (A0–A9, B0–B9)

Usage:
    cd /home/zzz0054/GoldenF
    .venv/bin/python KnowledgeGraph/kg_cluster_vis_sparse.py
"""

import json
import logging
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
KG_DIR = Path(__file__).resolve().parent
METRICS_DIR = KG_DIR / "cluster_metrics_out"
OUT_DIR = METRICS_DIR / "vis_sparse"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GROUPS = [f"A{i}" for i in range(10)] + [f"B{i}" for i in range(10)]

SPARSE_TYPES = {"gene_overlap", "pubmed_costudy", "shared_go_bp"}

EDGE_COLORS = {
    "gene_overlap":  "#e74c3c",
    "pubmed_costudy": "#3498db",
    "shared_go_bp":  "#1abc9c",
}
EDGE_LABELS = {
    "gene_overlap":  "Gene OL",
    "pubmed_costudy": "PubMed",
    "shared_go_bp":  "GO BP",
}
ALL_EDGE_TYPES = [
    "gene_overlap", "pubmed_costudy", "shared_mesh_term",
    "shared_pathway", "shared_disease", "shared_go_bp",
]
NODE_COLORS = {"up": "#27ae60", "dn": "#e74c3c"}


# ---------------------------------------------------------------------------
# Load shared data
# ---------------------------------------------------------------------------

def load_metadata() -> pd.DataFrame:
    """Load enriched metadata CSV and index by GS_ID."""
    from pathlib import Path as _Path
    import sys
    sys.path.insert(0, str(KG_DIR))
    from build_kg import METADATA_CSV
    meta = pd.read_csv(METADATA_CSV, low_memory=False)
    if "GS_ID" not in meta.columns:
        meta = meta.rename(columns={"gs_id": "GS_ID"})
    return meta.set_index("GS_ID")


def load_assignments() -> pd.DataFrame:
    return pd.read_csv(METRICS_DIR / "cluster_assignments_k10.csv")


def load_stats() -> pd.DataFrame:
    return pd.read_csv(METRICS_DIR / "graph_stats.csv")


def load_coverage() -> pd.DataFrame:
    return pd.read_csv(METRICS_DIR / "knowledge_coverage.csv")


# ---------------------------------------------------------------------------
# Edge info HTML helpers
# ---------------------------------------------------------------------------

def _edge_info_html(etype: str, row: pd.Series) -> str:
    if etype == "gene_overlap":
        sim = float(row.get("similarity", 0) or 0)
        return (
            f"<b>Gene Overlap</b><br>"
            f"<b>Similarity (Jaccard):</b> {sim:.3f}<br>"
            f"<b>Overlap count:</b> {row.get('overlap_count', '')}<br>"
            f"<b>−log p-val:</b> {float(row.get('neg_log_pval', 0) or 0):.2e}"
        )
    if etype == "pubmed_costudy":
        cnt = int(row.get("shared_pmid_count", 1) or 1)
        pmids = str(row.get("shared_pmids", "")).replace("|", ", ")[:300]
        return (
            f"<b>PubMed Co-study</b><br>"
            f"<b>Shared PMIDs:</b> {cnt}<br>"
            f"<b>PMIDs:</b> {pmids}"
        )
    if etype == "shared_go_bp":
        cnt = int(row.get("go_count", 0) or 0)
        terms = str(row.get("go_terms", "")).replace("|", "<br>• ")[:500]
        return (
            f"<b>Shared GO BP</b><br>"
            f"<b>Count:</b> {cnt}<br>"
            f"<b>Best adj-p:</b> {float(row.get('min_pval', 1) or 1):.2e}<br>"
            f"• {terms}"
        )
    return f"<b>{etype}</b>"


# ---------------------------------------------------------------------------
# Build one sparse HTML
# ---------------------------------------------------------------------------

def build_sparse_html(
    group_id: str,
    edges_df: pd.DataFrame,
    meta: pd.DataFrame,
    assignments: pd.DataFrame,
    stats_row: dict,
    coverage_row: dict,
) -> str:
    model_label = "GNN (text-only)" if group_id.startswith("A") else "GNN-beta (structlite)"
    cluster_col = "gnn_cluster" if group_id.startswith("A") else "structlite_cluster"
    cluster_num = int(group_id[1:])
    group_ids = assignments.loc[assignments[cluster_col] == cluster_num, "gs_id"].tolist()

    # ── Nodes ──────────────────────────────────────────────────────────────
    nodes_js = []
    for nid in group_ids:
        row = meta.loc[nid] if nid in meta.index else {}
        direction = str(row.get("direction", "")) if hasattr(row, "get") else ""
        try:
            gs_size = int(row.get("GS_SIZE", row.get("gs_size", 20)))
        except (ValueError, TypeError, AttributeError):
            gs_size = 20
        size_px = max(12, min(40, 12 + gs_size * 0.4))
        color = NODE_COLORS.get(direction, "#95a5a6")
        name = str(row.get("NAME", row.get("name", nid))) if hasattr(row, "get") else nid
        short = name.split("-")[0] if "-" in name else name[:30]
        geo_title = str(row.get("geo_series_title", "")).strip('"')[:120] if hasattr(row, "get") else ""
        info = (
            f"<b>{name}</b><hr style='border-color:#444;margin:4px 0'>"
            f"<b>GSE:</b> {row.get('gse_ids','') if hasattr(row,'get') else ''}<br>"
            f"<b>Study:</b> {geo_title}<br>"
            f"<b>Direction:</b> {direction} &nbsp; <b>Size:</b> {gs_size}"
        )
        nodes_js.append({
            "id": nid, "label": short,
            "color": {"background": color, "border": "#fff",
                      "highlight": {"background": color, "border": "#fff"}},
            "size": size_px,
            "font": {"color": "#ffffff", "size": 10},
            "shadow": True,
            "info": info,
        })

    # ── Edges (sparse types only) ──────────────────────────────────────────
    sparse_df = edges_df[edges_df["edge_type"].isin(SPARSE_TYPES)] if "edge_type" in edges_df.columns else pd.DataFrame()
    edges_js = []
    for _, erow in sparse_df.iterrows():
        etype = erow.get("edge_type", "")
        color = EDGE_COLORS.get(etype, "#aaa")
        label = EDGE_LABELS.get(etype, etype)
        if etype == "gene_overlap":
            width = max(1.0, float(erow.get("similarity", 0) or 0) * 6)
        elif etype == "pubmed_costudy":
            width = max(1.0, float(erow.get("shared_pmid_count", 1) or 1) * 1.5)
        else:
            width = 1.5
        info = _edge_info_html(etype, erow)
        edges_js.append({
            "from": str(erow["source"]), "to": str(erow["target"]),
            "color": {"color": color, "highlight": "#ffffff"},
            "width": width,
            "label": label,
            "font": {"size": 8, "color": color, "align": "middle",
                     "strokeWidth": 2, "strokeColor": "#1a1a2e"},
            "smooth": {"type": "continuous"},
            "info": info,
        })

    n_sparse = len(sparse_df)
    n_total = int(stats_row.get("n_edges", 0))

    # ── Stats sidebar content ──────────────────────────────────────────────
    stats_html = "".join(
        f"<div><b>{k.replace('n_edges_','').replace('_',' ')}:</b> "
        f"{int(v) if isinstance(v, float) and v == int(v) else round(v, 4)}</div>"
        for k, v in stats_row.items()
        if k not in ("group_id",) and v is not None
    )
    cov_html = "".join(
        f"<div><b>{et.replace('_',' ')}:</b> {float(coverage_row.get(et, 0)):.1%}</div>"
        for et in ALL_EDGE_TYPES
    )

    # ── Legend ─────────────────────────────────────────────────────────────
    legend_items = "".join(
        f'<div><span style="display:inline-block;width:16px;height:3px;background:{c};'
        f'vertical-align:middle;margin-right:5px;border-radius:2px"></span>{et.replace("_"," ")}</div>'
        for et, c in EDGE_COLORS.items()
    )
    node_legend = "".join(
        f'<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
        f'background:{c};vertical-align:middle;margin-right:5px"></span>{lbl}</div>'
        for lbl, c in [("Up-regulated", "#27ae60"), ("Down-regulated", "#e74c3c"), ("Other", "#95a5a6")]
    )

    nodes_json = json.dumps(nodes_js, ensure_ascii=False)
    edges_json = json.dumps(edges_js, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Group {group_id} — {model_label} (sparse)</title>
<script src="https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js"></script>
<link  href="https://unpkg.com/vis-network@9.1.9/dist/dist/vis-network.min.css" rel="stylesheet">
<style>
* {{ box-sizing:border-box;margin:0;padding:0; }}
body {{ background:#1a1a2e;font-family:'Segoe UI',sans-serif;display:flex;height:100vh;overflow:hidden; }}
#network {{ flex:1; }}
#sidebar {{ width:320px;background:#16213e;border-left:1px solid #0f3460;
  padding:14px;color:#ccc;font-size:12px;overflow-y:auto;display:flex;
  flex-direction:column;gap:10px; }}
#sidebar h2 {{ color:#e0e0e0;font-size:13px;border-bottom:1px solid #0f3460;padding-bottom:6px; }}
.section-title {{ color:#e0e0e0;font-weight:bold;font-size:11px;
  border-bottom:1px solid #0f3460;padding-bottom:3px;margin-top:4px; }}
#info-panel {{ line-height:1.6;color:#bbb;font-size:11px; }}
#info-panel b {{ color:#e0e0e0; }}
#info-panel hr {{ border-color:#0f3460;margin:5px 0; }}
.stats-block {{ font-size:11px;line-height:1.7;color:#bbb; }}
.stats-block b {{ color:#e0e0e0; }}
#legend {{ background:#0f3460;border-radius:5px;padding:9px;font-size:11px;line-height:1.9; }}
#hint {{ color:#555;font-size:11px;font-style:italic; }}
#stab-msg {{ color:#f39c12;font-size:11px;text-align:center; }}
.note {{ background:#1a2a1a;border:1px solid #2a4a2a;border-radius:4px;
  padding:6px 8px;font-size:10px;color:#888;line-height:1.5; }}
a.back {{ color:#3498db;font-size:11px; }}
a.csv-link {{ color:#f39c12;font-size:11px; }}
</style>
</head>
<body>
<div id="network"></div>
<div id="sidebar">
  <h2>Group {group_id} &mdash; {model_label}</h2>
  <a class="back" href="index_sparse.html">&#8592; Back to summary</a>
  <a class="csv-link" href="../edges_{group_id}.csv">&#8595; Full edge CSV ({n_total:,} edges)</a>
  <div id="stab-msg">&#9654; Stabilizing layout&hellip;</div>
  <div id="info-panel"><span style="color:#555">Click a node or edge for details.</span></div>
  <div>
    <div class="section-title">Graph stats</div>
    <div class="stats-block">{stats_html}</div>
  </div>
  <div>
    <div class="section-title">Knowledge coverage</div>
    <div class="stats-block">{cov_html}</div>
  </div>
  <div class="note">
    Showing {n_sparse:,} of {n_total:,} edges (gene overlap, PubMed, MeSH only).<br>
    Dense types (pathway, disease, GO BP) omitted for performance.<br>
    Download the CSV for full edge data.
  </div>
  <div id="legend">
    <div style="color:#e0e0e0;font-weight:bold;margin-bottom:3px">Nodes</div>
    {node_legend}
    <div style="color:#e0e0e0;font-weight:bold;margin:7px 0 3px">Edges shown</div>
    {legend_items}
    <div style="color:#666;font-size:10px;margin-top:5px">Node size ∝ gene set size</div>
  </div>
  <div id="hint">Drag nodes · scroll to zoom · click for details</div>
</div>
<script>
const nodesData={nodes_json};
const edgesData={edges_json};
const container=document.getElementById('network');
const data={{nodes:new vis.DataSet(nodesData),edges:new vis.DataSet(edgesData)}};
const options={{
  nodes:{{shape:'dot',borderWidth:1,shadow:true}},
  edges:{{arrows:{{to:{{enabled:false}}}}}},
  physics:{{
    enabled:true,
    barnesHut:{{gravitationalConstant:-8000,centralGravity:0.3,
      springLength:130,springConstant:0.05,damping:0.09,avoidOverlap:0.1}},
    stabilization:{{iterations:400,updateInterval:25}},
  }},
  interaction:{{hover:true,dragNodes:true,dragView:true,zoomView:true,
    navigationButtons:true,keyboard:true}},
}};
const network=new vis.Network(container,data,options);
network.on('stabilizationIterationsDone',function(){{
  network.setOptions({{physics:{{enabled:false}}}});
  document.getElementById('stab-msg').textContent='✓ Layout ready — drag to rearrange';
  setTimeout(()=>document.getElementById('stab-msg').style.display='none',2500);
}});
network.on('click',function(params){{
  const panel=document.getElementById('info-panel');
  if(params.nodes.length>0){{
    const n=data.nodes.get(params.nodes[0]);
    panel.innerHTML=n.info||'<i>No info</i>';
  }} else if(params.edges.length>0){{
    const e=data.edges.get(params.edges[0]);
    const fn=data.nodes.get(e.from), tn=data.nodes.get(e.to);
    panel.innerHTML=(e.info||'<i>No info</i>')
      +'<hr style="border-color:#0f3460;margin:8px 0">'
      +'<b>From:</b> '+(fn?fn.label:e.from)+'<br>'
      +'<b>To:</b> '  +(tn?tn.label:e.to);
  }} else {{
    panel.innerHTML='<span style="color:#555">Click a node or edge for details.</span>';
  }}
}});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Summary index
# ---------------------------------------------------------------------------

def build_index_html(
    groups: list[str],
    stats_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
) -> str:
    stats_map = stats_df.set_index("group_id").to_dict("index")
    cov_map = coverage_df.set_index("group_id").to_dict("index")

    rows_html = ""
    for gid in groups:
        model = "GNN (text-only)" if gid.startswith("A") else "GNN-beta (structlite)"
        s = stats_map.get(gid, {})
        c = cov_map.get(gid, {})
        cov_cells = "".join(
            f'<td style="text-align:right">{float(c.get(et, 0)):.2f}</td>'
            for et in ALL_EDGE_TYPES
        )
        rows_html += (
            f'<tr>'
            f'<td><a href="vis_sparse_{gid}.html">{gid}</a></td>'
            f'<td>{model}</td>'
            f'<td style="text-align:right">{s.get("n_nodes","")}</td>'
            f'<td style="text-align:right">{s.get("n_edges","")}</td>'
            f'<td style="text-align:right">{float(s.get("density",0)):.4f}</td>'
            f'<td style="text-align:right">{float(s.get("mean_degree",0)):.1f}</td>'
            f'<td style="text-align:right">{s.get("n_connected_components","")}</td>'
            f'{cov_cells}'
            f'<td><a href="../edges_{gid}.csv" style="color:#f39c12">CSV</a></td>'
            f'</tr>\n'
        )

    et_headers = "".join(f'<th>cov_{et}</th>' for et in ALL_EDGE_TYPES)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>KG Cluster Metrics — Sparse View (GNN vs GNN-beta, k=10)</title>
<style>
body{{background:#1a1a2e;color:#ccc;font-family:'Segoe UI',sans-serif;padding:24px;}}
h1{{color:#e0e0e0;margin-bottom:4px;font-size:20px;}}
p{{color:#888;font-size:12px;margin-bottom:18px;}}
table{{border-collapse:collapse;width:100%;font-size:12px;}}
th{{background:#0f3460;color:#e0e0e0;padding:7px 10px;text-align:left;
   border-bottom:2px solid #3498db;white-space:nowrap;}}
td{{padding:5px 10px;border-bottom:1px solid #0f3460;}}
tr:hover td{{background:#16213e;}}
a{{color:#3498db;text-decoration:none;}}
a:hover{{text-decoration:underline;}}
.note{{background:#1a2a1a;border:1px solid #2a4a2a;border-radius:4px;
  padding:8px 12px;font-size:11px;color:#888;margin-bottom:16px;}}
</style>
</head>
<body>
<h1>KG Cluster Metrics &mdash; Sparse View (k=10)</h1>
<div class="note">
  HTML visualizations show <b>gene overlap, PubMed co-study, and MeSH</b> edges only.<br>
  Dense edge types (pathway, disease, GO BP) are omitted from the HTML for performance —
  they are available in the per-group CSV files.
</div>
<table>
<thead><tr>
  <th>Group</th><th>Model</th>
  <th>Nodes</th><th>Edges (total)</th><th>Density</th><th>Mean&deg;</th><th>Components</th>
  {et_headers}
  <th>Full CSV</th>
</tr></thead>
<tbody>
{rows_html}
</tbody>
</table>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Loading shared data …")
    meta = load_metadata()
    assignments = load_assignments()
    stats_df = load_stats()
    coverage_df = load_coverage()
    stats_map = stats_df.set_index("group_id").to_dict("index")
    cov_map = coverage_df.set_index("group_id").to_dict("index")

    for group_id in GROUPS:
        edge_path = METRICS_DIR / f"edges_{group_id}.csv"
        if not edge_path.exists():
            log.warning("Missing edge CSV for %s — skipping", group_id)
            continue
        log.info("Building sparse HTML for group %s …", group_id)
        edges_df = pd.read_csv(edge_path, low_memory=False)
        html = build_sparse_html(
            group_id, edges_df, meta, assignments,
            stats_map.get(group_id, {}),
            cov_map.get(group_id, {}),
        )
        out_path = OUT_DIR / f"vis_sparse_{group_id}.html"
        out_path.write_text(html, encoding="utf-8")
        log.info("  Saved %s (%.0f KB)", out_path.name, out_path.stat().st_size / 1024)

    log.info("Building index_sparse.html …")
    index_html = build_index_html(GROUPS, stats_df, coverage_df)
    (OUT_DIR / "index_sparse.html").write_text(index_html, encoding="utf-8")
    log.info("Done. Open: %s/index_sparse.html", OUT_DIR)


if __name__ == "__main__":
    main()
