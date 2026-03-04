#!/usr/bin/env python3
"""
kg_cluster_metrics.py — Build KGs and compute metrics for all 20 cluster groups.

Groups:
  A0–A9  — GNN text-only (gnn_old_holdout) k-means clusters, k=10
  B0–B9  — GNN-beta (text_structlite) k-means clusters, k=10

Edge types per KG:
  1. gene_overlap       — PAGER gene-overlap (local file)
  2. pubmed_costudy     — shared PMIDs from metadata CSV (local, fast)
  3. shared_mesh_term   — shared MeSH descriptors from metadata (local, fast)
  4. shared_pathway     — Reactome pathway membership (cached)
  5. shared_disease     — Enrichr DisGeNET library (cached)
  6. shared_go_bp       — Enrichr GO_Biological_Process_2023 (cached)

Usage:
    cd /home/zzz0054/GoldenF
    .venv/bin/python KnowledgeGraph/kg_cluster_metrics.py 2>&1 | tee KnowledgeGraph/cluster_metrics_out/run.log

Outputs (KnowledgeGraph/cluster_metrics_out/):
    cluster_assignments_k10.csv  — gs_id, gnn_cluster, structlite_cluster
    knowledge_coverage.csv       — one row per group, coverage per edge type
    centrality.csv               — one row per (group, gs_id)
    graph_stats.csv              — one row per group (density, n_edges, etc.)
    edges_<GROUP>.csv            — full edge list for each of the 20 groups
    gene_overlap_all_groups.csv  — combined gene_overlap edges across all 20 groups
    vis_<GROUP>.html             — interactive vis.js KG for each group
    index.html                   — summary dashboard linking all 20 groups
"""

import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap: import shared functions + constants from build_kg.py
# ---------------------------------------------------------------------------
_KG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_KG_DIR))

from build_kg import (  # noqa: E402
    _best_kmeans,
    _parse_mesh_terms,
    _parse_pmids,
    _build_disease_edges_from_index,
    fetch_gene_members,
    build_pathway_edges,
    fetch_all_enrichr,
    build_enrichr_edges,
    NPZ_TEXT_ONLY,
    NPZ_STRUCTLITE,
    METADATA_CSV,
    EDGES_TXT,
    KG_DIR,
    NODE_COLS,
)

OUT_DIR = KG_DIR / "cluster_metrics_out"

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
K = 10
SEEDS = list(range(11, 16))

EDGE_TYPES = [
    "gene_overlap",
    "pubmed_costudy",
    "shared_mesh_term",
    "shared_pathway",
    "shared_disease",
    "shared_go_bp",
]

ENRICHR_LIBRARIES = ["DisGeNET", "GO_Biological_Process_2023"]


# ---------------------------------------------------------------------------
# Step 1 — Load embeddings and run k-means
# ---------------------------------------------------------------------------

def load_and_cluster():
    """
    Load both NPZ embeddings, inner-join on common IDs, run k-means (k=10)
    on each with seeds 11–15 (best-inertia run).

    Returns: (common_ids: list[str], labels_gnn: np.ndarray, labels_beta: np.ndarray)
    """
    log.info("Step 1: Loading embeddings …")
    a = np.load(NPZ_TEXT_ONLY, allow_pickle=True)
    b = np.load(NPZ_STRUCTLITE, allow_pickle=True)

    ids_a = a["ID"].tolist()
    emb_a = a["embeddings"]
    ids_b = b["ID"].tolist()
    emb_b = b["embeddings"]

    common = sorted(set(ids_a) & set(ids_b))
    log.info("  Inner-join: %d common IDs (GNN=%d, beta=%d)", len(common), len(ids_a), len(ids_b))

    idx_a = {v: i for i, v in enumerate(ids_a)}
    idx_b = {v: i for i, v in enumerate(ids_b)}
    emb_a_al = emb_a[[idx_a[c] for c in common]]
    emb_b_al = emb_b[[idx_b[c] for c in common]]

    log.info("  K-means GNN (text-only), k=%d:", K)
    labels_a = _best_kmeans(emb_a_al, K, SEEDS)
    log.info("  K-means GNN-beta (structlite), k=%d:", K)
    labels_b = _best_kmeans(emb_b_al, K, SEEDS)

    return common, labels_a, labels_b


# ---------------------------------------------------------------------------
# Step 2 — Define 20 groups
# ---------------------------------------------------------------------------

def make_groups(common: list[str], labels_a: np.ndarray, labels_b: np.ndarray) -> dict[str, list[str]]:
    """Return dict of group_id → list of gs_ids."""
    groups: dict[str, list[str]] = {}
    for c in range(K):
        groups[f"A{c}"] = [common[i] for i, lbl in enumerate(labels_a) if lbl == c]
    for c in range(K):
        groups[f"B{c}"] = [common[i] for i, lbl in enumerate(labels_b) if lbl == c]
    return groups


# ---------------------------------------------------------------------------
# Step 3 — Edge-building helpers (local, fast)
# ---------------------------------------------------------------------------

def _load_edges_df() -> pd.DataFrame:
    """Load PAGER edges file once."""
    df = pd.read_csv(EDGES_TXT, sep="\t", comment="#", low_memory=False)
    df.columns = [c.upper() for c in df.columns]
    return df


def build_gene_overlap_edges(group_ids: list[str], edges_df: pd.DataFrame) -> list[dict]:
    """Filter pre-loaded PAGER edges_df for intra-group pairs."""
    id_set = set(group_ids)
    edges = []
    for _, row in edges_df.iterrows():
        gs_a = str(row.get("GS_A_ID", "")).strip()
        gs_b = str(row.get("GS_B_ID", "")).strip()
        if gs_a in id_set and gs_b in id_set and gs_a != gs_b:
            edges.append({
                "source": gs_a,
                "target": gs_b,
                "edge_type": "gene_overlap",
                "overlap_count": int(row.get("OLAP", 0)),
                "similarity": float(row.get("SIMILARITY", 0.0)),
                "neg_log_pval": float(row.get("NLOGCDF", 0.0)),
            })
    return edges


def build_pubmed_edges_local(group_ids: list[str], node_table: pd.DataFrame) -> list[dict]:
    """Build pubmed_costudy edges from direct PMID sharing only (local, no NCBI API)."""
    pmid_to_gs: dict[str, list[str]] = defaultdict(list)
    for gs_id in group_ids:
        if gs_id not in node_table.index:
            continue
        pmid_val = node_table.loc[gs_id, "pmids"] if "pmids" in node_table.columns else ""
        for pmid in _parse_pmids(pmid_val):
            pmid_to_gs[pmid].append(gs_id)

    edges: list[dict] = []
    seen_pairs: set[frozenset] = set()

    for pmid, gs_list in pmid_to_gs.items():
        if len(gs_list) < 2:
            continue
        gs_sorted = sorted(gs_list)
        for i in range(len(gs_sorted)):
            for j in range(i + 1, len(gs_sorted)):
                a, b = gs_sorted[i], gs_sorted[j]
                pair = frozenset((a, b))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    edges.append({
                        "source": a,
                        "target": b,
                        "edge_type": "pubmed_costudy",
                        "shared_pmid_count": 1,
                        "shared_pmids": [pmid],
                    })
                else:
                    for e in edges:
                        if e["source"] == a and e["target"] == b:
                            e["shared_pmids"].append(pmid)
                            e["shared_pmid_count"] += 1
                            break
    return edges


def build_mesh_edges(group_ids: list[str], node_table: pd.DataFrame) -> list[dict]:
    """Build shared_mesh_term edges from mesh_descriptors column."""
    mesh_to_gs: dict[str, list[str]] = defaultdict(list)
    for gs_id in group_ids:
        if gs_id not in node_table.index:
            continue
        mesh_val = node_table.loc[gs_id, "mesh_descriptors"] if "mesh_descriptors" in node_table.columns else ""
        for term in _parse_mesh_terms(mesh_val):
            mesh_to_gs[term].append(gs_id)
    return _build_disease_edges_from_index(mesh_to_gs, {}, "shared_mesh_term")


# ---------------------------------------------------------------------------
# Step 4 — Assemble KG for a single group
# ---------------------------------------------------------------------------

def build_group_kg(
    group_id: str,
    group_gs_ids: list[str],
    node_table: pd.DataFrame,
    gene_members: dict[str, list[str]],
    edges_df: pd.DataFrame,
) -> nx.MultiGraph:
    """Build and return a nx.MultiGraph for one cluster group."""
    log.info("Building KG for group %s (%d nodes) …", group_id, len(group_gs_ids))
    grp_node_table = node_table[node_table.index.isin(group_gs_ids)]
    grp_gene_members = {gs_id: gene_members.get(gs_id, []) for gs_id in group_gs_ids}

    go_edges = build_gene_overlap_edges(group_gs_ids, edges_df)
    pm_edges = build_pubmed_edges_local(group_gs_ids, grp_node_table)
    mesh_edges = build_mesh_edges(group_gs_ids, grp_node_table)
    pw_edges = build_pathway_edges(group_gs_ids, grp_gene_members)

    dis_edges = build_enrichr_edges(
        group_gs_ids, grp_gene_members,
        library="DisGeNET",
        edge_type="shared_disease",
        term_field="disease_names",
    )
    gobp_edges = build_enrichr_edges(
        group_gs_ids, grp_gene_members,
        library="GO_Biological_Process_2023",
        edge_type="shared_go_bp",
        term_field="go_terms",
    )

    G = nx.MultiGraph()
    for gs_id in group_gs_ids:
        G.add_node(gs_id, gs_id=gs_id)

    all_edges = go_edges + pm_edges + mesh_edges + pw_edges + dis_edges + gobp_edges
    for e in all_edges:
        src, tgt = e["source"], e["target"]
        if src in G and tgt in G:
            attrs = {k: ("|".join(str(x) for x in v) if isinstance(v, list) else v)
                     for k, v in e.items() if k not in ("source", "target")}
            G.add_edge(src, tgt, **attrs)

    etype_counts: dict[str, int] = defaultdict(int)
    for _, _, data in G.edges(data=True):
        etype_counts[data.get("edge_type", "unknown")] += 1
    log.info(
        "  Group %s: %d nodes, %d edges %s",
        group_id, G.number_of_nodes(), G.number_of_edges(), dict(etype_counts),
    )
    return G


# ---------------------------------------------------------------------------
# Step 5 — Metrics
# ---------------------------------------------------------------------------

def compute_knowledge_coverage(G: nx.MultiGraph, group_gs_ids: list[str]) -> dict[str, float]:
    """Compute coverage rate (fraction of nodes with ≥1 edge) per edge type."""
    n_nodes = len(group_gs_ids)
    node_set = set(group_gs_ids)

    covered_by_type: dict[str, set] = defaultdict(set)
    for u, v, data in G.edges(data=True):
        et = data.get("edge_type", "unknown")
        covered_by_type[et].add(u)
        covered_by_type[et].add(v)

    coverage: dict[str, float] = {}
    for et in EDGE_TYPES:
        covered = covered_by_type.get(et, set()) & node_set
        coverage[et] = len(covered) / n_nodes if n_nodes > 0 else 0.0
    return coverage


def compute_graph_stats(G: nx.MultiGraph, group_gs_ids: list[str]) -> dict:
    """Compute graph-level statistics for a group KG."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    etype_counts: dict[str, int] = defaultdict(int)
    for _, _, data in G.edges(data=True):
        etype_counts[data.get("edge_type", "unknown")] += 1

    SG = nx.Graph(G)  # simple-graph projection
    density = nx.density(SG) if n_nodes > 1 else 0.0
    n_cc = nx.number_connected_components(SG) if n_nodes > 0 else 0
    degrees = [d for _, d in SG.degree()]
    mean_degree = sum(degrees) / n_nodes if n_nodes > 0 else 0.0
    avg_clustering = nx.average_clustering(SG) if n_nodes > 0 else 0.0

    stats: dict = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": density,
        "n_connected_components": n_cc,
        "mean_degree": mean_degree,
        "average_clustering": avg_clustering,
    }
    for et in EDGE_TYPES:
        stats[f"n_edges_{et}"] = etype_counts.get(et, 0)
    return stats


def compute_centrality(G: nx.MultiGraph, group_id: str, group_gs_ids: list[str]) -> list[dict]:
    """Compute degree, betweenness, closeness centrality for each node."""
    SG = nx.Graph(G)
    degree_c = nx.degree_centrality(SG)
    betweenness_c = nx.betweenness_centrality(SG, normalized=True)
    closeness_c = nx.closeness_centrality(SG)

    rows = []
    for gs_id in group_gs_ids:
        rows.append({
            "group_id": group_id,
            "gs_id": gs_id,
            "degree_centrality": degree_c.get(gs_id, 0.0),
            "betweenness_centrality": betweenness_c.get(gs_id, 0.0),
            "closeness_centrality": closeness_c.get(gs_id, 0.0),
        })
    return rows


# ---------------------------------------------------------------------------
# Output helpers — edge CSVs and web visualizations
# ---------------------------------------------------------------------------

_VIS_EDGE_COLORS = {
    "gene_overlap":    "#e74c3c",
    "pubmed_costudy":  "#3498db",
    "shared_mesh_term": "#f39c12",
    "shared_pathway":  "#27ae60",
    "shared_disease":  "#e67e22",
    "shared_go_bp":    "#1abc9c",
}
_VIS_EDGE_LABELS = {
    "gene_overlap":    "Gene OL",
    "pubmed_costudy":  "PubMed",
    "shared_mesh_term": "MeSH",
    "shared_pathway":  "Pathway",
    "shared_disease":  "Disease",
    "shared_go_bp":    "GO BP",
}
_NODE_COLORS = {"up": "#27ae60", "dn": "#e74c3c"}


def save_group_edges(group_id: str, G: nx.MultiGraph, out_dir: Path) -> None:
    """Save all edges for one group as a flat CSV."""
    rows = []
    for u, v, data in G.edges(data=True):
        row: dict = {"group_id": group_id, "source": u, "target": v}
        for k, val in data.items():
            row[k] = val
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / f"edges_{group_id}.csv", index=False)
    else:
        pd.DataFrame(columns=["group_id", "source", "target", "edge_type"]).to_csv(
            out_dir / f"edges_{group_id}.csv", index=False
        )


def _edge_info_html(etype: str, data: dict) -> str:
    """Build HTML info panel string for one edge in the vis.js tooltip."""
    if etype == "gene_overlap":
        sim = float(data.get("similarity", 0))
        return (
            f"<b>Gene Overlap</b><br>"
            f"<b>Similarity (Jaccard):</b> {sim:.3f}<br>"
            f"<b>Overlap count:</b> {data.get('overlap_count', '')}<br>"
            f"<b>−log p-val:</b> {float(data.get('neg_log_pval', 0)):.2e}"
        )
    if etype == "pubmed_costudy":
        cnt = int(data.get("shared_pmid_count", 1))
        pmids = str(data.get("shared_pmids", "")).replace("|", ", ")[:300]
        return (
            f"<b>PubMed Co-study</b><br>"
            f"<b>Shared PMIDs:</b> {cnt}<br>"
            f"<b>PMIDs:</b> {pmids}"
        )
    if etype == "shared_mesh_term":
        ids = str(data.get("disease_ids", "")).replace("|", ", ")[:300]
        return f"<b>Shared MeSH Term</b><br><b>Terms:</b> {ids}"
    if etype == "shared_pathway":
        names = str(data.get("pathway_names", "")).replace("|", "<br>• ")[:400]
        return f"<b>Shared Pathway</b><br>• {names}"
    if etype == "shared_disease":
        cnt = int(data.get("disease_count", 0))
        names = str(data.get("disease_names", "")).replace("|", "<br>• ")[:400]
        return (
            f"<b>Shared Disease (DisGeNET)</b><br>"
            f"<b>Count:</b> {cnt}<br>"
            f"<b>Best adj-p:</b> {float(data.get('min_pval', 1)):.2e}<br>"
            f"• {names}"
        )
    if etype == "shared_go_bp":
        cnt = int(data.get("go_count", 0))
        terms = str(data.get("go_terms", "")).replace("|", "<br>• ")[:500]
        return (
            f"<b>Shared GO BP</b><br>"
            f"<b>Count:</b> {cnt}<br>"
            f"<b>Best adj-p:</b> {float(data.get('min_pval', 1)):.2e}<br>"
            f"• {terms}"
        )
    return f"<b>{etype}</b>"


def make_group_vis_html(
    group_id: str,
    G: nx.MultiGraph,
    node_table: pd.DataFrame,
) -> str:
    """Generate a self-contained vis.js HTML for one cluster group's KG."""
    import json as _json

    nodes_js = []
    for nid in G.nodes():
        row = node_table.loc[nid] if nid in node_table.index else {}
        direction = str(row.get("direction", "")) if hasattr(row, "get") else ""
        try:
            gs_size = int(row.get("GS_SIZE", row.get("gs_size", 20)))
        except (ValueError, TypeError, AttributeError):
            gs_size = 20
        size_px = max(12, min(40, 12 + gs_size * 0.4))
        color = _NODE_COLORS.get(direction, "#95a5a6")
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

    edges_js = []
    for u, v, data in G.edges(data=True):
        etype = data.get("edge_type", "")
        color = _VIS_EDGE_COLORS.get(etype, "#aaa")
        label = _VIS_EDGE_LABELS.get(etype, etype)
        if etype == "gene_overlap":
            width = max(1.0, float(data.get("similarity", 0)) * 6)
        elif etype == "pubmed_costudy":
            width = max(1.0, float(data.get("shared_pmid_count", 1)) * 1.5)
        else:
            width = 1.5
        info = _edge_info_html(etype, data)
        edges_js.append({
            "from": u, "to": v,
            "color": {"color": color, "highlight": "#ffffff"},
            "width": width,
            "label": label,
            "font": {"size": 8, "color": color, "align": "middle",
                     "strokeWidth": 2, "strokeColor": "#1a1a2e"},
            "smooth": {"type": "continuous"},
            "info": info,
        })

    model_label = "GNN (text-only)" if group_id.startswith("A") else "GNN-beta (structlite)"
    nodes_json = _json.dumps(nodes_js, ensure_ascii=False)
    edges_json = _json.dumps(edges_js, ensure_ascii=False)

    legend_items = "".join(
        f'<div><span style="display:inline-block;width:16px;height:3px;background:{c};'
        f'vertical-align:middle;margin-right:5px;border-radius:2px"></span>{et.replace("_"," ")}</div>'
        for et, c in _VIS_EDGE_COLORS.items()
    )
    node_legend = "".join(
        f'<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
        f'background:{c};vertical-align:middle;margin-right:5px"></span>{lbl}</div>'
        for lbl, c in [("Up-regulated", "#27ae60"), ("Down-regulated", "#e74c3c"), ("Other", "#95a5a6")]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Group {group_id} — {model_label} KG</title>
<script src="https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js"></script>
<link  href="https://unpkg.com/vis-network@9.1.9/dist/dist/vis-network.min.css" rel="stylesheet">
<style>
* {{ box-sizing:border-box;margin:0;padding:0; }}
body {{ background:#1a1a2e;font-family:'Segoe UI',sans-serif;display:flex;height:100vh;overflow:hidden; }}
#network {{ flex:1; }}
#sidebar {{ width:310px;background:#16213e;border-left:1px solid #0f3460;
  padding:14px;color:#ccc;font-size:12px;overflow-y:auto;display:flex;
  flex-direction:column;gap:10px; }}
#sidebar h2 {{ color:#e0e0e0;font-size:13px;border-bottom:1px solid #0f3460;padding-bottom:6px; }}
#info-panel {{ line-height:1.6;color:#bbb; }}
#info-panel b {{ color:#e0e0e0; }}
#info-panel hr {{ border-color:#0f3460;margin:5px 0; }}
#legend {{ background:#0f3460;border-radius:5px;padding:9px;font-size:11px;line-height:1.9; }}
#hint {{ color:#555;font-size:11px;font-style:italic; }}
#stab-msg {{ color:#f39c12;font-size:11px;text-align:center; }}
a.back {{ color:#3498db;font-size:11px; }}
</style>
</head>
<body>
<div id="network"></div>
<div id="sidebar">
  <h2>Group {group_id} &mdash; {model_label}</h2>
  <a class="back" href="index.html">&#8592; Back to summary</a>
  <div id="stab-msg">&#9654; Stabilizing layout&hellip;</div>
  <div id="info-panel"><span style="color:#555">Click a node or edge for details.</span></div>
  <div id="legend">
    <div style="color:#e0e0e0;font-weight:bold;margin-bottom:3px">Nodes</div>
    {node_legend}
    <div style="color:#e0e0e0;font-weight:bold;margin:7px 0 3px">Edges</div>
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
  setTimeout(()=>document.getElementById('stab-msg').style.display='none',2000);
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


def make_index_html(
    groups: dict[str, list[str]],
    all_stats_rows: list[dict],
    all_coverage_rows: list[dict],
) -> str:
    """Generate a summary index HTML linking to all 20 group visualizations."""
    stats_by_group = {r["group_id"]: r for r in all_stats_rows}
    cov_by_group   = {r["group_id"]: r for r in all_coverage_rows}

    rows_html = ""
    for group_id in sorted(groups.keys()):
        model = "GNN (text-only)" if group_id.startswith("A") else "GNN-beta (structlite)"
        s = stats_by_group.get(group_id, {})
        c = cov_by_group.get(group_id, {})
        cov_cells = "".join(
            f'<td style="text-align:right">{c.get(et, 0):.2f}</td>'
            for et in EDGE_TYPES
        )
        rows_html += (
            f'<tr>'
            f'<td><a href="vis_{group_id}.html" style="color:#3498db">{group_id}</a></td>'
            f'<td>{model}</td>'
            f'<td style="text-align:right">{s.get("n_nodes","")}</td>'
            f'<td style="text-align:right">{s.get("n_edges","")}</td>'
            f'<td style="text-align:right">{s.get("density",0):.4f}</td>'
            f'<td style="text-align:right">{s.get("mean_degree",0):.1f}</td>'
            f'<td style="text-align:right">{s.get("n_connected_components","")}</td>'
            f'{cov_cells}'
            f'<td><a href="edges_{group_id}.csv" style="color:#aaa;font-size:11px">CSV</a></td>'
            f'</tr>\n'
        )

    et_headers = "".join(f'<th>cov_{et}</th>' for et in EDGE_TYPES)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>KG Cluster Metrics — GNN vs GNN-beta (k=10)</title>
<style>
body{{background:#1a1a2e;color:#ccc;font-family:'Segoe UI',sans-serif;padding:24px;}}
h1{{color:#e0e0e0;margin-bottom:6px;font-size:20px;}}
p{{color:#888;font-size:12px;margin-bottom:18px;}}
table{{border-collapse:collapse;width:100%;font-size:12px;}}
th{{background:#0f3460;color:#e0e0e0;padding:7px 10px;text-align:left;
   border-bottom:2px solid #3498db;white-space:nowrap;}}
td{{padding:5px 10px;border-bottom:1px solid #0f3460;}}
tr:hover td{{background:#16213e;}}
a{{color:#3498db;text-decoration:none;}}
a:hover{{text-decoration:underline;}}
.gnn{{background:#0f2a1a;}}
.beta{{background:#1a1a2e;}}
</style>
</head>
<body>
<h1>KG Cluster Metrics &mdash; GNN vs GNN-beta (k=10)</h1>
<p>20 groups: A0&ndash;A9 (GNN text-only) and B0&ndash;B9 (GNN-beta structlite).
Click a group ID to open its interactive KG, or CSV to download its edge list.</p>
<table>
<thead><tr>
  <th>Group</th><th>Model</th>
  <th>Nodes</th><th>Edges</th><th>Density</th><th>Mean&deg;</th><th>Components</th>
  {et_headers}
  <th>Edges</th>
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
    log.info("=" * 60)
    log.info("KG Cluster Metrics — GNN vs GNN-beta, k=%d", K)
    log.info("Output dir: %s", OUT_DIR)
    log.info("=" * 60)

    # ── Step 1: Cluster assignments ──────────────────────────────────────
    common, labels_a, labels_b = load_and_cluster()

    assignments_df = pd.DataFrame({
        "gs_id": common,
        "gnn_cluster": labels_a,
        "structlite_cluster": labels_b,
    })
    assignments_path = OUT_DIR / "cluster_assignments_k10.csv"
    assignments_df.to_csv(assignments_path, index=False)
    log.info("Saved cluster assignments: %s  (%d rows)", assignments_path, len(assignments_df))

    # ── Step 2: Define 20 groups ─────────────────────────────────────────
    groups = make_groups(common, labels_a, labels_b)
    for gid, members in sorted(groups.items()):
        log.info("  Group %-4s  %d members", gid, len(members))

    all_group_ids = list(common)

    # ── Step 3: Load shared resources ────────────────────────────────────
    log.info("Loading global metadata (%s) …", METADATA_CSV.name)
    meta = pd.read_csv(METADATA_CSV, low_memory=False)
    if "GS_ID" not in meta.columns:
        meta = meta.rename(columns={"gs_id": "GS_ID"})
    available_cols = [c for c in NODE_COLS if c in meta.columns]
    full_node_table = (
        meta[meta["GS_ID"].isin(set(all_group_ids))][available_cols]
        .copy()
        .set_index("GS_ID")
    )
    log.info("  Node table: %d rows, %d columns", len(full_node_table), len(full_node_table.columns))

    log.info("Loading PAGER edges file …")
    edges_df = _load_edges_df()
    log.info("  %d edges in PAGER file", len(edges_df))

    # ── Step 4: Fetch gene members (cached) ──────────────────────────────
    log.info("Fetching gene members for all %d gene sets …", len(all_group_ids))
    gene_members = fetch_gene_members(all_group_ids, full_node_table)

    # ── Step 5: Fetch Enrichr for ALL gene sets (one pass, cache shared) ─
    log.info("Fetching Enrichr annotations (amortized across all groups) …")
    fetch_all_enrichr(all_group_ids, gene_members, ENRICHR_LIBRARIES)

    # ── Step 6: Build KG and compute metrics per group ───────────────────
    all_coverage_rows: list[dict] = []
    all_centrality_rows: list[dict] = []
    all_stats_rows: list[dict] = []

    for group_id in sorted(groups.keys()):
        group_gs_ids = groups[group_id]
        log.info("─" * 50)
        log.info("Processing group %s …", group_id)

        G = build_group_kg(group_id, group_gs_ids, full_node_table, gene_members, edges_df)

        coverage = compute_knowledge_coverage(G, group_gs_ids)
        stats = compute_graph_stats(G, group_gs_ids)
        centrality_rows = compute_centrality(G, group_id, group_gs_ids)

        cov_row = {"group_id": group_id, "n_nodes": len(group_gs_ids)}
        cov_row.update(coverage)
        all_coverage_rows.append(cov_row)

        stats_row = {"group_id": group_id}
        stats_row.update(stats)
        all_stats_rows.append(stats_row)

        all_centrality_rows.extend(centrality_rows)

        # Per-group edge list CSV
        save_group_edges(group_id, G, OUT_DIR)
        log.info("  Saved edge CSV: edges_%s.csv (%d edges)", group_id, G.number_of_edges())

        # Per-group vis.js HTML
        html = make_group_vis_html(group_id, G, full_node_table)
        (OUT_DIR / f"vis_{group_id}.html").write_text(html, encoding="utf-8")
        log.info("  Saved vis HTML: vis_%s.html", group_id)

        log.info("Saved KG for group %s", group_id)

    # ── Step 7: Save aggregate CSVs ───────────────────────────────────────
    log.info("=" * 60)
    log.info("Saving final outputs …")

    pd.DataFrame(all_coverage_rows).to_csv(OUT_DIR / "knowledge_coverage.csv", index=False)
    log.info("  %s", OUT_DIR / "knowledge_coverage.csv")

    pd.DataFrame(all_centrality_rows).to_csv(OUT_DIR / "centrality.csv", index=False)
    log.info("  %s", OUT_DIR / "centrality.csv")

    pd.DataFrame(all_stats_rows).to_csv(OUT_DIR / "graph_stats.csv", index=False)
    log.info("  %s", OUT_DIR / "graph_stats.csv")

    # ── Step 8: Combined gene_overlap CSV (all 20 groups) ─────────────────
    log.info("Building combined gene_overlap CSV …")
    overlap_rows: list[dict] = []
    for group_id in sorted(groups.keys()):
        edge_path = OUT_DIR / f"edges_{group_id}.csv"
        if edge_path.exists():
            df_e = pd.read_csv(edge_path)
            go_df = df_e[df_e["edge_type"] == "gene_overlap"] if "edge_type" in df_e.columns else pd.DataFrame()
            if not go_df.empty:
                overlap_rows.append(go_df)
    if overlap_rows:
        all_overlap_df = pd.concat(overlap_rows, ignore_index=True)
        all_overlap_df.to_csv(OUT_DIR / "gene_overlap_all_groups.csv", index=False)
        log.info("  gene_overlap_all_groups.csv: %d rows", len(all_overlap_df))
    else:
        log.warning("  No gene_overlap edges found across any group.")

    # ── Step 9: Summary index HTML ────────────────────────────────────────
    log.info("Building index.html …")
    index_html = make_index_html(groups, all_stats_rows, all_coverage_rows)
    (OUT_DIR / "index.html").write_text(index_html, encoding="utf-8")
    log.info("  index.html")

    log.info("Done. All 20 groups processed.")
    log.info("Open: %s/index.html", OUT_DIR)


if __name__ == "__main__":
    main()
