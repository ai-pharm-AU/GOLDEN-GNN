#!/usr/bin/env python3
"""
vis_case_study_A8_B7.py — Case Study: A8 × B7 Overlap Visualization

A8 (GNN text-only, 261 nodes) and B7 (GNN-beta structlite, 268 nodes)
share 181 gene sets = 67.5% overlap.

Outputs:
    KnowledgeGraph/cluster_metrics_out/vis_sparse/case_study_A8_B7.html

Usage:
    cd /home/zzz0054/GoldenF
    .venv/bin/python KnowledgeGraph/vis_case_study_A8_B7.py
"""

import csv
import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
KG_DIR = Path(__file__).resolve().parent
METRICS_DIR = KG_DIR / "cluster_metrics_out"
VIS_DIR = METRICS_DIR / "vis_sparse"
OUT_HTML = VIS_DIR / "case_study_A8_B7.html"

METADATA_CSV = (
    KG_DIR.parent
    / "work_alpha_gnn_20260212"
    / "task_20260219_gnn_feature_ablation"
    / "human_only"
    / "subset085"
    / "inputs"
    / "enriched_gs_metadata_subset085_human.csv"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPARSE_TYPES = {"gene_overlap", "pubmed_costudy", "shared_go_bp"}
EDGE_COLORS = {
    "gene_overlap": "#e74c3c",
    "pubmed_costudy": "#3498db",
    "shared_go_bp": "#1abc9c",
}
MAX_EDGES = 2500
random.seed(42)


# ---------------------------------------------------------------------------
# Step 1: Load and classify nodes
# ---------------------------------------------------------------------------
def load_assignments():
    a8_nodes, b7_nodes = set(), set()
    with open(METRICS_DIR / "cluster_assignments_k10.csv") as f:
        for row in csv.DictReader(f):
            if row["gnn_cluster"] == "8":
                a8_nodes.add(row["gs_id"])
            if row["structlite_cluster"] == "7":
                b7_nodes.add(row["gs_id"])
    overlap = a8_nodes & b7_nodes
    a8_only = a8_nodes - overlap
    b7_only = b7_nodes - overlap
    union = a8_nodes | b7_nodes
    print(f"A8: {len(a8_nodes)}, B7: {len(b7_nodes)}, Overlap: {len(overlap)}, Union: {len(union)}")
    return a8_only, overlap, b7_only, union


# ---------------------------------------------------------------------------
# Step 2: Load centrality (degree_centrality per gs_id for A8/B7)
# ---------------------------------------------------------------------------
def load_centrality(union):
    centrality = {}
    with open(METRICS_DIR / "centrality.csv") as f:
        for row in csv.DictReader(f):
            if row["group_id"] in ("A8", "B7") and row["gs_id"] in union:
                gsid = row["gs_id"]
                val = float(row.get("degree_centrality", 0) or 0)
                # keep max if duplicate (node appears in both A8 and B7)
                if gsid not in centrality or val > centrality[gsid]:
                    centrality[gsid] = val
    return centrality


# ---------------------------------------------------------------------------
# Step 3: Load metadata (NAME + geo_series_summary per gs_id)
# ---------------------------------------------------------------------------
def load_metadata(union):
    meta = {}
    with open(METADATA_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gsid = row.get("GS_ID", row.get("gs_id", ""))
            if gsid in union:
                meta[gsid] = {
                    "name": row.get("NAME", row.get("name", gsid)),
                    "summary": row.get("geo_series_summary", ""),
                }
    return meta


# ---------------------------------------------------------------------------
# Step 4: Load edges (filter to union, filter edge types, cap 2500)
# ---------------------------------------------------------------------------
def load_edges(union):
    all_edges = []
    for group in ("A8", "B7"):
        path = METRICS_DIR / f"edges_{group}.csv"
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                etype = row.get("edge_type", "")
                if etype not in SPARSE_TYPES:
                    continue
                src, tgt = row.get("source", ""), row.get("target", "")
                if src not in union or tgt not in union:
                    continue
                try:
                    nlp = float(row.get("neg_log_pval", 0) or 0)
                except ValueError:
                    nlp = 0.0
                all_edges.append({
                    "source": src,
                    "target": tgt,
                    "edge_type": etype,
                    "neg_log_pval": nlp,
                    "similarity": row.get("similarity", ""),
                    "shared_pmid_count": row.get("shared_pmid_count", ""),
                    "go_count": row.get("go_count", ""),
                })

    # Deduplicate (same pair + same type)
    seen = set()
    deduped = []
    for e in all_edges:
        key = (min(e["source"], e["target"]), max(e["source"], e["target"]), e["edge_type"])
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    # Sort by neg_log_pval desc, cap at MAX_EDGES
    deduped.sort(key=lambda x: x["neg_log_pval"], reverse=True)
    return deduped[:MAX_EDGES]


# ---------------------------------------------------------------------------
# Step 5: Load cluster summaries for A8 and B7
# ---------------------------------------------------------------------------
def load_summaries():
    summaries = {}
    with open(METRICS_DIR / "cluster_summaries.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["group_id"] in ("A8", "B7"):
                summaries[row["group_id"]] = row.get("summary", "")
    return summaries


# ---------------------------------------------------------------------------
# Step 6: Top-5 pathways per region (from edges, by pathway names)
# ---------------------------------------------------------------------------
def top5_per_region(union, a8_only, overlap, b7_only):
    """Extract top pathway names per region from edges files."""
    from collections import Counter
    region_counters = {"a8_only": Counter(), "overlap": Counter(), "b7_only": Counter()}

    def get_region(gsid):
        if gsid in a8_only:
            return "a8_only"
        if gsid in overlap:
            return "overlap"
        if gsid in b7_only:
            return "b7_only"
        return None

    for group in ("A8", "B7"):
        path = METRICS_DIR / f"edges_{group}.csv"
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("edge_type") != "shared_pathway":
                    continue
                src, tgt = row.get("source", ""), row.get("target", "")
                if src not in union or tgt not in union:
                    continue
                r = get_region(src) or get_region(tgt)
                if r is None:
                    continue
                names = row.get("pathway_names", "")
                for n in names.split("|")[:5]:
                    n = n.strip()
                    if n:
                        region_counters[r][n] += 1

    result = {}
    for r, counter in region_counters.items():
        result[r] = [name for name, _ in counter.most_common(5)]
    return result


# ---------------------------------------------------------------------------
# Step 7: Build vis-network nodes + edges JSON
# ---------------------------------------------------------------------------
def build_nodes_json(a8_only, overlap, b7_only, centrality, meta):
    nodes = []

    def make_node(gsid, region, color, x_center):
        dc = centrality.get(gsid, 0.1)
        size = min(20, 8 + dc * 12)
        m = meta.get(gsid, {})
        name = m.get("name", gsid)
        summary = m.get("summary", "")[:200]
        label = name[:28]
        if region == "a8_only":
            badge = "A8 only"
        elif region == "overlap":
            badge = "Overlap"
        else:
            badge = "B7 only"
        tooltip = f"{name}\n{summary}\n[{badge}]"
        y = random.uniform(-300, 300)
        nodes.append({
            "id": gsid,
            "label": label,
            "x": x_center + random.uniform(-150, 150),
            "y": y,
            "color": {
                "background": color,
                "border": "#ffffff",
                "highlight": {"background": color, "border": "#ffffff"},
            },
            "size": size,
            "font": {"color": "#ffffff", "size": 9},
            "shadow": True,
            "title": tooltip,
            "region": region,
            "centrality": round(dc, 4),
            "full_name": name,
            "summary": m.get("summary", "")[:200],
        })

    for gsid in sorted(a8_only):
        make_node(gsid, "a8_only", "#4a9eff", -600)
    for gsid in sorted(overlap):
        make_node(gsid, "overlap", "#9b59b6", 0)
    for gsid in sorted(b7_only):
        make_node(gsid, "b7_only", "#ff8c42", 600)

    return nodes


def build_edges_json(edges):
    result = []
    for e in edges:
        etype = e["edge_type"]
        color = EDGE_COLORS.get(etype, "#aaa")
        result.append({
            "from": e["source"],
            "to": e["target"],
            "color": {"color": color, "highlight": "#ffffff", "opacity": 0.6},
            "width": 1,
            "arrows": "",
            "smooth": {"type": "continuous"},
            "edge_type": etype,
        })
    return result


# ---------------------------------------------------------------------------
# Step 8: Render HTML
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Case Study: A8 × B7 Overlap</title>
<script src="https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js"></script>
<link href="https://unpkg.com/vis-network@9.1.9/dist/dist/vis-network.min.css" rel="stylesheet">
<style>
* {{ box-sizing:border-box;margin:0;padding:0; }}
body {{ background:#1a1a2e;font-family:'Segoe UI',sans-serif;display:flex;height:100vh;overflow:hidden; }}
#network {{ flex:1; }}
#sidebar {{ width:320px;background:#16213e;border-left:1px solid #0f3460;
  padding:14px;color:#ccc;font-size:12px;overflow-y:auto;display:flex;
  flex-direction:column;gap:10px; }}
#sidebar h2 {{ color:#e0e0e0;font-size:14px;border-bottom:1px solid #0f3460;padding-bottom:6px; }}
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
a.back {{ color:#3498db;font-size:11px; }}
.overlap-box {{ background:#1a1a3e;border:1px solid #0f3460;border-radius:6px;padding:8px;font-size:11px; }}
.region-row {{ display:flex;align-items:center;gap:6px;padding:3px 0; }}
.region-dot {{ width:10px;height:10px;border-radius:50%;flex-shrink:0; }}
.badge {{ background:#9b59b6;color:#fff;border-radius:4px;padding:1px 6px;font-size:10px;font-weight:bold; }}
.summary-box {{ background:#0f2030;border-radius:4px;padding:7px;font-size:10px;color:#aaa;line-height:1.5; }}
details {{ border:1px solid #0f3460;border-radius:4px;overflow:hidden; }}
summary {{ cursor:pointer;padding:5px 8px;background:#0f2030;color:#bbb;font-size:11px;font-weight:bold; }}
summary:hover {{ background:#0f3460; }}
.pathway-list {{ padding:6px 10px;font-size:10px;color:#aaa;line-height:1.8; }}
</style>
</head>
<body>
<div id="network"></div>
<div id="sidebar">
  <h2>Case Study: A8 &times; B7</h2>
  <div style="display:flex;flex-direction:column;gap:4px">
    <a class="back" href="index_sparse.html">&larr; index_sparse.html</a>
    <a class="back" href="vis_sparse_A8.html">&rarr; vis_sparse_A8.html</a>
    <a class="back" href="vis_sparse_B7.html">&rarr; vis_sparse_B7.html</a>
  </div>
  <div id="stab-msg">&#9654; Stabilizing layout&hellip;</div>

  <div class="overlap-box">
    <div class="section-title" style="margin-top:0">Membership</div>
    <div class="region-row">
      <div class="region-dot" style="background:#4a9eff"></div>
      <span style="flex:1">A8 only (GNN text-only)</span>
      <b style="color:#4a9eff">{n_a8_only}</b>
    </div>
    <div class="region-row">
      <div class="region-dot" style="background:#9b59b6"></div>
      <span style="flex:1">Overlap</span>
      <b style="color:#9b59b6">{n_overlap}</b>
      <span class="badge">{overlap_pct}%</span>
    </div>
    <div class="region-row">
      <div class="region-dot" style="background:#ff8c42"></div>
      <span style="flex:1">B7 only (structlite)</span>
      <b style="color:#ff8c42">{n_b7_only}</b>
    </div>
  </div>

  <div>
    <div class="section-title">A8 Summary</div>
    <div class="summary-box">{summary_a8}</div>
  </div>
  <div>
    <div class="section-title">B7 Summary</div>
    <div class="summary-box">{summary_b7}</div>
  </div>

  <details>
    <summary>&#128200; Top Pathways: A8 only</summary>
    <div class="pathway-list">{top_a8_only}</div>
  </details>
  <details>
    <summary>&#128200; Top Pathways: Overlap</summary>
    <div class="pathway-list">{top_overlap}</div>
  </details>
  <details>
    <summary>&#128200; Top Pathways: B7 only</summary>
    <div class="pathway-list">{top_b7_only}</div>
  </details>

  <div id="hint">Click a node for details.</div>
  <div id="info-panel"><span style="color:#555">&#8681; Node info appears here</span></div>

  <div id="legend">
    <div class="section-title" style="margin-top:0">Node membership</div>
    <div class="region-row"><div class="region-dot" style="background:#4a9eff"></div> A8 only</div>
    <div class="region-row"><div class="region-dot" style="background:#9b59b6"></div> Overlap</div>
    <div class="region-row"><div class="region-dot" style="background:#ff8c42"></div> B7 only</div>
    <div class="section-title" style="margin-top:6px">Edge types</div>
    <div><span style="display:inline-block;width:16px;height:3px;background:#e74c3c;vertical-align:middle;margin-right:5px;border-radius:2px"></span>Gene overlap</div>
    <div><span style="display:inline-block;width:16px;height:3px;background:#3498db;vertical-align:middle;margin-right:5px;border-radius:2px"></span>PubMed co-study</div>
    <div><span style="display:inline-block;width:16px;height:3px;background:#1abc9c;vertical-align:middle;margin-right:5px;border-radius:2px"></span>Shared GO BP</div>
  </div>
</div>

<script>
const NODES_DATA = {nodes_json};
const EDGES_DATA = {edges_json};

const nodeMap = {{}};
NODES_DATA.forEach(n => {{ nodeMap[n.id] = n; }});

const container = document.getElementById('network');
const data = {{
  nodes: new vis.DataSet(NODES_DATA),
  edges: new vis.DataSet(EDGES_DATA),
}};

const options = {{
  physics: {{
    enabled: true,
    solver: 'forceAtlas2Based',
    forceAtlas2Based: {{
      gravitationalConstant: -80,
      centralGravity: 0.005,
      springLength: 100,
      springConstant: 0.08,
      damping: 0.4,
    }},
    stabilization: {{ iterations: 400, updateInterval: 25 }},
  }},
  nodes: {{
    shape: 'dot',
    borderWidth: 1.5,
    chosen: true,
  }},
  edges: {{
    arrows: {{ to: {{ enabled: false }} }},
    smooth: {{ type: 'continuous' }},
  }},
  interaction: {{
    hover: true,
    tooltipDelay: 100,
    hideEdgesOnDrag: true,
  }},
}};

const network = new vis.Network(container, data, options);

// ── Draw background ellipses ──────────────────────────────────────────────
network.on('beforeDrawing', function(ctx) {{
  const regions = [
    {{ cx: -600, label: 'A8 only · {n_a8_only}',     fill: 'rgba(74,158,255,0.07)',   stroke: 'rgba(74,158,255,0.4)' }},
    {{ cx:    0, label: 'Overlap · {n_overlap} (67.5%)', fill: 'rgba(155,89,182,0.10)',  stroke: 'rgba(155,89,182,0.5)' }},
    {{ cx:  600, label: 'B7 only · {n_b7_only}',      fill: 'rgba(255,140,66,0.07)',   stroke: 'rgba(255,140,66,0.4)' }},
  ];
  regions.forEach(r => {{
    const pos = network.canvasToDOM({{ x: r.cx, y: 0 }});
    const w = Math.abs(network.canvasToDOM({{ x: r.cx + 280, y: 0 }}).x - network.canvasToDOM({{ x: r.cx - 280, y: 0 }}).x) / 2;
    const h = Math.abs(network.canvasToDOM({{ x: 0, y: 350 }}).y - network.canvasToDOM({{ x: 0, y: -350 }}).y) / 2;
    ctx.save();
    ctx.translate(pos.x, pos.y);
    ctx.beginPath();
    ctx.ellipse(0, 0, w, h, 0, 0, Math.PI * 2);
    ctx.fillStyle = r.fill;
    ctx.fill();
    ctx.strokeStyle = r.stroke;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = r.stroke;
    ctx.font = 'bold 11px Segoe UI';
    ctx.textAlign = 'center';
    ctx.fillText(r.label, 0, -h - 8);
    ctx.restore();
  }});
}});

// ── Disable physics after stabilization ──────────────────────────────────
network.on('stabilizationIterationsDone', function() {{
  network.setOptions({{ physics: {{ enabled: false }} }});
  document.getElementById('stab-msg').style.display = 'none';
}});

// ── Node click handler ─────────────────────────────────────────────────
network.on('click', function(params) {{
  const panel = document.getElementById('info-panel');
  if (!params.nodes.length) {{
    panel.innerHTML = '<span style="color:#555">&#8681; Click a node for details</span>';
    return;
  }}
  const nid = params.nodes[0];
  const n = nodeMap[nid];
  if (!n) return;
  const regionLabel = n.region === 'a8_only' ? '<span style="color:#4a9eff">A8 only</span>'
    : n.region === 'overlap' ? '<span style="color:#9b59b6">Overlap</span>'
    : '<span style="color:#ff8c42">B7 only</span>';
  panel.innerHTML = `
    <b>${{n.full_name}}</b>
    <hr>
    <b>ID:</b> ${{nid}}<br>
    <b>Membership:</b> ${{regionLabel}}<br>
    <b>Centrality:</b> ${{n.centrality}}<br>
    <hr>
    <b>Summary:</b><br><span style="color:#aaa">${{n.summary || '—'}}</span>
  `;
}});
</script>
</body>
</html>
"""


def fmt_top5(pathways):
    if not pathways:
        return "<i style='color:#555'>No data</i>"
    return "".join(f"<div>&#8226; {p}</div>" for p in pathways)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading assignments...")
    a8_only, overlap, b7_only, union = load_assignments()

    print("Loading centrality...")
    centrality = load_centrality(union)

    print("Loading metadata...")
    meta = load_metadata(union)

    print("Loading edges...")
    edges = load_edges(union)
    print(f"  Edges after filter + cap: {len(edges)}")

    print("Loading summaries...")
    summaries = load_summaries()

    print("Computing top-5 pathways per region...")
    top5 = top5_per_region(union, a8_only, overlap, b7_only)

    print("Building nodes/edges JSON...")
    nodes_json = build_nodes_json(a8_only, overlap, b7_only, centrality, meta)
    edges_json = build_edges_json(edges)

    n_a8_only = len(a8_only)
    n_overlap = len(overlap)
    n_b7_only = len(b7_only)
    overlap_pct = round(n_overlap / (n_a8_only + n_overlap) * 100, 1)

    summary_a8 = summaries.get("A8", "")[:250]
    summary_b7 = summaries.get("B7", "")[:250]

    html = HTML_TEMPLATE.format(
        n_a8_only=n_a8_only,
        n_overlap=n_overlap,
        n_b7_only=n_b7_only,
        overlap_pct=overlap_pct,
        summary_a8=summary_a8,
        summary_b7=summary_b7,
        top_a8_only=fmt_top5(top5.get("a8_only", [])),
        top_overlap=fmt_top5(top5.get("overlap", [])),
        top_b7_only=fmt_top5(top5.get("b7_only", [])),
        nodes_json=json.dumps(nodes_json, ensure_ascii=False),
        edges_json=json.dumps(edges_json, ensure_ascii=False),
    )

    OUT_HTML.write_text(html, encoding="utf-8")
    size_kb = OUT_HTML.stat().st_size // 1024
    print(f"Written: {OUT_HTML}  ({size_kb} KB)")
    print("Done.")


if __name__ == "__main__":
    main()
