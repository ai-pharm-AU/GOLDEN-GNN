"""
Biomedical LLM Cluster Summarization using OpenBioLLM-8B (Llama-3-based).

Generates a 2-3 sentence biological summary for each of the 20 clusters
(A0-A9: GNN text-only, B0-B9: GNN-beta structlite).

Usage:
    .venv/bin/python KnowledgeGraph/kg_cluster_summarize.py
"""

import os
import re
import sys
import csv
import html
import logging
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
OUT_DIR = BASE / "cluster_metrics_out"
METADATA_PATH = (
    Path(__file__).parent.parent
    / "work_alpha_gnn_20260212"
    / "task_20260219_gnn_feature_ablation"
    / "human_only"
    / "subset085"
    / "inputs"
    / "enriched_gs_metadata_subset085_human.csv"
)

GROUPS = [f"A{i}" for i in range(10)] + [f"B{i}" for i in range(10)]
MODEL_LABEL = {"A": "GNN text-only", "B": "GNN-beta structlite"}

# ---------------------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------------------

def load_metadata(path: Path) -> dict:
    """Returns {gs_id: {'name': ..., 'summary': ...}}"""
    meta = {}
    df = pd.read_csv(path, usecols=["GS_ID", "NAME", "geo_series_title", "geo_series_summary"],
                     dtype=str, low_memory=False)
    for _, row in df.iterrows():
        gs_id = str(row["GS_ID"]).strip()
        name = str(row["NAME"]).strip() if pd.notna(row["NAME"]) else gs_id
        title = str(row["geo_series_title"]).strip() if pd.notna(row["geo_series_title"]) else ""
        summary = str(row["geo_series_summary"]).strip() if pd.notna(row["geo_series_summary"]) else ""
        # Truncate long summaries
        summary = summary[:300] + "..." if len(summary) > 300 else summary
        meta[gs_id] = {"name": name, "title": title, "summary": summary}
    return meta


def load_centrality(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"gs_id": str})


def load_graph_stats(path: Path) -> dict:
    df = pd.read_csv(path)
    return {row["group_id"]: row for _, row in df.iterrows()}


def get_top_terms_from_edges(edges_path: Path, edge_type: str, col: str, top_n: int = 10) -> list:
    """
    Parse pipe-delimited values from `col` in rows where edge_type matches.
    Return top_n by frequency.
    """
    if not edges_path.exists():
        return []
    df = pd.read_csv(edges_path, dtype=str, low_memory=False)
    subset = df[df["edge_type"] == edge_type]
    counter: Counter = Counter()
    for val in subset[col].dropna():
        for term in val.split("|"):
            term = term.strip()
            if term:
                counter[term] += 1
    return [t for t, _ in counter.most_common(top_n)]


# ---------------------------------------------------------------------------
# Step 2: Build prompt
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "You are an expert bioinformatician analyzing gene set clusters from GNN-based "
    "embedding analysis of human transcriptomics data.<|eot_id|>\n"
    "<|start_header_id|>user<|end_header_id|>\n"
    "Below is a cluster of {n_nodes} gene sets identified by {model_label} embedding.\n\n"
    "Top gene sets (by network centrality):\n{gene_set_list}\n\n"
    "Most frequently shared Reactome pathways:\n{pathway_list}\n\n"
    "Most frequently shared GO Biological Process terms:\n{go_list}\n\n"
    "Write a concise 2-3 sentence biological summary of what unifies this cluster. "
    "Focus on cell biology, tissue type, disease context, or molecular mechanisms "
    "that link these gene sets.<|eot_id|>\n"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)


def build_prompt(group_id: str, centrality_df: pd.DataFrame, metadata: dict,
                 graph_stats: dict) -> tuple[str, dict]:
    """
    Returns (prompt_str, context_dict) where context_dict has top gene sets,
    pathways, go terms for saving to CSV.
    """
    variant = MODEL_LABEL[group_id[0]]
    stats = graph_stats.get(group_id)
    n_nodes = int(stats["n_nodes"]) if stats is not None else "?"

    # Top 5 gene sets by degree_centrality for this group
    grp = centrality_df[centrality_df["group_id"] == group_id].copy()
    grp = grp.sort_values("degree_centrality", ascending=False).head(5)

    gs_lines = []
    gs_ids_top = []
    for _, row in grp.iterrows():
        gs_id = str(row["gs_id"]).strip()
        gs_ids_top.append(gs_id)
        m = metadata.get(gs_id, {})
        name = m.get("name", gs_id)
        summary = m.get("summary", "")
        line = f"  - {name}"
        if summary:
            line += f": {summary}"
        gs_lines.append(line)

    gene_set_list = "\n".join(gs_lines) if gs_lines else "  (none)"

    # Top pathways and GO terms from edges file
    edges_path = OUT_DIR / f"edges_{group_id}.csv"
    top_pathways = get_top_terms_from_edges(edges_path, "shared_pathway", "pathway_names", 10)
    top_go = get_top_terms_from_edges(edges_path, "shared_go_bp", "go_terms", 10)

    # Clean up GO term format: "Term Name (GO:XXXXXXX)" -> just term name
    top_go_clean = []
    for t in top_go:
        # Strip GO ID suffix for readability
        clean = re.sub(r"\s*\(GO:\d+\)\s*$", "", t).strip()
        top_go_clean.append(clean)

    pathway_list = "\n".join(f"  - {p}" for p in top_pathways) if top_pathways else "  (none)"
    go_list = "\n".join(f"  - {g}" for g in top_go_clean) if top_go_clean else "  (none)"

    prompt = PROMPT_TEMPLATE.format(
        n_nodes=n_nodes,
        model_label=variant,
        gene_set_list=gene_set_list,
        pathway_list=pathway_list,
        go_list=go_list,
    )

    context = {
        "n_nodes": n_nodes,
        "model_variant": variant,
        "top_gene_sets": "; ".join(
            metadata.get(g, {}).get("name", g) for g in gs_ids_top
        ),
        "top_pathways": "; ".join(top_pathways),
        "top_go_terms": "; ".join(top_go_clean),
    }
    return prompt, context


# ---------------------------------------------------------------------------
# Step 3: Load model
# ---------------------------------------------------------------------------

def load_model():
    model_id = "aaditya/Llama3-OpenBioLLM-8B"
    log.info(f"Loading tokenizer from {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    log.info("Loading model with 4-bit quantization ...")
    # Spread across GPUs 1-3, skipping GPU 0 (occupied by other processes).
    # 4-bit quantization peaks >10 GB during loading on a single 11 GB card.
    max_memory = {0: "0GiB", 1: "9GiB", 2: "9GiB", 3: "9GiB",
                  4: "0GiB", 5: "0GiB", 6: "0GiB", 7: "0GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        max_memory=max_memory,
    )
    model.eval()
    log.info("Model loaded.")
    return tokenizer, model


# ---------------------------------------------------------------------------
# Step 4: Generate summary
# ---------------------------------------------------------------------------

def generate_summary(prompt: str, tokenizer, model) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only new tokens
    new_tokens = output[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


# ---------------------------------------------------------------------------
# Step 5: Save outputs
# ---------------------------------------------------------------------------

def save_csv(rows: list[dict], out_path: Path):
    fieldnames = ["group_id", "model_variant", "n_nodes",
                  "top_gene_sets", "top_pathways", "top_go_terms", "summary"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"CSV saved: {out_path}")


def save_html(rows: list[dict], out_path: Path):
    def esc(s):
        return html.escape(str(s))

    cards = []
    for row in rows:
        group_id = row["group_id"]
        variant_badge = "A-series" if group_id.startswith("A") else "B-series"
        badge_color = "#4a9eff" if group_id.startswith("A") else "#ff8c42"

        gs_items = "".join(
            f"<li>{esc(g.strip())}</li>"
            for g in str(row["top_gene_sets"]).split(";") if g.strip()
        )
        pw_items = "".join(
            f"<li>{esc(p.strip())}</li>"
            for p in str(row["top_pathways"]).split(";") if p.strip()
        )
        go_items = "".join(
            f"<li>{esc(g.strip())}</li>"
            for g in str(row["top_go_terms"]).split(";") if g.strip()
        )

        card = f"""
        <div class="card">
          <div class="card-header">
            <span class="group-id">{esc(group_id)}</span>
            <span class="badge" style="background:{badge_color}">{variant_badge}</span>
            <span class="n-nodes">{esc(row['n_nodes'])} gene sets</span>
          </div>
          <div class="summary">{esc(row['summary'])}</div>
          <div class="details-grid">
            <div class="detail-col">
              <h4>Top Gene Sets</h4>
              <ul>{gs_items}</ul>
            </div>
            <div class="detail-col">
              <h4>Top Pathways</h4>
              <ul>{pw_items}</ul>
            </div>
            <div class="detail-col">
              <h4>Top GO BP Terms</h4>
              <ul>{go_items}</ul>
            </div>
          </div>
        </div>"""
        cards.append(card)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cluster Summaries — OpenBioLLM-8B</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0d1117;
    color: #c9d1d9;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    padding: 2rem;
  }}
  h1 {{
    color: #f0f6fc;
    text-align: center;
    margin-bottom: 0.5rem;
    font-size: 1.8rem;
  }}
  .subtitle {{
    text-align: center;
    color: #8b949e;
    margin-bottom: 2rem;
    font-size: 0.9rem;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(600px, 1fr));
    gap: 1.5rem;
    max-width: 1600px;
    margin: 0 auto;
  }}
  .card {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1.25rem;
  }}
  .card-header {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.75rem;
  }}
  .group-id {{
    font-size: 1.3rem;
    font-weight: 700;
    color: #f0f6fc;
  }}
  .badge {{
    padding: 0.2rem 0.6rem;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #0d1117;
  }}
  .n-nodes {{
    color: #8b949e;
    font-size: 0.85rem;
    margin-left: auto;
  }}
  .summary {{
    background: #1c2128;
    border-left: 3px solid #238636;
    padding: 0.75rem 1rem;
    border-radius: 0 4px 4px 0;
    margin-bottom: 1rem;
    font-size: 0.9rem;
    line-height: 1.6;
    color: #e6edf3;
  }}
  .details-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
  }}
  .detail-col h4 {{
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #8b949e;
    margin-bottom: 0.4rem;
  }}
  .detail-col ul {{
    list-style: none;
    padding: 0;
  }}
  .detail-col li {{
    font-size: 0.78rem;
    color: #c9d1d9;
    padding: 0.15rem 0;
    border-bottom: 1px solid #21262d;
    word-break: break-word;
  }}
  .detail-col li:last-child {{ border-bottom: none; }}
</style>
</head>
<body>
<h1>Cluster Biological Summaries</h1>
<p class="subtitle">Generated by OpenBioLLM-Llama3-8B &bull; 20 clusters (A0–A9: GNN text-only, B0–B9: GNN-beta structlite)</p>
<div class="grid">
{''.join(cards)}
</div>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    log.info(f"HTML saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== Cluster Summarization Start ===")

    # Load shared data
    log.info("Loading metadata ...")
    metadata = load_metadata(METADATA_PATH)
    log.info(f"  Loaded {len(metadata)} gene set metadata entries.")

    centrality_df = load_centrality(OUT_DIR / "centrality.csv")
    graph_stats = load_graph_stats(OUT_DIR / "graph_stats.csv")

    # Load model once
    tokenizer, model = load_model()

    rows = []
    for group_id in GROUPS:
        log.info(f"Processing group {group_id} ...")
        prompt, context = build_prompt(group_id, centrality_df, metadata, graph_stats)

        log.info(f"  Generating summary for {group_id} ...")
        summary = generate_summary(prompt, tokenizer, model)
        log.info(f"  Summary: {summary[:120]}...")

        rows.append({
            "group_id": group_id,
            "model_variant": context["model_variant"],
            "n_nodes": context["n_nodes"],
            "top_gene_sets": context["top_gene_sets"],
            "top_pathways": context["top_pathways"],
            "top_go_terms": context["top_go_terms"],
            "summary": summary,
        })

    # Save outputs
    save_csv(rows, OUT_DIR / "cluster_summaries.csv")
    save_html(rows, OUT_DIR / "cluster_summaries.html")

    log.info("=== Done ===")
    log.info(f"  Rows written: {len(rows)}")
    empty = sum(1 for r in rows if not r["summary"].strip())
    log.info(f"  Empty summaries: {empty}")


if __name__ == "__main__":
    main()
