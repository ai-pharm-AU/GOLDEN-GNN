#!/usr/bin/env python3
"""
build_kg.py — Build a multi-relational Knowledge Graph for Cluster 8 Super-PAG (Subset 085).

The super-PAG is the 181-gene-set intersection of:
  A8 (text-only GNN k-means cluster 8) ∩ B7 (text+structlite GNN k-means cluster 7)

Edge types:
  1. gene_overlap        — PAGER gene-overlap statistics (local file)
  2. pubmed_costudy      — shared PubMed citations / GSE associations
  3. shared_pathway      — Reactome pathway membership (via cache)
  4. shared_mesh_term    — shared MeSH descriptors (DISGENET fallback)
  5. shared_disease      — DisGeNET disease associations via Enrichr API
  6. shared_go_bp        — GO Biological Process terms via Enrichr API

Usage:
    cd /home/zzz0054/GoldenF
    .venv/bin/python KnowledgeGraph/build_kg.py

Outputs (all under KnowledgeGraph/):
    super_pag_ids.txt
    gene_members_cache.json
    enrichr_cache.json
    cluster8_subset085_kg.graphml
    cluster8_subset085_kg.json
    cluster8_subset085_kg_stats.json
"""

import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    PROJECT_ROOT
    / "work_alpha_gnn_20260212"
    / "task_20260219_gnn_feature_ablation"
    / "runs"
    / "single_085_090_human_only_20260220_233515"
)
SUBSET_DIR = (
    PROJECT_ROOT
    / "work_alpha_gnn_20260212"
    / "task_20260219_gnn_feature_ablation"
    / "human_only"
    / "subset085"
)
KG_DIR = PROJECT_ROOT / "KnowledgeGraph"

NPZ_TEXT_ONLY = RUN_DIR / "subset085" / "inputs" / "gnn_old_holdout_subset_085_human.npz"
NPZ_STRUCTLITE = RUN_DIR / "subset085" / "features" / "text_structlite_gnn_holdout_subset085.npz"
METADATA_CSV = SUBSET_DIR / "inputs" / "enriched_gs_metadata_subset085_human.csv"
EDGES_TXT = SUBSET_DIR / "inputs" / "edges_with_mtype_threshold_085_human.txt"

SUPER_PAG_IDS_TXT = KG_DIR / "super_pag_ids.txt"
GENE_MEMBERS_CACHE = KG_DIR / "gene_members_cache.json"
KG_GRAPHML = KG_DIR / "cluster8_subset085_kg.graphml"
KG_JSON = KG_DIR / "cluster8_subset085_kg.json"
KG_STATS_JSON = KG_DIR / "cluster8_subset085_kg_stats.json"

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
# API keys from environment
# ---------------------------------------------------------------------------
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
DISGENET_API_KEY = os.environ.get("DISGENET_API_KEY", "")

NCBI_RATE = 10 if NCBI_API_KEY else 3   # requests/sec
NCBI_DELAY = 1.0 / NCBI_RATE


# ---------------------------------------------------------------------------
# Step 1 — Re-derive super-PAG IDs
# ---------------------------------------------------------------------------

def _best_kmeans(emb: np.ndarray, k: int, seeds: list[int]) -> np.ndarray:
    """Run k-means with multiple seeds; return labels from lowest-inertia run."""
    best_labels = None
    best_inertia = float("inf")
    best_seed = seeds[0]
    for seed in seeds:
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        km.fit(emb)
        if km.inertia_ < best_inertia:
            best_inertia = km.inertia_
            best_labels = km.labels_.copy()
            best_seed = seed
    log.info("  KMeans best_seed=%d inertia=%.4f", best_seed, best_inertia)
    return best_labels


def get_cluster8_genesets() -> list[str]:
    """
    Re-run k-means on both GNN embeddings to identify the super-PAG:
    A8 (text-only cluster 8) ∩ B7 (text+structlite cluster 7) = ~181 gene sets.

    Returns sorted list of GS_IDs.
    """
    if SUPER_PAG_IDS_TXT.exists():
        ids = SUPER_PAG_IDS_TXT.read_text().strip().splitlines()
        log.info("Loaded %d super-PAG IDs from cache: %s", len(ids), SUPER_PAG_IDS_TXT)
        return ids

    log.info("Step 1: Re-deriving super-PAG IDs via k-means …")

    a = np.load(NPZ_TEXT_ONLY, allow_pickle=True)
    b = np.load(NPZ_STRUCTLITE, allow_pickle=True)

    ids_a = a["ID"].tolist()
    emb_a = a["embeddings"]
    ids_b = b["ID"].tolist()
    emb_b = b["embeddings"]

    # Inner-join alignment
    common = sorted(set(ids_a) & set(ids_b))
    log.info("  Inner-join: %d common gene sets (A=%d, B=%d)", len(common), len(ids_a), len(ids_b))

    idx_a = {v: i for i, v in enumerate(ids_a)}
    idx_b = {v: i for i, v in enumerate(ids_b)}
    emb_a_al = emb_a[[idx_a[c] for c in common]]
    emb_b_al = emb_b[[idx_b[c] for c in common]]

    k = 10
    seeds = list(range(11, 16))

    log.info("  K-means text-only (k=%d):", k)
    labels_a = _best_kmeans(emb_a_al, k, seeds)
    log.info("  K-means structlite (k=%d):", k)
    labels_b = _best_kmeans(emb_b_al, k, seeds)

    set_a8 = {common[i] for i, lbl in enumerate(labels_a) if lbl == 8}
    set_b7 = {common[i] for i, lbl in enumerate(labels_b) if lbl == 7}
    super_pag = sorted(set_a8 & set_b7)

    log.info(
        "  A8=%d  B7=%d  A8∩B7=%d  (%.1f%%)",
        len(set_a8), len(set_b7), len(super_pag),
        100 * len(super_pag) / max(len(set_a8), len(set_b7), 1),
    )

    SUPER_PAG_IDS_TXT.write_text("\n".join(super_pag) + "\n")
    log.info("  Saved: %s", SUPER_PAG_IDS_TXT)
    return super_pag


# ---------------------------------------------------------------------------
# Step 2 — Build node table
# ---------------------------------------------------------------------------

NODE_COLS = [
    "GS_ID", "NAME", "DESCRIPTION", "GS_SIZE", "ORGANISM",
    "PUBMED_ID", "gse_ids", "direction", "pmids", "pubmed_present",
    "geo_series_title", "geo_series_summary", "geo_overall_design",
    "mesh_descriptors", "pub_years", "pub_journals",
]


def build_node_table(super_pag_ids: list[str]) -> pd.DataFrame:
    """Join super-PAG IDs with enriched metadata. Returns DataFrame indexed by GS_ID."""
    log.info("Step 2: Building node table …")
    meta = pd.read_csv(METADATA_CSV, low_memory=False)
    # Normalise column name
    if "GS_ID" not in meta.columns:
        meta = meta.rename(columns={"gs_id": "GS_ID"})

    available = [c for c in NODE_COLS if c in meta.columns]
    node_df = meta[meta["GS_ID"].isin(set(super_pag_ids))][available].copy()
    node_df = node_df.set_index("GS_ID")
    missing = set(super_pag_ids) - set(node_df.index)
    log.info("  %d nodes (metadata missing for %d IDs)", len(node_df), len(missing))
    if missing:
        log.warning("  Missing metadata for: %s", sorted(missing)[:10])
    return node_df


# ---------------------------------------------------------------------------
# Step 3 — Gene-overlap edges (local PAGER file)
# ---------------------------------------------------------------------------

def build_gene_overlap_edges(super_pag_ids: list[str]) -> list[dict]:
    """Filter PAGER edge file for intra-cluster edges. Returns list of edge dicts."""
    log.info("Step 3: Building gene-overlap edges …")
    id_set = set(super_pag_ids)
    edges = []
    edges_df = pd.read_csv(EDGES_TXT, sep="\t", comment="#", low_memory=False)

    # Normalise column names
    col_map = {c.upper(): c for c in edges_df.columns}
    edges_df.columns = [c.upper() for c in edges_df.columns]

    for _, row in tqdm(edges_df.iterrows(), total=len(edges_df), desc="gene_overlap edges", leave=False):
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

    log.info("  %d gene_overlap edges", len(edges))
    return edges


# ---------------------------------------------------------------------------
# Step 4 — PubMed co-study edges
# ---------------------------------------------------------------------------

def _parse_pmids(pmid_str) -> set[str]:
    """Parse a PMID string (comma/semicolon separated) to a set of strings."""
    if pd.isna(pmid_str) or str(pmid_str).strip() in ("", "nan"):
        return set()
    return {p.strip() for p in str(pmid_str).replace(";", ",").split(",") if p.strip()}


def _ncbi_get(url: str, params: dict, retries: int = 3) -> dict | None:
    """GET from NCBI E-utilities with rate limiting and retries."""
    if NCBI_API_KEY:
        params = dict(params, api_key=NCBI_API_KEY)
    for attempt in range(retries):
        try:
            time.sleep(NCBI_DELAY)
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                log.debug("NCBI request failed: %s", e)
    return None


def build_pubmed_edges(node_table: pd.DataFrame) -> list[dict]:
    """
    Build pubmed_costudy edges:
      1. Direct shared PMIDs from metadata 'pmids' column.
      2. (Optional) NCBI elink to find publications citing both GSE datasets.
    """
    log.info("Step 4: Building PubMed co-study edges …")

    # Build PMID → gs_ids index from metadata
    pmid_to_gs: dict[str, list[str]] = defaultdict(list)
    for gs_id, row in node_table.iterrows():
        for pmid in _parse_pmids(row.get("pmids", "")):
            pmid_to_gs[pmid].append(gs_id)

    edges = []
    seen_pairs: set[frozenset] = set()

    # Direct PMID sharing
    for pmid, gs_list in pmid_to_gs.items():
        if len(gs_list) < 2:
            continue
        gs_list_sorted = sorted(gs_list)
        for i in range(len(gs_list_sorted)):
            for j in range(i + 1, len(gs_list_sorted)):
                a, b = gs_list_sorted[i], gs_list_sorted[j]
                pair = frozenset((a, b))
                if pair in seen_pairs:
                    # Accumulate shared PMIDs
                    for e in edges:
                        if e["source"] == a and e["target"] == b:
                            e["shared_pmids"].append(pmid)
                            e["shared_pmid_count"] += 1
                            break
                else:
                    seen_pairs.add(pair)
                    edges.append({
                        "source": a,
                        "target": b,
                        "edge_type": "pubmed_costudy",
                        "shared_pmid_count": 1,
                        "shared_pmids": [pmid],
                    })

    log.info("  %d pubmed_costudy edges (direct PMID sharing)", len(edges))

    # API enrichment: GSE-based PMID lookup via NCBI esearch
    nodes_without_pmids = [
        gs_id for gs_id, row in node_table.iterrows()
        if not _parse_pmids(row.get("pmids", ""))
        and not pd.isna(row.get("gse_ids", ""))
        and str(row.get("gse_ids", "")).strip()
    ]
    log.info("  %d nodes without direct PMIDs — querying NCBI esearch for GSE IDs …",
             len(nodes_without_pmids))

    base_esearch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    gse_pmids: dict[str, set[str]] = {}

    for gs_id in tqdm(nodes_without_pmids[:50], desc="NCBI esearch", leave=False):
        gse_id = str(node_table.loc[gs_id, "gse_ids"]).split(",")[0].strip()
        if not gse_id:
            continue
        data = _ncbi_get(base_esearch, {
            "db": "pubmed", "term": f"{gse_id}[All Fields]", "retmax": 10,
            "retmode": "json",
        })
        if data and "esearchresult" in data:
            pmids_found = data["esearchresult"].get("idlist", [])
            if pmids_found:
                gse_pmids[gs_id] = set(pmids_found)
                for pmid in pmids_found:
                    pmid_to_gs[pmid].append(gs_id)

    # Re-check for new shared PMIDs after API enrichment
    extra = 0
    for pmid, gs_list in pmid_to_gs.items():
        if len(gs_list) < 2:
            continue
        gs_list_sorted = sorted(gs_list)
        for i in range(len(gs_list_sorted)):
            for j in range(i + 1, len(gs_list_sorted)):
                a, b = gs_list_sorted[i], gs_list_sorted[j]
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
                    extra += 1

    if extra:
        log.info("  +%d pubmed_costudy edges from NCBI esearch enrichment", extra)
    log.info("  Total pubmed_costudy edges: %d", len(edges))
    return edges


# ---------------------------------------------------------------------------
# Step 5 — Fetch gene members
# ---------------------------------------------------------------------------

def _fetch_rummageo_members(gs_id: str) -> list[str] | None:
    """Query RummaGEO API for gene members of a single gene set."""
    url = f"https://rummageo.com/graphql"
    query = """
    query GeneSetGenes($term: String!) {
      geneSetByTerm(term: $term) {
        genes { nodes { symbol } }
      }
    }
    """
    try:
        r = requests.post(url, json={"query": query, "variables": {"term": gs_id}}, timeout=15)
        if r.status_code == 200:
            data = r.json()
            gs_data = data.get("data", {}).get("geneSetByTerm")
            if gs_data:
                return [n["symbol"] for n in gs_data["genes"]["nodes"]]
    except Exception:
        pass
    return None


def _fetch_rummageo_members_batch(gs_id_to_term: dict[str, str]) -> dict[str, list[str]]:
    """
    Query RummaGEO for multiple gene sets (GraphQL batch).
    gs_id_to_term maps GS_ID → RummaGEO term (the NAME field).
    """
    url = "https://rummageo.com/graphql"
    items = list(gs_id_to_term.items())
    aliases = []
    for i, (gs_id, term) in enumerate(items):
        escaped = term.replace('"', '\\"').replace("\\", "\\\\")
        aliases.append(
            f'gs{i}: geneSetByTerm(term: "{escaped}") {{ nGeneIds genes {{ nodes {{ symbol }} }} }}'
        )
    query = "query { " + " ".join(aliases) + " }"
    try:
        r = requests.post(url, json={"query": query}, timeout=60)
        if r.status_code == 200:
            data = r.json().get("data", {})
            result = {}
            for i, (gs_id, term) in enumerate(items):
                node = data.get(f"gs{i}")
                if node and node.get("genes"):
                    result[gs_id] = [n["symbol"] for n in node["genes"]["nodes"]]
            return result
    except Exception as e:
        log.debug("RummaGEO batch failed: %s", e)
    return {}


def fetch_gene_members(super_pag_ids: list[str], node_table: pd.DataFrame | None = None) -> dict[str, list[str]]:
    """
    Fetch gene member lists for each super-PAG gene set.
    1. Try RummaGEO GraphQL API (batch).
    2. Cache to gene_members_cache.json.
    """
    log.info("Step 5: Fetching gene members …")

    cache: dict[str, list[str]] = {}
    if GENE_MEMBERS_CACHE.exists():
        cache = json.loads(GENE_MEMBERS_CACHE.read_text())
        log.info("  Loaded %d cached gene sets", len(cache))

    missing = [gs_id for gs_id in super_pag_ids if gs_id not in cache]
    log.info("  %d gene sets need fetching", len(missing))

    if missing:
        # Build GS_ID → term mapping using NAME column from node_table
        id_to_term: dict[str, str] = {}
        if node_table is not None:
            for gs_id in missing:
                if gs_id in node_table.index and "NAME" in node_table.columns:
                    name = str(node_table.loc[gs_id, "NAME"]).strip()
                    if name and name != "nan":
                        id_to_term[gs_id] = name
        # Fallback: use GS_ID as term for any still missing
        for gs_id in missing:
            if gs_id not in id_to_term:
                id_to_term[gs_id] = gs_id

        batch_size = 20
        items = list(id_to_term.items())
        batches = [dict(items[i:i + batch_size]) for i in range(0, len(items), batch_size)]
        fetched = 0
        failed = 0
        for batch in tqdm(batches, desc="RummaGEO gene members", leave=False):
            result = _fetch_rummageo_members_batch(batch)
            for gs_id in batch:
                if gs_id in result and result[gs_id]:
                    cache[gs_id] = result[gs_id]
                    fetched += 1
                else:
                    cache[gs_id] = []  # mark as attempted
                    failed += 1
            time.sleep(0.3)

        log.info("  Fetched %d, empty/failed %d", fetched, failed)
        GENE_MEMBERS_CACHE.write_text(json.dumps(cache, indent=2))
        log.info("  Cache saved: %s", GENE_MEMBERS_CACHE)

    # Report coverage
    non_empty = sum(1 for gs_id in super_pag_ids if cache.get(gs_id))
    log.info(
        "  Gene member coverage: %d/%d (%.0f%%)",
        non_empty, len(super_pag_ids),
        100 * non_empty / max(len(super_pag_ids), 1),
    )
    return {gs_id: cache.get(gs_id, []) for gs_id in super_pag_ids}


# ---------------------------------------------------------------------------
# Step 6 — Shared pathway edges (KEGG + Reactome)
# ---------------------------------------------------------------------------

def _kegg_get(path: str, retries: int = 3) -> str | None:
    url = f"https://rest.kegg.jp/{path}"
    for attempt in range(retries):
        try:
            time.sleep(0.34)  # KEGG limit ~3/sec
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                return r.text
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                log.debug("KEGG request failed: %s", e)
    return None


def _genes_to_kegg_pathways(genes: list[str]) -> dict[str, str]:
    """Return {pathway_id: pathway_name} for a list of human gene symbols."""
    if not genes:
        return {}
    # Convert symbols to KEGG IDs via find
    pathways: dict[str, str] = {}
    for gene in genes[:30]:  # limit per gene set to avoid timeouts
        data = _kegg_get(f"find/hsa/{gene}")
        if not data:
            continue
        kegg_ids = [line.split("\t")[0] for line in data.strip().splitlines() if "\t" in line]
        for kegg_id in kegg_ids[:3]:
            pw_data = _kegg_get(f"link/pathway/{kegg_id}")
            if not pw_data:
                continue
            for line in pw_data.strip().splitlines():
                parts = line.split("\t")
                if len(parts) >= 2:
                    pw_id = parts[1]
                    if pw_id.startswith("path:hsa"):
                        pathways[pw_id] = pw_id  # name lookup below
    return pathways


def _reactome_get_pathways(gene_list: list[str]) -> dict[str, str]:
    """Return {pathway_id: pathway_name} from Reactome AnalysisService for a gene list."""
    if not gene_list:
        return {}
    url = "https://reactome.org/AnalysisService/identifiers/projection"
    try:
        r = requests.post(
            url,
            headers={"Content-Type": "text/plain", "Accept": "application/json"},
            data="\n".join(gene_list[:500]),
            params={"pageSize": 200, "page": 1},
            timeout=30,
        )
        if r.status_code != 200:
            return {}
        data = r.json()
        pathways = {}
        for pw in data.get("pathways", []):
            pw_id = pw.get("stId", "")
            pw_name = pw.get("name", pw_id)
            if pw_id:
                pathways[pw_id] = pw_name
        return pathways
    except Exception as e:
        log.debug("Reactome request failed: %s", e)
        return {}


def build_pathway_edges(
    super_pag_ids: list[str],
    gene_members: dict[str, list[str]],
) -> list[dict]:
    """
    Build shared_pathway edges via KEGG + Reactome.
    Uses Reactome AnalysisService (bulk) + KEGG link API (per-gene).
    """
    log.info("Step 6: Building shared pathway edges …")

    ids_with_genes = [gs_id for gs_id in super_pag_ids if gene_members.get(gs_id)]
    log.info("  %d gene sets have gene member data", len(ids_with_genes))

    # Pathway annotation cache
    pw_cache_path = KG_DIR / "pathway_cache.json"
    pw_cache: dict[str, dict[str, str]] = {}
    if pw_cache_path.exists():
        pw_cache = json.loads(pw_cache_path.read_text())

    # Reactome first (bulk, faster)
    reactome_updated = False
    for gs_id in tqdm(ids_with_genes, desc="Reactome pathways", leave=False):
        if gs_id in pw_cache:
            continue
        genes = gene_members[gs_id]
        pw = _reactome_get_pathways(genes)
        pw_cache[gs_id] = pw
        reactome_updated = True
        time.sleep(0.5)

    if reactome_updated:
        pw_cache_path.write_text(json.dumps(pw_cache, indent=2))

    # Build pathway → [gs_ids] index
    pathway_to_gs: dict[str, list[str]] = defaultdict(list)
    pathway_names: dict[str, str] = {}
    for gs_id, pws in pw_cache.items():
        if gs_id not in set(super_pag_ids):
            continue
        for pw_id, pw_name in pws.items():
            pathway_to_gs[pw_id].append(gs_id)
            pathway_names[pw_id] = pw_name

    edges = []
    seen_pairs: set[frozenset] = set()
    pair_pathways: dict[frozenset, dict] = {}

    for pw_id, gs_list in pathway_to_gs.items():
        if len(gs_list) < 2:
            continue
        gs_list_sorted = sorted(gs_list)
        for i in range(len(gs_list_sorted)):
            for j in range(i + 1, len(gs_list_sorted)):
                a, b = gs_list_sorted[i], gs_list_sorted[j]
                pair = frozenset((a, b))
                if pair not in pair_pathways:
                    pair_pathways[pair] = {
                        "source": a, "target": b,
                        "edge_type": "shared_pathway",
                        "pathway_ids": [], "pathway_names": [], "source_db": [],
                    }
                pair_pathways[pair]["pathway_ids"].append(pw_id)
                pair_pathways[pair]["pathway_names"].append(pathway_names.get(pw_id, pw_id))
                src = "KEGG" if pw_id.startswith("path:") else "Reactome"
                pair_pathways[pair]["source_db"].append(src)

    for pair_data in pair_pathways.values():
        # Deduplicate
        pair_data["pathway_ids"] = list(dict.fromkeys(pair_data["pathway_ids"]))
        pair_data["pathway_names"] = list(dict.fromkeys(pair_data["pathway_names"]))
        pair_data["source_db"] = list(dict.fromkeys(pair_data["source_db"]))
        edges.append(pair_data)

    log.info("  %d shared_pathway edges (Reactome)", len(edges))
    return edges


# ---------------------------------------------------------------------------
# Step 7 — Shared disease edges (DisGeNET or MESH fallback)
# ---------------------------------------------------------------------------

def _disgenet_get_disease_genes(gene_list: list[str], min_score: float = 0.3) -> dict[str, str]:
    """
    Query DisGeNET API for disease associations for a list of gene symbols.
    Returns {disease_id: disease_name} for diseases with score >= min_score.
    """
    if not DISGENET_API_KEY:
        return {}
    url = "https://api.disgenet.com/api/v1/gda/gene"
    headers = {"Authorization": f"Bearer {DISGENET_API_KEY}", "accept": "application/json"}
    result = {}
    for gene in gene_list[:20]:
        try:
            time.sleep(0.5)
            r = requests.get(url, headers=headers, params={"gene_symbol": gene, "min_score": min_score}, timeout=15)
            if r.status_code == 200:
                for item in r.json().get("payload", []):
                    d_id = item.get("diseaseid", "")
                    d_name = item.get("diseasename", d_id)
                    if d_id:
                        result[d_id] = d_name
        except Exception as e:
            log.debug("DisGeNET request failed for %s: %s", gene, e)
    return result


def _parse_mesh_terms(mesh_str) -> set[str]:
    """Parse semi-colon delimited MESH terms string."""
    if pd.isna(mesh_str) or str(mesh_str).strip() in ("", "nan"):
        return set()
    return {t.strip() for t in str(mesh_str).split(";") if t.strip()}


def build_disease_edges(
    super_pag_ids: list[str],
    node_table: pd.DataFrame,
    gene_members: dict[str, list[str]],
) -> list[dict]:
    """
    Build shared_disease edges via DisGeNET API, or MESH term fallback.
    """
    log.info("Step 7: Building shared disease edges …")

    if DISGENET_API_KEY:
        log.info("  DisGeNET API key found — querying disease associations …")
        disease_cache_path = KG_DIR / "disease_cache.json"
        disease_cache: dict[str, dict[str, str]] = {}
        if disease_cache_path.exists():
            disease_cache = json.loads(disease_cache_path.read_text())

        ids_with_genes = [gs_id for gs_id in super_pag_ids if gene_members.get(gs_id)]
        for gs_id in tqdm(ids_with_genes, desc="DisGeNET", leave=False):
            if gs_id not in disease_cache:
                disease_cache[gs_id] = _disgenet_get_disease_genes(gene_members[gs_id])
        disease_cache_path.write_text(json.dumps(disease_cache, indent=2))

        disease_to_gs: dict[str, list[str]] = defaultdict(list)
        disease_names: dict[str, str] = {}
        for gs_id, diseases in disease_cache.items():
            if gs_id not in set(super_pag_ids):
                continue
            for d_id, d_name in diseases.items():
                disease_to_gs[d_id].append(gs_id)
                disease_names[d_id] = d_name

        edges = _build_disease_edges_from_index(
            disease_to_gs, disease_names, "shared_disease"
        )
        log.info("  %d shared_disease edges (DisGeNET)", len(edges))
    else:
        log.warning("  DISGENET_API_KEY not set — using MESH term fallback")
        # MESH fallback
        mesh_to_gs: dict[str, list[str]] = defaultdict(list)
        for gs_id in super_pag_ids:
            if gs_id not in node_table.index:
                continue
            mesh_col = node_table.loc[gs_id, "mesh_descriptors"] if "mesh_descriptors" in node_table.columns else ""
            for term in _parse_mesh_terms(mesh_col):
                mesh_to_gs[term].append(gs_id)

        edges = _build_disease_edges_from_index(
            mesh_to_gs, {}, "shared_mesh_term"
        )
        log.info("  %d shared_mesh_term edges (MESH fallback)", len(edges))

    return edges


def _build_disease_edges_from_index(
    disease_to_gs: dict[str, list[str]],
    disease_names: dict[str, str],
    edge_type: str,
) -> list[dict]:
    """Build edge list from disease → gs_ids index."""
    pair_diseases: dict[frozenset, dict] = {}
    for d_id, gs_list in disease_to_gs.items():
        if len(gs_list) < 2:
            continue
        gs_list_sorted = sorted(gs_list)
        for i in range(len(gs_list_sorted)):
            for j in range(i + 1, len(gs_list_sorted)):
                a, b = gs_list_sorted[i], gs_list_sorted[j]
                pair = frozenset((a, b))
                if pair not in pair_diseases:
                    pair_diseases[pair] = {
                        "source": a, "target": b,
                        "edge_type": edge_type,
                        "disease_ids": [], "disease_names": [],
                    }
                pair_diseases[pair]["disease_ids"].append(d_id)
                pair_diseases[pair]["disease_names"].append(disease_names.get(d_id, d_id))

    edges = []
    for pair_data in pair_diseases.values():
        pair_data["disease_ids"] = list(dict.fromkeys(pair_data["disease_ids"]))
        pair_data["disease_names"] = list(dict.fromkeys(pair_data["disease_names"]))
        edges.append(pair_data)
    return edges


# ---------------------------------------------------------------------------
# Step 8 — Enrichr-based edges (TF co-regulation, disease, GO BP)
# ---------------------------------------------------------------------------

ENRICHR_BASE = "https://maayanlab.cloud/Enrichr"
ENRICHR_CACHE = KG_DIR / "enrichr_cache.json"


def _enrichr_add_list(genes: list[str], description: str = "") -> str | None:
    """POST gene list to Enrichr; return userListId string."""
    # Filter to proper gene symbols only (drop ENSG IDs, ncRNA names, etc.)
    symbols = [g for g in genes if g and not g.startswith("ENSG") and not g.startswith("ENST")]
    if not symbols:
        return None
    for attempt in range(3):
        try:
            time.sleep(0.5)
            # Enrichr requires multipart/form-data (files=), not application/x-www-form-urlencoded
            r = requests.post(
                f"{ENRICHR_BASE}/addList",
                files={"list": (None, "\n".join(symbols)),
                       "description": (None, description)},
                timeout=20,
            )
            r.raise_for_status()
            return str(r.json()["userListId"])
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                log.debug("Enrichr addList failed: %s", e)
    return None


def _enrichr_get_results(user_list_id: str, library: str) -> list[list]:
    """GET Enrichr enrichment results for a library. Returns raw result rows."""
    for attempt in range(3):
        try:
            time.sleep(0.3)
            r = requests.get(
                f"{ENRICHR_BASE}/enrich",
                params={"userListId": user_list_id, "backgroundType": library},
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
            return data.get(library, [])
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                log.debug("Enrichr enrich failed (%s): %s", library, e)
    return []


def _load_enrichr_cache() -> dict:
    if ENRICHR_CACHE.exists():
        return json.loads(ENRICHR_CACHE.read_text())
    return {}


def _save_enrichr_cache(cache: dict) -> None:
    ENRICHR_CACHE.write_text(json.dumps(cache, indent=2))


def _query_all_libraries(
    gs_id: str,
    genes: list[str],
    libraries: list[str],
    adj_pval_cutoff: float,
) -> dict[str, list]:
    """
    Upload one gene list to Enrichr, then query all requested libraries.
    Returns {library: [{"term": ..., "adj_pval": ...}, ...]} for significant hits.
    """
    uid = _enrichr_add_list(genes, description=gs_id)
    results: dict[str, list] = {lib: [] for lib in libraries}
    if uid is None:
        return results
    for lib in libraries:
        rows = _enrichr_get_results(uid, lib)
        results[lib] = [
            {"term": r[1], "adj_pval": r[6]}
            for r in rows
            if len(r) > 6 and isinstance(r[6], (int, float)) and r[6] <= adj_pval_cutoff
        ]
    return results


def fetch_all_enrichr(
    super_pag_ids: list[str],
    gene_members: dict[str, list[str]],
    libraries: list[str],
    adj_pval_cutoff: float = 0.05,
) -> None:
    """
    Query Enrichr for all gene sets + all libraries in one pass
    (one addList call per gene set, reused for all libraries).
    Results saved to enrichr_cache.json.
    """
    log.info("Step 8: Fetching Enrichr annotations for %d libraries …", len(libraries))
    cache = _load_enrichr_cache()
    ids_with_genes = [gs_id for gs_id in super_pag_ids if gene_members.get(gs_id)]
    log.info("  %d gene sets have gene members", len(ids_with_genes))

    updated = False
    needs_query = [
        gs_id for gs_id in ids_with_genes
        if any(lib not in cache.get(gs_id, {}) for lib in libraries)
    ]
    log.info("  %d gene sets need Enrichr queries", len(needs_query))

    for gs_id in tqdm(needs_query, desc="Enrichr (all libraries)", leave=False):
        if gs_id not in cache:
            cache[gs_id] = {}
        missing_libs = [lib for lib in libraries if lib not in cache[gs_id]]
        if not missing_libs:
            continue
        lib_results = _query_all_libraries(
            gs_id, gene_members[gs_id], missing_libs, adj_pval_cutoff
        )
        cache[gs_id].update(lib_results)
        updated = True

    if updated:
        _save_enrichr_cache(cache)
        log.info("  Enrichr cache saved.")


def build_enrichr_edges(
    super_pag_ids: list[str],
    gene_members: dict[str, list[str]],
    library: str,
    edge_type: str,
    term_field: str,
    max_term_prevalence: float = 0.80,
) -> list[dict]:
    """
    Build edges from pre-fetched Enrichr cache for one library.
    Terms appearing in > max_term_prevalence fraction of gene sets are excluded.
    Call fetch_all_enrichr() first to populate the cache.
    """
    log.info("  Building %s edges from '%s' …", edge_type, library)

    cache = _load_enrichr_cache()
    ids_with_genes = [gs_id for gs_id in super_pag_ids if gene_members.get(gs_id)]
    n_total = len(ids_with_genes)

    # Count term prevalence
    term_set_count: dict[str, int] = defaultdict(int)
    for gs_id in ids_with_genes:
        for entry in cache.get(gs_id, {}).get(library, []):
            term_set_count[entry["term"]] += 1

    max_count = int(max_term_prevalence * n_total)
    filtered_terms = {t for t, c in term_set_count.items() if c <= max_count}
    log.info(
        "    Terms passing prevalence filter (<%.0f%%): %d / %d",
        max_term_prevalence * 100,
        len(filtered_terms),
        len(term_set_count),
    )

    # Build term → [gs_ids] index
    term_to_gs: dict[str, list[str]] = defaultdict(list)
    term_pvals: dict[str, dict[str, float]] = defaultdict(dict)
    for gs_id in ids_with_genes:
        for entry in cache.get(gs_id, {}).get(library, []):
            term = entry["term"]
            if term in filtered_terms:
                term_to_gs[term].append(gs_id)
                term_pvals[term][gs_id] = float(entry["adj_pval"])

    # Build pair → shared terms
    pair_data: dict[frozenset, dict] = {}
    for term, gs_list in term_to_gs.items():
        if len(gs_list) < 2:
            continue
        gs_list_sorted = sorted(gs_list)
        for i in range(len(gs_list_sorted)):
            for j in range(i + 1, len(gs_list_sorted)):
                a, b = gs_list_sorted[i], gs_list_sorted[j]
                pair = frozenset((a, b))
                if pair not in pair_data:
                    pair_data[pair] = {
                        "source": a, "target": b,
                        "edge_type": edge_type,
                        term_field: [],
                        "min_pval": 1.0,
                    }
                pair_data[pair][term_field].append(term)
                best_pval = min(
                    term_pvals[term].get(a, 1.0),
                    term_pvals[term].get(b, 1.0),
                )
                pair_data[pair]["min_pval"] = min(pair_data[pair]["min_pval"], best_pval)

    edges = []
    for pd_entry in pair_data.values():
        terms = pd_entry[term_field]
        pd_entry[term_field] = list(dict.fromkeys(terms))
        count_key = term_field.replace("_names", "_count").replace("_terms", "_count")
        pd_entry[count_key] = len(pd_entry[term_field])
        edges.append(pd_entry)

    log.info("    %d %s edges", len(edges), edge_type)
    return edges


# ---------------------------------------------------------------------------
# Step 9 — Assemble and save KG
# ---------------------------------------------------------------------------

def _safe_str(val) -> str:
    """Convert a value to a GraphML-safe string."""
    if isinstance(val, list):
        return "|".join(str(v) for v in val)
    if pd.isna(val) if not isinstance(val, (list, dict)) else False:
        return ""
    return str(val)


def build_kg() -> nx.MultiGraph:
    """Main entrypoint: assemble multi-relational KG and save outputs."""
    log.info("=" * 60)
    log.info("Building Knowledge Graph for Cluster 8 Super-PAG (Subset 085)")
    log.info("=" * 60)

    KG_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1
    super_pag_ids = get_cluster8_genesets()
    log.info("Super-PAG: %d gene sets", len(super_pag_ids))

    # Step 2
    node_table = build_node_table(super_pag_ids)

    # Step 3
    gene_overlap_edges = build_gene_overlap_edges(super_pag_ids)

    # Step 4
    pubmed_edges = build_pubmed_edges(node_table)

    # Step 5
    gene_members = fetch_gene_members(super_pag_ids, node_table)

    # Step 6
    pathway_edges = build_pathway_edges(super_pag_ids, gene_members)

    # Step 7
    disease_edges = build_disease_edges(super_pag_ids, node_table, gene_members)

    # Step 8 — Enrichr edges (one addList per gene set, shared across all libraries)
    enrichr_libraries = ["ChEA_2022", "DisGeNET", "GO_Biological_Process_2023"]
    fetch_all_enrichr(super_pag_ids, gene_members, enrichr_libraries)
    tf_edges = build_enrichr_edges(
        super_pag_ids, gene_members,
        library="ChEA_2022",
        edge_type="shared_tf_regulator",
        term_field="tf_names",
    )
    disease_enr_edges = build_enrichr_edges(
        super_pag_ids, gene_members,
        library="DisGeNET",
        edge_type="shared_disease",
        term_field="disease_names",
    )
    go_bp_edges = build_enrichr_edges(
        super_pag_ids, gene_members,
        library="GO_Biological_Process_2023",
        edge_type="shared_go_bp",
        term_field="go_terms",
    )

    # Assemble
    log.info("Step 9: Assembling KG …")
    G = nx.MultiGraph()

    # Add nodes
    for gs_id in super_pag_ids:
        attrs = {}
        if gs_id in node_table.index:
            for col in node_table.columns:
                attrs[col.lower()] = _safe_str(node_table.loc[gs_id, col])
        attrs["gs_id"] = gs_id
        G.add_node(gs_id, **attrs)

    # Add edges
    all_edges = (gene_overlap_edges + pubmed_edges + pathway_edges + disease_edges
                 + tf_edges + disease_enr_edges + go_bp_edges)
    skipped = 0
    for e in all_edges:
        src, tgt = e["source"], e["target"]
        if src not in G or tgt not in G:
            skipped += 1
            continue
        edge_attrs = {k: _safe_str(v) if isinstance(v, list) else v
                      for k, v in e.items() if k not in ("source", "target")}
        G.add_edge(src, tgt, **edge_attrs)

    if skipped:
        log.warning("  Skipped %d edges with unknown nodes", skipped)

    # Summary
    edge_type_counts: dict[str, int] = defaultdict(int)
    for u, v, data in G.edges(data=True):
        edge_type_counts[data.get("edge_type", "unknown")] += 1

    log.info("")
    log.info("=== Knowledge Graph Summary ===")
    log.info("  Nodes: %d", G.number_of_nodes())
    log.info("  Edges: %d total", G.number_of_edges())
    for et, cnt in sorted(edge_type_counts.items()):
        log.info("    %-25s %d", et, cnt)

    # Save GraphML
    log.info("Saving outputs …")
    nx.write_graphml(G, str(KG_GRAPHML))
    log.info("  %s", KG_GRAPHML)

    # Save JSON (node-link format)
    kg_data = nx.node_link_data(G)
    KG_JSON.write_text(json.dumps(kg_data, indent=2, default=str))
    log.info("  %s", KG_JSON)

    # Save stats
    stats = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "edge_type_counts": dict(edge_type_counts),
        "gene_member_coverage": sum(1 for gs_id in super_pag_ids if gene_members.get(gs_id)),
        "gene_member_coverage_pct": round(
            100 * sum(1 for gs_id in super_pag_ids if gene_members.get(gs_id)) / max(len(super_pag_ids), 1), 1
        ),
    }
    KG_STATS_JSON.write_text(json.dumps(stats, indent=2))
    log.info("  %s", KG_STATS_JSON)

    return G


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_kg() -> None:
    """Load saved GraphML and print verification summary."""
    log.info("")
    log.info("=== Verification ===")
    G = nx.read_graphml(str(KG_GRAPHML))
    log.info("  G.number_of_nodes() = %d", G.number_of_nodes())
    log.info("  G.number_of_edges() = %d", G.number_of_edges())
    etype_counts: dict[str, int] = defaultdict(int)
    for u, v, data in G.edges(data=True):
        etype_counts[data.get("edge_type", "unknown")] += 1
    for et, cnt in sorted(etype_counts.items()):
        log.info("    %-25s %d", et, cnt)

    assert G.number_of_nodes() >= 150, f"Expected ~181 nodes, got {G.number_of_nodes()}"
    assert G.number_of_edges() > 0, "KG has no edges"
    log.info("  All verification checks passed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    G = build_kg()
    verify_kg()
    log.info("Done.")
