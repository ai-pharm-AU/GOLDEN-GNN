#!/usr/bin/env python3
"""Build node feature recipes for GNN ablation study.

Outputs NPZ files with keys:
- ID: object array of GS IDs
- embeddings: float32 matrix
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer


def load_npz_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "ID" in data:
        ids = data["ID"]
    elif "GOID" in data:
        ids = data["GOID"]
    else:
        raise KeyError(f"Missing ID/GOID key in {path}")

    if "embeddings" in data:
        emb = data["embeddings"]
    elif "emb" in data:
        emb = data["emb"]
    else:
        raise KeyError(f"Missing embeddings/emb key in {path}")

    ids_out = [str(x) for x in ids]
    emb_out = np.asarray(emb, dtype=np.float32)
    if emb_out.ndim != 2:
        raise ValueError(f"Expected 2D embeddings in {path}, got shape={emb_out.shape}")
    if len(ids_out) != emb_out.shape[0]:
        raise ValueError(
            f"ID count mismatch in {path}: ids={len(ids_out)} rows={emb_out.shape[0]}"
        )
    return ids_out, emb_out


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms <= 0.0, 1.0, norms)
    return x / norms


def zscore_cols(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    return (x - mu) / sd


def robust_scale_cols(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    med = np.median(x, axis=0, keepdims=True)
    q1 = np.percentile(x, 25.0, axis=0, keepdims=True)
    q3 = np.percentile(x, 75.0, axis=0, keepdims=True)
    iqr = q3 - q1
    iqr = np.where(iqr <= 1e-12, 1.0, iqr)
    return (x - med) / iqr


def rankgauss_cols(x: np.ndarray, seed: int) -> np.ndarray:
    if x.size == 0:
        return x
    n = x.shape[0]
    if n < 8:
        return zscore_cols(x)
    n_quantiles = min(1000, n)
    model = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution="normal",
        random_state=seed,
        copy=True,
    )
    out = model.fit_transform(x)
    return np.asarray(out, dtype=np.float32)


def normalize_cols(x: np.ndarray, mode: str, seed: int) -> np.ndarray:
    m = mode.strip().lower()
    if m == "zscore":
        return zscore_cols(x)
    if m == "robust":
        return robust_scale_cols(x)
    if m == "rankgauss":
        return rankgauss_cols(x, seed=seed)
    raise ValueError(f"Unknown normalization mode: {mode}")


def align_block(base_ids: list[str], src_ids: list[str], src_emb: np.ndarray) -> tuple[np.ndarray, int]:
    src = {gid: i for i, gid in enumerate(src_ids)}
    out = np.zeros((len(base_ids), src_emb.shape[1]), dtype=np.float32)
    miss = 0
    for i, gid in enumerate(base_ids):
        j = src.get(gid)
        if j is None:
            miss += 1
            continue
        out[i] = src_emb[j]
    return out, miss


def safe_float(text: object, default: float = math.nan) -> float:
    if text is None:
        return default
    s = str(text).strip()
    if not s or s.lower() == "nan":
        return default
    try:
        return float(s)
    except ValueError:
        return default


def parse_year(value: object) -> float:
    s = str(value).strip() if value is not None else ""
    if not s or s.lower() == "nan":
        return math.nan
    parts = [p.strip() for p in s.replace("|", ";").split(";") if p.strip()]
    years: list[int] = []
    for p in parts:
        if p.isdigit() and len(p) == 4:
            years.append(int(p))
    if years:
        return float(np.mean(years))
    if s.isdigit() and len(s) == 4:
        return float(int(s))
    return math.nan


def build_meta_features(base_ids: list[str], meta_df: pd.DataFrame) -> tuple[np.ndarray, list[str], dict[str, int]]:
    cols_needed = [
        "GS_ID",
        "GS_SIZE",
        "direction",
        "ORGANISM",
        "pubmed_present",
        "pub_years",
    ]
    missing_cols = [c for c in cols_needed if c not in meta_df.columns]
    if missing_cols:
        raise KeyError(f"Missing metadata columns: {missing_cols}")

    meta = meta_df.copy()
    meta["GS_ID"] = meta["GS_ID"].astype(str)
    meta = meta.set_index("GS_ID", drop=False)

    organism_vals = (
        meta["ORGANISM"].astype(str).replace("nan", "").replace("None", "").fillna("")
    )
    top_organisms = [k for k, _ in organism_vals.value_counts().head(8).items() if k]
    org_to_idx = {org: i for i, org in enumerate(top_organisms)}

    feat_names = [
        "log1p_gs_size",
        "direction_up",
        "direction_dn",
        "pubmed_present",
        "pub_year_norm",
        "pub_year_missing",
    ] + [f"organism_{o}" for o in top_organisms]

    x = np.zeros((len(base_ids), len(feat_names)), dtype=np.float32)
    stats = {"meta_missing_rows": 0, "meta_missing_year": 0}

    for i, gid in enumerate(base_ids):
        if gid not in meta.index:
            stats["meta_missing_rows"] += 1
            continue
        row = meta.loc[gid]

        gs_size = safe_float(row.get("GS_SIZE"), default=0.0)
        x[i, 0] = np.float32(np.log1p(max(gs_size, 0.0)))

        direction = str(row.get("direction", "")).strip().lower()
        if direction == "up":
            x[i, 1] = 1.0
        elif direction == "dn":
            x[i, 2] = 1.0

        pub_present = safe_float(row.get("pubmed_present"), default=0.0)
        x[i, 3] = np.float32(1.0 if pub_present > 0 else 0.0)

        year = parse_year(row.get("pub_years"))
        if math.isnan(year):
            x[i, 5] = 1.0
            stats["meta_missing_year"] += 1
        else:
            # Stable scaling to roughly [0,1] for modern biomedical years.
            x[i, 4] = np.float32((year - 1990.0) / 40.0)

        org = str(row.get("ORGANISM", "")).strip()
        j = org_to_idx.get(org)
        if j is not None:
            x[i, 6 + j] = 1.0

    x = zscore_cols(x)
    return x, feat_names, stats


def build_struct_features(
    base_ids: list[str],
    edge_file: Path,
    weight_column: str,
    struct_features: str = "basic",
    struct_norm: str = "zscore",
    seed: int = 42,
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    edge_df = pd.read_csv(edge_file, sep="\t")
    required = {"GS_A_ID", "GS_B_ID", weight_column}
    miss = [c for c in required if c not in edge_df.columns]
    if miss:
        raise KeyError(f"Missing edge columns: {miss} in {edge_file}")

    id_set = set(base_ids)
    G = nx.Graph()
    for gid in base_ids:
        G.add_node(gid)

    aligned_edges = 0
    for _, row in edge_df.iterrows():
        a = str(row["GS_A_ID"])
        b = str(row["GS_B_ID"])
        if a not in id_set or b not in id_set or a == b:
            continue
        w = safe_float(row[weight_column], default=0.0)
        if math.isnan(w):
            w = 0.0
        if G.has_edge(a, b):
            # Keep max for duplicate pairs.
            if w > G[a][b].get("weight", 0.0):
                G[a][b]["weight"] = w
        else:
            G.add_edge(a, b, weight=float(w))
            aligned_edges += 1

    degree = dict(G.degree())
    strength = dict(G.degree(weight="weight"))
    clustering = nx.clustering(G, weight="weight")
    pagerank = nx.pagerank(G, alpha=0.85, weight="weight", max_iter=100)

    mode = struct_features.strip().lower()
    if mode not in {"basic", "basic_plus", "centrality_plus"}:
        raise ValueError(f"Unsupported struct_features: {struct_features}")

    feat_names = [
        "log1p_degree",
        "log1p_strength",
        "clustering_weighted",
        "pagerank",
    ]

    core_number: dict[str, float] = {}
    triangles: dict[str, float] = {}
    closeness: dict[str, float] = {}
    betweenness: dict[str, float] = {}

    if mode in {"basic_plus", "centrality_plus"}:
        core_number = nx.core_number(G) if G.number_of_edges() > 0 else {gid: 0.0 for gid in base_ids}
        triangles = nx.triangles(G)
        feat_names.extend(["core_number", "log1p_triangles"])

    if mode == "centrality_plus":
        closeness = nx.closeness_centrality(G)
        if G.number_of_edges() > 0 and G.number_of_nodes() > 2:
            k_bt = min(256, max(16, int(math.sqrt(G.number_of_nodes()))))
            betweenness = nx.betweenness_centrality(
                G,
                k=k_bt,
                normalized=True,
                weight=None,
                seed=seed,
            )
        else:
            betweenness = {gid: 0.0 for gid in base_ids}
        feat_names.extend(["closeness", "betweenness_approx"])

    x = np.zeros((len(base_ids), len(feat_names)), dtype=np.float32)
    for i, gid in enumerate(base_ids):
        d = float(degree.get(gid, 0.0))
        s = float(strength.get(gid, 0.0))
        c = float(clustering.get(gid, 0.0))
        p = float(pagerank.get(gid, 0.0))
        row_vals: list[float] = [
            float(np.log1p(max(d, 0.0))),
            float(np.log1p(max(s, 0.0))),
            c,
            p,
        ]
        if mode in {"basic_plus", "centrality_plus"}:
            row_vals.extend(
                [
                    float(core_number.get(gid, 0.0)),
                    float(np.log1p(max(float(triangles.get(gid, 0.0)), 0.0))),
                ]
            )
        if mode == "centrality_plus":
            row_vals.extend(
                [
                    float(closeness.get(gid, 0.0)),
                    float(betweenness.get(gid, 0.0)),
                ]
            )
        x[i] = np.asarray(row_vals, dtype=np.float32)

    x = normalize_cols(x, mode=struct_norm, seed=seed)
    stats = {
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "graph_aligned_edges": aligned_edges,
        "struct_features_mode": mode,
        "struct_norm_mode": struct_norm,
    }
    return x, feat_names, stats


def reduce_block(x: np.ndarray, dim: int, seed: int) -> np.ndarray:
    if dim <= 0:
        return x
    n, d = x.shape
    if d <= dim:
        return x
    n_comp = min(dim, n, d)
    if n_comp <= 1:
        return x
    model = PCA(n_components=n_comp, random_state=seed)
    out = model.fit_transform(x)
    return np.asarray(out, dtype=np.float32)


def parse_recipes(raw: str) -> list[str]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    if not items:
        raise ValueError("No recipes provided")
    return items


def resolve_default_paths(base_dir: Path, subset: str) -> dict[str, Path]:
    if subset == "090":
        return {
            "go_npz": base_dir / "subset_85_90_20260213/embeddings/go_embeddings_subset_090_l2.npz",
            "pager_npz": base_dir / "subset_85_90_20260213/embeddings/pager_embeddings_subset_090_l2.npz",
            "meta_csv": base_dir / "reports/bio_meta_cache_090/enriched_gs_metadata.csv",
            "edge_file": base_dir / "subset_85_90_20260213/csv/edges_with_mtype_threshold_090.txt",
        }
    if subset == "085":
        return {
            "go_npz": base_dir / "subset_85_90_20260213/embeddings/go_embeddings_subset_085_l2.npz",
            "pager_npz": base_dir / "subset_85_90_20260213/embeddings/pager_embeddings_subset_085_l2.npz",
            "meta_csv": base_dir
            / "reports/bio_meta_cache_085/bio_meta_cache_085/enriched_gs_metadata.csv",
            "edge_file": base_dir / "subset_85_90_20260213/csv/edges_with_mtype_threshold_085.txt",
        }
    raise ValueError(f"Unsupported subset: {subset}")


def build_recipes(
    base_ids: list[str],
    go_emb: np.ndarray,
    pager_emb: np.ndarray,
    meta_feat: np.ndarray,
    struct_feat: np.ndarray,
    recipes: Iterable[str],
    pager_dim: int,
    seed: int,
    go_scale: float,
    meta_scale: float,
    struct_scale: float,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    pager_small = reduce_block(pager_emb, pager_dim, seed=seed)
    go_block = np.asarray(go_emb, dtype=np.float32) * np.float32(go_scale)
    meta_block = np.asarray(meta_feat, dtype=np.float32) * np.float32(meta_scale)
    struct_block = np.asarray(struct_feat, dtype=np.float32) * np.float32(struct_scale)

    out: dict[str, np.ndarray] = {}
    for recipe in recipes:
        if recipe == "text_only":
            mat = go_block
        elif recipe == "text_meta":
            mat = np.concatenate([go_block, meta_block], axis=1)
        elif recipe == "text_structlite":
            mat = np.concatenate([go_block, struct_block], axis=1)
        elif recipe == "text_meta_structlite":
            mat = np.concatenate([go_block, meta_block, struct_block], axis=1)
        elif recipe == "node2vec_only":
            mat = pager_emb
        elif recipe == "text_node2vec":
            mat = np.concatenate([go_block, pager_small], axis=1)
        elif recipe == "shuffle_control":
            perm = rng.permutation(go_block.shape[0])
            mat = go_block[perm]
        elif recipe == "random_control":
            mat = rng.normal(size=go_block.shape).astype(np.float32)
        else:
            raise ValueError(f"Unknown recipe: {recipe}")

        mat = np.asarray(mat, dtype=np.float32)
        mat = l2_normalize_rows(mat)
        out[recipe] = mat
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build node feature recipes for ablation.")
    parser.add_argument("--subset", required=True, choices=["085", "090"])
    parser.add_argument(
        "--base_dir",
        default="/home/zzz0054/GoldenF/work_alpha_gnn_20260212",
        help="Base directory for work_alpha_gnn_20260212",
    )
    parser.add_argument("--go_npz", default="")
    parser.add_argument("--pager_npz", default="")
    parser.add_argument("--meta_csv", default="")
    parser.add_argument("--edge_file", default="")
    parser.add_argument("--weight_column", default="NLOGPMF")
    parser.add_argument(
        "--recipes",
        default=(
            "text_only,text_meta,text_structlite,text_meta_structlite,"
            "node2vec_only,text_node2vec,shuffle_control,random_control"
        ),
    )
    parser.add_argument("--pager_dim", type=int, default=64)
    parser.add_argument("--struct_features", default="basic", choices=["basic", "basic_plus", "centrality_plus"])
    parser.add_argument("--struct_norm", default="zscore", choices=["zscore", "robust", "rankgauss"])
    parser.add_argument("--go_scale", type=float, default=1.0)
    parser.add_argument("--meta_scale", type=float, default=1.0)
    parser.add_argument("--struct_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    defaults = resolve_default_paths(base_dir, args.subset)

    go_npz = Path(args.go_npz).resolve() if args.go_npz else defaults["go_npz"]
    pager_npz = Path(args.pager_npz).resolve() if args.pager_npz else defaults["pager_npz"]
    meta_csv = Path(args.meta_csv).resolve() if args.meta_csv else defaults["meta_csv"]
    edge_file = Path(args.edge_file).resolve() if args.edge_file else defaults["edge_file"]
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    recipes = parse_recipes(args.recipes)

    go_ids, go_emb_raw = load_npz_embeddings(go_npz)
    pager_ids, pager_emb_raw = load_npz_embeddings(pager_npz)

    base_ids = list(go_ids)
    go_emb = l2_normalize_rows(np.asarray(go_emb_raw, dtype=np.float32))
    pager_aligned, pager_missing = align_block(base_ids, pager_ids, pager_emb_raw)
    pager_emb = l2_normalize_rows(pager_aligned)

    meta_df = pd.read_csv(meta_csv)
    meta_feat, meta_names, meta_stats = build_meta_features(base_ids, meta_df)
    struct_feat, struct_names, struct_stats = build_struct_features(
        base_ids=base_ids,
        edge_file=edge_file,
        weight_column=args.weight_column,
        struct_features=args.struct_features,
        struct_norm=args.struct_norm,
        seed=args.seed,
    )

    recipe_mats = build_recipes(
        base_ids=base_ids,
        go_emb=go_emb,
        pager_emb=pager_emb,
        meta_feat=meta_feat,
        struct_feat=struct_feat,
        recipes=recipes,
        pager_dim=args.pager_dim,
        seed=args.seed,
        go_scale=args.go_scale,
        meta_scale=args.meta_scale,
        struct_scale=args.struct_scale,
    )

    manifest: dict[str, object] = {
        "subset": args.subset,
        "inputs": {
            "go_npz": str(go_npz),
            "pager_npz": str(pager_npz),
            "meta_csv": str(meta_csv),
            "edge_file": str(edge_file),
            "weight_column": args.weight_column,
            "struct_features": args.struct_features,
            "struct_norm": args.struct_norm,
            "go_scale": args.go_scale,
            "meta_scale": args.meta_scale,
            "struct_scale": args.struct_scale,
        },
        "stats": {
            "num_nodes": len(base_ids),
            "go_dim": int(go_emb.shape[1]),
            "pager_dim": int(pager_emb.shape[1]),
            "pager_missing_for_base_ids": int(pager_missing),
            **meta_stats,
            **struct_stats,
            "meta_feature_names": meta_names,
            "struct_feature_names": struct_names,
        },
        "recipes": {},
    }

    for recipe, mat in recipe_mats.items():
        out_npz = out_dir / f"{recipe}_subset{args.subset}.npz"
        np.savez(out_npz, ID=np.asarray(base_ids, dtype=object), embeddings=mat.astype(np.float32))
        manifest["recipes"][recipe] = {
            "output_npz": str(out_npz),
            "shape": [int(mat.shape[0]), int(mat.shape[1])],
        }
        print(f"wrote {out_npz} shape={mat.shape}")

    manifest_path = out_dir / f"feature_manifest_subset{args.subset}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
