import argparse
import math
import os
import pathlib
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import hypergeom
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import re

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.data_loader import DataLoader

_SPLIT_RE = re.compile(r"[;|,\s]+")


def to_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def parse_int_values(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def split_values(raw: str) -> set[str]:
    s = to_text(raw)
    if not s:
        return set()
    return {t for t in _SPLIT_RE.split(s) if t}


def primary_value(raw: str) -> str:
    vals = [t for t in _SPLIT_RE.split(to_text(raw)) if t]
    return vals[0] if vals else ""


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    n = p.size
    if n == 0:
        return np.array([], dtype=np.float64)
    order = np.argsort(p)
    ranked_p = p[order]
    ranks = np.arange(1, n + 1, dtype=np.float64)
    q_ranked = ranked_p * n / ranks
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    q = np.empty_like(q_ranked)
    q[order] = q_ranked
    return q


def build_attr_matrix(values_per_item: list[set[str]], min_support: int) -> tuple[list[str], csr_matrix, np.ndarray]:
    counter: Counter[str] = Counter()
    for s in values_per_item:
        for v in s:
            counter[v] += 1
    kept = [v for v, c in counter.items() if c >= min_support]
    kept.sort()
    v_to_j = {v: j for j, v in enumerate(kept)}

    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []
    for i, s in enumerate(values_per_item):
        for v in s:
            j = v_to_j.get(v)
            if j is None:
                continue
            rows.append(i)
            cols.append(j)
            data.append(1)
    n = len(values_per_item)
    V = len(kept)
    A = csr_matrix((data, (rows, cols)), shape=(n, V), dtype=np.int8)
    global_counts = np.asarray(A.sum(axis=0)).ravel().astype(np.int64)
    return kept, A, global_counts


def enrichment_cluster_stats(
    labels: np.ndarray,
    k: int,
    values: list[str],
    A: csr_matrix,
    global_counts: np.ndarray,
    fdr_q: float,
) -> dict[int, dict[str, object]]:
    # Returns per-cluster: sig_count, best_value, best_q, best_er, best_overlap.
    N = int(labels.shape[0])
    cluster_sizes = np.bincount(labels, minlength=k).astype(np.int64)

    # counts[c, j] = overlap
    C = csr_matrix(
        (np.ones(N, dtype=np.int8), (labels.astype(np.int64), np.arange(N, dtype=np.int64))),
        shape=(k, N),
        dtype=np.int8,
    )
    counts = (C @ A).toarray().astype(np.int64)  # K x V
    V = int(A.shape[1])

    p_all = np.ones((k, V), dtype=np.float64)
    er_all = np.zeros((k, V), dtype=np.float64)
    for c in range(k):
        n_c = int(cluster_sizes[c])
        if n_c <= 0:
            continue
        a = counts[c]
        g = global_counts
        mask = a > 0
        if np.any(mask):
            p_all[c, mask] = hypergeom.sf(a[mask] - 1, N, g[mask], n_c)
        denom = g.astype(np.float64) / float(N)
        with np.errstate(divide="ignore", invalid="ignore"):
            er = (a.astype(np.float64) / float(n_c)) / denom
        er_all[c] = np.where(np.isfinite(er), er, 0.0)

    q_flat = benjamini_hochberg(p_all.ravel())
    q_all = q_flat.reshape((k, V))

    out: dict[int, dict[str, object]] = {}
    for c in range(k):
        q_c = q_all[c]
        a_c = counts[c]
        sig_idx = np.where((a_c > 0) & (q_c <= fdr_q))[0]
        if sig_idx.size == 0:
            out[c] = {
                "sig_count": 0,
                "best_value": "",
                "best_q": 1.0,
                "best_er": float("nan"),
                "best_overlap": 0,
            }
            continue
        # best by q then -ER.
        order = sig_idx[np.lexsort((-er_all[c, sig_idx], q_c[sig_idx]))]
        best = int(order[0])
        out[c] = {
            "sig_count": int(sig_idx.size),
            "best_value": values[best],
            "best_q": float(q_c[best]),
            "best_er": float(er_all[c, best]),
            "best_overlap": int(a_c[best]),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze confounding: compare MeSH enrichment strength vs study (GSE) enrichment/purity per cluster."
    )
    parser.add_argument("--embedding_npz", required=True)
    parser.add_argument("--enriched_metadata_csv", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--min_support", type=int, default=5)
    parser.add_argument("--fdr_q", type=float, default=0.05)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_md), exist_ok=True)

    ids, emb = DataLoader.load_embeddings(args.embedding_npz)
    ids = [str(x) for x in ids]
    x = np.asarray(emb)

    meta = pd.read_csv(args.enriched_metadata_csv)
    meta["GS_ID"] = meta["GS_ID"].astype(str)
    meta = meta.set_index("GS_ID", drop=False)
    keep = [i for i, gid in enumerate(ids) if gid in meta.index]
    ids_keep = [ids[i] for i in keep]
    x_keep = x[keep]
    df = meta.loc[ids_keep].reset_index(drop=True)
    N = len(df)

    labels = KMeans(n_clusters=args.k, random_state=args.seed, n_init="auto").fit_predict(x_keep)
    df["cluster"] = labels

    # Primary confound labels.
    primary_gse = [primary_value(v) for v in df.get("gse_ids", pd.Series([""] * N)).tolist()]
    primary_platform = [primary_value(v) for v in df.get("geo_platform_ids", pd.Series([""] * N)).tolist()]

    # NMI/ARI against primary confounds (items with empty label are dropped for this calc).
    def score_against_primary(primary: list[str]) -> tuple[float, float, int, int]:
        idx = [i for i, v in enumerate(primary) if v]
        if len(idx) < 2:
            return float("nan"), float("nan"), len(idx), len(set(primary)) - (1 if "" in primary else 0)
        y = [primary[i] for i in idx]
        lab = labels[idx]
        nmi = float(normalized_mutual_info_score(y, lab))
        # ARI expects integer labels; map strings.
        mapping = {v: j for j, v in enumerate(sorted(set(y)))}
        y_int = [mapping[v] for v in y]
        ari = float(adjusted_rand_score(y_int, lab))
        return nmi, ari, len(idx), len(mapping)

    nmi_gse, ari_gse, n_gse_items, n_gse_unique = score_against_primary(primary_gse)
    nmi_plat, ari_plat, n_plat_items, n_plat_unique = score_against_primary(primary_platform)

    # Enrichment stats for MeSH and GSE and platform.
    mesh_sets = [split_values(v) for v in df.get("mesh_descriptors", pd.Series([""] * N)).tolist()]
    gse_sets = [split_values(v) for v in df.get("gse_ids", pd.Series([""] * N)).tolist()]
    plat_sets = [split_values(v) for v in df.get("geo_platform_ids", pd.Series([""] * N)).tolist()]

    mesh_vals, mesh_A, mesh_g = build_attr_matrix(mesh_sets, args.min_support)
    gse_vals, gse_A, gse_g = build_attr_matrix(gse_sets, args.min_support)
    plat_vals, plat_A, plat_g = build_attr_matrix(plat_sets, args.min_support)

    mesh_stats = enrichment_cluster_stats(labels, args.k, mesh_vals, mesh_A, mesh_g, args.fdr_q)
    gse_stats = enrichment_cluster_stats(labels, args.k, gse_vals, gse_A, gse_g, args.fdr_q)
    plat_stats = enrichment_cluster_stats(labels, args.k, plat_vals, plat_A, plat_g, args.fdr_q)

    # Cluster-level purity by primary GSE.
    cluster_sizes = np.bincount(labels, minlength=args.k)
    purity: dict[int, tuple[str, float, int]] = {}
    for c in range(args.k):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            purity[c] = ("", float("nan"), 0)
            continue
        cnt = Counter(primary_gse[i] for i in idx if primary_gse[i])
        if not cnt:
            purity[c] = ("", 0.0, int(idx.size))
            continue
        top, top_n = cnt.most_common(1)[0]
        purity[c] = (top, float(top_n / idx.size), int(idx.size))

    with open(args.output_md, "w", encoding="utf-8") as handle:
        handle.write("# Mesh vs GSE Confound Check\n\n")
        handle.write(f"- embedding: `{args.embedding_npz}`\n")
        handle.write(f"- enriched metadata: `{args.enriched_metadata_csv}`\n")
        handle.write(f"- seed: {args.seed} | k: {args.k} | N: {N}\n")
        handle.write(f"- min_support: {args.min_support} | fdr_q: {args.fdr_q}\n\n")

        handle.write("## Confound Association (Primary Labels)\n\n")
        handle.write(
            f"- primary GSE: items={n_gse_items} | unique={n_gse_unique} | NMI={nmi_gse:.3f} | ARI={ari_gse:.3f}\n"
        )
        handle.write(
            f"- primary platform: items={n_plat_items} | unique={n_plat_unique} | NMI={nmi_plat:.3f} | ARI={ari_plat:.3f}\n\n"
        )

        handle.write("## Cluster Summary\n\n")
        handle.write(
            "| cluster | size | top_gse | gse_purity | mesh_sig | best_mesh (q,ER) | gse_sig | best_gse (q,ER) | plat_sig | best_plat (q,ER) |\n"
        )
        handle.write("|---:|---:|---|---:|---:|---|---:|---|---:|---|\n")
        for c in range(args.k):
            top_gse, pur, size = purity[c]
            m = mesh_stats[c]
            g = gse_stats[c]
            p = plat_stats[c]
            mesh_desc = (
                f"{m['best_value']} ({m['best_q']:.1e},{m['best_er']:.2f})" if m["best_value"] else ""
            )
            gse_desc = (
                f"{g['best_value']} ({g['best_q']:.1e},{g['best_er']:.2f})" if g["best_value"] else ""
            )
            plat_desc = (
                f"{p['best_value']} ({p['best_q']:.1e},{p['best_er']:.2f})" if p["best_value"] else ""
            )
            handle.write(
                f"| {c} | {size} | {top_gse} | {pur:.3f} | {int(m['sig_count'])} | {mesh_desc} | "
                f"{int(g['sig_count'])} | {gse_desc} | {int(p['sig_count'])} | {plat_desc} |\n"
            )

        handle.write("\n## Interpretation Guide\n\n")
        handle.write(
            "- High `mesh_sig` with moderate/low `gse_purity` suggests biology-like themes not tied to one study.\n"
            "- Very high `gse_purity` and strong `best_gse` indicates a study/batch-driven cluster (still real structure, but a confound for cross-study biology).\n"
            "- Platform enrichment is a common technical confound; treat strong `best_plat` similarly.\n"
        )

    print(f"Wrote: {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
