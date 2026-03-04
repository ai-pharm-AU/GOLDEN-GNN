import argparse
import csv
import math
import os
import pathlib
import re
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix, hstack
from scipy.stats import hypergeom
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.data_loader import DataLoader


GSE_RE = re.compile(r"GSE\d+")


def to_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


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


def clean_text(text: str) -> str:
    s = text or ""
    s = GSE_RE.sub(" ", s)
    s = re.sub(r"\d+", " ", s)
    s = s.replace("_", " ").replace("-", " ")
    return s


def parse_int_values(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]

def parse_int_pair(raw: str) -> tuple[int, int]:
    parts = parse_int_values(raw)
    if len(parts) != 2:
        raise ValueError(f"Expected two comma-separated ints, got: {raw!r}")
    return int(parts[0]), int(parts[1])


def split_values(raw: str) -> set[str]:
    s = to_text(raw)
    if not s:
        return set()
    return {t.strip() for t in s.split(";") if t.strip()}


def build_docs(df: pd.DataFrame, cols: list[str], clean: bool) -> list[str]:
    docs: list[str] = []
    for _, row in df.iterrows():
        parts: list[str] = []
        for c in cols:
            parts.append(to_text(row.get(c, "")))
        doc = " ".join([p for p in parts if p]).strip()
        docs.append(clean_text(doc) if clean else doc)
    return docs


def build_text_matrix(
    docs: list[str],
    min_df: int,
    max_features: int,
    word_ngrams: tuple[int, int],
    char_ngrams: tuple[int, int],
) -> csr_matrix:
    vec_word = TfidfVectorizer(
        stop_words="english",
        min_df=min_df,
        max_features=max_features if max_features > 0 else None,
        ngram_range=word_ngrams,
        norm="l2",
    )
    vec_char = TfidfVectorizer(
        analyzer="char_wb",
        min_df=min_df,
        max_features=max_features if max_features > 0 else None,
        ngram_range=char_ngrams,
        norm="l2",
    )
    x_word = vec_word.fit_transform(docs)
    x_char = vec_char.fit_transform(docs)
    return hstack([x_word, x_char], format="csr")


def coherence_by_centroid(x: csr_matrix, labels: np.ndarray, k: int) -> float:
    weighted_sum = 0.0
    total = 0
    for c in range(k):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        x_c = x[idx]
        centroid = np.asarray(x_c.mean(axis=0))
        sims = cosine_similarity(x_c, centroid).ravel()
        weighted_sum += float(sims.mean()) * int(idx.size)
        total += int(idx.size)
    return weighted_sum / total if total > 0 else float("nan")


def empirical_p_greater(null: np.ndarray, obs: float) -> float:
    if null.size == 0 or math.isnan(obs):
        return float("nan")
    return float((np.sum(null >= obs) + 1) / (null.size + 1))


@dataclass(frozen=True)
class AttrMatrix:
    attr_type: str
    values: list[str]
    A: csr_matrix  # N x V binary
    global_counts: np.ndarray  # (V,)


def build_attr_matrix(values_per_item: list[set[str]], min_support: int, attr_type: str) -> AttrMatrix:
    # Global counts (items containing v).
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
    return AttrMatrix(attr_type=attr_type, values=kept, A=A, global_counts=global_counts)


def cluster_indicator(labels: np.ndarray, k: int) -> csr_matrix:
    n = labels.shape[0]
    rows = labels.astype(np.int64)
    cols = np.arange(n, dtype=np.int64)
    data = np.ones(n, dtype=np.int8)
    return csr_matrix((data, (rows, cols)), shape=(k, n), dtype=np.int8)


def meta_enrichment_for_seed(
    labels: np.ndarray,
    k: int,
    attr_mats: list[AttrMatrix],
    fdr_q: float,
    top_n: int,
) -> tuple[list[dict], dict[str, dict[str, float]]]:
    N = int(labels.shape[0])
    cluster_sizes = np.bincount(labels, minlength=k).astype(np.int64)

    rows: list[dict] = []
    summary: dict[str, dict[str, float]] = {}

    for attr in attr_mats:
        V = int(attr.A.shape[1])
        if V == 0:
            summary[attr.attr_type] = {
                "tested": 0,
                "significant": 0,
                "max_er": float("nan"),
                "min_q": 1.0,
            }
            continue

        C = cluster_indicator(labels, k)
        counts = (C @ attr.A).toarray().astype(np.int64)  # K x V

        # Compute p-values for all cluster/value tests.
        p_all = np.ones((k, V), dtype=np.float64)
        er_all = np.zeros((k, V), dtype=np.float64)
        for c in range(k):
            n_c = int(cluster_sizes[c])
            if n_c <= 0:
                continue
            a = counts[c]  # overlaps for all values
            g = attr.global_counts
            mask = a > 0
            if np.any(mask):
                p_all[c, mask] = hypergeom.sf(a[mask] - 1, N, g[mask], n_c)
            # Enrichment ratio: (a/n) / (g/N)
            denom = (g.astype(np.float64) / float(N))
            with np.errstate(divide="ignore", invalid="ignore"):
                er = (a.astype(np.float64) / float(n_c)) / denom
            er_all[c] = np.where(np.isfinite(er), er, 0.0)

        q_flat = benjamini_hochberg(p_all.ravel())
        q_all = q_flat.reshape((k, V))

        sig_mask = q_all <= fdr_q
        n_sig = int(np.sum(sig_mask))
        min_q = float(np.min(q_all)) if q_all.size else 1.0
        max_er = float(np.max(er_all[sig_mask])) if n_sig > 0 else float("nan")

        # Emit rows (optional): top-N per cluster (top_n>0), none (top_n==0), or all (top_n<0).
        if top_n != 0:
            for c in range(k):
                q_c = q_all[c]
                er_c = er_all[c]
                a_c = counts[c]
                sig_idx = np.where(q_c <= fdr_q)[0]
                if sig_idx.size == 0:
                    continue
                order = sig_idx[np.lexsort((-er_c[sig_idx], q_c[sig_idx]))]
                chosen = order[:top_n] if top_n > 0 else order
                for j in chosen:
                    v = attr.values[int(j)]
                    g = int(attr.global_counts[int(j)])
                    n_c = int(cluster_sizes[c])
                    a = int(a_c[int(j)])
                    rows.append(
                        {
                            "attr_type": attr.attr_type,
                            "cluster": int(c),
                            "value": v,
                            "N_total": N,
                            "n_cluster": n_c,
                            "g_global": g,
                            "a_overlap": a,
                            "p_value": float(p_all[c, int(j)]),
                            "q_value": float(q_c[int(j)]),
                            "enrichment_ratio": float(er_c[int(j)]),
                        }
                    )

        summary[attr.attr_type] = {
            "tested": float(k * V),
            "significant": float(n_sig),
            "max_er": max_er,
            "min_q": min_q,
        }

    return rows, summary


def meta_null_summary(
    labels: np.ndarray,
    k: int,
    attr_mats: list[AttrMatrix],
    fdr_q: float,
    n_perm: int,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    # Returns per-attr null distribution stats for "significant count" and "max -log10(q)".
    if n_perm <= 0:
        return {}

    null_sig: dict[str, list[float]] = {a.attr_type: [] for a in attr_mats}
    null_maxlogq: dict[str, list[float]] = {a.attr_type: [] for a in attr_mats}

    for _ in range(n_perm):
        perm_labels = rng.permutation(labels)
        _, summary = meta_enrichment_for_seed(
            labels=perm_labels,
            k=k,
            attr_mats=attr_mats,
            fdr_q=fdr_q,
            top_n=0,  # no row emission, summary only
        )
        for attr_type, s in summary.items():
            sig = float(s.get("significant", 0.0))
            min_q = float(s.get("min_q", 1.0))
            null_sig[attr_type].append(sig)
            null_maxlogq[attr_type].append(-math.log10(max(min_q, 1e-300)))

    out: dict[str, dict[str, float]] = {}
    for attr_type in null_sig:
        sig_arr = np.asarray(null_sig[attr_type], dtype=np.float64)
        maxlogq_arr = np.asarray(null_maxlogq[attr_type], dtype=np.float64)
        out[attr_type] = {
            "null_sig_mean": float(sig_arr.mean()) if sig_arr.size else float("nan"),
            "null_sig_std": float(sig_arr.std()) if sig_arr.size else float("nan"),
            "null_maxlogq_mean": float(maxlogq_arr.mean()) if maxlogq_arr.size else float("nan"),
            "null_maxlogq_std": float(maxlogq_arr.std()) if maxlogq_arr.size else float("nan"),
            "null_sig_values": sig_arr,
            "null_maxlogq_values": maxlogq_arr,
        }
    return out


def best_cluster_mapping(a_sets: dict[int, set[int]], b_sets: dict[int, set[int]], k: int) -> dict[int, int]:
    counts = np.zeros((k, k), dtype=np.int64)
    for i in range(k):
        a_set = a_sets.get(i, set())
        if not a_set:
            continue
        for j in range(k):
            b_set = b_sets.get(j, set())
            if not b_set:
                continue
            counts[i, j] = len(a_set.intersection(b_set))
    row_ind, col_ind = linear_sum_assignment(-counts)
    return {int(r): int(c) for r, c in zip(row_ind, col_ind)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate clustering via text coherence + metadata enrichment (with permutation null and seed stability)."
    )
    parser.add_argument("--embedding_npz", required=True)
    parser.add_argument("--enriched_metadata_csv", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seeds", default="11,12,13,14,15")
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--text_cols", default="NAME,DESCRIPTION,geo_series_title,geo_series_summary,geo_overall_design")
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--max_features", type=int, default=200000)
    parser.add_argument("--word_ngrams", default="1,2")
    parser.add_argument("--char_ngrams", default="3,5")

    parser.add_argument("--min_support", type=int, default=5)
    parser.add_argument("--fdr_q", type=float, default=0.05)
    parser.add_argument("--top_n_per_cluster", type=int, default=10)

    parser.add_argument("--n_perm", type=int, default=200)
    parser.add_argument("--random_seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    seeds = parse_int_values(args.seeds)
    word_lo, word_hi = parse_int_pair(args.word_ngrams)
    char_lo, char_hi = parse_int_pair(args.char_ngrams)
    text_cols = [c.strip() for c in args.text_cols.split(",") if c.strip()]

    ids, emb = DataLoader.load_embeddings(args.embedding_npz)
    ids = [str(x) for x in ids]
    x = np.asarray(emb)
    print(f"[load] embedding: n={x.shape[0]:,} dim={x.shape[1]}", flush=True)

    meta = pd.read_csv(args.enriched_metadata_csv)
    meta["GS_ID"] = meta["GS_ID"].astype(str)
    id_to_row = meta.set_index("GS_ID", drop=False)
    keep = [i for i, gid in enumerate(ids) if gid in id_to_row.index]
    ids_keep = [ids[i] for i in keep]
    x_keep = x[keep]
    df = id_to_row.loc[ids_keep].reset_index(drop=True)
    N = len(ids_keep)
    print(f"[align] enriched metadata aligned: n={N:,}", flush=True)

    # Build text matrices once (raw and cleaned), independent of clustering seed.
    docs_raw = build_docs(df, cols=text_cols, clean=False)
    docs_clean = build_docs(df, cols=text_cols, clean=True)
    x_text_raw = build_text_matrix(
        docs_raw,
        min_df=args.min_df,
        max_features=args.max_features,
        word_ngrams=(word_lo, word_hi),
        char_ngrams=(char_lo, char_hi),
    )
    x_text_clean = build_text_matrix(
        docs_clean,
        min_df=args.min_df,
        max_features=args.max_features,
        word_ngrams=(word_lo, word_hi),
        char_ngrams=(char_lo, char_hi),
    )
    print(
        f"[text] raw_dim={x_text_raw.shape[1]:,} clean_dim={x_text_clean.shape[1]:,}",
        flush=True,
    )

    # Build attribute matrices.
    def values_for(attr: str) -> list[set[str]]:
        if attr == "gse_id":
            return [split_values(v) for v in df.get("gse_ids", pd.Series([""] * N)).tolist()]
        if attr == "direction":
            return [{to_text(v).lower()} if to_text(v) else set() for v in df.get("direction", pd.Series([""] * N)).tolist()]
        if attr == "geo_platform":
            return [split_values(v) for v in df.get("geo_platform_ids", pd.Series([""] * N)).tolist()]
        if attr == "geo_organism":
            return [split_values(v) for v in df.get("geo_sample_organisms", pd.Series([""] * N)).tolist()]
        if attr == "mesh":
            return [split_values(v) for v in df.get("mesh_descriptors", pd.Series([""] * N)).tolist()]
        if attr == "pubmed_present":
            return [{"present"} if int(v) == 1 else {"missing"} for v in df.get("pubmed_present", pd.Series([0] * N)).tolist()]
        if attr == "organism":
            return [{to_text(v)} if to_text(v) else set() for v in df.get("ORGANISM", pd.Series([""] * N)).tolist()]
        raise ValueError(f"Unknown attr: {attr}")

    attr_types = ["mesh", "geo_platform", "gse_id", "direction", "pubmed_present", "organism"]
    attr_mats: list[AttrMatrix] = []
    for t in attr_types:
        mat = build_attr_matrix(values_for(t), min_support=args.min_support, attr_type=t)
        attr_mats.append(mat)
        print(f"[attr] {t}: values_kept={len(mat.values):,} nnz={mat.A.nnz:,}", flush=True)

    # Main loop: per seed compute clustering, coherence, enrichment, nulls.
    rng_master = np.random.default_rng(args.random_seed)
    seed_to_labels: dict[int, np.ndarray] = {}
    text_rows: list[dict] = []
    enrich_rows: list[dict] = []
    enrich_seed_summary_rows: list[dict] = []
    null_rows: list[dict] = []

    for seed in seeds:
        model = KMeans(n_clusters=args.k, random_state=seed, n_init="auto")
        labels = model.fit_predict(x_keep)
        seed_to_labels[seed] = labels

        coh_raw = coherence_by_centroid(x_text_raw, labels, args.k)
        coh_clean = coherence_by_centroid(x_text_clean, labels, args.k)

        rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        null_raw = np.zeros(args.n_perm, dtype=np.float64) if args.n_perm > 0 else np.array([], dtype=np.float64)
        null_clean = np.zeros(args.n_perm, dtype=np.float64) if args.n_perm > 0 else np.array([], dtype=np.float64)
        for i in range(null_raw.size):
            perm = rng.permutation(labels)
            null_raw[i] = coherence_by_centroid(x_text_raw, perm, args.k)
            null_clean[i] = coherence_by_centroid(x_text_clean, perm, args.k)

        text_rows.append(
            {
                "seed": seed,
                "n": N,
                "k": args.k,
                "text_coherence_raw": coh_raw,
                "text_null_raw_mean": float(null_raw.mean()) if null_raw.size else float("nan"),
                "text_null_raw_std": float(null_raw.std()) if null_raw.size else float("nan"),
                "text_null_raw_p": empirical_p_greater(null_raw, coh_raw),
                "text_coherence_clean": coh_clean,
                "text_null_clean_mean": float(null_clean.mean()) if null_clean.size else float("nan"),
                "text_null_clean_std": float(null_clean.std()) if null_clean.size else float("nan"),
                "text_null_clean_p": empirical_p_greater(null_clean, coh_clean),
            }
        )

        rows, summary = meta_enrichment_for_seed(
            labels=labels,
            k=args.k,
            attr_mats=attr_mats,
            fdr_q=args.fdr_q,
            top_n=args.top_n_per_cluster,
        )
        for r in rows:
            r["seed"] = seed
        enrich_rows.extend(rows)

        # Null summaries for meta enrichment (per attr type).
        meta_null = meta_null_summary(
            labels=labels,
            k=args.k,
            attr_mats=attr_mats,
            fdr_q=args.fdr_q,
            n_perm=args.n_perm,
            rng=rng,
        )

        for attr_type, s in summary.items():
            observed_sig = float(s.get("significant", 0.0))
            observed_min_q = float(s.get("min_q", 1.0))
            observed_maxlogq = -math.log10(max(observed_min_q, 1e-300))

            null_sig_values = np.asarray(meta_null.get(attr_type, {}).get("null_sig_values", np.array([])), dtype=np.float64)
            null_maxlogq_values = np.asarray(
                meta_null.get(attr_type, {}).get("null_maxlogq_values", np.array([])), dtype=np.float64
            )

            enrich_seed_summary_rows.append(
                {
                    "seed": seed,
                    "attr_type": attr_type,
                    "tested": int(s.get("tested", 0.0)),
                    "significant": int(observed_sig),
                    "max_er": float(s.get("max_er", float("nan"))),
                    "min_q": observed_min_q,
                    "max_log10q": observed_maxlogq,
                    "null_sig_mean": float(meta_null.get(attr_type, {}).get("null_sig_mean", float("nan"))),
                    "null_sig_std": float(meta_null.get(attr_type, {}).get("null_sig_std", float("nan"))),
                    "null_sig_p": empirical_p_greater(null_sig_values, observed_sig),
                    "null_maxlog10q_mean": float(meta_null.get(attr_type, {}).get("null_maxlogq_mean", float("nan"))),
                    "null_maxlog10q_std": float(meta_null.get(attr_type, {}).get("null_maxlogq_std", float("nan"))),
                    "null_maxlog10q_p": empirical_p_greater(null_maxlogq_values, observed_maxlogq),
                }
            )

        print(
            f"[seed={seed}] coh_raw={coh_raw:.4f} coh_clean={coh_clean:.4f} "
            f"mesh_sig={int(summary.get('mesh', {}).get('significant', 0))}",
            flush=True,
        )

    # Stability: pairwise ARI across seeds.
    ari_rows: list[dict] = []
    for a, b in combinations(seeds, 2):
        ari = float(adjusted_rand_score(seed_to_labels[a], seed_to_labels[b]))
        ari_rows.append({"seed_a": a, "seed_b": b, "ari": ari})

    # Optional: mapping-based replication summary can be layered later; for now, include ARI only.
    out_text_csv = os.path.join(args.output_dir, "text_coherence_by_seed.csv")
    out_enrich_csv = os.path.join(args.output_dir, "meta_enrichment_results.csv")
    out_enrich_summary_csv = os.path.join(args.output_dir, "meta_enrichment_seed_summary.csv")
    out_ari_csv = os.path.join(args.output_dir, "pairwise_ari.csv")
    out_report_md = os.path.join(args.output_dir, "validation_report.md")

    def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(
        out_text_csv,
        text_rows,
        [
            "seed",
            "n",
            "k",
            "text_coherence_raw",
            "text_null_raw_mean",
            "text_null_raw_std",
            "text_null_raw_p",
            "text_coherence_clean",
            "text_null_clean_mean",
            "text_null_clean_std",
            "text_null_clean_p",
        ],
    )
    if enrich_rows:
        write_csv(
            out_enrich_csv,
            enrich_rows,
            [
                "seed",
                "attr_type",
                "cluster",
                "value",
                "N_total",
                "n_cluster",
                "g_global",
                "a_overlap",
                "p_value",
                "q_value",
                "enrichment_ratio",
            ],
        )
    else:
        write_csv(out_enrich_csv, [], ["seed", "attr_type", "cluster", "value"])

    write_csv(
        out_enrich_summary_csv,
        enrich_seed_summary_rows,
        [
            "seed",
            "attr_type",
            "tested",
            "significant",
            "max_er",
            "min_q",
            "max_log10q",
            "null_sig_mean",
            "null_sig_std",
            "null_sig_p",
            "null_maxlog10q_mean",
            "null_maxlog10q_std",
            "null_maxlog10q_p",
        ],
    )
    write_csv(out_ari_csv, ari_rows, ["seed_a", "seed_b", "ari"])

    # Markdown report (compact).
    with open(out_report_md, "w", encoding="utf-8") as handle:
        handle.write("# Cluster Validation Report\n\n")
        handle.write(f"- embedding: `{args.embedding_npz}`\n")
        handle.write(f"- enriched metadata: `{args.enriched_metadata_csv}`\n")
        handle.write(f"- k: {args.k}\n")
        handle.write(f"- seeds: {', '.join(str(s) for s in seeds)}\n")
        handle.write(f"- n_perm: {args.n_perm}\n")
        handle.write(f"- min_support: {args.min_support}\n")
        handle.write(f"- fdr_q: {args.fdr_q}\n\n")

        handle.write("## Text Coherence\n\n")
        handle.write("| seed | coh_raw | p_raw | coh_clean | p_clean |\n")
        handle.write("|---:|---:|---:|---:|---:|\n")
        for r in text_rows:
            handle.write(
                f"| {int(r['seed'])} | {float(r['text_coherence_raw']):.4f} | "
                f"{float(r['text_null_raw_p']):.4f} | {float(r['text_coherence_clean']):.4f} | "
                f"{float(r['text_null_clean_p']):.4f} |\n"
            )
        handle.write("\n## Seed Stability (ARI)\n\n")
        if not ari_rows:
            handle.write("Only one seed provided.\n\n")
        else:
            ari_vals = [float(r["ari"]) for r in ari_rows]
            handle.write(
                f"- pairs: {len(ari_rows)} | ARI mean={float(np.mean(ari_vals)):.3f} std={float(np.std(ari_vals)):.3f}\n\n"
            )

        handle.write("## Metadata Enrichment Summary\n\n")
        handle.write("| seed | attr_type | significant | null_sig_mean | p_sig | max_log10q | p_maxlog10q |\n")
        handle.write("|---:|---|---:|---:|---:|---:|---:|\n")
        for r in enrich_seed_summary_rows:
            handle.write(
                f"| {int(r['seed'])} | {r['attr_type']} | {int(r['significant'])} | "
                f"{float(r['null_sig_mean']):.2f} | {float(r['null_sig_p']):.4f} | "
                f"{float(r['max_log10q']):.2f} | {float(r['null_maxlog10q_p']):.4f} |\n"
            )

    print(f"Wrote: {out_text_csv}", flush=True)
    print(f"Wrote: {out_enrich_csv}", flush=True)
    print(f"Wrote: {out_enrich_summary_csv}", flush=True)
    print(f"Wrote: {out_ari_csv}", flush=True)
    print(f"Wrote: {out_report_md}", flush=True)


if __name__ == "__main__":
    main()
