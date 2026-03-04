import argparse
import csv
import os
import pathlib
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import hypergeom
from sklearn.cluster import KMeans

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.data_loader import DataLoader


def parse_embedding_specs(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid embedding spec '{item}'. Expected format: threshold=path"
            )
        threshold, path = item.split("=", 1)
        threshold = threshold.strip()
        path = path.strip()
        if not threshold or not path:
            raise ValueError(
                f"Invalid embedding spec '{item}'. Expected format: threshold=path"
            )
        out[threshold] = path
    if not out:
        raise ValueError("No embedding specs were provided")
    return out


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


def load_term_metadata(meta_csv: str) -> dict[str, tuple[str, str]]:
    meta: dict[str, tuple[str, str]] = {}
    with open(meta_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"GS_ID", "NAME", "DESCRIPTION"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Metadata file missing required columns: {sorted(missing)}"
            )
        for row in reader:
            gs_id = str(row.get("GS_ID", "")).strip()
            if not gs_id:
                continue
            name = str(row.get("NAME", "")).strip()
            desc = str(row.get("DESCRIPTION", "")).strip()
            meta[gs_id] = (name, desc)
    return meta


def cluster_embedding(
    embedding_file: str, k: int, seed: int
) -> tuple[list[str], np.ndarray]:
    ids, emb = DataLoader.load_embeddings(embedding_file)
    ids = [str(x) for x in ids]
    emb = np.asarray(emb)
    labels = KMeans(n_clusters=k, random_state=seed, n_init="auto").fit_predict(emb)
    return ids, labels


def write_cluster_assignments(
    output_csv: str,
    threshold_to_ids: dict[str, list[str]],
    threshold_to_labels: dict[str, np.ndarray],
    threshold_to_embedding: dict[str, str],
) -> None:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["threshold", "embedding_file", "gs_id", "cluster"])
        for threshold in sorted(threshold_to_ids.keys()):
            ids = threshold_to_ids[threshold]
            labels = threshold_to_labels[threshold]
            emb_file = threshold_to_embedding[threshold]
            for gs_id, cluster in zip(ids, labels):
                writer.writerow([threshold, emb_file, gs_id, int(cluster)])


def build_gs_memberships(
    threshold_to_ids: dict[str, list[str]], threshold_to_labels: dict[str, np.ndarray]
) -> dict[str, list[tuple[str, int]]]:
    out: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for threshold in threshold_to_ids:
        ids = threshold_to_ids[threshold]
        labels = threshold_to_labels[threshold]
        for gs_id, cluster in zip(ids, labels):
            out[gs_id].append((threshold, int(cluster)))
    return out


def first_pass_cluster_genes(
    membership_csv: str,
    gs_memberships: dict[str, list[tuple[str, int]]],
    thresholds: list[str],
    log_every: int,
) -> tuple[dict[str, set[str]], dict[str, dict[int, set[str]]]]:
    universe_genes: dict[str, set[str]] = {t: set() for t in thresholds}
    cluster_genes: dict[str, dict[int, set[str]]] = {t: defaultdict(set) for t in thresholds}
    total_rows = 0
    matched_rows = 0

    with open(membership_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"GS_ID", "GENE_SYMBOL"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Membership file missing required columns: {sorted(missing)}"
            )
        for row in reader:
            total_rows += 1
            gs_id = str(row.get("GS_ID", "")).strip()
            gene = str(row.get("GENE_SYMBOL", "")).strip()
            if not gs_id or not gene:
                continue
            contexts = gs_memberships.get(gs_id)
            if not contexts:
                if log_every > 0 and total_rows % log_every == 0:
                    print(
                        f"[pass1] processed {total_rows:,} rows; matched {matched_rows:,}",
                        flush=True,
                    )
                continue
            for threshold, cluster in contexts:
                universe_genes[threshold].add(gene)
                cluster_genes[threshold][cluster].add(gene)
            matched_rows += 1
            if log_every > 0 and total_rows % log_every == 0:
                print(
                    f"[pass1] processed {total_rows:,} rows; matched {matched_rows:,}",
                    flush=True,
                )

    print(
        f"[pass1] done: processed {total_rows:,} rows; matched {matched_rows:,}",
        flush=True,
    )
    return universe_genes, cluster_genes


def build_gene_to_clusters(
    cluster_genes: dict[str, dict[int, set[str]]]
) -> dict[str, dict[str, list[int]]]:
    out: dict[str, dict[str, list[int]]] = {}
    for threshold, cluster_map in cluster_genes.items():
        gene_map: dict[str, list[int]] = defaultdict(list)
        for cluster, genes in cluster_map.items():
            for gene in genes:
                gene_map[gene].append(cluster)
        for gene in gene_map:
            gene_map[gene].sort()
        out[threshold] = dict(gene_map)
    return out


def second_pass_term_counts(
    membership_csv: str,
    universe_genes: dict[str, set[str]],
    gene_to_clusters: dict[str, dict[str, list[int]]],
    thresholds: list[str],
    log_every: int,
) -> tuple[dict[str, Counter], dict[str, dict[int, Counter]]]:
    term_counts: dict[str, Counter] = {t: Counter() for t in thresholds}
    overlap_counts: dict[str, dict[int, Counter]] = {
        t: defaultdict(Counter) for t in thresholds
    }

    total_rows = 0
    with open(membership_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"GS_ID", "GENE_SYMBOL"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Membership file missing required columns: {sorted(missing)}"
            )
        for row in reader:
            total_rows += 1
            term_id = str(row.get("GS_ID", "")).strip()
            gene = str(row.get("GENE_SYMBOL", "")).strip()
            if not term_id or not gene:
                continue

            for threshold in thresholds:
                if gene in universe_genes[threshold]:
                    term_counts[threshold][term_id] += 1
                cluster_list = gene_to_clusters[threshold].get(gene)
                if cluster_list:
                    c_map = overlap_counts[threshold]
                    for cluster in cluster_list:
                        c_map[cluster][term_id] += 1

            if log_every > 0 and total_rows % log_every == 0:
                print(f"[pass2] processed {total_rows:,} rows", flush=True)

    print(f"[pass2] done: processed {total_rows:,} rows", flush=True)
    return term_counts, overlap_counts


def run_enrichment(
    threshold: str,
    term_counts: Counter,
    overlap_counts: dict[int, Counter],
    cluster_genes: dict[int, set[str]],
    cluster_node_counts: Counter,
    term_meta: dict[str, tuple[str, str]],
    fdr_q: float,
    min_term_genes: int,
    max_term_genes: int,
    top_n: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    M = len({g for genes in cluster_genes.values() for g in genes})
    if M == 0:
        return [], [], []

    terms = []
    m_values = []
    for term_id, m in term_counts.items():
        if m < min_term_genes:
            continue
        if max_term_genes > 0 and m > max_term_genes:
            continue
        terms.append(term_id)
        m_values.append(m)

    if not terms:
        return [], [], []

    terms_arr = np.array(terms, dtype=object)
    m_arr = np.array(m_values, dtype=np.int64)
    n_tested = int(len(terms_arr))

    results_rows: list[dict] = []
    summary_rows: list[dict] = []
    top_rows: list[dict] = []

    for cluster in sorted(cluster_genes.keys()):
        n = int(len(cluster_genes[cluster]))
        if n == 0:
            summary_rows.append(
                {
                    "threshold": threshold,
                    "cluster": cluster,
                    "cluster_nodes": int(cluster_node_counts.get(cluster, 0)),
                    "cluster_genes": 0,
                    "universe_genes": M,
                    "tested_terms": n_tested,
                    "significant_terms": 0,
                    "best_term_id": "",
                    "best_term_name": "",
                    "best_q_value": 1.0,
                    "best_p_value": 1.0,
                    "best_overlap": 0,
                    "best_enrichment_ratio": 0.0,
                }
            )
            continue

        overlap_map = overlap_counts.get(cluster, Counter())
        k_arr = np.array([overlap_map.get(t, 0) for t in terms_arr], dtype=np.int64)

        p_arr = np.ones(n_tested, dtype=np.float64)
        mask = k_arr > 0
        if np.any(mask):
            p_arr[mask] = hypergeom.sf(k_arr[mask] - 1, M, m_arr[mask], n)

        q_arr = benjamini_hochberg(p_arr)
        enrich_arr = (k_arr / n) / (m_arr / M)

        sig_mask = (k_arr > 0) & (q_arr <= fdr_q)
        sig_idx = np.where(sig_mask)[0]
        if sig_idx.size > 0:
            order = sig_idx[np.lexsort((-enrich_arr[sig_idx], p_arr[sig_idx], q_arr[sig_idx]))]
            chosen = order[:top_n] if top_n > 0 else order
        else:
            chosen = np.array([], dtype=np.int64)

        for idx in chosen:
            term_id = str(terms_arr[idx])
            name, desc = term_meta.get(term_id, ("", ""))
            row = {
                "threshold": threshold,
                "cluster": cluster,
                "cluster_nodes": int(cluster_node_counts.get(cluster, 0)),
                "cluster_genes": n,
                "universe_genes": M,
                "tested_terms": n_tested,
                "term_id": term_id,
                "term_name": name,
                "term_description": desc,
                "k_overlap": int(k_arr[idx]),
                "n_cluster_genes": n,
                "m_term_genes": int(m_arr[idx]),
                "M_universe_genes": M,
                "p_value": float(p_arr[idx]),
                "q_value": float(q_arr[idx]),
                "enrichment_ratio": float(enrich_arr[idx]),
            }
            results_rows.append(row)
            top_rows.append(row)

        if sig_idx.size > 0:
            best_idx = int(sig_idx[np.argmin(q_arr[sig_idx])])
            best_term = str(terms_arr[best_idx])
            best_name, _ = term_meta.get(best_term, ("", ""))
            summary_rows.append(
                {
                    "threshold": threshold,
                    "cluster": cluster,
                    "cluster_nodes": int(cluster_node_counts.get(cluster, 0)),
                    "cluster_genes": n,
                    "universe_genes": M,
                    "tested_terms": n_tested,
                    "significant_terms": int(sig_idx.size),
                    "best_term_id": best_term,
                    "best_term_name": best_name,
                    "best_q_value": float(q_arr[best_idx]),
                    "best_p_value": float(p_arr[best_idx]),
                    "best_overlap": int(k_arr[best_idx]),
                    "best_enrichment_ratio": float(enrich_arr[best_idx]),
                }
            )
        else:
            summary_rows.append(
                {
                    "threshold": threshold,
                    "cluster": cluster,
                    "cluster_nodes": int(cluster_node_counts.get(cluster, 0)),
                    "cluster_genes": n,
                    "universe_genes": M,
                    "tested_terms": n_tested,
                    "significant_terms": 0,
                    "best_term_id": "",
                    "best_term_name": "",
                    "best_q_value": 1.0,
                    "best_p_value": 1.0,
                    "best_overlap": 0,
                    "best_enrichment_ratio": 0.0,
                }
            )

    return results_rows, summary_rows, top_rows


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(path: str, top_rows: list[dict], summary_rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    by_group: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in top_rows:
        by_group[(row["threshold"], int(row["cluster"]))].append(row)

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# GNN Enrichment Report\n\n")
        handle.write("Cluster-level ORA with hypergeometric test and BH-FDR.\n\n")
        for row in sorted(summary_rows, key=lambda x: (x["threshold"], x["cluster"])):
            threshold = row["threshold"]
            cluster = int(row["cluster"])
            handle.write(f"## Threshold {threshold} / Cluster {cluster}\n\n")
            handle.write(
                f"- cluster nodes: {row['cluster_nodes']}\n"
                f"- cluster genes: {row['cluster_genes']}\n"
                f"- universe genes: {row['universe_genes']}\n"
                f"- tested terms: {row['tested_terms']}\n"
                f"- significant terms (FDR): {row['significant_terms']}\n\n"
            )
            hits = by_group.get((threshold, cluster), [])
            if not hits:
                handle.write("No significant terms.\n\n")
                continue
            handle.write(
                "| term_id | term_name | k_overlap | q_value | enrichment_ratio |\n"
            )
            handle.write("|---|---|---:|---:|---:|\n")
            for hit in hits:
                term_id = str(hit["term_id"]).replace("|", " ")
                term_name = str(hit["term_name"]).replace("|", " ")
                handle.write(
                    f"| {term_id} | {term_name} | {hit['k_overlap']} | "
                    f"{hit['q_value']:.3e} | {hit['enrichment_ratio']:.3f} |\n"
                )
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cluster-level ORA enrichment on GNN embeddings for multiple thresholds."
    )
    parser.add_argument(
        "--embedding_specs",
        required=True,
        help="Comma-separated list of threshold=embedding_npz",
    )
    parser.add_argument("--membership_csv", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--shuffle_labels",
        action="store_true",
        help=(
            "Permutation-null control: shuffle cluster labels across IDs after KMeans "
            "(preserves cluster size distribution, breaks biological structure)."
        ),
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=-1,
        help="Seed for label shuffling (-1 uses --seed).",
    )
    parser.add_argument("--fdr_q", type=float, default=0.05)
    parser.add_argument("--min_term_genes", type=int, default=5)
    parser.add_argument("--max_term_genes", type=int, default=5000)
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=2_000_000)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    threshold_to_embedding = parse_embedding_specs(args.embedding_specs)
    thresholds = sorted(threshold_to_embedding.keys())
    os.makedirs(args.output_dir, exist_ok=True)

    term_meta = load_term_metadata(args.metadata_csv)
    print(f"Loaded metadata terms: {len(term_meta):,}", flush=True)

    shuffle_seed = args.seed if args.shuffle_seed < 0 else args.shuffle_seed
    shuffle_rng = np.random.default_rng(shuffle_seed)

    threshold_to_ids: dict[str, list[str]] = {}
    threshold_to_labels: dict[str, np.ndarray] = {}
    threshold_to_cluster_node_counts: dict[str, Counter] = {}
    for threshold in thresholds:
        emb_file = threshold_to_embedding[threshold]
        ids, labels = cluster_embedding(emb_file, args.k, args.seed)
        if args.shuffle_labels:
            labels = shuffle_rng.permutation(labels)
        threshold_to_ids[threshold] = ids
        threshold_to_labels[threshold] = labels
        threshold_to_cluster_node_counts[threshold] = Counter(labels.tolist())
        print(
            f"[cluster] threshold={threshold} samples={len(ids):,} clusters={len(set(labels.tolist()))}",
            flush=True,
        )

    prefix = f"gnn_enrichment_k{args.k}_seed{args.seed}"
    if args.shuffle_labels:
        prefix = f"{prefix}_shuffled"
    assignment_csv = os.path.join(args.output_dir, f"{prefix}_cluster_assignments.csv")
    write_cluster_assignments(
        assignment_csv, threshold_to_ids, threshold_to_labels, threshold_to_embedding
    )
    print(f"Wrote: {assignment_csv}", flush=True)

    gs_memberships = build_gs_memberships(threshold_to_ids, threshold_to_labels)
    universe_genes, cluster_genes = first_pass_cluster_genes(
        membership_csv=args.membership_csv,
        gs_memberships=gs_memberships,
        thresholds=thresholds,
        log_every=args.log_every,
    )

    for threshold in thresholds:
        print(
            f"[pass1] threshold={threshold} universe_genes={len(universe_genes[threshold]):,}",
            flush=True,
        )

    gene_to_clusters = build_gene_to_clusters(cluster_genes)
    term_counts, overlap_counts = second_pass_term_counts(
        membership_csv=args.membership_csv,
        universe_genes=universe_genes,
        gene_to_clusters=gene_to_clusters,
        thresholds=thresholds,
        log_every=args.log_every,
    )

    all_results: list[dict] = []
    all_summary: list[dict] = []
    all_top: list[dict] = []
    for threshold in thresholds:
        results_rows, summary_rows, top_rows = run_enrichment(
            threshold=threshold,
            term_counts=term_counts[threshold],
            overlap_counts=overlap_counts[threshold],
            cluster_genes=cluster_genes[threshold],
            cluster_node_counts=threshold_to_cluster_node_counts[threshold],
            term_meta=term_meta,
            fdr_q=args.fdr_q,
            min_term_genes=args.min_term_genes,
            max_term_genes=args.max_term_genes,
            top_n=args.top_n,
        )
        all_results.extend(results_rows)
        all_summary.extend(summary_rows)
        all_top.extend(top_rows)
        sig_total = sum(int(r["significant_terms"]) for r in summary_rows)
        print(
            f"[enrich] threshold={threshold} clusters={len(summary_rows)} sig_terms_total={sig_total}",
            flush=True,
        )

    results_csv = os.path.join(args.output_dir, f"{prefix}_results.csv")
    summary_csv = os.path.join(args.output_dir, f"{prefix}_summary.csv")
    report_md = os.path.join(args.output_dir, f"{prefix}_report.md")

    if all_results:
        write_csv(
            results_csv,
            all_results,
            [
                "threshold",
                "cluster",
                "cluster_nodes",
                "cluster_genes",
                "universe_genes",
                "tested_terms",
                "term_id",
                "term_name",
                "term_description",
                "k_overlap",
                "n_cluster_genes",
                "m_term_genes",
                "M_universe_genes",
                "p_value",
                "q_value",
                "enrichment_ratio",
            ],
        )
    else:
        write_csv(
            results_csv,
            [],
            [
                "threshold",
                "cluster",
                "cluster_nodes",
                "cluster_genes",
                "universe_genes",
                "tested_terms",
                "term_id",
                "term_name",
                "term_description",
                "k_overlap",
                "n_cluster_genes",
                "m_term_genes",
                "M_universe_genes",
                "p_value",
                "q_value",
                "enrichment_ratio",
            ],
        )

    write_csv(
        summary_csv,
        all_summary,
        [
            "threshold",
            "cluster",
            "cluster_nodes",
            "cluster_genes",
            "universe_genes",
            "tested_terms",
            "significant_terms",
            "best_term_id",
            "best_term_name",
            "best_q_value",
            "best_p_value",
            "best_overlap",
            "best_enrichment_ratio",
        ],
    )
    write_markdown_report(report_md, all_top, all_summary)

    print(f"Wrote: {results_csv}", flush=True)
    print(f"Wrote: {summary_csv}", flush=True)
    print(f"Wrote: {report_md}", flush=True)


if __name__ == "__main__":
    main()
