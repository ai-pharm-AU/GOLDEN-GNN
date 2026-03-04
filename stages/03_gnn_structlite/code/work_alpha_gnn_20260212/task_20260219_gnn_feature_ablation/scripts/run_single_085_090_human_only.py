#!/usr/bin/env python3
"""Rebuild single_085_090 results using human-only nodes/edges.

This script regenerates the full artifact set in a new folder:
- UMAP plots (subset085/090 x fusion/gnn/gnn_structlite)
- internal metrics tables (csv + md)
- additional metrics comparison tables (csv + md)

Rules:
- Keep a node only if ORGANISM == "Homo sapiens" in master metadata.
- Keep an edge only if both endpoints are kept human nodes.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class SubsetConfig:
    subset: str
    edge_file: Path
    meta_csv: Path
    go_npz: Path
    pager_npz: Path
    fusion_npz: Path
    gnn_npz: Path


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd), check=False, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def load_npz(path: Path) -> tuple[list[str], np.ndarray]:
    data = np.load(path, allow_pickle=True)
    ids = data["ID"] if "ID" in data else data["GOID"]
    emb = data["embeddings"] if "embeddings" in data else data["emb"]
    return [str(x) for x in ids], np.asarray(emb, dtype=np.float32)


def write_npz(path: Path, ids: list[str], emb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ID=np.asarray(ids, dtype=object), embeddings=np.asarray(emb, dtype=np.float32))


def filter_npz_by_ids(src_npz: Path, keep_ids: set[str], dst_npz: Path) -> tuple[int, int]:
    ids, emb = load_npz(src_npz)
    keep_idx = [i for i, gid in enumerate(ids) if gid in keep_ids]
    ids_keep = [ids[i] for i in keep_idx]
    emb_keep = emb[keep_idx]
    write_npz(dst_npz, ids_keep, emb_keep)
    return len(ids), len(ids_keep)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_"
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df.iterrows():
        vals: list[str] = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if math.isnan(v):
                    vals.append("")
                else:
                    vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def parse_confound(md_path: Path) -> dict[str, float]:
    txt = md_path.read_text(encoding="utf-8")
    gse = re.search(r"primary GSE:.*?NMI=([0-9.]+|nan) \| ARI=([0-9.]+|nan)", txt)
    pla = re.search(r"primary platform:.*?NMI=([0-9.]+|nan) \| ARI=([0-9.]+|nan)", txt)
    out: dict[str, float] = {}
    if gse:
        out["primary_gse_nmi"] = float(gse.group(1))
        out["primary_gse_ari"] = float(gse.group(2))
    if pla:
        out["primary_platform_nmi"] = float(pla.group(1))
        out["primary_platform_ari"] = float(pla.group(2))
    return out


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return x / norms


def text_coherence(ids: list[str], labels: np.ndarray, id_to_desc: dict[str, str]) -> float:
    docs = [id_to_desc.get(i, "") for i in ids]
    vec = TfidfVectorizer(stop_words="english", min_df=2)
    x = vec.fit_transform(docs)
    if x.shape[1] == 0:
        return float("nan")

    weighted_sum = 0.0
    total = 0
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        x_c = x[idx]
        centroid = np.asarray(x_c.mean(axis=0))
        sims = cosine_similarity(x_c, centroid).ravel()
        weighted_sum += float(sims.mean()) * len(idx)
        total += len(idx)
    return weighted_sum / total if total > 0 else float("nan")


def load_desc_map(desc_tsv: Path) -> dict[str, str]:
    df = pd.read_csv(desc_tsv, sep="\t")
    return {str(i): str(d) for i, d in zip(df["GOID"], df["DESCRIPTION"])}


def cluster_balance_text_coh(emb_npz: Path, k: int, seeds: list[int], id_to_desc: dict[str, str]) -> float:
    ids, emb = load_npz(emb_npz)
    keep = [i for i, gid in enumerate(ids) if gid in id_to_desc]
    ids2 = [ids[i] for i in keep]
    x = l2_normalize_rows(emb[keep])
    vals: list[float] = []
    for seed in seeds:
        labels = KMeans(n_clusters=k, random_state=seed, n_init="auto").fit_predict(x)
        vals.append(text_coherence(ids2, labels, id_to_desc))
    return float(np.mean(vals)) if vals else float("nan")


def summarize_internal(eval_csv: Path, subset_label: str, method: str) -> dict[str, float | int | str]:
    d = pd.read_csv(eval_csv)
    d = d[d["k"] == 10]
    return {
        "subset": subset_label,
        "method": method,
        "n_seeds": int(d["seed"].nunique()),
        "silhouette_euclidean_mean": float(d["silhouette_euclidean"].mean()),
        "silhouette_cosine_mean": float(d["silhouette_cosine"].mean()),
        "davies_bouldin_mean": float(d["davies_bouldin"].mean()),
        "calinski_harabasz_mean": float(d["calinski_harabasz"].mean()),
        "silhouette_euclidean_std": float(d["silhouette_euclidean"].std()),
        "silhouette_cosine_std": float(d["silhouette_cosine"].std()),
        "davies_bouldin_std": float(d["davies_bouldin"].std()),
        "calinski_harabasz_std": float(d["calinski_harabasz"].std()),
    }


def mean_text_stability(validation_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    t = pd.read_csv(validation_dir / "text_coherence_by_seed.csv")
    out["text_coherence_raw_mean"] = float(t["text_coherence_raw"].mean())
    out["text_null_raw_p_mean"] = float(t["text_null_raw_p"].mean())
    out["text_coherence_clean_mean"] = float(t["text_coherence_clean"].mean())
    out["text_null_clean_p_mean"] = float(t["text_null_clean_p"].mean())

    a = pd.read_csv(validation_dir / "pairwise_ari.csv")
    out["pairwise_ari_mean"] = float(a["ari"].mean())
    out["pairwise_ari_std"] = float(a["ari"].std())
    return out


def mean_enrichment(validation_dir: Path) -> dict[tuple[str, str], float]:
    d = pd.read_csv(validation_dir / "meta_enrichment_seed_summary.csv")
    out: dict[tuple[str, str], float] = {}
    for attr in sorted(d["attr_type"].unique()):
        s = d[d["attr_type"] == attr]
        out[("significant_mean", attr)] = float(s["significant"].mean())
        out[("null_sig_mean", attr)] = float(s["null_sig_mean"].mean())
        out[("p_sig_mean", attr)] = float(s["null_sig_p"].mean())
        out[("max_log10q_mean", attr)] = float(s["max_log10q"].mean())
        out[("p_maxlog10q_mean", attr)] = (
            float(s["null_maxlog10q_p"].mean()) if "null_maxlog10q_p" in s.columns else float("nan")
        )
    return out


def build_additional_table(
    subset: str,
    methods: list[str],
    method_to_emb: dict[str, Path],
    method_to_val_dir: dict[str, Path],
    method_to_confound: dict[str, Path],
    desc_map: dict[str, str],
    seeds: list[int],
) -> pd.DataFrame:
    text_tfidf: dict[str, float] = {}
    txt_stab: dict[str, dict[str, float]] = {}
    enrich: dict[str, dict[tuple[str, str], float]] = {}
    conf: dict[str, dict[str, float]] = {}

    for m in methods:
        text_tfidf[m] = cluster_balance_text_coh(method_to_emb[m], k=10, seeds=seeds, id_to_desc=desc_map)
        txt_stab[m] = mean_text_stability(method_to_val_dir[m])
        enrich[m] = mean_enrichment(method_to_val_dir[m])
        conf[m] = parse_confound(method_to_confound[m])

    rows: list[dict[str, object]] = []
    rows.append(
        {
            "subset": subset,
            "category": "text_coherence_tfidf",
            "metric": "text_coherence_weighted_mean_k10",
            "attr_type": "",
            **{m: text_tfidf[m] for m in methods},
        }
    )

    for metric in [
        "text_coherence_clean_mean",
        "text_coherence_raw_mean",
        "text_null_clean_p_mean",
        "text_null_raw_p_mean",
    ]:
        rows.append(
            {
                "subset": subset,
                "category": "text_coherence_meta",
                "metric": metric,
                "attr_type": "",
                **{m: txt_stab[m].get(metric, float("nan")) for m in methods},
            }
        )

    for metric in ["pairwise_ari_mean", "pairwise_ari_std"]:
        rows.append(
            {
                "subset": subset,
                "category": "stability",
                "metric": metric,
                "attr_type": "",
                **{m: txt_stab[m].get(metric, float("nan")) for m in methods},
            }
        )

    attr_types = sorted({k[1] for m in methods for k in enrich[m].keys()})
    for attr in attr_types:
        for metric in [
            "max_log10q_mean",
            "null_sig_mean",
            "p_maxlog10q_mean",
            "p_sig_mean",
            "significant_mean",
        ]:
            rows.append(
                {
                    "subset": subset,
                    "category": "meta_enrichment",
                    "metric": metric,
                    "attr_type": attr,
                    **{m: enrich[m].get((metric, attr), float("nan")) for m in methods},
                }
            )

    for metric in [
        "primary_gse_ari",
        "primary_gse_nmi",
        "primary_platform_ari",
        "primary_platform_nmi",
    ]:
        rows.append(
            {
                "subset": subset,
                "category": "confound",
                "metric": metric,
                "attr_type": "",
                **{m: conf[m].get(metric, float("nan")) for m in methods},
            }
        )

    return pd.DataFrame(rows)


def write_md_from_df(title: str, df: pd.DataFrame, out_md: Path) -> None:
    lines = [f"# {title}", "", markdown_table(df)]
    out_md.write_text("\n".join(lines), encoding="utf-8")


def method_label(method: str) -> str:
    if method == "fusion":
        return "fusion_alpha05"
    if method == "gnn":
        return "gnn_old_holdout"
    if method == "gnn_structlite":
        return "gnn_structlite"
    return method


def main() -> None:
    parser = argparse.ArgumentParser(description="Run human-only single_085_090 regeneration.")
    parser.add_argument("--python_bin", default="/home/zzz0054/GoldenF/.venv/bin/python")
    parser.add_argument("--master_meta_csv", default="/home/zzz0054/GoldenF/data/mcxqrhgm61ztvegwj8ltzbcvv1h06vni.csv")
    parser.add_argument("--seeds", default="11,12,13,14,15")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--n_perm", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    py = Path(args.python_bin)
    if not py.exists():
        raise FileNotFoundError(f"python_bin not found: {py}")

    root = Path("/home/zzz0054/GoldenF")
    base = root / "work_alpha_gnn_20260212"
    task = base / "task_20260219_gnn_feature_ablation"
    scripts_base = base / "scripts"
    scripts_task = task / "scripts"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_root = task / "runs" / f"single_085_090_human_only_{ts}"
    plot_dir = task / "reports" / "plots" / f"single_085_090_human_only_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    seeds_csv = ",".join(str(x) for x in seeds)

    subset_cfg = {
        "085": SubsetConfig(
            subset="085",
            edge_file=base / "subset_85_90_20260213/csv/edges_with_mtype_threshold_085.txt",
            meta_csv=base / "reports/bio_meta_cache_085/bio_meta_cache_085/enriched_gs_metadata.csv",
            go_npz=base / "subset_85_90_20260213/embeddings/go_embeddings_subset_085_l2.npz",
            pager_npz=base / "subset_85_90_20260213/embeddings/pager_embeddings_subset_085_l2.npz",
            fusion_npz=base / "task_20260217_metrics_080_085/embeddings/fusion_concat_postl2_alpha0.5_085.npz",
            gnn_npz=base / "subset_85_90_20260213/embeddings/gnn_holdout_subset_085_l2.npz",
        ),
        "090": SubsetConfig(
            subset="090",
            edge_file=base / "subset_85_90_20260213/csv/edges_with_mtype_threshold_090.txt",
            meta_csv=base / "reports/bio_meta_cache_090/enriched_gs_metadata.csv",
            go_npz=base / "subset_85_90_20260213/embeddings/go_embeddings_subset_090_l2.npz",
            pager_npz=base / "subset_85_90_20260213/embeddings/pager_embeddings_subset_090_l2.npz",
            fusion_npz=base / "task_20260217_alpha05_subset090/embeddings/fusion_concat_postl2_alpha0.5_subset_090.npz",
            gnn_npz=base / "subset_85_90_20260213/embeddings/gnn_holdout_subset_090_l2.npz",
        ),
    }

    master = pd.read_csv(args.master_meta_csv, usecols=["GS_ID", "ORGANISM"])
    master["GS_ID"] = master["GS_ID"].astype(str)
    master_org = master.set_index("GS_ID")["ORGANISM"].fillna("").astype(str).str.strip()

    methods = ["fusion", "gnn", "gnn_structlite"]
    all_internal_rows: list[dict[str, object]] = []
    additional_tables: list[pd.DataFrame] = []
    readme_lines = [
        "# Single UMAP + Internal Metrics Tables (subset 085/090, human-only)",
        "",
        "Definition used in this package:",
        "- `fusion` = fusion alpha=0.5 embedding",
        "- `gnn` = old gnn holdout embedding",
        "- `gnn_structlite` = `text_structlite_gnn_holdout` embedding retrained on human-only graph",
        "- all points in plots are `Homo sapiens` only",
        "- all edges used for gnn_structlite training require both endpoints `Homo sapiens`",
        "",
        "## UMAP plots",
        "",
    ]

    desc_map = load_desc_map(root / "data/RummaGEO/rummageo_descriptions.tsv")

    for subset in ["085", "090"]:
        cfg = subset_cfg[subset]
        sub = run_root / f"subset{subset}"
        inputs_dir = sub / "inputs"
        feat_dir = sub / "features"
        csv_dir = sub / "csv"
        reports_dir = sub / "reports"
        val_root = reports_dir / "validation"
        conf_root = reports_dir / "confound"
        for d in [inputs_dir, feat_dir, csv_dir, reports_dir, val_root, conf_root]:
            d.mkdir(parents=True, exist_ok=True)

        go_ids, _ = load_npz(cfg.go_npz)
        human_ids = [gid for gid in go_ids if master_org.get(gid, "") == "Homo sapiens"]
        human_set = set(human_ids)

        human_ids_file = inputs_dir / f"human_ids_subset{subset}.txt"
        human_ids_file.write_text("\n".join(human_ids) + "\n", encoding="utf-8")

        meta_df = pd.read_csv(cfg.meta_csv)
        meta_df["GS_ID"] = meta_df["GS_ID"].astype(str)
        meta_h = meta_df[(meta_df["GS_ID"].isin(human_set)) & (meta_df["ORGANISM"].fillna("").astype(str).str.strip() == "Homo sapiens")].copy()
        meta_h = meta_h.drop_duplicates(subset=["GS_ID"]).reset_index(drop=True)
        meta_h_path = inputs_dir / f"enriched_gs_metadata_subset{subset}_human.csv"
        meta_h.to_csv(meta_h_path, index=False)

        edge_df = pd.read_csv(cfg.edge_file, sep="\t")
        edge_h = edge_df[
            edge_df["GS_A_ID"].astype(str).isin(human_set)
            & edge_df["GS_B_ID"].astype(str).isin(human_set)
        ].copy()
        edge_h_path = inputs_dir / f"edges_with_mtype_threshold_{subset}_human.txt"
        edge_h.to_csv(edge_h_path, sep="\t", index=False)

        go_h = inputs_dir / f"go_embeddings_subset_{subset}_human_l2.npz"
        pager_h = inputs_dir / f"pager_embeddings_subset_{subset}_human_l2.npz"
        fusion_h = inputs_dir / f"fusion_subset_{subset}_human.npz"
        gnn_h = inputs_dir / f"gnn_old_holdout_subset_{subset}_human.npz"

        _, go_n = filter_npz_by_ids(cfg.go_npz, human_set, go_h)
        _, pager_n = filter_npz_by_ids(cfg.pager_npz, human_set, pager_h)
        _, fusion_n = filter_npz_by_ids(cfg.fusion_npz, human_set, fusion_h)
        _, gnn_n = filter_npz_by_ids(cfg.gnn_npz, human_set, gnn_h)

        stats_md = inputs_dir / f"filter_stats_subset{subset}.md"
        stats_md.write_text(
            "\n".join(
                [
                    f"# Human-only filter stats (subset {subset})",
                    "",
                    f"- go_human_nodes: {go_n}",
                    f"- pager_human_nodes: {pager_n}",
                    f"- fusion_human_nodes: {fusion_n}",
                    f"- gnn_human_nodes: {gnn_n}",
                    f"- meta_human_rows: {len(meta_h)}",
                    f"- edge_total: {len(edge_df)}",
                    f"- edge_human_only: {len(edge_h)}",
                ]
            ),
            encoding="utf-8",
        )

        # Build text_structlite features.
        run_cmd(
            [
                str(py),
                str(scripts_task / "build_node_features_from_meta.py"),
                "--subset",
                subset,
                "--go_npz",
                str(go_h),
                "--pager_npz",
                str(pager_h),
                "--meta_csv",
                str(meta_h_path),
                "--edge_file",
                str(edge_h_path),
                "--recipes",
                "text_structlite",
                "--weight_column",
                "NLOGPMF",
                "--output_dir",
                str(feat_dir),
            ],
            cwd=base,
        )

        text_structlite_npz = feat_dir / f"text_structlite_subset{subset}.npz"
        text_structlite_gnn_npz = feat_dir / f"text_structlite_gnn_holdout_subset{subset}.npz"
        text_structlite_linkpred_csv = csv_dir / f"text_structlite_linkpred_subset{subset}.csv"

        # Train text_structlite GNN on human-only graph.
        run_cmd(
            [
                str(py),
                str(scripts_base / "gnn_linkpred_holdout.py"),
                "--edge_file",
                str(edge_h_path),
                "--feature_npz",
                str(text_structlite_npz),
                "--weight_column",
                "NLOGPMF",
                "--edge_weight_transform",
                "log1p",
                "--hidden_dim",
                "256",
                "--num_layers",
                "2",
                "--dropout",
                "0.1",
                "--lr",
                "1e-3",
                "--epochs",
                str(args.epochs),
                "--eval_every",
                str(args.eval_every),
                "--val_ratio",
                "0.1",
                "--pos_samples_per_epoch",
                "50000",
                "--seed",
                "42",
                "--device",
                args.device,
                "--output_csv",
                str(text_structlite_linkpred_csv),
                "--output_npz",
                str(text_structlite_gnn_npz),
            ],
            cwd=base,
        )

        method_to_emb = {
            "fusion": fusion_h,
            "gnn": gnn_h,
            "gnn_structlite": text_structlite_gnn_npz,
        }
        method_to_eval_csv: dict[str, Path] = {}
        method_to_val_dir: dict[str, Path] = {}
        method_to_confound: dict[str, Path] = {}

        for method in methods:
            emb_path = method_to_emb[method]
            eval_csv = csv_dir / f"{method}_eval_subset{subset}.csv"
            val_dir = val_root / f"{method}_validation_subset{subset}"
            conf_md = conf_root / f"{method}_confound_subset{subset}.md"
            val_dir.mkdir(parents=True, exist_ok=True)

            run_cmd(
                [
                    str(py),
                    str(scripts_base / "evaluate_single_embedding.py"),
                    "--embedding_file",
                    str(emb_path),
                    "--algos",
                    "kmeans",
                    "--k_list",
                    "10",
                    "--seeds",
                    seeds_csv,
                    "--output_csv",
                    str(eval_csv),
                ],
                cwd=base,
            )

            run_cmd(
                [
                    str(py),
                    str(scripts_base / "validate_cluster_meta_enrichment.py"),
                    "--embedding_npz",
                    str(emb_path),
                    "--enriched_metadata_csv",
                    str(meta_h_path),
                    "--k",
                    "10",
                    "--seeds",
                    seeds_csv,
                    "--n_perm",
                    str(args.n_perm),
                    "--output_dir",
                    str(val_dir),
                ],
                cwd=base,
            )

            run_cmd(
                [
                    str(py),
                    str(scripts_base / "analyze_mesh_vs_gse_confound.py"),
                    "--embedding_npz",
                    str(emb_path),
                    "--enriched_metadata_csv",
                    str(meta_h_path),
                    "--k",
                    "10",
                    "--seed",
                    "11",
                    "--min_support",
                    "5",
                    "--fdr_q",
                    "0.05",
                    "--output_md",
                    str(conf_md),
                ],
                cwd=base,
            )

            out_png = plot_dir / f"umap_subset{subset}_{method_label(method)}.png"
            run_cmd(
                [
                    str(py),
                    str(scripts_base / "plot_embedding_umap_scatter.py"),
                    "--embedding_npz",
                    str(emb_path),
                    "--output_png",
                    str(out_png),
                    "--k",
                    "10",
                    "--seed",
                    "11",
                    "--title",
                    f"subset{subset} {method}",
                    "--color_mode",
                    "label",
                ],
                cwd=base,
            )

            method_to_eval_csv[method] = eval_csv
            method_to_val_dir[method] = val_dir
            method_to_confound[method] = conf_md

            all_internal_rows.append(
                summarize_internal(
                    eval_csv,
                    subset_label=str(int(subset)),
                    method=method,
                )
            )

        add_df = build_additional_table(
            subset=subset,
            methods=methods,
            method_to_emb=method_to_emb,
            method_to_val_dir=method_to_val_dir,
            method_to_confound=method_to_confound,
            desc_map=desc_map,
            seeds=seeds,
        )
        additional_tables.append(add_df)

        add_csv = plot_dir / f"additional_metrics_comparison_subset{subset}.csv"
        add_md = plot_dir / f"additional_metrics_comparison_subset{subset}.md"
        add_df.to_csv(add_csv, index=False)
        write_md_from_df(
            title=f"Additional Metrics Comparison (subset{subset})",
            df=add_df,
            out_md=add_md,
        )

        readme_lines.extend(
            [
                f"### subset {subset}",
                f"- `umap_subset{subset}_fusion_alpha05.png`",
                f"- `umap_subset{subset}_gnn_old_holdout.png`",
                f"- `umap_subset{subset}_gnn_structlite.png`",
                "",
            ]
        )

    internal_df = pd.DataFrame(all_internal_rows)
    col_order = [
        "subset",
        "method",
        "n_seeds",
        "silhouette_euclidean_mean",
        "silhouette_cosine_mean",
        "davies_bouldin_mean",
        "calinski_harabasz_mean",
        "silhouette_euclidean_std",
        "silhouette_cosine_std",
        "davies_bouldin_std",
        "calinski_harabasz_std",
    ]
    internal_df = internal_df[col_order]

    for subset in ["85", "90"]:
        sub_df = internal_df[internal_df["subset"] == subset].copy()
        csv_path = plot_dir / f"internal_metrics_table_subset0{subset}.csv"
        md_path = plot_dir / f"internal_metrics_table_subset0{subset}.md"
        sub_df.to_csv(csv_path, index=False)
        write_md_from_df(f"Internal Metrics Table (subset0{subset})", sub_df, md_path)

    internal_comb = internal_df.copy()
    internal_comb.to_csv(plot_dir / "internal_metrics_table_subset085_090_combined.csv", index=False)

    add_comb = pd.concat(additional_tables, axis=0, ignore_index=True)
    add_comb.to_csv(plot_dir / "additional_metrics_comparison_subset085_090_combined.csv", index=False)
    write_md_from_df(
        "Additional Metrics Comparison (subset085_090 combined)",
        add_comb,
        plot_dir / "additional_metrics_comparison_subset085_090_combined.md",
    )

    readme_lines.extend(
        [
            "## Internal metrics tables (4 metrics)",
            "- `internal_metrics_table_subset085.csv`",
            "- `internal_metrics_table_subset090.csv`",
            "- `internal_metrics_table_subset085_090_combined.csv`",
            "",
            "Markdown render copies:",
            "- `internal_metrics_table_subset085.md`",
            "- `internal_metrics_table_subset090.md`",
            "",
            "## Additional comparison metrics (3 algorithms)",
            "- `additional_metrics_comparison_subset085.csv`",
            "- `additional_metrics_comparison_subset090.csv`",
            "- `additional_metrics_comparison_subset085_090_combined.csv`",
            "",
            "Markdown render copies:",
            "- `additional_metrics_comparison_subset085.md`",
            "- `additional_metrics_comparison_subset090.md`",
            "- `additional_metrics_comparison_subset085_090_combined.md`",
            "",
            f"## Run directories",
            f"- run_root: `{run_root}`",
            f"- plot_dir: `{plot_dir}`",
        ]
    )
    (plot_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    manifest = run_root / "run_manifest_human_only.md"
    manifest.write_text(
        "\n".join(
            [
                "# Human-only single_085_090 run manifest",
                "",
                f"- python_bin: `{py}`",
                f"- master_meta_csv: `{args.master_meta_csv}`",
                f"- seeds: {seeds_csv}",
                f"- epochs: {args.epochs}",
                f"- eval_every: {args.eval_every}",
                f"- n_perm: {args.n_perm}",
                f"- device: {args.device}",
                f"- output_plot_dir: `{plot_dir}`",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[done] run_root={run_root}")
    print(f"[done] plot_dir={plot_dir}")


if __name__ == "__main__":
    main()
