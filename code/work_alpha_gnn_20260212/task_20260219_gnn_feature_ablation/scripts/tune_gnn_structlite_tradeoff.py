#!/usr/bin/env python3
"""Tune gnn_structlite to improve separation while preserving additional metrics."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def run_cmd(cmd: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            handle.write(line)
        ret = proc.wait()
        handle.write(f"\n[exit_code] {ret}\n")
    if ret != 0:
        raise RuntimeError(f"Command failed ({ret}): {' '.join(cmd)}")


def load_npz(path: Path) -> tuple[list[str], np.ndarray]:
    data = np.load(path, allow_pickle=True)
    ids = data["ID"] if "ID" in data else data["GOID"]
    emb = data["embeddings"] if "embeddings" in data else data["emb"]
    return [str(x) for x in ids], np.asarray(emb, dtype=np.float32)


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return x / norms


def load_desc_map(path: Path) -> dict[str, str]:
    df = pd.read_csv(path, sep="\t")
    return {str(i): str(d) for i, d in zip(df["GOID"], df["DESCRIPTION"])}


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


def text_coherence_weighted_mean(emb_npz: Path, seeds: list[int], k: int, id_to_desc: dict[str, str]) -> float:
    ids, emb = load_npz(emb_npz)
    keep = [i for i, gid in enumerate(ids) if gid in id_to_desc]
    ids_keep = [ids[i] for i in keep]
    x = l2_normalize_rows(emb[keep])
    vals: list[float] = []
    for seed in seeds:
        labels = KMeans(n_clusters=k, random_state=seed, n_init="auto").fit_predict(x)
        vals.append(text_coherence(ids_keep, labels, id_to_desc))
    return float(np.mean(vals)) if vals else float("nan")


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


def mean_internal(eval_csv: Path, k: int) -> dict[str, float]:
    d = pd.read_csv(eval_csv)
    d = d[d["k"] == k]
    return {
        "silhouette_euclidean_mean": float(d["silhouette_euclidean"].mean()),
        "silhouette_cosine_mean": float(d["silhouette_cosine"].mean()),
        "davies_bouldin_mean": float(d["davies_bouldin"].mean()),
        "calinski_harabasz_mean": float(d["calinski_harabasz"].mean()),
    }


def mean_validation(validation_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    t = pd.read_csv(validation_dir / "text_coherence_by_seed.csv")
    out["text_coherence_raw_mean"] = float(t["text_coherence_raw"].mean())
    out["text_null_raw_p_mean"] = float(t["text_null_raw_p"].mean())
    out["text_coherence_clean_mean"] = float(t["text_coherence_clean"].mean())
    out["text_null_clean_p_mean"] = float(t["text_null_clean_p"].mean())

    a = pd.read_csv(validation_dir / "pairwise_ari.csv")
    out["pairwise_ari_mean"] = float(a["ari"].mean())

    m = pd.read_csv(validation_dir / "meta_enrichment_seed_summary.csv")
    for attr in ["mesh", "gse_id"]:
        s = m[m["attr_type"] == attr]
        out[f"{attr}_significant_mean"] = float(s["significant"].mean())
        out[f"{attr}_p_sig_mean"] = float(s["null_sig_p"].mean())
    return out


def collect_metrics(
    eval_csv: Path,
    val_dir: Path,
    confound_md: Path,
    emb_npz: Path,
    seeds: list[int],
    k: int,
    id_to_desc: dict[str, str],
) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update(mean_internal(eval_csv, k=k))
    out.update(mean_validation(val_dir))
    out.update(parse_confound(confound_md))
    out["text_coherence_weighted_mean_k10"] = text_coherence_weighted_mean(
        emb_npz, seeds=seeds, k=k, id_to_desc=id_to_desc
    )
    return out


def internal_score(candidate: dict[str, float], baseline: dict[str, float]) -> float:
    def safe_rel(new: float, base: float) -> float:
        den = abs(base) if abs(base) > 1e-12 else 1.0
        return (new - base) / den

    z_sil_euc = safe_rel(candidate["silhouette_euclidean_mean"], baseline["silhouette_euclidean_mean"])
    z_sil_cos = safe_rel(candidate["silhouette_cosine_mean"], baseline["silhouette_cosine_mean"])
    z_db = safe_rel(baseline["davies_bouldin_mean"], candidate["davies_bouldin_mean"])
    z_ch = safe_rel(candidate["calinski_harabasz_mean"], baseline["calinski_harabasz_mean"])
    return float(0.3 * z_sil_euc + 0.3 * z_sil_cos + 0.2 * z_db + 0.2 * z_ch)


def tol(base: float, rel: float, abs_floor: float) -> float:
    return max(abs_floor, rel * abs(base))


def check_constraints(
    candidate: dict[str, float],
    baseline: dict[str, float],
    rel: float,
    abs_floor: float,
) -> tuple[bool, list[dict[str, object]]]:
    regular_metrics = [
        "text_coherence_weighted_mean_k10",
        "text_coherence_clean_mean",
        "text_coherence_raw_mean",
        "text_null_clean_p_mean",
        "text_null_raw_p_mean",
        "pairwise_ari_mean",
        "mesh_significant_mean",
        "gse_id_significant_mean",
        "mesh_p_sig_mean",
        "gse_id_p_sig_mean",
    ]
    confound_metrics = ["primary_gse_nmi", "primary_platform_nmi"]

    violations: list[dict[str, object]] = []
    for m in regular_metrics:
        b = baseline.get(m, float("nan"))
        c = candidate.get(m, float("nan"))
        if not (np.isfinite(b) and np.isfinite(c)):
            violations.append({"metric": m, "reason": "nan", "base": b, "candidate": c})
            continue
        t = tol(b, rel=rel, abs_floor=abs_floor)
        d = c - b
        if abs(d) > t:
            violations.append(
                {
                    "metric": m,
                    "reason": "drift",
                    "base": b,
                    "candidate": c,
                    "delta": d,
                    "tol": t,
                }
            )

    for m in confound_metrics:
        b = baseline.get(m, float("nan"))
        c = candidate.get(m, float("nan"))
        if not (np.isfinite(b) and np.isfinite(c)):
            violations.append({"metric": m, "reason": "nan", "base": b, "candidate": c})
            continue
        t = tol(b, rel=rel, abs_floor=abs_floor)
        d = c - b
        if d > t:
            violations.append(
                {
                    "metric": m,
                    "reason": "confound_up",
                    "base": b,
                    "candidate": c,
                    "delta": d,
                    "tol": t,
                }
            )
    return len(violations) == 0, violations


def default_search_space() -> dict[str, list[Any]]:
    return {
        "hidden_dim": [256, 320],
        "num_layers": [2, 3],
        "dropout": [0.05, 0.1],
        "lr": [1e-3, 5e-4],
        "weight_decay": [0.0],
        "edge_weight_transform": ["log1p", "sqrt"],
        "struct_features": ["basic", "basic_plus"],
        "struct_norm": ["zscore", "robust"],
        "go_scale": [1.0],
        "meta_scale": [1.0],
        "struct_scale": [0.8, 1.0, 1.2],
    }


def load_search_space(path: str) -> dict[str, list[Any]]:
    if not path:
        return default_search_space()
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("search_space_json must contain an object")
    out: dict[str, list[Any]] = {}
    for k, v in payload.items():
        if not isinstance(v, list) or not v:
            raise ValueError(f"search space key `{k}` must be a non-empty list")
        out[k] = v
    return out


def expand_candidates(space: dict[str, list[Any]], max_candidates: int) -> list[dict[str, Any]]:
    keys = sorted(space.keys())
    vals = [space[k] for k in keys]
    out: list[dict[str, Any]] = []
    for items in itertools.product(*vals):
        out.append({k: v for k, v in zip(keys, items)})
        if max_candidates > 0 and len(out) >= max_candidates:
            break
    return out


@dataclass(frozen=True)
class PathsPerSubset:
    subset: str
    input_dir: Path
    out_dir: Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune gnn_structlite tradeoff.")
    parser.add_argument(
        "--python_bin",
        default="/home/zzz0054/GoldenF/.venv/bin/python",
    )
    parser.add_argument(
        "--base_dir",
        default="/home/zzz0054/GoldenF/work_alpha_gnn_20260212",
    )
    parser.add_argument(
        "--base_run_dir",
        default=(
            "/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/"
            "runs/single_085_090_human_only_20260220_233515"
        ),
    )
    parser.add_argument(
        "--output_root",
        default="/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation/reports",
    )
    parser.add_argument("--subsets", default="085,090")
    parser.add_argument("--seeds", default="11,12,13,14,15")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n_perm", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--pos_samples_per_epoch", type=int, default=50000)
    parser.add_argument("--train_seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--constraint_rel", type=float, default=0.03)
    parser.add_argument("--constraint_abs", type=float, default=0.01)
    parser.add_argument("--search_space_json", default="")
    parser.add_argument("--max_candidates", type=int, default=0)
    parser.add_argument("--internal_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    py = Path(args.python_bin)
    if not py.exists():
        raise FileNotFoundError(f"python_bin not found: {py}")

    base_dir = Path(args.base_dir).resolve()
    base_run = Path(args.base_run_dir).resolve()
    output_root = Path(args.output_root).resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_root / f"tuning_structlite_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_list(args.seeds)
    subsets = parse_csv_list(args.subsets)
    if not subsets:
        raise ValueError("No subsets provided")

    desc_tsv = Path("/home/zzz0054/GoldenF/data/RummaGEO/rummageo_descriptions.tsv")
    id_to_desc = load_desc_map(desc_tsv)

    space = load_search_space(args.search_space_json)
    candidates = expand_candidates(space, max_candidates=args.max_candidates)
    if not candidates:
        raise RuntimeError("No candidates to evaluate")

    # Load baseline metrics from base_run_dir.
    baseline: dict[str, dict[str, float]] = {}
    for subset in subsets:
        sub = base_run / f"subset{subset}"
        eval_csv = sub / "csv" / f"gnn_structlite_eval_subset{subset}.csv"
        if args.internal_only:
            baseline[subset] = mean_internal(eval_csv, k=args.k)
        else:
            val_dir = sub / "reports" / "validation" / f"gnn_structlite_validation_subset{subset}"
            conf_md = sub / "reports" / "confound" / f"gnn_structlite_confound_subset{subset}.md"
            emb_npz = sub / "features" / f"text_structlite_gnn_holdout_subset{subset}.npz"
            baseline[subset] = collect_metrics(
                eval_csv=eval_csv,
                val_dir=val_dir,
                confound_md=conf_md,
                emb_npz=emb_npz,
                seeds=seeds,
                k=args.k,
                id_to_desc=id_to_desc,
            )

    baseline_json = out_dir / "baseline_metrics.json"
    baseline_json.write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    if args.dry_run:
        dry_info = {
            "num_candidates": len(candidates),
            "subsets": subsets,
            "seeds": seeds,
            "k": args.k,
            "output_dir": str(out_dir),
            "search_space": space,
        }
        (out_dir / "dry_run.json").write_text(json.dumps(dry_info, indent=2), encoding="utf-8")
        print(f"[dry_run] wrote {out_dir / 'dry_run.json'}")
        return

    build_script = base_dir / "task_20260219_gnn_feature_ablation/scripts/build_node_features_from_meta.py"
    train_script = base_dir / "scripts/gnn_linkpred_holdout.py"
    eval_script = base_dir / "scripts/evaluate_single_embedding.py"
    val_script = base_dir / "scripts/validate_cluster_meta_enrichment.py"
    confound_script = base_dir / "scripts/analyze_mesh_vs_gse_confound.py"

    score_rows: list[dict[str, object]] = []
    violation_rows: list[dict[str, object]] = []

    for idx, cfg in enumerate(candidates, start=1):
        cand_id = f"cand_{idx:04d}"
        cand_root = out_dir / "candidates" / cand_id
        cand_root.mkdir(parents=True, exist_ok=True)

        subset_scores: dict[str, float] = {}
        subset_valid: dict[str, bool] = {}
        subset_metrics: dict[str, dict[str, float]] = {}

        for subset in subsets:
            subset_root = cand_root / f"subset{subset}"
            features_dir = subset_root / "features"
            csv_dir = subset_root / "csv"
            logs_dir = subset_root / "logs"
            reports_dir = subset_root / "reports"
            val_dir = reports_dir / "validation" / f"gnn_structlite_validation_subset{subset}"
            conf_dir = reports_dir / "confound"
            for d in [features_dir, csv_dir, logs_dir, val_dir, conf_dir]:
                d.mkdir(parents=True, exist_ok=True)

            in_dir = base_run / f"subset{subset}" / "inputs"
            go_npz = in_dir / f"go_embeddings_subset_{subset}_human_l2.npz"
            pager_npz = in_dir / f"pager_embeddings_subset_{subset}_human_l2.npz"
            meta_csv = in_dir / f"enriched_gs_metadata_subset{subset}_human.csv"
            edge_file = in_dir / f"edges_with_mtype_threshold_{subset}_human.txt"
            for p in [go_npz, pager_npz, meta_csv, edge_file]:
                if not p.exists():
                    raise FileNotFoundError(f"Missing required input: {p}")

            feat_npz = features_dir / f"text_structlite_subset{subset}.npz"
            emb_npz = features_dir / f"text_structlite_gnn_holdout_subset{subset}.npz"
            linkpred_csv = csv_dir / f"text_structlite_linkpred_subset{subset}.csv"
            eval_csv = csv_dir / f"gnn_structlite_eval_subset{subset}.csv"
            conf_md = conf_dir / f"gnn_structlite_confound_subset{subset}.md"

            run_cmd(
                [
                    str(py),
                    str(build_script),
                    "--subset",
                    subset,
                    "--go_npz",
                    str(go_npz),
                    "--pager_npz",
                    str(pager_npz),
                    "--meta_csv",
                    str(meta_csv),
                    "--edge_file",
                    str(edge_file),
                    "--recipes",
                    "text_structlite",
                    "--weight_column",
                    "NLOGPMF",
                    "--struct_features",
                    str(cfg["struct_features"]),
                    "--struct_norm",
                    str(cfg["struct_norm"]),
                    "--go_scale",
                    str(cfg["go_scale"]),
                    "--meta_scale",
                    str(cfg["meta_scale"]),
                    "--struct_scale",
                    str(cfg["struct_scale"]),
                    "--output_dir",
                    str(features_dir),
                ],
                cwd=base_dir,
                log_path=logs_dir / "build.log",
            )

            run_cmd(
                [
                    str(py),
                    str(train_script),
                    "--edge_file",
                    str(edge_file),
                    "--feature_npz",
                    str(feat_npz),
                    "--weight_column",
                    "NLOGPMF",
                    "--edge_weight_transform",
                    str(cfg["edge_weight_transform"]),
                    "--hidden_dim",
                    str(cfg["hidden_dim"]),
                    "--num_layers",
                    str(cfg["num_layers"]),
                    "--dropout",
                    str(cfg["dropout"]),
                    "--lr",
                    str(cfg["lr"]),
                    "--weight_decay",
                    str(cfg["weight_decay"]),
                    "--patience",
                    "0",
                    "--epochs",
                    str(args.epochs),
                    "--eval_every",
                    str(args.eval_every),
                    "--val_ratio",
                    "0.1",
                    "--pos_samples_per_epoch",
                    str(args.pos_samples_per_epoch),
                    "--seed",
                    str(args.train_seed),
                    "--device",
                    args.device,
                    "--output_csv",
                    str(linkpred_csv),
                    "--output_npz",
                    str(emb_npz),
                ],
                cwd=base_dir,
                log_path=logs_dir / "train.log",
            )

            run_cmd(
                [
                    str(py),
                    str(eval_script),
                    "--embedding_file",
                    str(emb_npz),
                    "--algos",
                    "kmeans",
                    "--k_list",
                    str(args.k),
                    "--seeds",
                    ",".join(str(s) for s in seeds),
                    "--output_csv",
                    str(eval_csv),
                ],
                cwd=base_dir,
                log_path=logs_dir / "eval.log",
            )
            if args.internal_only:
                metrics = mean_internal(eval_csv, k=args.k)
            else:
                run_cmd(
                    [
                        str(py),
                        str(val_script),
                        "--embedding_npz",
                        str(emb_npz),
                        "--enriched_metadata_csv",
                        str(meta_csv),
                        "--k",
                        str(args.k),
                        "--seeds",
                        ",".join(str(s) for s in seeds),
                        "--n_perm",
                        str(args.n_perm),
                        "--output_dir",
                        str(val_dir),
                    ],
                    cwd=base_dir,
                    log_path=logs_dir / "validate.log",
                )

                run_cmd(
                    [
                        str(py),
                        str(confound_script),
                        "--embedding_npz",
                        str(emb_npz),
                        "--enriched_metadata_csv",
                        str(meta_csv),
                        "--k",
                        str(args.k),
                        "--seed",
                        "11",
                        "--min_support",
                        "5",
                        "--fdr_q",
                        "0.05",
                        "--output_md",
                        str(conf_md),
                    ],
                    cwd=base_dir,
                    log_path=logs_dir / "confound.log",
                )

                metrics = collect_metrics(
                    eval_csv=eval_csv,
                    val_dir=val_dir,
                    confound_md=conf_md,
                    emb_npz=emb_npz,
                    seeds=seeds,
                    k=args.k,
                    id_to_desc=id_to_desc,
                )
            subset_metrics[subset] = metrics

            score = internal_score(metrics, baseline[subset])
            subset_scores[subset] = score

            if args.internal_only:
                subset_valid[subset] = True
            else:
                ok, viols = check_constraints(
                    candidate=metrics,
                    baseline=baseline[subset],
                    rel=args.constraint_rel,
                    abs_floor=args.constraint_abs,
                )
                subset_valid[subset] = ok
                for v in viols:
                    violation_rows.append(
                        {
                            "candidate_id": cand_id,
                            "subset": subset,
                            **v,
                        }
                    )

        valid = all(subset_valid.get(s, False) for s in subsets)
        score_total = float(np.mean([subset_scores[s] for s in subsets]))
        row: dict[str, object] = {
            "candidate_id": cand_id,
            "valid": valid,
            "score_total": score_total,
            "config_json": json.dumps(cfg, sort_keys=True),
            **{f"score_{s}": subset_scores[s] for s in subsets},
            **{f"valid_{s}": subset_valid[s] for s in subsets},
        }
        for subset in subsets:
            for k_name, k_val in subset_metrics[subset].items():
                row[f"{subset}_{k_name}"] = k_val
        score_rows.append(row)

    score_df = pd.DataFrame(score_rows).sort_values(
        by=["valid", "score_total"],
        ascending=[False, False],
    )
    score_csv = out_dir / "candidate_scores.csv"
    score_df.to_csv(score_csv, index=False)

    viol_df = pd.DataFrame(violation_rows)
    viol_csv = out_dir / "constraint_violations.csv"
    if viol_df.empty:
        viol_df = pd.DataFrame(columns=["candidate_id", "subset", "metric", "reason", "base", "candidate", "delta", "tol"])
    viol_df.to_csv(viol_csv, index=False)

    if not score_df.empty:
        best = score_df.iloc[0].to_dict()
        best_json = out_dir / "best_config.json"
        best_json.write_text(
            json.dumps(
                {
                    "candidate_id": best["candidate_id"],
                    "valid": bool(best["valid"]),
                    "score_total": float(best["score_total"]),
                    "config": json.loads(str(best["config_json"])),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        best_json = out_dir / "best_config.json"
        best_json.write_text("{}", encoding="utf-8")

    summary_lines = [
        "# gnn_structlite Tuning Summary",
        "",
        f"- output_dir: `{out_dir}`",
        f"- candidates: {len(score_rows)}",
        f"- valid_candidates: {int(score_df['valid'].sum()) if not score_df.empty else 0}",
        f"- score_csv: `{score_csv}`",
        f"- violations_csv: `{viol_csv}`",
        f"- best_config: `{best_json}`",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
