#!/usr/bin/env python3
"""Run feature ablation pipeline for subset 085/090.

Stages:
- build: generate feature recipe NPZ files
- train: run gnn_linkpred_holdout.py for each recipe
- eval: run evaluate_single_embedding.py on trained GNN embeddings
- validate: run validate_cluster_meta_enrichment.py
- confound: run analyze_mesh_vs_gse_confound.py
- summary: aggregate outputs into leaderboard/report
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path


def parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def default_paths(base_dir: Path, subset: str) -> dict[str, Path]:
    if subset == "090":
        return {
            "edge_file": base_dir / "subset_85_90_20260213/csv/edges_with_mtype_threshold_090.txt",
            "meta_csv": base_dir / "reports/bio_meta_cache_090/enriched_gs_metadata.csv",
            "gs_file": base_dir / "task_20260217_alpha05_subset090/gs_ids_subset_090_clean.txt",
        }
    if subset == "085":
        return {
            "edge_file": base_dir / "subset_85_90_20260213/csv/edges_with_mtype_threshold_085.txt",
            "meta_csv": base_dir
            / "reports/bio_meta_cache_085/bio_meta_cache_085/enriched_gs_metadata.csv",
            "gs_file": Path(""),
        }
    raise ValueError(f"Unsupported subset: {subset}")


def human_only_paths(task_dir: Path, subset: str) -> dict[str, Path]:
    inputs_dir = task_dir / "human_only" / f"subset{subset}" / "inputs"
    return {
        "inputs_dir": inputs_dir,
        "go_npz": inputs_dir / f"go_embeddings_subset_{subset}_human_l2.npz",
        "pager_npz": inputs_dir / f"pager_embeddings_subset_{subset}_human_l2.npz",
        "meta_csv": inputs_dir / f"enriched_gs_metadata_subset{subset}_human.csv",
        "edge_file": inputs_dir / f"edges_with_mtype_threshold_{subset}_human.txt",
        "gs_file": inputs_dir / f"human_ids_subset{subset}.txt",
    }


def run_cmd(cmd: list[str], log_file: Path, cwd: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
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
            lf.write(line)
        ret = proc.wait()
        lf.write(f"\n[exit_code] {ret}\n")
    if ret != 0:
        raise RuntimeError(f"Command failed ({ret}): {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run feature ablation pipeline for one subset.")
    parser.add_argument("--subset", required=True, choices=["085", "090"])
    parser.add_argument("--python_bin", default="python")
    parser.add_argument(
        "--base_dir",
        default="/home/zzz0054/GoldenF/work_alpha_gnn_20260212",
        help="Base folder for work_alpha_gnn_20260212",
    )
    parser.add_argument(
        "--task_dir",
        default="/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation",
    )
    parser.add_argument(
        "--recipes",
        default=(
            "text_only,text_meta,text_structlite,text_meta_structlite,"
            "node2vec_only,text_node2vec,shuffle_control,random_control"
        ),
    )
    parser.add_argument(
        "--stages",
        default="build,train,eval,validate,confound,summary",
        help="Comma-separated stages",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--seeds", default="11,12,13,14,15")
    parser.add_argument("--human_only", action="store_true", help="Use Homo sapiens-only nodes/edges/metadata")
    parser.add_argument(
        "--master_meta_csv",
        default="/home/zzz0054/GoldenF/data/mcxqrhgm61ztvegwj8ltzbcvv1h06vni.csv",
    )
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--train_seeds", default="11,12,13,14,15,16,17,18,19,20")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--train_multiseed",
        action="store_true",
        help="Train link prediction across multiple random seeds (mean±std + representative embedding).",
    )
    group.add_argument(
        "--train_single_seed",
        action="store_true",
        help="Force single-seed link prediction training (legacy behavior).",
    )
    parser.add_argument("--k_list", default="10-50 step 10")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--pager_dim", type=int, default=64)
    parser.add_argument("--struct_features", default="basic", choices=["basic", "basic_plus", "centrality_plus"])
    parser.add_argument("--struct_norm", default="zscore", choices=["zscore", "robust", "rankgauss"])
    parser.add_argument("--go_scale", type=float, default=1.0)
    parser.add_argument("--meta_scale", type=float, default=1.0)
    parser.add_argument("--struct_scale", type=float, default=1.0)
    parser.add_argument("--weight_column", default="NLOGPMF")
    parser.add_argument("--edge_weight_transform", default="log1p", choices=["none", "log1p", "sqrt"])
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="Short run for quick pipeline validation")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    task_dir = Path(args.task_dir).resolve()
    paths = default_paths(base_dir, args.subset)

    recipes = parse_csv_list(args.recipes)
    stages = set(parse_csv_list(args.stages))

    # Output root: keep legacy layout by default; write human-only results to a separate tree.
    task_root = task_dir
    if args.human_only:
        task_root = task_dir / "human_only" / f"subset{args.subset}"

    feature_dir = task_root / "features"
    csv_dir = task_root / "csv"
    log_dir = task_root / "logs"
    report_dir = task_root / "reports"
    config_dir = task_root / "configs"
    for d in [feature_dir, csv_dir, log_dir, report_dir, config_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Ensure human-only inputs are present (or refresh on build stage).
    if args.human_only:
        build_inputs_script = task_dir / "scripts/build_human_only_inputs.py"
        h = human_only_paths(task_dir, args.subset)
        required_inputs = [h["go_npz"], h["pager_npz"], h["meta_csv"], h["edge_file"], h["gs_file"]]
        missing = [p for p in required_inputs if not p.exists()]
        if ("build" in stages) or missing:
            cmd = [
                args.python_bin,
                str(build_inputs_script),
                "--subset",
                args.subset,
                "--base_dir",
                str(base_dir),
                "--task_dir",
                str(task_dir),
                "--master_meta_csv",
                str(Path(args.master_meta_csv).resolve()),
            ]
            run_cmd(cmd, log_dir / "build_human_only_inputs.log", cwd=base_dir)
        paths = human_only_paths(task_dir, args.subset)

    run_ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_config_path = config_dir / f"run_config_{run_ts}.json"
    train_multiseed = args.train_multiseed or (args.human_only and not args.train_single_seed)
    run_config = {
        "subset": args.subset,
        "recipes": recipes,
        "stages": sorted(stages),
        "smoke": args.smoke,
        "device": args.device,
        "epochs": args.epochs,
        "eval_every": args.eval_every,
        "seeds": args.seeds,
        "human_only": args.human_only,
        "master_meta_csv": args.master_meta_csv,
        "test_ratio": args.test_ratio,
        "train_seeds": args.train_seeds,
        "train_multiseed": train_multiseed,
        "k_list": args.k_list,
        "k": args.k,
        "pager_dim": args.pager_dim,
        "struct_features": args.struct_features,
        "struct_norm": args.struct_norm,
        "go_scale": args.go_scale,
        "meta_scale": args.meta_scale,
        "struct_scale": args.struct_scale,
        "weight_column": args.weight_column,
        "edge_weight_transform": args.edge_weight_transform,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "paths": {
            "base_dir": str(base_dir),
            "task_dir": str(task_dir),
            "task_root": str(task_root),
            "edge_file": str(paths["edge_file"]),
            "meta_csv": str(paths["meta_csv"]),
            "gs_file": str(paths["gs_file"]),
            "go_npz": str(paths.get("go_npz", "")),
            "pager_npz": str(paths.get("pager_npz", "")),
            "inputs_dir": str(paths.get("inputs_dir", "")),
        },
    }
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    run_manifest = report_dir / "run_manifest.md"
    with open(run_manifest, "a", encoding="utf-8") as mf:
        mf.write(f"\n## Run {dt.datetime.now().isoformat(timespec='seconds')}\n")
        mf.write(f"- subset: {args.subset}\n")
        mf.write(f"- stages: {','.join(sorted(stages))}\n")
        mf.write(f"- recipes: {','.join(recipes)}\n")
        mf.write(f"- smoke: {args.smoke}\n")
        mf.write(f"- run_config: `{run_config_path}`\n")

    builder_script = task_dir / "scripts/build_node_features_from_meta.py"
    summary_script = task_dir / "scripts/summarize_feature_ablation.py"
    multiseed_script = task_dir / "scripts/run_linkpred_multiseed.py"
    gnn_script = base_dir / "scripts/gnn_linkpred_holdout.py"
    eval_script = base_dir / "scripts/evaluate_single_embedding.py"
    validate_script = base_dir / "scripts/validate_cluster_meta_enrichment.py"
    confound_script = base_dir / "scripts/analyze_mesh_vs_gse_confound.py"

    epochs = 20 if args.smoke else args.epochs
    eval_every = 5 if args.smoke else args.eval_every
    n_perm = 50 if args.smoke else 200
    pos_samples = 10000 if args.smoke else 50000
    seeds = "11,12" if args.smoke else args.seeds
    train_seeds = "11,12" if args.smoke else args.train_seeds

    if "build" in stages:
        cmd = [
            args.python_bin,
            str(builder_script),
            "--subset",
            args.subset,
            "--base_dir",
            str(base_dir),
            "--edge_file",
            str(paths["edge_file"]),
            "--meta_csv",
            str(paths["meta_csv"]),
            "--recipes",
            ",".join(recipes),
            "--pager_dim",
            str(args.pager_dim),
            "--struct_features",
            args.struct_features,
            "--struct_norm",
            args.struct_norm,
            "--go_scale",
            str(args.go_scale),
            "--meta_scale",
            str(args.meta_scale),
            "--struct_scale",
            str(args.struct_scale),
            "--weight_column",
            args.weight_column,
        ]
        if args.human_only:
            cmd.extend(["--go_npz", str(paths["go_npz"])])
            cmd.extend(["--pager_npz", str(paths["pager_npz"])])
        cmd.extend(
            [
            "--output_dir",
            str(feature_dir),
            ]
        )
        run_cmd(cmd, log_dir / "build_features.log", cwd=base_dir)

    for recipe in recipes:
        input_feature_npz = feature_dir / f"{recipe}_subset{args.subset}.npz"
        trained_npz = feature_dir / f"{recipe}_gnn_holdout_subset{args.subset}.npz"
        linkpred_csv = csv_dir / f"{recipe}_linkpred_subset{args.subset}.csv"
        eval_csv = csv_dir / f"{recipe}_eval_subset{args.subset}.csv"
        val_dir = report_dir / f"{recipe}_validation_subset{args.subset}"
        confound_md = report_dir / f"{recipe}_confound_subset{args.subset}.md"

        if "train" in stages:
            if train_multiseed:
                tag = f"{recipe}_subset{args.subset}"
                cmd = [
                    args.python_bin,
                    str(multiseed_script),
                    "--python_bin",
                    args.python_bin,
                    "--gnn_script",
                    str(gnn_script),
                    "--cwd",
                    str(base_dir),
                    "--edge_file",
                    str(paths["edge_file"]),
                    "--feature_npz",
                    str(input_feature_npz),
                    "--weight_column",
                    args.weight_column,
                    "--edge_weight_transform",
                    args.edge_weight_transform,
                    "--hidden_dim",
                    str(args.hidden_dim),
                    "--num_layers",
                    str(args.num_layers),
                    "--dropout",
                    str(args.dropout),
                    "--lr",
                    str(args.lr),
                    "--weight_decay",
                    str(args.weight_decay),
                    "--epochs",
                    str(epochs),
                    "--eval_every",
                    str(eval_every),
                    "--patience",
                    str(args.patience),
                    "--val_ratio",
                    "0.1",
                    "--test_ratio",
                    str(args.test_ratio),
                    "--pos_samples_per_epoch",
                    str(pos_samples),
                    "--device",
                    args.device,
                    "--seeds",
                    train_seeds,
                    "--split_seed_mode",
                    "same_as_train_seed",
                    "--out_dir",
                    str(csv_dir),
                    "--tag",
                    tag,
                ]
                run_cmd(cmd, log_dir / f"{recipe}_train_multiseed.log", cwd=base_dir)

                selected_curve = csv_dir / f"selected_curve_{tag}.csv"
                selected_emb = csv_dir / f"selected_embedding_{tag}.npz"
                if not selected_curve.exists():
                    raise FileNotFoundError(f"Missing selected_curve: {selected_curve}")
                if not selected_emb.exists():
                    raise FileNotFoundError(f"Missing selected_embedding: {selected_emb}")
                shutil.copyfile(selected_curve, linkpred_csv)
                shutil.copyfile(selected_emb, trained_npz)
            else:
                cmd = [
                    args.python_bin,
                    str(gnn_script),
                    "--edge_file",
                    str(paths["edge_file"]),
                    "--feature_npz",
                    str(input_feature_npz),
                    "--weight_column",
                    args.weight_column,
                    "--edge_weight_transform",
                    args.edge_weight_transform,
                    "--hidden_dim",
                    str(args.hidden_dim),
                    "--num_layers",
                    str(args.num_layers),
                    "--dropout",
                    str(args.dropout),
                    "--lr",
                    str(args.lr),
                    "--weight_decay",
                    str(args.weight_decay),
                    "--patience",
                    str(args.patience),
                    "--epochs",
                    str(epochs),
                    "--eval_every",
                    str(eval_every),
                    "--val_ratio",
                    "0.1",
                    "--test_ratio",
                    str(args.test_ratio),
                    "--pos_samples_per_epoch",
                    str(pos_samples),
                    "--seed",
                    "42",
                    "--split_seed",
                    "42",
                    "--device",
                    args.device,
                    "--output_csv",
                    str(linkpred_csv),
                    "--output_npz",
                    str(trained_npz),
                ]
                run_cmd(cmd, log_dir / f"{recipe}_train.log", cwd=base_dir)

        if "eval" in stages:
            emb_for_eval = trained_npz if trained_npz.exists() else input_feature_npz
            cmd = [
                args.python_bin,
                str(eval_script),
                "--embedding_file",
                str(emb_for_eval),
                "--algos",
                "kmeans",
                "--k_list",
                args.k_list,
                "--seeds",
                seeds,
                "--output_csv",
                str(eval_csv),
            ]
            gs_file = Path(paths.get("gs_file", ""))
            if str(gs_file) and gs_file.is_file():
                cmd.extend(["--gs_file", str(gs_file)])
            run_cmd(cmd, log_dir / f"{recipe}_eval.log", cwd=base_dir)

        if "validate" in stages:
            emb_for_val = trained_npz if trained_npz.exists() else input_feature_npz
            val_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                args.python_bin,
                str(validate_script),
                "--embedding_npz",
                str(emb_for_val),
                "--enriched_metadata_csv",
                str(paths["meta_csv"]),
                "--k",
                str(args.k),
                "--seeds",
                seeds,
                "--n_perm",
                str(n_perm),
                "--output_dir",
                str(val_dir),
            ]
            run_cmd(cmd, log_dir / f"{recipe}_validate.log", cwd=base_dir)

        if "confound" in stages:
            emb_for_conf = trained_npz if trained_npz.exists() else input_feature_npz
            cmd = [
                args.python_bin,
                str(confound_script),
                "--embedding_npz",
                str(emb_for_conf),
                "--enriched_metadata_csv",
                str(paths["meta_csv"]),
                "--k",
                str(args.k),
                "--seed",
                "11",
                "--min_support",
                "5",
                "--fdr_q",
                "0.05",
                "--output_md",
                str(confound_md),
            ]
            run_cmd(cmd, log_dir / f"{recipe}_confound.log", cwd=base_dir)

    if "summary" in stages:
        cmd = [
            args.python_bin,
            str(summary_script),
            "--subset",
            args.subset,
            "--task_dir",
            str(task_root),
            "--recipes",
            ",".join(recipes),
            "--k",
            str(args.k),
        ]
        run_cmd(cmd, log_dir / "summary.log", cwd=base_dir)

    print(f"Done. Outputs under: {task_root}")


if __name__ == "__main__":
    main()
