#!/usr/bin/env python3
"""Build human-only (Homo sapiens) inputs for subset 085/090.

Outputs a stable directory structure consumed by the feature ablation pipeline:

  human_only/subset{SUBSET}/inputs/
    - go_embeddings_subset_{SUBSET}_human_l2.npz
    - pager_embeddings_subset_{SUBSET}_human_l2.npz
    - enriched_gs_metadata_subset{SUBSET}_human.csv
    - edges_with_mtype_threshold_{SUBSET}_human.txt
    - human_ids_subset{SUBSET}.txt
    - filter_stats_subset{SUBSET}.md
    - inputs_manifest.json

Rule:
  Keep node if master_meta_csv has ORGANISM == "Homo sapiens" for the GS_ID.
  Keep edge only if both endpoints are kept nodes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _default_paths(base_dir: Path, subset: str) -> dict[str, Path]:
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
        raise ValueError(f"ID count mismatch in {path}: ids={len(ids_out)} rows={emb_out.shape[0]}")
    return ids_out, emb_out


def write_npz(path: Path, ids: list[str], emb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        ID=np.asarray(ids, dtype=object),
        embeddings=np.asarray(emb, dtype=np.float32),
    )


def filter_npz_keep_ids(
    src_npz: Path,
    keep_ids_order: list[str],
    dst_npz: Path,
) -> dict[str, int]:
    ids, emb = load_npz_embeddings(src_npz)
    idx = {gid: i for i, gid in enumerate(ids)}
    out = np.zeros((len(keep_ids_order), emb.shape[1]), dtype=np.float32)
    missing = 0
    for i, gid in enumerate(keep_ids_order):
        j = idx.get(gid)
        if j is None:
            missing += 1
            continue
        out[i] = emb[j]
    write_npz(dst_npz, keep_ids_order, out)
    return {"src_nodes": len(ids), "dst_nodes": len(keep_ids_order), "missing": missing}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build human-only inputs for subset 085/090.")
    parser.add_argument("--subset", required=True, choices=["085", "090"])
    parser.add_argument("--base_dir", default="/home/zzz0054/GoldenF/work_alpha_gnn_20260212")
    parser.add_argument(
        "--task_dir",
        default="/home/zzz0054/GoldenF/work_alpha_gnn_20260212/task_20260219_gnn_feature_ablation",
    )
    parser.add_argument(
        "--master_meta_csv",
        default="/home/zzz0054/GoldenF/data/mcxqrhgm61ztvegwj8ltzbcvv1h06vni.csv",
        help="Master metadata with GS_ID and ORGANISM columns",
    )
    parser.add_argument("--go_npz", default="")
    parser.add_argument("--pager_npz", default="")
    parser.add_argument("--meta_csv", default="")
    parser.add_argument("--edge_file", default="")
    args = parser.parse_args()

    subset = args.subset
    base_dir = Path(args.base_dir).resolve()
    task_dir = Path(args.task_dir).resolve()
    defaults = _default_paths(base_dir, subset=subset)

    go_npz = Path(args.go_npz).resolve() if args.go_npz else defaults["go_npz"]
    pager_npz = Path(args.pager_npz).resolve() if args.pager_npz else defaults["pager_npz"]
    meta_csv = Path(args.meta_csv).resolve() if args.meta_csv else defaults["meta_csv"]
    edge_file = Path(args.edge_file).resolve() if args.edge_file else defaults["edge_file"]
    master_meta_csv = Path(args.master_meta_csv).resolve()

    out_dir = task_dir / "human_only" / f"subset{subset}" / "inputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(master_meta_csv, usecols=["GS_ID", "ORGANISM"])
    master["GS_ID"] = master["GS_ID"].astype(str)
    master["ORGANISM"] = master["ORGANISM"].fillna("").astype(str).str.strip()
    human_master = set(master.loc[master["ORGANISM"] == "Homo sapiens", "GS_ID"].tolist())

    go_ids, _ = load_npz_embeddings(go_npz)
    keep_ids = [gid for gid in go_ids if gid in human_master]
    keep_set = set(keep_ids)

    (out_dir / f"human_ids_subset{subset}.txt").write_text(
        "\n".join(keep_ids) + "\n", encoding="utf-8"
    )

    go_out = out_dir / f"go_embeddings_subset_{subset}_human_l2.npz"
    go_stats = filter_npz_keep_ids(go_npz, keep_ids, go_out)

    pager_out = out_dir / f"pager_embeddings_subset_{subset}_human_l2.npz"
    pager_stats = filter_npz_keep_ids(pager_npz, keep_ids, pager_out)

    meta_df = pd.read_csv(meta_csv)
    if "GS_ID" not in meta_df.columns:
        raise KeyError(f"Missing GS_ID column in metadata: {meta_csv}")
    meta_df["GS_ID"] = meta_df["GS_ID"].astype(str)
    meta_keep = meta_df[meta_df["GS_ID"].isin(keep_set)].copy()
    meta_out = out_dir / f"enriched_gs_metadata_subset{subset}_human.csv"
    meta_keep.to_csv(meta_out, index=False)

    edge_df = pd.read_csv(edge_file, sep="\t")
    required = {"GS_A_ID", "GS_B_ID"}
    missing = [c for c in required if c not in edge_df.columns]
    if missing:
        raise KeyError(f"Missing columns in edge_file: {missing} ({edge_file})")
    edge_df["GS_A_ID"] = edge_df["GS_A_ID"].astype(str)
    edge_df["GS_B_ID"] = edge_df["GS_B_ID"].astype(str)
    edge_keep = edge_df[edge_df["GS_A_ID"].isin(keep_set) & edge_df["GS_B_ID"].isin(keep_set)].copy()
    edge_out = out_dir / f"edges_with_mtype_threshold_{subset}_human.txt"
    edge_keep.to_csv(edge_out, sep="\t", index=False)

    stats_md = out_dir / f"filter_stats_subset{subset}.md"
    stats_md.write_text(
        "\n".join(
            [
                f"# Human-only filter stats (subset {subset})",
                "",
                f"- go_src_nodes: {go_stats['src_nodes']}",
                f"- go_human_nodes: {go_stats['dst_nodes']}",
                f"- pager_src_nodes: {pager_stats['src_nodes']}",
                f"- pager_human_nodes: {pager_stats['dst_nodes']}",
                f"- pager_missing_for_go_human_ids: {pager_stats['missing']}",
                f"- meta_src_rows: {len(meta_df)}",
                f"- meta_human_rows: {len(meta_keep)}",
                f"- edge_total: {len(edge_df)}",
                f"- edge_human_only: {len(edge_keep)}",
                "",
                "## Rule",
                "",
                '- Keep node iff master_meta_csv has ORGANISM == "Homo sapiens".',
                "- Keep edge iff both endpoints kept.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest = {
        "subset": subset,
        "rule": {"organism_equals": "Homo sapiens"},
        "sources": {
            "master_meta_csv": str(master_meta_csv),
            "go_npz": str(go_npz),
            "pager_npz": str(pager_npz),
            "meta_csv": str(meta_csv),
            "edge_file": str(edge_file),
        },
        "outputs": {
            "out_dir": str(out_dir),
            "human_ids": str(out_dir / f"human_ids_subset{subset}.txt"),
            "go_npz": str(go_out),
            "pager_npz": str(pager_out),
            "meta_csv": str(meta_out),
            "edge_file": str(edge_out),
            "filter_stats_md": str(stats_md),
        },
        "counts": {
            "human_ids": len(keep_ids),
            "go_src_nodes": go_stats["src_nodes"],
            "go_human_nodes": go_stats["dst_nodes"],
            "pager_src_nodes": pager_stats["src_nodes"],
            "pager_human_nodes": pager_stats["dst_nodes"],
            "pager_missing_for_go_human_ids": pager_stats["missing"],
            "meta_src_rows": int(len(meta_df)),
            "meta_human_rows": int(len(meta_keep)),
            "edge_total": int(len(edge_df)),
            "edge_human_only": int(len(edge_keep)),
        },
    }

    (out_dir / "inputs_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"[ok] wrote human-only inputs: {out_dir}", flush=True)


if __name__ == "__main__":
    main()

