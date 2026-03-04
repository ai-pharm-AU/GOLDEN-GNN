import argparse
import csv
import math
import os
import pathlib
import sys

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.data_loader import DataLoader


def parse_k_list(raw: str) -> list[int]:
    text = raw.strip().lower()
    if "," in text:
        return [int(part) for part in text.split(",") if part.strip()]
    text = text.replace("step", ":").replace(" ", "")
    if "-" in text:
        if ":" in text:
            range_part, step_part = text.split(":", 1)
            step = int(step_part)
        else:
            range_part = text
            step = 1
        start_s, end_s = range_part.split("-", 1)
        return list(range(int(start_s), int(end_s) + 1, step))
    return [int(text)]


def parse_int_values(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return x / norms


def load_gs_ids(gs_file: str) -> list[str]:
    with open(gs_file, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def cluster_labels(algo: str, x: np.ndarray, k: int, seed: int) -> np.ndarray:
    name = algo.lower()
    if name == "kmeans":
        return KMeans(n_clusters=k, random_state=seed, n_init="auto").fit_predict(x)
    if name == "agglomerative":
        return AgglomerativeClustering(n_clusters=k).fit_predict(x)
    if name == "kmedoids":
        from sklearn_extra.cluster import KMedoids  # type: ignore

        return KMedoids(
            n_clusters=k, random_state=seed, metric="euclidean"
        ).fit_predict(x)
    raise ValueError(f"Unsupported algo: {algo}")


def compute_metrics(
    x: np.ndarray, labels: np.ndarray, sample_size: int | None, seed: int
) -> dict[str, float]:
    if sample_size is not None and x.shape[0] > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x.shape[0], size=sample_size, replace=False)
        x_eval = x[idx]
        y_eval = labels[idx]
    else:
        x_eval = x
        y_eval = labels

    if len(np.unique(y_eval)) < 2:
        return {
            "silhouette_euclidean": math.nan,
            "silhouette_cosine": math.nan,
            "davies_bouldin": math.nan,
            "calinski_harabasz": math.nan,
        }
    x_cos = l2_normalize_rows(x_eval)
    return {
        "silhouette_euclidean": float(
            silhouette_score(x_eval, y_eval, metric="euclidean")
        ),
        "silhouette_cosine": float(silhouette_score(x_cos, y_eval, metric="cosine")),
        "davies_bouldin": float(davies_bouldin_score(x_eval, y_eval)),
        "calinski_harabasz": float(calinski_harabasz_score(x_eval, y_eval)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one embedding file with clustering metrics."
    )
    parser.add_argument("--embedding_file", required=True)
    parser.add_argument("--gs_file", default="")
    parser.add_argument("--algos", default="kmeans")
    parser.add_argument("--k_list", default="10-50 step 10")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--sample_size", type=int, default=0)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    ids, emb = DataLoader.load_embeddings(args.embedding_file)
    ids = [str(x) for x in ids]

    if args.gs_file:
        gs_ids = set(load_gs_ids(args.gs_file))
        keep = [i for i, gid in enumerate(ids) if gid in gs_ids]
        ids = [ids[i] for i in keep]
        emb = emb[keep]

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    k_values = parse_k_list(args.k_list)
    seeds = parse_int_values(args.seeds)
    sample_size = args.sample_size if args.sample_size > 0 else None

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    header = [
        "embedding_file",
        "algo",
        "k",
        "seed",
        "n_samples",
        "dim",
        "silhouette_euclidean",
        "silhouette_cosine",
        "davies_bouldin",
        "calinski_harabasz",
    ]

    with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for algo in algos:
            for k in k_values:
                for seed in seeds:
                    labels = cluster_labels(algo, emb, k, seed)
                    metrics = compute_metrics(emb, labels, sample_size, seed)
                    writer.writerow(
                        [
                            args.embedding_file,
                            algo,
                            k,
                            seed,
                            emb.shape[0],
                            emb.shape[1],
                            metrics["silhouette_euclidean"],
                            metrics["silhouette_cosine"],
                            metrics["davies_bouldin"],
                            metrics["calinski_harabasz"],
                        ]
                    )
                    print(
                        f"algo={algo} k={k} seed={seed} "
                        f"sil_euc={metrics['silhouette_euclidean']:.4f} "
                        f"sil_cos={metrics['silhouette_cosine']:.4f} "
                        f"db={metrics['davies_bouldin']:.4f} "
                        f"ch={metrics['calinski_harabasz']:.2f}",
                        flush=True,
                    )

    print(f"Wrote: {args.output_csv}")


if __name__ == "__main__":
    main()
