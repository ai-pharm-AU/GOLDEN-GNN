import argparse
import csv
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.data_loader import DataLoader


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return x / norms


def transform_weights(w: np.ndarray, method: str) -> np.ndarray:
    m = method.lower()
    if m == "none":
        return w
    if m == "log1p":
        return np.log1p(np.maximum(w, 0.0))
    if m == "sqrt":
        return np.sqrt(np.maximum(w, 0.0))
    raise ValueError(f"Unknown edge_weight_transform: {method}")


def build_graph_from_edges(
    edge_file: str,
    weight_column: str,
    id_to_idx: dict[str, int],
    edge_weight_transform: str,
) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(edge_file, delimiter="\t")
    required = {"GS_A_ID", "GS_B_ID", weight_column}
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in edge_file: {missing}")

    a = df["GS_A_ID"].astype(str).to_numpy()
    b = df["GS_B_ID"].astype(str).to_numpy()
    w = df[weight_column].astype(float).to_numpy()

    rows = []
    for u_id, v_id, weight in zip(a, b, w):
        if u_id not in id_to_idx or v_id not in id_to_idx:
            continue
        u = id_to_idx[u_id]
        v = id_to_idx[v_id]
        if u == v:
            continue
        if u > v:
            u, v = v, u
        rows.append((u, v, float(weight)))

    if not rows:
        raise ValueError("No aligned edges remained after filtering by feature IDs")

    edge_df = pd.DataFrame(rows, columns=["u", "v", "w"])
    edge_df = edge_df.groupby(["u", "v"], as_index=False)["w"].max()
    edge_df["w"] = transform_weights(edge_df["w"].to_numpy(), edge_weight_transform)

    edge_index = edge_df[["u", "v"]].to_numpy(dtype=np.int64)
    edge_weight = edge_df["w"].to_numpy(dtype=np.float32)
    return edge_index, edge_weight


def build_normalized_adjacency(
    num_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    u = edge_index[:, 0]
    v = edge_index[:, 1]

    row = np.concatenate([u, v, np.arange(num_nodes, dtype=np.int64)])
    col = np.concatenate([v, u, np.arange(num_nodes, dtype=np.int64)])
    val = np.concatenate(
        [edge_weight, edge_weight, np.ones(num_nodes, dtype=np.float32)]
    )

    row_t = torch.from_numpy(row).to(device)
    col_t = torch.from_numpy(col).to(device)
    val_t = torch.from_numpy(val).to(device)

    deg = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    deg.index_add_(0, row_t, val_t)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1e-12), -0.5)
    norm_val = deg_inv_sqrt[row_t] * val_t * deg_inv_sqrt[col_t]

    idx = torch.stack([row_t, col_t], dim=0)
    adj = torch.sparse_coo_tensor(idx, norm_val, (num_nodes, num_nodes), device=device)
    adj = adj.coalesce()
    return adj


class GraphSageEncoder(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.dropout = dropout
        self.layers = nn.ModuleList()

        in_dim = input_dim
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(in_dim * 2, hidden_dim))
            in_dim = hidden_dim
        self.out = nn.Linear(in_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            neigh = torch.sparse.mm(adj, h)
            h = torch.cat([h, neigh], dim=1)
            h = layer(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        neigh = torch.sparse.mm(adj, h)
        h = torch.cat([h, neigh], dim=1)
        z = self.out(h)
        return z


def encode_pair_key(u: np.ndarray, v: np.ndarray, n: int) -> np.ndarray:
    return u.astype(np.int64) * np.int64(n) + v.astype(np.int64)


def sample_negative_edges(
    num_nodes: int,
    n_samples: int,
    edge_key_set: set[int],
    rng: np.random.Generator,
) -> np.ndarray:
    out_u = np.empty(n_samples, dtype=np.int64)
    out_v = np.empty(n_samples, dtype=np.int64)
    filled = 0

    while filled < n_samples:
        batch = max(4096, (n_samples - filled) * 2)
        u = rng.integers(0, num_nodes, size=batch, endpoint=False)
        v = rng.integers(0, num_nodes, size=batch, endpoint=False)
        mask = u != v
        if not np.any(mask):
            continue
        u = u[mask]
        v = v[mask]
        lo = np.minimum(u, v)
        hi = np.maximum(u, v)
        keys = encode_pair_key(lo, hi, num_nodes)

        keep_mask = np.array([int(k not in edge_key_set) for k in keys], dtype=bool)
        if not np.any(keep_mask):
            continue

        lo = lo[keep_mask]
        hi = hi[keep_mask]
        take = min(len(lo), n_samples - filled)
        out_u[filled : filled + take] = lo[:take]
        out_v[filled : filled + take] = hi[:take]
        filled += take

    return np.stack([out_u, out_v], axis=1)


def dot_decoder(z: torch.Tensor, edge_idx: torch.Tensor) -> torch.Tensor:
    src = z[edge_idx[:, 0]]
    dst = z[edge_idx[:, 1]]
    return (src * dst).sum(dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unsupervised GraphSAGE-style embedding pilot via edge reconstruction."
    )
    parser.add_argument("--edge_file", required=True)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--weight_column", default="NLOGPMF")
    parser.add_argument(
        "--edge_weight_transform", default="log1p", choices=["none", "log1p", "sqrt"]
    )
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pos_samples_per_epoch", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--metrics_csv", required=True)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ids, emb = DataLoader.load_embeddings(args.feature_npz)
    ids = [str(x) for x in ids]
    x_np = l2_normalize_rows(np.asarray(emb, dtype=np.float32))
    num_nodes = x_np.shape[0]
    input_dim = x_np.shape[1]
    id_to_idx = {gid: i for i, gid in enumerate(ids)}

    edge_index, edge_weight = build_graph_from_edges(
        args.edge_file,
        args.weight_column,
        id_to_idx,
        args.edge_weight_transform,
    )

    adj = build_normalized_adjacency(num_nodes, edge_index, edge_weight, device)
    x = torch.from_numpy(x_np).to(device)

    model = GraphSageEncoder(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    u = edge_index[:, 0]
    v = edge_index[:, 1]
    edge_keys = set(encode_pair_key(u, v, num_nodes).tolist())

    out_metrics_dir = os.path.dirname(args.metrics_csv)
    if out_metrics_dir:
        os.makedirs(out_metrics_dir, exist_ok=True)
    out_emb_dir = os.path.dirname(args.output_npz)
    if out_emb_dir:
        os.makedirs(out_emb_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    total_edges = edge_index.shape[0]

    print(
        f"Device={device} nodes={num_nodes} input_dim={input_dim} edges={total_edges} "
        f"hidden_dim={args.hidden_dim} layers={args.num_layers}",
        flush=True,
    )

    with open(args.metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "loss", "pos_mean_logit", "neg_mean_logit", "pos_samples"]
        )

        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad()

            z = model(x, adj)

            n_pos = min(args.pos_samples_per_epoch, total_edges)
            pos_idx = rng.choice(total_edges, size=n_pos, replace=False)
            pos_edges = edge_index[pos_idx]
            neg_edges = sample_negative_edges(num_nodes, n_pos, edge_keys, rng)

            pos_t = torch.from_numpy(pos_edges).to(device)
            neg_t = torch.from_numpy(neg_edges).to(device)

            pos_logits = dot_decoder(z, pos_t)
            neg_logits = dot_decoder(z, neg_t)

            pos_loss = F.binary_cross_entropy_with_logits(
                pos_logits, torch.ones_like(pos_logits)
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_logits, torch.zeros_like(neg_logits)
            )
            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()

            pos_mean = float(pos_logits.detach().mean().item())
            neg_mean = float(neg_logits.detach().mean().item())
            writer.writerow([epoch, float(loss.item()), pos_mean, neg_mean, n_pos])
            f.flush()

            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(
                    f"epoch={epoch:03d} loss={loss.item():.4f} "
                    f"pos_logit={pos_mean:.4f} neg_logit={neg_mean:.4f}",
                    flush=True,
                )

    model.eval()
    with torch.no_grad():
        z_final = model(x, adj).cpu().numpy().astype(np.float32)
    np.savez(args.output_npz, ID=np.asarray(ids, dtype=object), embeddings=z_final)

    norms = np.linalg.norm(z_final, axis=1)
    print(
        f"Saved embeddings: {args.output_npz} | shape={z_final.shape} "
        f"norm_mean={float(norms.mean()):.4f} norm_std={float(norms.std()):.4f}",
        flush=True,
    )
    print(f"Saved training curve: {args.metrics_csv}", flush=True)


if __name__ == "__main__":
    main()
