import argparse
import os
import pickle
import random
from collections import deque

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def load_data(processed_dir):
    npz = os.path.join(processed_dir, "hetionet_v1_baseline.npz")
    pkl = os.path.join(processed_dir, "hetionet_v1_baseline_maps.pkl")
    if not (os.path.isfile(npz) and os.path.isfile(pkl)):
        raise FileNotFoundError(f"Brak {npz} lub {pkl} — odpal prepare_hetionet.py")
    z = np.load(npz)
    with open(pkl, "rb") as f:
        maps = pickle.load(f)
    return z["edge_index"], z["edge_type"], z["node_kind"], maps


def csr(edge_index, edge_type, undirected=True):
    s = edge_index[0].astype(np.int64, copy=False)
    d = edge_index[1].astype(np.int64, copy=False)
    t = edge_type.astype(np.int64, copy=False)
    if undirected:
        s = np.concatenate([s, d])
        d = np.concatenate([d, s])
        t = np.concatenate([t, t])
    order = np.argsort(s, kind="mergesort")
    s = s[order]
    d = d[order]
    t = t[order]
    n = int(max(s.max(), d.max())) + 1
    counts = np.bincount(s, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts)
    return indptr, d, t, n


def bfs_nodes(indptr, dst, seeds, hops, max_nodes):
    seen = set(seeds)
    q = deque((x, 0) for x in seeds)
    while q:
        u, h = q.popleft()
        if h >= hops:
            continue
        lo, hi = int(indptr[u]), int(indptr[u + 1])
        for j in range(lo, hi):
            v = int(dst[j])
            if v in seen:
                continue
            if len(seen) >= max_nodes:
                return seen
            seen.add(v)
            q.append((v, h + 1))
    return seen


def edges_inside(edge_index, edge_type, nodes, int_to_meta):
    arr = np.fromiter(nodes, dtype=np.int64, count=len(nodes))
    s, d = edge_index[0], edge_index[1]
    mask = np.isin(s, arr) & np.isin(d, arr)
    out = []
    for i in np.nonzero(mask)[0]:
        out.append((int(s[i]), int(d[i]), int_to_meta[int(edge_type[i])]))
    return out


def label(i, idx_to_id, node_kind, int_to_kind):
    raw = idx_to_id.get(i, str(i))
    k = int_to_kind.get(int(node_kind[i]), "?")
    if "::" in raw:
        tail = raw.split("::", 1)[-1]
        tail = tail[:16] + "…" if len(tail) > 18 else tail
        return f"{k[:3]}\n{tail}"
    raw = raw[:20] + "…" if len(raw) > 22 else raw
    return f"{k[:3]}\n{raw}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--processed-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "processed"),
    )
    ap.add_argument("--output", default="figures/hetionet_subgraph.png")
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--max-nodes", type=int, default=90)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    ddir = os.path.abspath(args.processed_dir)
    edge_index, edge_type, node_kind, maps = load_data(ddir)
    int_to_meta = maps["int_to_metaedge"]
    int_to_kind = maps["int_to_kind"]
    idx_to_id = maps["idx_to_id"]
    meta_to_int = maps["metaedge_to_int"]

    if "CtD" not in meta_to_int:
        raise KeyError("Brak CtD — zrób pełne prepare (bez --sample).")
    ctd = meta_to_int["CtD"]
    cands = np.where(edge_type == ctd)[0]
    if len(cands) == 0:
        raise RuntimeError("Brak krawędzi CtD.")

    ei = int(np.random.choice(cands))
    s0, d0 = int(edge_index[0, ei]), int(edge_index[1, ei])
    print(f"Seed CtD: {idx_to_id[s0]} -> {idx_to_id[d0]} (#{ei})")

    indptr, dst, _t, _n = csr(edge_index, edge_type, True)
    nodes = bfs_nodes(indptr, dst, [s0, d0], args.hops, args.max_nodes)
    nodes.update([s0, d0])
    sub_edges = edges_inside(edge_index, edge_type, nodes, int_to_meta)

    G = nx.MultiGraph()
    G.add_nodes_from(nodes)
    for u, v, meta in sub_edges:
        G.add_edge(u, v, rel=meta)

    plt.figure(figsize=(14, 10))
    k = 0.55 / max(len(nodes) ** 0.5, 1)
    pos = nx.spring_layout(G, seed=args.seed, k=k, iterations=60)

    kinds_here = {int(node_kind[n]) for n in nodes}
    cmap = mpl.colormaps["tab20"].resampled(20)
    colors = [cmap(int(node_kind[n]) % 20) for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=380, alpha=0.92)
    nx.draw_networkx_edges(G, pos, alpha=0.35, width=0.8)
    nx.draw_networkx_labels(
        G, pos, {n: label(n, idx_to_id, node_kind, int_to_kind) for n in G.nodes()}, font_size=5.5
    )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cmap(k % 20),
            markersize=8,
            label=int_to_kind.get(k, str(k)),
        )
        for k in sorted(kinds_here)
    ]
    plt.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9, title="Typ węzła")

    a, b = idx_to_id[s0], idx_to_id[d0]
    line = f"{a[:40]}… → {b[:40]}…" if len(a) > 40 or len(b) > 40 else f"{a} → {b}"
    plt.title(f"Hetionet, BFS {args.hops} hop, ≤{args.max_nodes} węzłów, seed CtD\n{line}", fontsize=10)
    plt.axis("off")
    plt.tight_layout()

    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"Zapisano: {out} ({G.number_of_nodes()} węzłów, {G.number_of_edges()} krawędzi)")


if __name__ == "__main__":
    main()
