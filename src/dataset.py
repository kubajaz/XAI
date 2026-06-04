"""Hetionet (.npz + .pkl) → HeteroData."""

from __future__ import annotations

import os
import pickle
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.data import HeteroData

NPZ = "hetionet_v1_baseline.npz"
PKL = "hetionet_v1_baseline_maps.pkl"


def get_hetionet_data(processed_dir: str) -> HeteroData:
    processed_dir = os.path.abspath(processed_dir)
    npz_path = os.path.join(processed_dir, NPZ)
    pkl_path = os.path.join(processed_dir, PKL)
    if not os.path.isfile(npz_path) or not os.path.isfile(pkl_path):
        raise FileNotFoundError(
            f"Brak danych w {processed_dir} — uruchom: python scripts/prepare_hetionet.py"
        )

    z = np.load(npz_path)
    with open(pkl_path, "rb") as f:
        maps = pickle.load(f)

    edge_index = z["edge_index"]
    edge_type = z["edge_type"]
    node_kind = z["node_kind"]
    kind_to_int = maps["kind_to_int"]
    int_to_metaedge = maps["int_to_metaedge"]

    global_to_local: dict[int, tuple[str, int]] = {}
    for kind_str in sorted(kind_to_int.keys()):
        k = kind_to_int[kind_str]
        for local_i, g in enumerate(np.where(node_kind == k)[0].tolist()):
            global_to_local[int(g)] = (kind_str, local_i)

    buckets: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    for e in range(edge_index.shape[1]):
        s, d = int(edge_index[0, e]), int(edge_index[1, e])
        rel = int_to_metaedge[int(edge_type[e])]
        sk, ls = global_to_local[s]
        dk, ld = global_to_local[d]
        buckets[(sk, rel, dk)].append((ls, ld))

    data = HeteroData()
    for kind_str in sorted(kind_to_int.keys()):
        data[kind_str].num_nodes = int(np.sum(node_kind == kind_to_int[kind_str]))

    for (src, rel, dst), pairs in sorted(buckets.items()):
        s, t = zip(*pairs)
        data[src, rel, dst].edge_index = torch.tensor([s, t], dtype=torch.long)

    return data
