import argparse
import gzip
import os
import pickle
import urllib.request

import numpy as np

NODES_URL = (
    "https://raw.githubusercontent.com/hetio/hetionet/master/hetnet/tsv/"
    "hetionet-v1.0-nodes.tsv"
)
EDGES_URL = (
    "https://github.com/hetio/hetionet/raw/master/hetnet/tsv/"
    "hetionet-v1.0-edges.sif.gz"
)


def download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isfile(dest):
        print(f"Już jest: {dest}")
        return
    print(f"Pobieram {url}")
    urllib.request.urlretrieve(url, dest)


def load_nodes(path):
    id_to_idx = {}
    idx_to_kind = []
    idx_to_id = {}
    with open(path, encoding="utf-8") as f:
        if not f.readline().lower().startswith("id"):
            raise ValueError("Zły nagłówek w pliku węzłów")
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 3:
                continue
            nid, _, kind = parts[0], parts[1], parts[2]
            i = len(id_to_idx)
            id_to_idx[nid] = i
            idx_to_kind.append(kind)
            idx_to_id[i] = nid
    return id_to_idx, idx_to_kind, idx_to_id


def iter_edges(path, id_to_idx, limit=None):
    n = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        if "source" not in f.readline().lower():
            raise ValueError("Zły nagłówek w pliku krawędzi")
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 3:
                continue
            src, meta, dst = parts[0], parts[1], parts[2]
            if src not in id_to_idx or dst not in id_to_idx:
                continue
            yield id_to_idx[src], id_to_idx[dst], meta
            n += 1
            if limit is not None and n >= limit:
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data"),
    )
    ap.add_argument("--sample", type=int, default=None, metavar="N")
    args = ap.parse_args()

    data = os.path.abspath(args.data_dir)
    raw = os.path.join(data, "raw")
    proc = os.path.join(data, "processed")
    os.makedirs(proc, exist_ok=True)

    nodes_path = os.path.join(raw, "hetionet-v1.0-nodes.tsv")
    edges_path = os.path.join(raw, "hetionet-v1.0-edges.sif.gz")
    download(NODES_URL, nodes_path)
    download(EDGES_URL, edges_path)

    print("Węzły...")
    id_to_idx, idx_to_kind, idx_to_id = load_nodes(nodes_path)
    print(f"  {len(id_to_idx)} węzłów")

    kinds = sorted(set(idx_to_kind))
    kind_to_int = {k: i for i, k in enumerate(kinds)}
    node_kind = np.array([kind_to_int[k] for k in idx_to_kind], dtype=np.int64)

    if args.sample is not None:
        print("  (--sample = pierwsze N linii krawędzi, rozkład relacji może być dziwny)")

    print("Krawędzie...")
    srcs, dsts, metas_str = [], [], []
    for s, d, m in iter_edges(edges_path, id_to_idx, limit=args.sample):
        srcs.append(s)
        dsts.append(d)
        metas_str.append(m)

    metas = sorted(set(metas_str))
    meta_to_int = {m: i for i, m in enumerate(metas)}
    edge_type = np.array([meta_to_int[m] for m in metas_str], dtype=np.int64)
    edge_index = np.array([srcs, dsts], dtype=np.int64)
    print(f"  {edge_index.shape[1]} krawędzi, {len(metas)} typów relacji")

    npz_path = os.path.join(proc, "hetionet_v1_baseline.npz")
    pkl_path = os.path.join(proc, "hetionet_v1_baseline_maps.pkl")

    np.savez_compressed(
        npz_path,
        edge_index=edge_index,
        edge_type=edge_type,
        node_kind=node_kind,
        num_nodes=np.array([len(id_to_idx)], dtype=np.int64),
    )
    with open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "num_nodes": len(id_to_idx),
                "kind_to_int": kind_to_int,
                "int_to_kind": {v: k for k, v in kind_to_int.items()},
                "metaedge_to_int": meta_to_int,
                "int_to_metaedge": {v: k for k, v in meta_to_int.items()},
                "id_to_idx": id_to_idx,
                "idx_to_id": idx_to_id,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"Zapis: {npz_path}, {pkl_path}")
    print(f"  CtD w tym zbiorze: {sum(1 for m in metas_str if m == 'CtD')}")


if __name__ == "__main__":
    main()
