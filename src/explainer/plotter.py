from __future__ import annotations

import os

import matplotlib.pyplot as plt
import networkx as nx
from torch import Tensor

from src.explainer.types import RankedEdge, node_name


def plot_gnn(
    batch,
    maps: dict,
    ranked: list[RankedEdge],
    compound: int,
    side_effect: int,
    local_eli: Tensor,
    score: float,
    fidelity_delta: float,
    out: str,
    processed_dir: str,
) -> None:
    G = nx.DiGraph()
    top_set = {(e.src_type, e.rel, e.dst_type, e.src, e.dst) for e in ranked}

    c_local = int(local_eli[0])
    se_local = int(local_eli[1])

    c_id = f"Compound:{c_local}"
    se_id = f"Side Effect:{se_local}"
    G.add_node(
        c_id, kind="Compound", label=node_name(maps, "Compound", compound, processed_dir)
    )
    G.add_node(
        se_id,
        kind="Side Effect",
        label=node_name(maps, "Side Effect", side_effect, processed_dir),
    )
    G.add_edge(c_id, se_id, rel="CcSE (pred.)", hi=True, predicted=True)

    for e in ranked:
        ns, nd = f"{e.src_type}:{e.src}", f"{e.dst_type}:{e.dst}"
        if ns not in G:
            G.add_node(
                ns,
                kind=e.src_type,
                label=node_name(maps, e.src_type, e.src, processed_dir, n_id=e.src_id),
            )
        if nd not in G:
            G.add_node(
                nd,
                kind=e.dst_type,
                label=node_name(maps, e.dst_type, e.dst, processed_dir, n_id=e.dst_id),
            )
        key = (e.src_type, e.rel, e.dst_type, e.src, e.dst)
        G.add_edge(ns, nd, rel=e.rel, hi=key in top_set, w=e.weight, predicted=False)

    pos = nx.spring_layout(G.to_undirected(), seed=42, k=1.2)
    fig, ax = plt.subplots(figsize=(13, 8))
    colors = {
        "Compound": "#22c55e",
        "Side Effect": "#f97316",
        "Gene": "#3b82f6",
        "Pathway": "#a855f7",
    }

    for n, attr in G.nodes(data=True):
        big = attr["kind"] in ("Compound", "Side Effect")
        nx.draw_networkx_nodes(
            G, pos, nodelist=[n],
            node_color=[colors.get(attr["kind"], "#94a3b8")],
            node_size=900 if big else 420, ax=ax,
        )
    max_w = max((a.get("w", 1.0) for _, _, a in G.edges(data=True)), default=1.0)
    for u, v, attr in G.edges(data=True):
        w = 3.5 if attr.get("predicted") else 1.0 + 3.0 * (attr.get("w", 0.0) / max(max_w, 1e-6))
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], width=w,
            style="dashed" if attr.get("predicted") else "solid",
            alpha=0.95 if attr.get("hi", False) else 0.35,
            ax=ax, arrows=True,
        )
    nx.draw_networkx_labels(G, pos, {n: d["label"] for n, d in G.nodes(data=True)}, font_size=7, ax=ax)

    ax.set_title(
        f"GNNExplainer CcSE: {node_name(maps, 'Compound', compound, processed_dir)} → "
        f"{node_name(maps, 'Side Effect', side_effect, processed_dir)}\n"
        f"score={score:.2f} | fidelity(top)={fidelity_delta:.2f} | krawędzi={len(ranked)}",
        fontsize=11,
    )
    ax.axis("off")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
