"""GNNExplainer dla predykcji CcSE: podgraf, ranking krawędzi, wizualizacja."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig, ThresholdConfig
from torch_geometric.loader import LinkNeighborLoader

from src.dataset import get_hetionet_data
from src.model import CCSE, EdgeType, Model
from src.paths import DEFAULT_EXPLANATION_OUTPUT
from src.train_utils import (
    LinkSplitName,
    TrainConfig,
    get_link_split_graph,
    load_checkpoint,
    resolve_processed_dir,
)

_KIND_GLOBALS: dict[str, np.ndarray] = {}


@dataclass
class RankedEdge:
    src_type: str
    rel: str
    dst_type: str
    src: int
    dst: int
    src_id: int
    dst_id: int
    weight: float


@dataclass
class ExplainInputs:
    compound: int
    side_effect: int
    explainer_epochs: int = 100
    top_k: int = 15
    output: str = DEFAULT_EXPLANATION_OUTPUT


class CcSEExplainWrapper(nn.Module):
    """Forward zgodny z PyG Explainer: x_dict + edge_index_dict + para lek–skutek."""

    def __init__(self, model: Model) -> None:
        super().__init__()
        self.core = model

    def forward(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[EdgeType, Tensor],
        edge_label_index: Tensor,
    ) -> Tensor:
        z = self.core.encode_features(x_dict, edge_index_dict)
        c = edge_label_index[0].long()
        se = edge_label_index[1].long()
        h = z["Compound"][c].unsqueeze(0)
        t = z["Side Effect"][se].unsqueeze(0)
        return self.core.decoder.score(h, CCSE, t).view(-1)


def load_maps(processed_dir: str) -> dict:
    with open(os.path.join(processed_dir, "hetionet_v1_baseline_maps.pkl"), "rb") as f:
        return pickle.load(f)


def node_name(
    maps: dict,
    ntype: str,
    local: int,
    processed_dir: str,
    n_id: int | None = None,
) -> str:
    if n_id is not None:
        return maps["idx_to_id"][int(n_id)].split("::")[-1][:32]
    if ntype not in _KIND_GLOBALS:
        z = np.load(os.path.join(processed_dir, "hetionet_v1_baseline.npz"), mmap_mode="r")
        kind = maps["kind_to_int"][ntype]
        _KIND_GLOBALS[ntype] = np.where(z["node_kind"] == kind)[0]
    g = int(_KIND_GLOBALS[ntype][local])
    return maps["idx_to_id"][g].split("::")[-1][:32]


def get_batch(
    data: HeteroData,
    model: Model,
    device: torch.device,
    compound: int,
    side_effect: int,
    num_neighbors: list[int],
):
    eli = torch.tensor([[compound], [side_effect]], dtype=torch.long)
    loader = LinkNeighborLoader(
        data,
        edge_label_index=(CCSE, eli),
        edge_label=torch.tensor([1.0]),
        num_neighbors=num_neighbors,
        batch_size=1,
        shuffle=False,
        neg_sampling_ratio=0.0,
    )
    batch = next(iter(loader)).to(device)
    x_dict = {
        nt: model.encoder.embeddings[nt](batch[nt].n_id).detach()
        for nt in batch.node_types
    }
    edge_index_dict = {
        et: batch[et].edge_index for et in batch.edge_types if batch[et].num_edges > 0
    }
    local_eli = batch[CCSE].edge_label_index[:, 0]
    return batch, x_dict, edge_index_dict, local_eli


def ranked_edges(explanation, batch, top_k: int) -> list[RankedEdge]:
    rows: list[RankedEdge] = []
    for store in explanation.edge_stores:
        mask = getattr(store, "edge_mask", None)
        if mask is None or store.num_edges == 0:
            continue
        st, rel, dt = store._key
        ei = store.edge_index
        n_id_src = batch[st].n_id
        n_id_dst = batch[dt].n_id
        for i in range(store.num_edges):
            w = float(mask[i])
            if w <= 0:
                continue
            s, d = int(ei[0, i]), int(ei[1, i])
            rows.append(
                RankedEdge(
                    st, rel, dt, s, d,
                    int(n_id_src[s]), int(n_id_dst[d]),
                    w,
                )
            )
    rows.sort(key=lambda r: r.weight, reverse=True)
    return rows[:top_k]


def fidelity(explainer, x_dict, edge_index_dict, local_eli, explanation, top_k: int) -> float:
    full = float(explainer.get_prediction(x_dict, edge_index_dict, edge_label_index=local_eli))
    edge_mask = {}
    for store in explanation.edge_stores:
        mask = getattr(store, "edge_mask", None)
        if mask is not None:
            edge_mask[store._key] = mask
    if not edge_mask:
        return 0.0
    flat = torch.cat([m.flatten() for m in edge_mask.values()])
    k = min(top_k, flat.numel())
    thresh = torch.topk(flat, k=k).values.min()
    masked = {et: m * (m < thresh).float() for et, m in edge_mask.items()}
    reduced = float(
        explainer.get_masked_prediction(
            x_dict, edge_index_dict, edge_mask=masked, edge_label_index=local_eli
        )
    )
    return full - reduced


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


def log_explanation_wandb(
    *,
    score: float,
    fidelity_delta: float,
    ranked: list[RankedEdge],
    maps: dict,
    processed_dir: str,
    compound: int,
    side_effect: int,
    compound_name: str,
    side_effect_name: str,
    n_nodes: int,
    n_edges: int,
    output: str,
    top_k: int,
    explainer_epochs: int,
) -> None:
    import wandb

    if wandb.run is None:
        return

    score_masked = score - fidelity_delta
    wandb.log(
        {
            "score": score,
            "score_after_topk_mask": score_masked,
            "fidelity": fidelity_delta,
            "subgraph_nodes": n_nodes,
            "subgraph_edges": n_edges,
            "num_ranked_edges": len(ranked),
            "explanation_plot": wandb.Image(
                output,
                caption=f"{compound_name} → {side_effect_name}",
            ),
        }
    )

    edge_table = wandb.Table(
        columns=["rank", "weight", "src_type", "relation", "dst_type", "src", "dst"]
    )
    for i, e in enumerate(ranked, 1):
        edge_table.add_data(
            i,
            e.weight,
            e.src_type,
            e.rel,
            e.dst_type,
            node_name(maps, e.src_type, e.src, processed_dir, n_id=e.src_id),
            node_name(maps, e.dst_type, e.dst, processed_dir, n_id=e.dst_id),
        )
    wandb.log({"top_edges": edge_table})

    artifact = wandb.Artifact(
        name=f"gnn-explainer-c{compound}-se{side_effect}",
        type="explanation",
        metadata={
            "compound": compound,
            "side_effect": side_effect,
            "compound_name": compound_name,
            "side_effect_name": side_effect_name,
            "score": score,
            "score_after_topk_mask": score_masked,
            "fidelity": fidelity_delta,
            "top_k": top_k,
            "explainer_epochs": explainer_epochs,
        },
    )
    artifact.add_file(output, name=os.path.basename(output))
    wandb.log_artifact(artifact)


def build_explain_wrapper(
    checkpoint: str,
    split: LinkSplitName = "test",
    device: torch.device | None = None,
) -> tuple[CcSEExplainWrapper, HeteroData, dict, TrainConfig, str, torch.device]:
    """Ładuje model; zwraca wrapper i graf CcSE z wybranego podziału (train/val/test)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt, train_config = load_checkpoint(checkpoint, device)
    processed_dir = resolve_processed_dir(train_config)

    full_data = get_hetionet_data(processed_dir)
    split_data = get_link_split_graph(full_data, train_config.seed, split)
    maps = load_maps(processed_dir)
    model = Model(full_data, dim=train_config.embed_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    wrapper = CcSEExplainWrapper(model).to(device)
    for p in wrapper.parameters():
        p.requires_grad = False
    return wrapper, split_data, maps, train_config, processed_dir, device


def run_explanation(
    wrapper: CcSEExplainWrapper,
    data: HeteroData,
    maps: dict,
    train_config: TrainConfig,
    processed_dir: str,
    device: torch.device,
    inputs: ExplainInputs,
    *,
    use_wandb: bool = False,
    compound_name: str | None = None,
    side_effect_name: str | None = None,
) -> None:
    compound = inputs.compound
    side_effect = inputs.side_effect
    c_name = compound_name or node_name(maps, "Compound", compound, processed_dir)
    se_name = side_effect_name or node_name(maps, "Side Effect", side_effect, processed_dir)

    model = wrapper.core
    print(
        f"Budowa podgrafu (num_neighbors={train_config.num_neighbors})…",
        flush=True,
    )
    batch, x_dict, edge_index_dict, local_eli = get_batch(
        data,
        model,
        device,
        compound,
        side_effect,
        train_config.num_neighbors,
    )
    n_edges = sum(batch[et].edge_index.size(1) for et in batch.edge_types)
    print(f"Podgraf: {batch.num_nodes} węzłów, {n_edges} krawędzi", flush=True)

    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(
            epochs=inputs.explainer_epochs, lr=0.01, edge_size=0.005
        ),
        explanation_type="model",
        edge_mask_type="object",
        model_config=ModelConfig(
            mode="regression",
            task_level="edge",
            return_type="raw",
        ),
        threshold_config=ThresholdConfig(threshold_type="topk", value=inputs.top_k),
    )

    print(f"GNNExplainer ({inputs.explainer_epochs} epok)…", flush=True)
    with torch.no_grad():
        score = float(
            explainer.get_prediction(x_dict, edge_index_dict, edge_label_index=local_eli)
        )
    explanation = explainer(x_dict, edge_index_dict, edge_label_index=local_eli)

    ranked = ranked_edges(explanation, batch, inputs.top_k)
    fid = fidelity(explainer, x_dict, edge_index_dict, local_eli, explanation, inputs.top_k)

    print(f"\nScore modelu: {score:.3f}")
    print(f"Score po odcięciu top-{inputs.top_k} krawędzi: {score - fid:.3f}")
    print(f"Fidelity+ (spadek score po usunięciu WAŻNYCH krawędzi): {fid:.3f}")
    print(f"\nTop-{len(ranked)} krawędzi istotnych dla predykcji:")
    for i, e in enumerate(ranked, 1):
        print(
            f"  {i:2d}. [{e.weight:.3f}] "
            f"{node_name(maps, e.src_type, e.src, processed_dir, n_id=e.src_id)} "
            f"--{e.rel}--> "
            f"{node_name(maps, e.dst_type, e.dst, processed_dir, n_id=e.dst_id)}"
        )

    print("\nRysowanie…", flush=True)
    plot_gnn(
        batch,
        maps,
        ranked,
        compound,
        side_effect,
        local_eli,
        score,
        fid,
        inputs.output,
        processed_dir,
    )
    print(f"Zapisano: {inputs.output}")

    if use_wandb:
        log_explanation_wandb(
            score=score,
            fidelity_delta=fid,
            ranked=ranked,
            maps=maps,
            processed_dir=processed_dir,
            compound=compound,
            side_effect=side_effect,
            compound_name=c_name,
            side_effect_name=se_name,
            n_nodes=batch.num_nodes,
            n_edges=n_edges,
            output=inputs.output,
            top_k=inputs.top_k,
            explainer_epochs=inputs.explainer_epochs,
        )
