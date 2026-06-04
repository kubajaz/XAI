"""GNNExplainer dla predykcji CcSE: podgraf, ranking krawędzi, wizualizacja."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig, ThresholdConfig
from torch_geometric.loader import LinkNeighborLoader

from src.dataset import get_hetionet_data
from src.explainer.plotter import plot_gnn
from src.explainer.types import ExplainInputs, RankedEdge, load_maps, node_name
from src.explainer.wandb_logger import log_explanation_wandb
from src.model import CCSE, EdgeType, Model
from src.train_utils import (
    LinkSplitName,
    TrainConfig,
    get_link_split_graph,
    load_checkpoint,
    resolve_processed_dir,
)

__all__ = [
    "CcSEExplainWrapper",
    "build_explain_wrapper",
    "run_explanation",
]


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
