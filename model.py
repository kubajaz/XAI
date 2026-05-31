"""RGCN (encoder) + DistMult (decoder) dla Hetionet."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, RGCNConv

EdgeType = Tuple[str, str, str]
# Compound causes Side Effect
CCSE = ("Compound", "CcSE", "Side Effect")


class RGCNEncoder(nn.Module):
    def __init__(
        self,
        num_nodes: Dict[str, int],
        edge_types: List[EdgeType],
        dim: int = 64,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleDict(
            {t: nn.Embedding(n, dim) for t, n in sorted(num_nodes.items())}
        )

        def layer(in_d: int, out_d: int) -> HeteroConv:
            return HeteroConv(
                {
                    et: RGCNConv((in_d, in_d), out_d, num_relations=1, aggr="mean")
                    for et in edge_types
                },
                aggr="sum",
            )

        self.conv1 = layer(dim, dim)
        self.conv2 = layer(dim, dim)

    def forward(self, data: HeteroData) -> Dict[str, Tensor]:
        device = self.device
        x = {}
        for t in data.node_types:
            store = data[t]
            idx = getattr(store, "n_id", None)
            if idx is None:
                idx = torch.arange(store.num_nodes, device=device)
            else:
                idx = idx.to(device)
            x[t] = self.embeddings[t](idx)

        ei = {et: data[et].edge_index for et in data.edge_types}
        et = {
            e: torch.zeros(data[e].edge_index.size(1), dtype=torch.long, device=device)
            for e in ei
        }

        h = self.conv1(x, ei, edge_type_dict=et)
        x = {t: F.relu(h[t]) if t in h else x[t] for t in x}
        h = self.conv2(x, ei, edge_type_dict=et)
        return {t: h[t] if t in h else x[t] for t in x}

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class DistMultDecoder(nn.Module):
    def __init__(self, edge_types: List[EdgeType], dim: int) -> None:
        super().__init__()
        ets = sorted(edge_types)
        self.index = {et: i for i, et in enumerate(ets)}
        self.rel = nn.Embedding(len(ets), dim)

    def score(self, h: Tensor, et: EdgeType, t: Tensor) -> Tensor:
        r = self.rel.weight[self.index[et]].unsqueeze(0).expand_as(h)
        return (h * r * t).sum(dim=-1)


class Model(nn.Module):
    def __init__(self, data: HeteroData, dim: int = 64) -> None:
        super().__init__()
        num_nodes = {t: data[t].num_nodes for t in data.node_types}
        edge_types = list(data.edge_types)
        self.encoder = RGCNEncoder(num_nodes, edge_types, dim)
        self.decoder = DistMultDecoder(edge_types, dim)

    def encode(self, data: HeteroData) -> Dict[str, Tensor]:
        return self.encoder(data)

    def encode_features(
        self, x_dict: Dict[str, Tensor], edge_index_dict: Dict[EdgeType, Tensor]
    ) -> Dict[str, Tensor]:
        """RGCN na podanych cechach węzłów (używane przez GNNExplainer)."""
        device = next(self.encoder.parameters()).device
        et = {
            e: torch.zeros(ei.size(1), dtype=torch.long, device=device)
            for e, ei in edge_index_dict.items()
        }
        h = self.encoder.conv1(x_dict, edge_index_dict, edge_type_dict=et)
        x = {t: F.relu(h[t]) if t in h else x_dict[t] for t in x_dict}
        h = self.encoder.conv2(x, edge_index_dict, edge_type_dict=et)
        return {t: h[t] if t in h else x[t] for t in x_dict}

    def predict(
        self,
        z: Dict[str, Tensor],
        compound_idx: Tensor,
        side_effect_idx: Tensor,
    ) -> Tensor:
        h = z["Compound"][compound_idx]
        t = z["Side Effect"][side_effect_idx]
        return self.decoder.score(h, CCSE, t)
