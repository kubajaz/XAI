"""Wspólne typy i mapowanie węzłów dla modułu wyjaśnień (bez importów PyG/torch)."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import numpy as np

from src.paths import DEFAULT_EXPLANATION_OUTPUT

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
