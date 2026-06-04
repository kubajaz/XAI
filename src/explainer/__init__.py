"""GNNExplainer dla predykcji CcSE."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.explainer.types import ExplainInputs, RankedEdge, load_maps, node_name

if TYPE_CHECKING:
    from src.explainer.explain_utils import CcSEExplainWrapper

__all__ = [
    "CcSEExplainWrapper",
    "ExplainInputs",
    "RankedEdge",
    "build_explain_wrapper",
    "load_maps",
    "node_name",
    "run_explanation",
]

_LAZY = frozenset({"CcSEExplainWrapper", "build_explain_wrapper", "run_explanation"})


def __getattr__(name: str):
    if name in _LAZY:
        from src.explainer import explain_utils

        return getattr(explain_utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
