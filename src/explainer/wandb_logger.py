from __future__ import annotations

import os

from src.explainer.types import RankedEdge, node_name


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
