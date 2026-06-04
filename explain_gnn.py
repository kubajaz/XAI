"""
GNNExplainer dla predykcji CcSE — entry point.

Uruchomienie: python explain_gnn.py [--checkpoint ...] [--example]
Logika: src/explain_utils.py
"""

from __future__ import annotations

import argparse
import os

from src.explain_utils import (
    ExplainInputs,
    build_explain_wrapper,
    node_name,
    run_explanation,
)
from src.model import CCSE
from src.paths import DEFAULT_CHECKPOINT, DEFAULT_EXPLANATION_OUTPUT, WANDB_PROJECT_DEFAULT


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="GNNExplainer dla pary lek → skutek uboczny")
    ap.add_argument("--compound", type=int, default=None, help="Indeks leku w HeteroData")
    ap.add_argument("--side-effect", type=int, default=None, help="Indeks skutku w HeteroData")
    ap.add_argument("--example", action="store_true", help="Użyj pierwszej pary CcSE z grafu")
    ap.add_argument("--epochs", type=int, default=100, help="Epoki optymalizacji masek")
    ap.add_argument("--top-k", type=int, default=15, help="Ile najważniejszych krawędzi pokazać")
    ap.add_argument("--output", default=DEFAULT_EXPLANATION_OUTPUT)
    ap.add_argument(
        "--checkpoint",
        default=os.environ.get("CHECKPOINT", DEFAULT_CHECKPOINT),
        help="Ścieżka checkpointu (config + wagi)",
    )
    ap.add_argument("--no-wandb", action="store_true", help="Wyłącz logowanie W&B")
    ap.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", WANDB_PROJECT_DEFAULT),
    )
    ap.add_argument("--wandb-run-name", default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    use_wandb = not args.no_wandb

    print(f"Checkpoint: {args.checkpoint}", flush=True)
    print("Ładowanie grafu i modelu…", flush=True)

    wrapper, data, maps, train_config, processed_dir, device = build_explain_wrapper(
        args.checkpoint
    )
    print(
        f"Config treningu: embed_dim={train_config.embed_dim}, "
        f"num_neighbors={train_config.num_neighbors}, "
        f"processed_dir={processed_dir}",
        flush=True,
    )
    print(f"Urządzenie: {device}", flush=True)

    if args.example or args.compound is None or args.side_effect is None:
        ei = data[CCSE].edge_index
        compound = int(ei[0, 0])
        side_effect = int(ei[1, 0])
        print(f"Para przykładowa z CcSE: compound={compound}, side-effect={side_effect}")
    else:
        compound = args.compound
        side_effect = args.side_effect

    c_name = node_name(maps, "Compound", compound, processed_dir)
    se_name = node_name(maps, "Side Effect", side_effect, processed_dir)
    print(f"Para: {c_name} → {se_name}", flush=True)

    inputs = ExplainInputs(
        compound=compound,
        side_effect=side_effect,
        explainer_epochs=args.epochs,
        top_k=args.top_k,
        output=args.output,
    )

    if use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            job_type="explain",
            config={
                "checkpoint": args.checkpoint,
                "train_config": train_config.to_dict(),
                "compound": compound,
                "side_effect": side_effect,
                "compound_name": c_name,
                "side_effect_name": se_name,
                "epochs": args.epochs,
                "top_k": args.top_k,
                "output": args.output,
                "example": args.example,
            },
        )

    try:
        run_explanation(
            wrapper,
            data,
            maps,
            train_config,
            processed_dir,
            device,
            inputs,
            use_wandb=use_wandb,
            compound_name=c_name,
            side_effect_name=se_name,
        )
    finally:
        if use_wandb:
            import wandb

            if wandb.run is not None:
                wandb.finish()


if __name__ == "__main__":
    main()
