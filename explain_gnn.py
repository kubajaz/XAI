"""
GNNExplainer dla predykcji CcSE — entry point.

Uruchomienie: python explain_gnn.py [--checkpoint ...] [--example | --n-pairs N]
Logika: src/explainer/
"""

from __future__ import annotations

import argparse
import os

from src.explainer import (
    ExplainInputs,
    build_explain_wrapper,
    node_name,
    run_explanation,
)
from src.paths import DEFAULT_CHECKPOINT, DEFAULT_EXPLANATION_OUTPUT, WANDB_PROJECT_DEFAULT
from src.train_utils import LinkSplitName, first_positive_ccse_pair, first_n_positive_ccse_pairs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="GNNExplainer dla pary lek → skutek uboczny")
    ap.add_argument("--compound", type=int, default=None, help="Indeks leku w HeteroData")
    ap.add_argument("--side-effect", type=int, default=None, help="Indeks skutku w HeteroData")
    ap.add_argument("--example", action="store_true", help="Pierwsza pozytywna para CcSE z podziału")
    ap.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Podział linków jak w treningu (domyślnie test, zgodnie z metrykami)",
    )
    ap.add_argument("--n-pairs", type=int, default=None, help="Iteruj po pierwszych n pozytywnych parach CcSE (zamiast --example)")
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

    split: LinkSplitName = args.split
    wrapper, data, maps, train_config, processed_dir, device = build_explain_wrapper(
        args.checkpoint, split=split
    )
    print(
        f"Config treningu: embed_dim={train_config.embed_dim}, "
        f"num_neighbors={train_config.num_neighbors}, "
        f"split={split}, seed={train_config.seed}, "
        f"processed_dir={processed_dir}",
        flush=True,
    )
    print(f"Urządzenie: {device}", flush=True)

    # Build list of (compound, side_effect) pairs to explain
    if args.n_pairs is not None:
        pairs = first_n_positive_ccse_pairs(data, n=args.n_pairs)
        print(f"Pierwsze {len(pairs)} pozytywnych par CcSE ({split})", flush=True)
    elif args.example:
        compound, side_effect = first_positive_ccse_pair(data)
        print(
            f"Para przykładowa ({split}, pozytywna CcSE): "
            f"compound={compound}, side-effect={side_effect}"
        )
        pairs = [(compound, side_effect)]
    elif args.compound is not None and args.side_effect is not None:
        pairs = [(args.compound, args.side_effect)]
    else:
        raise SystemExit(
            "error: podaj --compound i --side-effect, --example albo --n-pairs"
        )

    # Derive output base: strip extension so we can suffix per pair
    output_base, output_ext = os.path.splitext(args.output)
    if not output_ext:
        output_ext = ".png"

    for idx, (compound, side_effect) in enumerate(pairs):
        require_positive = (args.compound is None) and split in ("val", "test")

        c_name = node_name(maps, "Compound", compound, processed_dir)
        se_name = node_name(maps, "Side Effect", side_effect, processed_dir)
        print(f"\n[{idx + 1}/{len(pairs)}] Para: {c_name} → {se_name}", flush=True)

        out_path = (
            f"{output_base}_{idx}{output_ext}"
            if len(pairs) > 1
            else f"{output_base}{output_ext}"
        )

        inputs = ExplainInputs(
            compound=compound,
            side_effect=side_effect,
            explainer_epochs=args.epochs,
            top_k=args.top_k,
            output=out_path,
            require_positive_ccse=require_positive,
        )

        wandb = None
        if use_wandb:
            import wandb as _wandb

            wandb = _wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                job_type="explain",
                config={
                    "checkpoint": args.checkpoint,
                    "train_config": train_config.to_dict(),
                    "split": split,
                    "compound": compound,
                    "side_effect": side_effect,
                    "compound_name": c_name,
                    "side_effect_name": se_name,
                    "epochs": args.epochs,
                    "top_k": args.top_k,
                    "output": out_path,
                    "pair_index": idx,
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
            if wandb is not None and wandb.run is not None:
                wandb.finish()


if __name__ == "__main__":
    main()
