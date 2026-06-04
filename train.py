"""
Trening: przewidywanie CcSE (Compound causes Side Effect) na Hetionet.

Uruchomienie: python train.py [--batch-size ...]
Logika treningu: train_utils.py
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from train_utils import (
    DEFAULT_CHECKPOINT,
    PROCESSED,
    WANDB_PROJECT_DEFAULT,
    TrainConfig,
    run_training,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Trening link prediction CcSE na Hetionet (RGCN + DistMult)"
    )
    ap.add_argument("--processed-dir", default=PROCESSED, help="Katalog z .npz/.pkl")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Ścieżka zapisu modelu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument(
        "--num-neighbors",
        type=int,
        nargs="+",
        default=[15, 10],
        metavar="N",
        help="Liczba sąsiadów na hop (np. 15 10)",
    )
    ap.add_argument("--no-wandb", action="store_true", help="Wyłącz logowanie W&B")
    ap.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", WANDB_PROJECT_DEFAULT),
    )
    ap.add_argument("--wandb-run-name", default=None)
    return ap.parse_args(argv)


def args_to_config(args: argparse.Namespace) -> TrainConfig:
    processed = os.environ.get("DATA_PROCESSED", args.processed_dir)
    return TrainConfig(
        processed_dir=processed,
        checkpoint=args.checkpoint,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_neighbors=list(args.num_neighbors),
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    config = args_to_config(args)

    if config.use_wandb:
        import wandb

        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=config.to_dict(),
        )

    try:
        return run_training(config, trial=None)
    finally:
        if config.use_wandb:
            import wandb

            if wandb.run is not None:
                wandb.finish()


if __name__ == "__main__":
    main()
