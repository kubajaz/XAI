#!/usr/bin/env python3
"""Optuna: jeden trial na zadanie SLURM (study.ask / study.tell)."""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPTS)

from optuna_utils import (
    DEFAULT_STORAGE,
    DEFAULT_STUDY_NAME,
    create_study,
    print_best_trial,
    run_one_ask_tell,
)
from train_utils import CHECKPOINTS_DIR, WANDB_PROJECT_DEFAULT, TrainConfig, promote_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Optuna: jeden trial (SLURM array) lub podsumowanie study"
    )
    ap.add_argument("--study-name", default=DEFAULT_STUDY_NAME)
    ap.add_argument("--storage", default=os.environ.get("OPTUNA_STORAGE", DEFAULT_STORAGE))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", WANDB_PROJECT_DEFAULT),
    )
    ap.add_argument(
        "--show-best",
        action="store_true",
        help="Wypisz najlepszy trial (login node, po array)",
    )
    ap.add_argument(
        "--promote-best",
        action="store_true",
        help="Skopiuj najlepszy checkpoint do model.pth",
    )
    args = ap.parse_args()

    if args.show_best:
        study = create_study(args.storage, args.study_name, use_pruner=not args.no_prune)
        best = print_best_trial(study)
        if args.promote_best:
            ckpt = os.path.join(CHECKPOINTS_DIR, f"trial_{best.number}.pth")
            if os.path.isfile(ckpt):
                promote_checkpoint(ckpt)
        return

    run_name = os.environ.get("SLURM_ARRAY_TASK_ID") or os.environ.get("SLURM_JOB_ID")
    base = TrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=f"slurm-{run_name}" if run_name else None,
    )
    run_one_ask_tell(
        args.storage,
        args.study_name,
        base,
        use_pruner=not args.no_prune,
    )


if __name__ == "__main__":
    main()
