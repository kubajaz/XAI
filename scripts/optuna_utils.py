"""Wspólna logika Optuna dla study i pojedynczych trialów (SLURM)."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import optuna
from optuna.trial import TrialState

from train_utils import (
    CHECKPOINTS_DIR,
    TrainConfig,
    config_from_optuna_trial,
    run_training,
)

DEFAULT_STORAGE = f"sqlite:///{os.path.join(ROOT, 'outputs', 'optuna', 'hetionet_ccse.db')}"
DEFAULT_STUDY_NAME = "hetionet-ccse"


def create_study(
    storage: str,
    study_name: str,
    use_pruner: bool = True,
) -> optuna.Study:
    os.makedirs(os.path.join(ROOT, "outputs", "optuna"), exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
        if use_pruner
        else None
    )
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        pruner=pruner,
    )


def run_objective(trial: optuna.Trial, base: TrainConfig) -> float:
    config = config_from_optuna_trial(trial, base)
    config.checkpoint = os.path.join(CHECKPOINTS_DIR, f"trial_{trial.number}.pth")
    if base.wandb_run_name is None:
        config.wandb_run_name = f"trial-{trial.number}"
    else:
        config.wandb_run_name = f"{base.wandb_run_name}-{trial.number}"

    if config.use_wandb:
        import wandb

        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=config.to_dict(),
            reinit=True,
        )

    try:
        results = run_training(config, trial=trial)
        if config.use_wandb:
            import wandb

            if wandb.run is not None:
                wandb.log(results)
        return float(results["best_val_ap"])
    finally:
        if config.use_wandb:
            import wandb

            if wandb.run is not None:
                wandb.finish()


def print_best_trial(study: optuna.Study) -> optuna.trial.FrozenTrial:
    trial = study.best_trial
    print(f"Najlepszy trial #{trial.number}: val AP = {trial.value:.4f}")
    print(f"  Parametry: {trial.params}")
    ckpt = os.path.join(CHECKPOINTS_DIR, f"trial_{trial.number}.pth")
    if os.path.isfile(ckpt):
        print(f"  Checkpoint: {ckpt}")
    return trial


def run_one_ask_tell(
    storage: str,
    study_name: str,
    base: TrainConfig,
    use_pruner: bool,
) -> None:
    """Jeden trial — SLURM job array (study.ask / study.tell)."""
    study = create_study(storage, study_name, use_pruner=use_pruner)
    trial = study.ask()
    try:
        value = run_objective(trial, base)
        study.tell(trial, value)
    except optuna.TrialPruned:
        study.tell(trial, state=TrialState.PRUNED)
