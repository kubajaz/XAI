"""Ścieżki względem katalogu głównego repozytorium."""

from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")
DEFAULT_CHECKPOINT = os.path.join(ROOT, "model.pth")
WANDB_PROJECT_DEFAULT = "zzsn-gnn-xai"
CHECKPOINTS_DIR = os.path.join(ROOT, "outputs", "checkpoints")
DEFAULT_EXPLANATION_OUTPUT = os.path.join(ROOT, "outputs", "explanation_gnn.png")
