"""Logika treningu CcSE: konfiguracja, pętla uczenia, ewaluacja, checkpointy."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader

from src.dataset import get_hetionet_data
from src.model import CCSE, Model
from src.paths import DEFAULT_CHECKPOINT, PROCESSED, WANDB_PROJECT_DEFAULT

LinkSplitName = Literal["train", "val", "test"]

__all__ = [
    "DEFAULT_CHECKPOINT",
    "LinkSplitName",
    "PROCESSED",
    "WANDB_PROJECT_DEFAULT",
    "TrainConfig",
    "first_positive_ccse_pair",
    "get_link_split_graph",
    "link_split",
    "load_checkpoint",
    "resolve_processed_dir",
    "run_training",
    "set_seed",
]


@dataclass
class TrainConfig:
    processed_dir: str = PROCESSED
    checkpoint: str = DEFAULT_CHECKPOINT
    seed: int = 42
    epochs: int = 50
    patience: int = 10
    batch_size: int = 256
    embed_dim: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    num_neighbors: list[int] = field(default_factory=lambda: [15, 10])
    use_wandb: bool = True
    wandb_project: str = WANDB_PROJECT_DEFAULT
    wandb_run_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_processed_dir(config: TrainConfig) -> str:
    return os.environ.get("DATA_PROCESSED", config.processed_dir)


def load_checkpoint(path: str, device: torch.device) -> tuple[dict, TrainConfig]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Brak {path} — uruchom: python train.py")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw = ckpt.get("config")
    if raw is None:
        print("UWAGA: checkpoint bez config — używam domyślnych hiperparametrów treningu")
        return ckpt, TrainConfig(checkpoint=path)
    return ckpt, TrainConfig(**raw)


def link_split(
    data: HeteroData, seed: int
) -> tuple[HeteroData, HeteroData, HeteroData]:
    """Ten sam podział CcSE co w treningu (RandomLinkSplit + seed)."""
    set_seed(seed)
    split = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        disjoint_train_ratio=0.3,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0,
        edge_types=CCSE,
    )
    return split(data)


def get_link_split_graph(data: HeteroData, seed: int, split: LinkSplitName) -> HeteroData:
    train_data, val_data, test_data = link_split(data, seed)
    return {"train": train_data, "val": val_data, "test": test_data}[split]


def first_positive_ccse_pair(graph: HeteroData) -> tuple[int, int]:
    """Pierwsza pozytywna para CcSE z podziału (indeksy w HeteroData)."""
    store = graph[CCSE]
    labels = store.edge_label
    pos_idx = (labels == 1).nonzero(as_tuple=False).view(-1)
    if pos_idx.numel() == 0:
        raise RuntimeError("Brak pozytywnych par CcSE w tym podziale link split.")
    i = int(pos_idx[0])
    eli = store.edge_label_index
    return int(eli[0, i]), int(eli[1, i])


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    scores, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        z = model.encode(batch)
        eli = batch[CCSE].edge_label_index
        y = batch[CCSE].edge_label
        s = model.predict(z, eli[0], eli[1])
        scores.append(s.cpu())
        labels.append(y.cpu())
    y_score = torch.cat(scores).numpy()
    y_true = torch.cat(labels).numpy()
    return roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)


def save_checkpoint(
    path: str,
    model: Model,
    epoch: int,
    val_ap: float,
    val_auc: float,
    config: TrainConfig,
) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "val_ap": val_ap,
            "val_auc": val_auc,
            "config": config.to_dict(),
        },
        path,
    )


def run_training(config: TrainConfig) -> dict[str, float | int | str | bool]:
    """Pełny przebieg treningu."""
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie: {device}")

    data = get_hetionet_data(config.processed_dir)
    print(f"Graf: {data.num_nodes} węzłów, {data.num_edges} krawędzi")
    print(f"CcSE: {data[CCSE].edge_index.size(1)} relacji lek→skutek uboczny")
    print(
        f"      {data['Compound'].num_nodes} leków, "
        f"{data['Side Effect'].num_nodes} skutków ubocznych"
    )

    train_data, val_data, test_data = link_split(data, config.seed)

    n_train = int((train_data[CCSE].edge_label == 1).sum())
    n_val = int(val_data[CCSE].edge_label.numel())
    print(
        f"Split CcSE: train {n_train} poz., "
        f"val {n_val} par, test {int(test_data[CCSE].edge_label.numel())} par"
    )

    loader_kw = dict(num_neighbors=config.num_neighbors, batch_size=config.batch_size)
    train_loader = LinkNeighborLoader(
        train_data,
        edge_label_index=(CCSE, train_data[CCSE].edge_label_index),
        edge_label=train_data[CCSE].edge_label,
        shuffle=True,
        neg_sampling_ratio=1.0,
        **loader_kw,
    )
    val_loader = LinkNeighborLoader(
        val_data,
        edge_label_index=(CCSE, val_data[CCSE].edge_label_index),
        edge_label=val_data[CCSE].edge_label,
        shuffle=False,
        neg_sampling_ratio=0.0,
        **loader_kw,
    )
    test_loader = LinkNeighborLoader(
        test_data,
        edge_label_index=(CCSE, test_data[CCSE].edge_label_index),
        edge_label=test_data[CCSE].edge_label,
        shuffle=False,
        neg_sampling_ratio=0.0,
        **loader_kw,
    )
    print(f"Batchy treningowe na epokę: {len(train_loader)}")

    model = Model(data, dim=config.embed_dim).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    best_ap = -1.0
    best_auc = 0.0
    best_epoch = 0
    stale = 0
    stopped_early = False

    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            z = model.encode(batch)
            eli = batch[CCSE].edge_label_index
            y = batch[CCSE].edge_label.float()
            logits = model.predict(z, eli[0], eli[1])
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.numel()
            n += y.numel()

        train_loss = loss_sum / max(n, 1)
        val_auc, val_ap = evaluate(model, val_loader, device)
        print(
            f"Epoka {epoch:03d} | loss {train_loss:.4f} | "
            f"val AUC {val_auc:.4f} | val AP {val_ap:.4f}"
        )

        if config.use_wandb:
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_auc": val_auc,
                        "val_ap": val_ap,
                    },
                    step=epoch,
                )

        if val_ap > best_ap:
            best_ap = val_ap
            best_auc = val_auc
            best_epoch = epoch
            stale = 0
            save_checkpoint(
                config.checkpoint,
                model,
                epoch,
                val_ap,
                val_auc,
                config,
            )
            print(f"  → zapisano {config.checkpoint}")
        else:
            stale += 1
            if stale >= config.patience:
                print(
                    f"Early stopping (brak poprawy val AP przez {config.patience} epok)."
                )
                stopped_early = True
                break

    if not os.path.isfile(config.checkpoint):
        raise RuntimeError("Brak zapisanego checkpointu — trening nie poprawił val AP.")

    ckpt = torch.load(config.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    val_auc, val_ap = evaluate(model, val_loader, device)
    test_auc, test_ap = evaluate(model, test_loader, device)
    print(f"\nNajlepszy checkpoint (epoka {ckpt['epoch']})")
    print(f"  Val  | AUC {val_auc:.4f} | AP {val_ap:.4f}")
    print(f"  Test | AUC {test_auc:.4f} | AP {test_ap:.4f}")

    results = {
        "best_val_ap": best_ap,
        "best_val_auc": best_auc,
        "best_epoch": best_epoch,
        "val_auc": val_auc,
        "val_ap": val_ap,
        "test_auc": test_auc,
        "test_ap": test_ap,
        "checkpoint": config.checkpoint,
        "stopped_early": stopped_early,
    }

    if config.use_wandb:
        import wandb

        if wandb.run is not None:
            wandb.log(
                {
                    "best_val_ap": best_ap,
                    "best_val_auc": best_auc,
                    "best_epoch": best_epoch,
                    "final_val_auc": val_auc,
                    "final_val_ap": val_ap,
                    "test_auc": test_auc,
                    "test_ap": test_ap,
                }
            )
            wandb.save(config.checkpoint)

    return results
