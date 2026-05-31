"""
Trening: przewidywanie CcSE (Compound causes Side Effect) na Hetionet.

1. Ukrywamy część krawędzi CcSE (znane powiązania lek → skutek uboczny).
2. Model (RGCN) uczy się z pozostałego grafu (~2 mln krawędzi).
3. DistMult ocenia, czy dany lek może powodować dany skutek uboczny.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import LinkNeighborLoader

from dataset import get_hetionet_data
from model import CCSE, Model

ROOT = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(ROOT, "data", "processed")
CHECKPOINT = os.path.join(ROOT, "model.pth")

SEED = 42
EPOCHS = 50
PATIENCE = 10
BATCH_SIZE = 256
EMBED_DIM = 64
LR = 1e-3
WEIGHT_DECAY = 1e-5


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie: {device}")

    data = get_hetionet_data(PROCESSED)
    print(f"Graf: {data.num_nodes} węzłów, {data.num_edges} krawędzi")
    print(f"CcSE: {data[CCSE].edge_index.size(1)} relacji lek→skutek uboczny")
    print(f"      {data['Compound'].num_nodes} leków, {data['Side Effect'].num_nodes} skutków ubocznych")

    split = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        disjoint_train_ratio=0.3,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0,
        edge_types=CCSE,
    )
    train_data, val_data, test_data = split(data)

    n_train = int((train_data[CCSE].edge_label == 1).sum())
    n_val = int(val_data[CCSE].edge_label.numel())
    print(
        f"Split CcSE: train {n_train} poz., "
        f"val {n_val} par, test {int(test_data[CCSE].edge_label.numel())} par"
    )

    train_kw = dict(num_neighbors=[15, 10], batch_size=BATCH_SIZE, neg_sampling_ratio=1.0)
    eval_kw = dict(num_neighbors=[15, 10], batch_size=BATCH_SIZE, neg_sampling_ratio=0.0)

    train_loader = LinkNeighborLoader(
        train_data,
        edge_label_index=(CCSE, train_data[CCSE].edge_label_index),
        edge_label=train_data[CCSE].edge_label,
        shuffle=True,
        **train_kw,
    )
    val_loader = LinkNeighborLoader(
        val_data,
        edge_label_index=(CCSE, val_data[CCSE].edge_label_index),
        edge_label=val_data[CCSE].edge_label,
        shuffle=False,
        **eval_kw,
    )
    test_loader = LinkNeighborLoader(
        test_data,
        edge_label_index=(CCSE, test_data[CCSE].edge_label_index),
        edge_label=test_data[CCSE].edge_label,
        shuffle=False,
        **eval_kw,
    )
    print(f"Batchy treningowe na epokę: {len(train_loader)}")

    model = Model(data, dim=EMBED_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_ap = -1.0
    stale = 0
    for epoch in range(1, EPOCHS + 1):
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

        val_auc, val_ap = evaluate(model, val_loader, device)
        print(
            f"Epoka {epoch:03d} | loss {loss_sum / max(n, 1):.4f} | "
            f"val AUC {val_auc:.4f} | val AP {val_ap:.4f}"
        )
        if val_ap > best_ap:
            best_ap = val_ap
            stale = 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_ap": val_ap, "val_auc": val_auc},
                CHECKPOINT,
            )
            print(f"  → zapisano {CHECKPOINT}")
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"Early stopping (brak poprawy val AP przez {PATIENCE} epok).")
                break

    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    val_auc, val_ap = evaluate(model, val_loader, device)
    test_auc, test_ap = evaluate(model, test_loader, device)
    print(f"\nNajlepszy checkpoint (epoka {ckpt['epoch']})")
    print(f"  Val  | AUC {val_auc:.4f} | AP {val_ap:.4f}")
    print(f"  Test | AUC {test_auc:.4f} | AP {test_ap:.4f}")


if __name__ == "__main__":
    main()
