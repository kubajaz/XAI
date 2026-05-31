# Hetionet + XAI — predykcja skutków ubocznych leków

Projekt przewiduje relację **CcSE** (*Compound causes Side Effect*) na grafie wiedzy i wyjaśnia predykcje.

## Co robi kod

| Plik | Opis |
|------|------|
| `scripts/prepare_hetionet.py` | Jednorazowe obranie i preprocessing danych |
| `dataset.py` | Hetionet (`.npz` + `.pkl`) → graf `HeteroData` dla PyTorch Geometric |
| `model.py` | RGCN (encoder) + DistMult (decoder) — score dla pary lek–skutek uboczny |
| `train.py` | Link prediction CcSE: split, trening, ewaluacja, zapis `model.pth` |
| `explain_gnn.py` | GNNExplainer: które krawędzie podgrafu uzasadniają score modelu |

## Uruchomienie

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python scripts/prepare_hetionet.py

python train.py

python explain_gnn.py --example
python explain_gnn.py --compound 322 --side-effect 1245 --epochs 100 --top-k 15
```

**Wyniki:** `model.pth`, `outputs/explanation.png`, `outputs/explanation_gnn.png`

Indeksy `--compound` / `--side-effect` to numery węzłów w `HeteroData`. Flaga `--example` bierze pierwszą parę CcSE z grafu.

## Zadanie

Model ukrywa część krawędzi CcSE, uczy się z reszty grafu (~2,25 mln krawędzi, 24 relacje) i ocenia, czy dany lek powoduje dany skutek uboczny.
