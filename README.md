# Hetionet + XAI — predykcja skutków ubocznych leków

Projekt przewiduje relację **CcSE** (*Compound causes Side Effect*) na grafie wiedzy i wyjaśnia predykcje.

## Co robi kod

| Plik | Opis |
|------|------|
| `scripts/prepare_hetionet.py` | Jednorazowe pobranie i preprocessing danych |
| `src/dataset.py` | Hetionet (`.npz` + `.pkl`) → graf `HeteroData` |
| `src/model.py` | RGCN (encoder) + DistMult (decoder) |
| `src/train_utils.py` | Config, pętla treningu, ewaluacja, checkpointy |
| `src/explain_utils.py` | GNNExplainer, ranking krawędzi, wizualizacja, W&B |
| `src/paths.py` | Ścieżki projektu (`data/`, `outputs/`, `model.pth`) |
| `train.py` | CLI treningu (entry point) |
| `explain_gnn.py` | CLI wyjaśnień (entry point) |
| `scripts/slurm/train_grid.sh` | Siatka hiperparametrów (lokalnie i SLURM) |

## Uruchomienie

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python scripts/prepare_hetionet.py

# Trening (domyślne hiperparametry)
wandb login   # jednorazowo
python train.py

# Własne hiperparametry
python train.py --help
python train.py \
  --batch-size 512 \
  --embed-dim 128 \
  --lr 5e-4 \
  --weight-decay 1e-4 \
  --num-neighbors 20 15 \
  --wandb-run-name my-run

python explain_gnn.py --example
python explain_gnn.py --compound 322 --side-effect 1245 --epochs 100 --top-k 15
```

**Wyniki:** `model.pth`, logi w projekcie W&B `zzsn-gnn-xai`, `outputs/explanation_gnn.png`

Indeksy `--compound` / `--side-effect` to numery węzłów w `HeteroData`. Flaga `--example` bierze pierwszą parę CcSE z grafu.

## Weights & Biases

- Domyślny projekt: **`zzsn-gnn-xai`** (nadpisanie: `--wandb-project` lub `WANDB_PROJECT`)
- Wyłączenie: `--no-wandb`
- Zmienne: `WANDB_API_KEY` (wymagane przy logowaniu)

## Siatka hiperparametrów

Wartości siatki są w **`scripts/slurm/train_grid.sh`** (jedno źródło dla lokalnego uruchomienia i SLURM).

Lokalnie (pętla w jednym procesie, z katalogu głównego repozytorium):

```bash
bash scripts/slurm/train_grid.sh
```

Na klastrze — **jedno** zadanie `sbatch`, ta sama pętla sekwencyjnie (jeśli masz wrapper SLURM):

```bash
sbatch scripts/slurm/train.sh   # pojedynczy trening
```

Checkpointy: `outputs/checkpoints/` z tagiem `bs*_dim*_lr*_wd*_nh*` (na SLURM z prefiksem `grid_<jobId>_<n>_...`).

## SLURM

Na klastrze (login node, raz na środowisko):

```bash
cd /path/to/XAI
bash scripts/slurm/setup_env.sh
bash scripts/slurm/download_data.sh
```

Środowisko conda trafia na scratch (`/net/tscratch/people/$USER/conda/py311_env` domyślnie). Dane Hetionet — na scratch w `$SCRATCH_BASE/data/hetionet/` (raw + `processed/`). Zadania `sbatch` ładują env i `DATA_PROCESSED` przez `source scripts/slurm/activate_env.sh`.

```bash
export WANDB_API_KEY=...
# Uruchamiaj sbatch z katalogu głównego repozytorium (ustawia SLURM_SUBMIT_DIR):
cd /path/to/XAI
sbatch scripts/slurm/train.sh

# Siatka (wiele treningów po kolei — dostosuj #SBATCH --time)
bash scripts/slurm/train_grid.sh
```

Zmienne środowiskowe:

| Zmienna | Opis |
|---------|------|
| `WANDB_API_KEY` | Autoryzacja W&B |
| `WANDB_PROJECT` | Nadpisuje domyślny projekt |
| `DATA_PROCESSED` | Katalog z `.npz`/`.pkl` (domyślnie `$SCRATCH_BASE/data/hetionet/processed`) |
| `DATA_DIR` | Korzeń danych na scratch (domyślnie `$SCRATCH_BASE/data/hetionet`) |
| `SCRATCH_BASE` | Baza scratch (domyślnie `/net/tscratch/people/$USER`) |
| `ENV_PREFIX` | Prefiks conda env (domyślnie `$SCRATCH_BASE/conda/py311_env`) |

## Zadanie

Model ukrywa część krawędzi CcSE, uczy się z reszty grafu (~2,25 mln krawędzi, 24 relacje) i ocenia, czy dany lek powoduje dany skutek uboczny.
