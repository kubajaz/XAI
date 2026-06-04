# Hetionet + XAI — predykcja skutków ubocznych leków

Projekt przewiduje relację **CcSE** (*Compound causes Side Effect*) na grafie wiedzy i wyjaśnia predykcje.

## Co robi kod

| Plik | Opis |
|------|------|
| `scripts/prepare_hetionet.py` | Jednorazowe obranie i preprocessing danych |
| `dataset.py` | Hetionet (`.npz` + `.pkl`) → graf `HeteroData` dla PyTorch Geometric |
| `model.py` | RGCN (encoder) + DistMult (decoder) — score dla pary lek–skutek uboczny |
| `train.py` | CLI treningu (entry point) |
| `train_utils.py` | Logika treningu: config, pętla, ewaluacja, checkpointy |
| `scripts/run_optuna_study.py` | Optuna — wiele trialów lokalnie |
| `scripts/run_optuna_trial.py` | Optuna — jeden trial (SLURM array) lub `--show-best` |
| `explain_gnn.py` | GNNExplainer: które krawędzie podgrafu uzasadniają score modelu |

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

## Optuna

Lokalnie (jeden proces, wiele trialów):

```bash
python scripts/run_optuna_study.py --n-trials 20 --promote-best
```

Checkpoints trialów: `outputs/checkpoints/trial_N.pth`. Study DB: `outputs/optuna/hetionet_ccse.db`.

## SLURM

Na klastrze (login node, raz na środowisko):

```bash
cd /path/to/XAI
bash scripts/slurm/setup_env.sh
```

Środowisko conda trafia na scratch (`/net/tscratch/people/$USER/conda/py311_env` domyślnie). Zadania `sbatch` ładują je przez `source scripts/slurm/activate_env.sh`.

```bash
export WANDB_API_KEY=...
sbatch scripts/slurm/train.sh

# Siatka ręczna (4 zadania)
sbatch scripts/slurm/train_array.sh

# Optuna: jeden trial na zadanie array
sbatch scripts/slurm/optuna_array.sh
# Po zakończeniu array (login node, po activate_env):
source scripts/slurm/activate_env.sh
python scripts/run_optuna_trial.py --show-best --promote-best
```

Dla równoległych trialów Optuny na wielu węzłach preferuj **`OPTUNA_STORAGE`** z PostgreSQL zamiast SQLite na NFS.

Zmienne środowiskowe:

| Zmienna | Opis |
|---------|------|
| `WANDB_API_KEY` | Autoryzacja W&B |
| `WANDB_PROJECT` | Nadpisuje domyślny projekt |
| `OPTUNA_STORAGE` | URL bazy study (np. `postgresql+psycopg2://...`) |
| `DATA_PROCESSED` | Ścieżka do `data/processed` na węźle obliczeniowym |
| `SCRATCH_BASE` | Baza scratch (domyślnie `/net/tscratch/people/$USER`) |
| `ENV_PREFIX` | Prefiks conda env (domyślnie `$SCRATCH_BASE/conda/py311_env`) |

Przykładowa siatka lokalna: `bash scripts/train_grid.sh`

## Zadanie

Model ukrywa część krawędzi CcSE, uczy się z reszty grafu (~2,25 mln krawędzi, 24 relacje) i ocenia, czy dany lek powoduje dany skutek uboczny.
