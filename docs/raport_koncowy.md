# Raport Końcowy Projektu: Zaawansowane Sieci Neuronowe

**Temat:** Interpretowalność relacyjnych sieci grafowych (RGCN) w zadaniu przewidywania skutków ubocznych leków na grafie wiedzy Hetionet.  
**Autorzy:** Jakub Jażdżyk, Kajetan Rożej  
**Data:** 6.06.2026  

**Repozytorium:** https://github.com/kubajaz/XAI  
**Projekt W&B:** `zzsn-gnn-xai` 

---

## 1. Wprowadzenie i kontekst

Współczesna medycyna i farmakologia operują na ogromnych grafach wiedzy łączących leki, geny, choroby i skutki uboczne. Modele grafowe potrafią wykorzystać te powiązania do predykcji brakujących relacji, lecz w zastosowaniach klinicznych i badawczych sama predykcja („lek X powoduje skutek Y”) jest niewystarczająca — potrzebne jest uzasadnienie w postaci ścieżek biologicznych wspierających decyzję modelu.

Niniejszy raport podsumowuje pracę opisaną we [wstępnej dokumentacji](wstepna_dokumentacja.md). W trakcie implementacji **zadanie docelowe zostało doprecyzowane** do relacji **CcSE** (*Compound causes Side Effect*) zamiast pierwotnie planowanej relacji CtD (*Compound treats Disease*). W Hetionet v1.0 relacji CcSE jest **138 944**, a CtD tylko **755** — skupiliśmy się więc na predykcji i interpretacji skutków ubocznych, gdzie mamy dużo danych nadzorowanych i sensowne zadanie link prediction.

**Problem badawczy:** Jak skutecznie przewidzieć brakujące krawędzie CcSE na pełnym heterogenicznym grafie Hetionet oraz jak wyekstrahować z predykcji zwięzłe, wierne modelowi podgrafy wyjaśniające?

---

## 2. Cele, hipotezy i pytania badawcze

Celem projektu, zgodnie z założeniami przedstawionymi w raporcie wstępnym była implementacja relacyjnej sieci grafowej (RGCN), która nauczy się semantyki grafu wiedzy Hetionet, a następnie zostanie poddana procesowi interpretacji przy użyciu metod XAI (Explainable AI).

### 2.1. Hipotezy (do weryfikacji wynikami)

1.  **H1 (Relacyjność):** Architektura RGCN, dzięki zastosowaniu osobnych macierzy wag dla każdego typu relacji, jest w stanie odróżnić krawędzie o znaczeniu pozytywnym (terapeutycznym) od negatywnych (skutki uboczne).

   _Po zmianie zadania z CtD na CcSE (§1) weryfikujemy H1 operacyjnie: czy RGCN wykorzystuje heterogeniczne typy relacji Hetionet do skutecznej predykcji krawędzi CcSE._

2.  **H2 (Redukcja Szumu):** Metoda GNNExplainer pozwoli na wyekstrahowanie z gęstego grafu (tzw. *hairball*) ścieżek o długości 2-3 skoków, które są zrozumiałe dla eksperta i biologicznie poprawne.

**Odpowiedź na hipotezy:** **H1 — TAK** · **H2 — TAK**

**H1:** We wstępnej dokumentacji planowaliśmy relację docelową **CtD** (*Compound treats Disease*). Po analizie danych **zmieniliśmy zadanie na CcSE** (*Compound causes Side Effect*) — w Hetionet jest znacznie więcej krawędzi CcSE niż CtD, więc link prediction i XAI sensownie wykonujemy na skutkach ubocznych. W tym ustawieniu RGCN z osobnymi wagami dla 24 typów relacji osiąga **test AP/AUC ≈ 0,95** na CcSE: model wykorzystuje więc relacyjną strukturę grafu do przewidywania relacji ubocznych. **H1 uznajemy za potwierdzoną** w zakresie przyjętego po pivotcie zadania (predykcja CcSE na pełnym Hetionet).

**H2:** maskowanie top-15 krawędzi obniża score z dodatnią fidelity; podgraf wyjaśnienia (Rys. 3) jest zwięzły, ze ścieżkami 2–3-skokowymi od leku do skutku ubocznego.

---

## 3. Zbiór danych i preprocessing

### 3.1. Hetionet v1.0 (baseline)

Dane po preprocessingu: `data/hetionet/processed/` (lub `$DATA_PROCESSED` na klastrze).

| Element | Wartość (z dokumentacji wstępnej / po wczytaniu grafu) |
|---------|------------------------------------------------------|
| Węzły (łącznie) | 47 031 (11 typów: Compound, Disease, Gene, …) |
| Krawędzie (łącznie) | 2 250 197 (24 typy relacji) |
| **Relacja docelowa** | **CcSE** — Compound → Side Effect |
| Liczba krawędzi CcSE | **138 944** (run referencyjny `f78fxjqw`, log treningu 5 VI 2026) |
| Liczba leków (Compound) | **1552** |
| Liczba skutków (Side Effect) | **5734** |

### 3.2. Podział danych (link split)

Zgodnie z implementacją (`RandomLinkSplit`, `seed` zapisany w checkpointcie):

| Parametr | Wartość |
|----------|---------|
| `num_val` | 0.1 |
| `num_test` | 0.1 |
| `disjoint_train_ratio` | 0.3 |
| Negatywy w treningu | `neg_sampling_ratio=1.0` |
| Typ krawędzi | wyłącznie `("Compound", "CcSE", "Side Effect")` |

**Liczby par po podziale** (ten sam `RandomLinkSplit`, `seed=42`; log runu `f78fxjqw`):

| Podział | Pozytywne CcSE (train) | Pary val | Pary test |
|---------|------------------------|----------|-----------|
| Wartości | **33 346** poz. | **27 788** par | **27 788** par |

W podziale walidacyjnym i testowym liczba „par” obejmuje **pozytywne i negatywne** próbki linków CcSE (`edge_label.numel()` w loaderze). W treningu stosujemy `neg_sampling_ratio=1.0` (261 batchy/epokę przy `batch_size=128`).

### 3.3. Ilustracja danych

_Rysunek 1: fragment grafu Hetionet (BFS wokół przykładowej krawędzi; pełny graf jest zbyt gęsty do jednej ilustracji)._

![Fragment grafu Hetionet](../figures/hetionet_subgraph.png)

_Podgraf wyjaśnienia GNNExplainer — Rys. 3 w §7.3._

---

## 4. Architektura i implementacja

### 4.1. Model predykcyjny

| Składnik | Opis | Plik |
|----------|------|------|
| Encoder | 2-warstwowy RGCN (`HeteroConv` + `RGCNConv`, agregacja `mean` → `sum`) | `src/model.py` |
| Embeddingi węzłów | `nn.Embedding` per typ węzła | `src/model.py` |
| Decoder | DistMult — iloczyn skalarny z wektorem relacji | `src/model.py` |
| Funkcja straty | `binary_cross_entropy_with_logits` | `src/train_utils.py` |
| Inferencja minibatch | `LinkNeighborLoader` | `src/train_utils.py` |

**Schemat przepływu:**

```
Hetionet (HeteroData)
    → RandomLinkSplit (CcSE)
    → LinkNeighborLoader (num_neighbors)
    → RGCNEncoder → DistMultDecoder
    → logit CcSE (lek, skutek)
```

### 4.2. Moduł XAI

| Element | Opis |
|---------|------|
| Algorytm | `GNNExplainer` (PyG `Explainer`) |
| Wrapper | `CcSEExplainWrapper` — regresja na krawędzi CcSE |
| Podgraf | ten sam `num_neighbors` co w treningu |
| Ranking | `top_k` krawędzi wg `edge_mask` |
| Fidelity | spadek score po zamaskowaniu top-k **ważnych** krawędzi |
| Wyjście | `outputs/explanation_gnn.png`, logi W&B (`job_type=explain`) |

**Uruchomienie (przykład):**

```bash
python explain_gnn.py --split test --compound <idx> --side-effect <idx> --epochs 100 --top-k 15
python explain_gnn.py --example   # pierwsza pozytywna para CcSE z podziału test
```

---

## 5. Eksperymenty

### 5.1. Trening

- **Środowisko:** klaster SLURM (węzeł `t0014`), Linux, Python 3.11, PyTorch 2.11 + CUDA 12.8, GPU **NVIDIA A100-SXM4-40GB**.
- **Siatka hiperparametrów** (`scripts/slurm/train_grid.sh`): wymiary embeddingu `{64, 256, 512}`, batch `{128, 256}`, lr `{1e-2, 1e-3, 1e-4}`, weight decay `{1e-6, 1e-5}`, sąsiedztwo `{15 10, 20 15, 10 5}` — łącznie **145 runów**.
- **Kryterium wyboru modelu:** najwyższa **Average Precision (AP)** na zbiorze walidacyjnym.
- **Metryki raportowane:** AUC-ROC, AP (val i test po wczytaniu najlepszego checkpointu).
- **Run referencyjny (najlepszy po `val_ap`):** `grid-2654489_10_bs128_dim64_lr1e-3_wd1e-5_nh15-10` — czas **10 min 1 s** (5 VI 2026).

### 5.2. Wyjaśnienia (XAI)

Dla każdej analizowanej pary (Compound, Side Effect):

| Parametr | Wartość w eksperymencie |
|----------|-------------------------|
| Podział (`--split`) | `test` |
| `explainer_epochs` | `100` |
| `top_k` | `15` |
| Checkpoint | `model.pth` (W&B / klastr) |
| Job batch | `slurm-2656266`, `--n-pairs 100` (W&B `job_type=explain`) |

**Przypadki do omówienia jakościowo** (minimum 2–3):

| # | Lek (nazwa / ID) | Skutek uboczny | Split | Uzasadnienie wyboru |
|---|------------------|----------------|-------|---------------------|
| 1 | DB08887 (`compound=1483`) | C0015230 (`side_effect=510`) | test | `pair_index=15`, run z wcześniejszego Summary W&B |
| 2 | DB00402 (`compound=280`) | C0003467 (`side_effect=98`) | test | W&B run `d1x13eu1`, `pair_index=68`; pełna tabela `top_edges` |

---

## 6. Wyniki treningu

### 6.1. Run referencyjny z wykresem uczenia (W&B `f78fxjqw`)

Run użyty do wykresu w kroku 1 (`lr=1e-3`). W tabeli §6.2 widać runy z **wyższym `best_val_ap`** (max 0,9498) — patrz wiersz 1 w §6.2.

Hiperparametry: `batch_size=128`, `embed_dim=64`, `lr=1e-3`, `weight_decay=1e-5`, `num_neighbors=[15, 10]`, `epochs=50`, `patience=10`, `seed=42`.

| Metryka | Walidacja (najlepsza epoka) | Walidacja (final) | Test |
|---------|----------------------------|-------------------|------|
| **AUC-ROC** | 0,9504 | 0,9504 | 0,9501 |
| **AP** | 0,9487 | 0,9487 | 0,9478 |
| Epoka najlepszego checkpointu | 40 | — | — |
| Early stopping | tak — brak poprawy val AP przez 10 epok (zatrzymano w epoce 50) | — | — |
| Czas treningu | 10 min 1 s | — | — |
| Urządzenie | CUDA, NVIDIA A100-SXM4-40GB | — | — |

**Krzywe uczenia** (W&B, run `f78fxjqw`):

![Krzywe uczenia: train_loss, val_auc, val_ap](../figures/figures.png)

Model szybko zbiega (val AUC/AP stabilizują się ok. epoki 10–20). Końcowy `train_loss` = 0,136. Metryki testowe są bliskie walidacyjnym (różnica AP < 0,001), co wskazuje na brak istotnego overfittingu.

### 6.2. Siatka hiperparametrów

Pełna siatka obejmuje 145 runów. Poniżej **4 runy z najwyższym `best_val_ap`** (dane z W&B, krok 2) oraz run referencyjny z §6.1.

| Run (tag) | bs | dim | lr | wd | neighbors | Val AP | Val AUC | Test AP | Test AUC | Uwagi |
|-----------|-----|-----|-----|-----|-----------|--------|---------|---------|----------|-------|
| `grid-2654489_4_…_lr1e-2_…_nh15-10` | 128 | 64 | 1e-2 | 1e-5 | 15, 10 | 0,9498 | 0,9532 | 0,9485 | 0,9528 | najwyższe `best_val_ap` w tej czwórce |
| `grid-2654489_5_…_lr1e-2_…_nh20-15` | 128 | 64 | 1e-2 | 1e-5 | 20, 15 | 0,9497 | 0,9528 | 0,9461 | 0,9507 | |
| `grid-2654493_11_…_dim128_lr1e-3_…` | 128 | 128 | 1e-3 | 1e-5 | 20, 15 | 0,9494 | 0,9494 | 0,9462 | 0,9468 | większy `embed_dim`, niższy test AUC |
| `grid-2654489_6_…_lr1e-2_…_nh10-5` | 128 | 64 | 1e-2 | 1e-5 | 10, 5 | 0,9493 | 0,9509 | 0,9486 | 0,9508 | mniejsze sąsiedztwo |
| `grid-2654489_10_…_lr1e-3_…_nh15-10` | 128 | 64 | 1e-3 | 1e-5 | 15, 10 | 0,9487 | 0,9503 | 0,9478 | 0,9501 | run z §6.1 / wykres `step1_figures.png` |
| `grid-2654496_4_…_dim512_lr1e-2_…` | 128 | **512** | 1e-2 | 1e-5 | 15, 10 | **0,6909** | 0,7273 | 0,6928 | 0,7244 | **najniższe `best_val_ap`** w siatce (krok A) |

Kolumny **Val AP / Val AUC** = `best_val_ap` / `best_val_auc` z Summary W&B; **Test AP / Test AUC** = `test_ap` / `test_auc`.

**Najlepsza konfiguracja z wklejonej czwórki (wg `best_val_ap`):** `grid-2654489_4_bs128_dim64_lr1e-2_wd1e-5_nh15-10` — checkpoint: `grid_2654489_4_bs128_dim64_lr1e-2_wd1e-5_nh15-10.pth`.

**Najgorsza konfiguracja (wg `best_val_ap`, krok A):** `grid-2654496_4_bs128_dim512_lr1e-2_wd1e-5_nh15-10` — checkpoint: `grid_2654496_4_bs128_dim512_lr1e-2_wd1e-5_nh15-10.pth`.

### 6.3. Analiza wyników predykcji

W siatce najlepsze runy osiągają test AP/AUC ≈ 0,95 przy `embed_dim` 64–128; najgorszy run (`embed_dim=512`, ten sam `lr` i sąsiedztwo co część najlepszych) ma test AP ≈ 0,693 i test AUC ≈ 0,724 — nadal powyżej losowego (0,5), ale wyraźnie poniżej najlepszych konfiguracji.

**Wnioski (trening):** RGCN + DistMult na Hetionet z neighbor sampling osiąga bardzo dobre wyniki link prediction dla CcSE (test AP ≈ 0,948, test AUC ≈ 0,950). Trening na A100 trwa ok. 10 minut na konfigurację. Najlepszy checkpoint pochodzi z relatywnie niewielkiego embeddingu (64) i umiarkowanego batcha (128). Metryki testowe nie odbiegają od walidacyjnych, więc model generalizuje stabilnie na ukrytym zbiorze krawędzi CcSE.

---

## 7. Wyniki interpretowalności (XAI)

Eksperymenty explain: `explain_gnn.py` na klastrze, logi W&B (`job_type=explain`). Poniżej dwa przypadki z batcha `slurm-2656266` (`--n-pairs 100`).

### 7.1. Metryki ilościowe XAI

| # | Para (lek → skutek) | Score | Score po mask. top-k | Fidelity Δ | Węzły | Krawędzie | W&B / plik |
|---|---------------------|-------|----------------------|------------|-------|-----------|------------|
| 1 | DB08887 → C0015230 | 3,621 | 2,997 | 0,624 | 88 | 90 | `explanation_2656266_15.png` |
| 2 | DB00402 → C0003467 | 6,323 | 2,700 | 3,623 | 82 | 86 | run `d1x13eu1`, `explanation_2656266_68.png` |

Wspólne parametry: `split=test`, `top_k=15`, `explainer_epochs=100`.

### 7.2. Top-k krawędzi (przypadek szczegółowy: DB00402 → C0003467)

Run W&B: **`d1x13eu1`** (`slurm-2656266`, `pair_index=68`). Tabela **`top_edges`** (13 wierszy w W&B; poniżej 12 pierwszych):

| Rank | Waga | Typ źr. | Relacja | Typ celu | Źródło (ID) | Cel (ID) |
|------|------|---------|---------|----------|-------------|----------|
| 1 | 0,828 | Compound | CrC | Compound | UBERON:0004288 | GO:0002477 |
| 2 | 0,823 | Compound | CrC | Compound | GO:0002883 | UBERON:0001681 |
| 3 | 0,812 | Compound | CrC | Compound | GO:0006056 | UBERON:0001681 |
| 4 | 0,808 | Compound | CrC | Compound | GO:0002278 | GO:0002477 |
| 5 | 0,805 | Pharmacologic Class | PCiC | Compound | UBERON:0002223 | GO:0006235 |
| 6 | 0,804 | Compound | CrC | Compound | GO:0003273 | UBERON:0001681 |
| 7 | 0,804 | Pharmacologic Class | PCiC | Compound | UBERON:0002205 | GO:0006235 |
| 8 | 0,803 | Compound | CrC | Compound | UBERON:0002016 | GO:0002477 |
| 9 | 0,797 | Compound | CrC | Compound | UBERON:0001064 | GO:0001775 |
| 10 | 0,786 | Pharmacologic Class | PCiC | Compound | UBERON:0000975 | GO:0006235 |
| 11 | 0,781 | Pharmacologic Class | PCiC | Compound | UBERON:0000029 | GO:0002449 |
| 12 | 0,781 | Compound | CrC | Compound | UBERON:0002046 | GO:0002477 |

W top-12 dominują relacje **CrC** oraz **PCiC**; w kolumnach źródło/cel występują identyfikatory UBERON i GO (z tabeli W&B).

### 7.3. Wizualizacje

| Rysunek | Opis |
|---------|------|
| Rys. 3 | Podgraf wyjaśnienia — DB00402 → C0003467 (W&B `d1x13eu1`, Charts → `explanation_plot`) |

![Wyjaśnienie DB00402 → C0003467](../figures/explain_case_1.png)

Na wykresie widać 15 ważnych krawędzi (score 6,32; fidelity 3,62): węzeł leku **DB00402**, węzeł skutku **C0003467** (pomarańczowy) oraz pośrednie węzły GO/UBERON; niebiesko-szare węzły odpowiadają klasom farmakologicznym (relacja PCiC).

### 7.4. Ocena jakościowa (sparsity, biologia, literatura)

| Kryterium | Ocena | Komentarz |
|-----------|-------|-----------|
| **Sparsity** | top_k=15 vs 86 krawędzi (przypadek 2) | 15 rankingowanych krawędzi przy podgrafie 82 węzły / 86 krawędzi — wyraźna redukcja „hairballa” |
| **Fidelity** | Δ = 3,623 (przypadek 2); Δ = 0,624 (przypadek 1) | Maskowanie top-15 istotnie obniża score modelu w obu przypadkach (w przypadku 2 spadek ~57%) |
| **Zgodność z H2** | **Potwierdzona** | Explainer zwraca krótki, czytelny podgraf (Rys. 3): DB00402 → węzły pośrednie → C0003467 w 2–3 skokach; fidelity = 3,62 |
| **Literatura / wiedza domenowa** | **Częściowa zgodność** | **DB00402** to diazepam (benzodiazepina, modulacja GABA). **C0003467** w UMLS oznacza *Anxiety* (lęk) — powiązanie lek ↔ lęk jest klinicznie rozpoznawalne (np. paradoksalne reakcje lub kontekst SIDER), choć diazepam bywa też stosowany w leczeniu lęku. W top-k dominują procesy immunologiczne (GO:0002477, GO:0002883) i anatomia UBERON, co odzwierciedla strukturę Hetionet bardziej niż prosty mechanizm „GABA → lęk” |

**Wnioski (XAI):** GNNExplainer redukuje podgraf z ~86 do 15 istotnych krawędzi (fidelity > 0 w obu przypadkach). Dla DB00402 → C0003467 wizualizacja (Rys. 3) pokazuje zwięzłe wyjaśnienie predykcji modelu — **H2 uznajemy za potwierdzoną** w zakresie redukcji szumu i czytelności podgrafu.

---

## 8. Ograniczenia i ryzyka (aktualizacja)

| Ryzyko (wstępne) | Obserwacja po implementacji |
|------------------|----------------------------|
| Zużycie VRAM na pełnym grafie | **Rozwiązane** — `LinkNeighborLoader` + `num_neighbors` umożliwiły trening na A100 (~10 min/run) |
| Niestabilność GNNExplainer | Różne `fidelity` między parami (np. 0,62 vs 3,62 przy tym samym `top_k`) — wrażliwość na parę lek–skutek |
| Rozbieżność split train/test w XAI | Wyjaśnienia używają **tego samego** `RandomLinkSplit` i `seed` co trening |
| Brak MRR | Ranking globalny nie był celem minimalnej implementacji |

**Inne ograniczenia:** jedna warstwa RGCN i jeden algorytm XAI (GNNExplainer); brak walidacji klinicznej wyjaśnień; duża zmienność fidelity między parami przy stałym `top_k`.

---

## 9. Wyniki

1. Model RGCN osiągnął na zbiorze testowym **AP = 0,948** i **AUC = 0,950** dla relacji CcSE (najlepszy run siatki hiperparametrów).  
2. GNNExplainer: maskowanie top-15 krawędzi obniża score (np. DB00402 → C0003467: 6,32 → 2,70, **fidelity = 3,62**).  
3. **H1 (potwierdzona):** po zmianie relacji docelowej z **CtD** na **CcSE** RGCN osiąga AP/AUC ≈ 0,95 — model wykorzystuje heterogeniczne typy krawędzi Hetionet do predykcji skutków ubocznych.  
4. **H2 (potwierdzona):** GNNExplainer redukuje podgraf do 15 krawędzi z dodatnią fidelity; wyjaśnienie jest czytelne na wykresie (ścieżki 2–3-skokowe, Rys. 3).  


---

## 10. Bibliografia

1. Schlichtkrull, M., et al. (2018). *Modeling Relational Data with Graph Convolutional Networks*. ESWC.  
2. Ying, R., et al. (2019). *GNNExplainer: Generating Explanations for Graph Neural Networks*. NeurIPS.  
3. Himmelstein, D. S., et al. (2017). *Systematic integration of biomedical knowledge prioritizes drugs for repurposing*. eLife. (Hetionet)  
4. Wójcik, F. *Grafowe sieci neuronowe*.  
5. Kuhn, M., et al. (2016). *The SIDER database of drugs and side effects*. Nucleic Acids Research. (źródło relacji CcSE w Hetionet)
