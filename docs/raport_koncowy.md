# Raport Końcowy Projektu: Zaawansowane Sieci Neuronowe

**Temat:** Interpretowalność relacyjnych sieci grafowych (RGCN) w zadaniu przewidywania skutków ubocznych leków na grafie wiedzy Hetionet.  
**Autorzy:** Jakub Jażdżyk, Kajetan Rożej  
**Data:** _[uzupełnij datę złożenia]_  

**Repozytorium:** `/path/to/XAI`  
**Projekt W&B:** `zzsn-gnn-xai`  

---

## 1. Wprowadzenie i kontekst

Współczesna medycyna i farmakologia operują na ogromnych grafach wiedzy łączących leki, geny, choroby i skutki uboczne. Modele grafowe potrafią wykorzystać te powiązania do predykcji brakujących relacji, lecz w zastosowaniach klinicznych i badawczych sama predykcja („lek X powoduje skutek Y”) jest niewystarczająca — potrzebne jest uzasadnienie w postaci ścieżek biologicznych wspierających decyzję modelu.

Niniejszy raport podsumowuje pracę opisaną we [wstępnej dokumentacji](wstepna_dokumentacja.md). W trakcie implementacji **zadanie docelowe zostało doprecyzowane** do relacji **CcSE** (*Compound causes Side Effect*) zamiast pierwotnie planowanej relacji CtD (*Compound treats Disease*). Było to spowodowane zauważoną w trakcie eksperymentów znaczną dysproporcją pomiędzy liczbą relacji Ctd as CcSE zawartą w badnym grafie.

**Problem badawczy:** Jak skutecznie przewidzieć brakujące krawędzie CcSE na pełnym heterogenicznym grafie Hetionet oraz jak wyekstrahować z predykcji zwięzłe, wierne modelowi podgrafy wyjaśniające?

---

## 2. Cele, hipotezy i pytania badawcze

Celem projektu, zgodnie z założeniami przedstawionymi w raporcie wstępnym była implementacja relacyjnej sieci grafowej (RGCN), która nauczy się semantyki grafu wiedzy Hetionet, a następnie zostanie poddana procesowi interpretacji przy użyciu metod XAI (Explainable AI).

### 2.1. Hipotezy (do weryfikacji wynikami)

1.  **H1 (Relacyjność):** Architektura RGCN, dzięki zastosowaniu osobnych macierzy wag dla każdego typu relacji, jest w stanie odróżnić krawędzie o znaczeniu pozytywnym (terapeutycznym) od negatywnych (skutki uboczne).

2.  **H2 (Redukcja Szumu):** Metoda GNNExplainer pozwoli na wyekstrahowanie z gęstego grafu (tzw. *hairball*) ścieżek o długości 2-3 skoków, które są zrozumiałe dla eksperta i biologicznie poprawne.

**Odpowiedź na hipotezy:** _[po analizie sekcji 6 i 7 — TAK/NIE/CZĘŚCIOWO + uzasadnienie]_

---

## 3. Zbiór danych i preprocessing

### 3.1. Hetionet v1.0 (baseline)

Dane po preprocessingu: `data/hetionet/processed/` (lub `$DATA_PROCESSED` na klastrze).

| Element | Wartość (z dokumentacji wstępnej / po wczytaniu grafu) |
|---------|------------------------------------------------------|
| Węzły (łącznie) | 47 031 (11 typów: Compound, Disease, Gene, …) |
| Krawędzie (łącznie) | ~2 250 197 (24 typy relacji) |
| **Relacja docelowa** | **CcSE** — Compound → Side Effect |
| Liczba krawędzi CcSE | _[uzupełnij z logu `train.py`: „CcSE: … relacji”]_ |
| Liczba leków (Compound) | _[uzupełnij]_ |
| Liczba skutków (Side Effect) | _[uzupełnij]_ |

### 3.2. Podział danych (link split)

Zgodnie z implementacją (`RandomLinkSplit`, `seed` zapisany w checkpointcie):

| Parametr | Wartość |
|----------|---------|
| `num_val` | 0.1 |
| `num_test` | 0.1 |
| `disjoint_train_ratio` | 0.3 |
| Negatywy w treningu | `neg_sampling_ratio=1.0` |
| Typ krawędzi | wyłącznie `("Compound", "CcSE", "Side Effect")` |

**Liczby par po podziale:**

| Podział | Pozytywne CcSE (train) | Pary val | Pary test |
|---------|------------------------|----------|-----------|
| Wartości | _[uzupełnij z logu treningu]_ | _[…]_ | _[…]_ |

### 3.3. Ilustracja danych

_Rysunek 1: fragment grafu Hetionet (opcjonalnie ten sam co we wstępnej dokumentacji)._

![Fragment grafu Hetionet](../figures/hetionet_subgraph.png)

_Rysunek 2: przykładowy podgraf z neighbor loadera dla wyjaśnienia — do wstawienia po uruchomieniu `explain_gnn.py`._

![Podgraf wyjaśnienia](../figures/_PLACEHOLDER_explanation_subgraph.png)

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

- **Środowisko:** _[lokalne / SLURM, GPU, wersja PyTorch, commit git: `…`]_
- **Siatka hiperparametrów** (`scripts/slurm/train_grid.sh`): wymiary embeddingu `{64, 256, 512}`, batch `{128, 256}`, lr `{1e-2, 1e-3, 1e-4}`, weight decay `{1e-6, 1e-5}`, sąsiedztwo `{15 10, 20 15, 10 5}`.
- **Kryterium wyboru modelu:** najwyższa **Average Precision (AP)** na zbiorze walidacyjnym.
- **Metryki raportowane:** AUC-ROC, AP (val i test po wczytaniu najlepszego checkpointu).

### 5.2. Wyjaśnienia (XAI)

Dla każdej analizowanej pary (Compound, Side Effect):

| Parametr | Wartość w eksperymencie |
|----------|-------------------------|
| Podział (`--split`) | _[train / val / test]_ |
| `explainer_epochs` | _[np. 100]_ |
| `top_k` | _[np. 15]_ |
| Nazwy węzłów (Hetionet ID) | z `hetionet_v1_baseline_maps.pkl` |

**Przypadki do omówienia jakościowo** (minimum 2–3):

| # | Lek (nazwa / ID) | Skutek uboczny | Split | Uzasadnienie wyboru |
|---|------------------|----------------|-------|---------------------|
| 1 | _[…]_ | _[…]_ | test | _[np. wysoki score, znany z literatury]_ |
| 2 | _[…]_ | _[…]_ | test | _[…]_ |
| 3 | _[…]_ | _[…]_ | val | _[opcjonalnie: fałszywy pozytyw / niski score]_ |

---

## 6. Wyniki treningu

> **Sekcja do uzupełnienia po zakończeniu eksperymentów.** Skopiuj wartości z konsoli `train.py`, checkpointu lub panelu W&B.

### 6.1. Run referencyjny (domyślne hiperparametry)

| Metryka | Walidacja (najlepsza epoka) | Walidacja (final) | Test |
|---------|----------------------------|-------------------|------|
| **AUC-ROC** | _[…]_ | _[…]_ | _[…]_ |
| **AP** | _[…]_ | _[…]_ | _[…]_ |
| Epoka najlepszego checkpointu | _[…]_ | — | — |
| Early stopping | _[tak/nie]_ | — | — |
| Czas treningu | _[…]_ | — | — |
| Urządzenie | _[CPU/CUDA, model GPU]_ | — | — |

**Krzywe uczenia** (opcjonalnie wstaw wykresy z W&B):

- `train_loss` vs epoka: _[link do panelu / załącznik]_
- `val_auc`, `val_ap` vs epoka: _[…]_

### 6.2. Siatka hiperparametrów

| Run (tag) | bs | dim | lr | wd | neighbors | Val AP | Val AUC | Test AP | Test AUC | Uwagi |
|-----------|-----|-----|-----|-----|-----------|--------|---------|---------|----------|-------|
| _bs256_dim64_lr1e-3_…_ | _[…]_ | _[…]_ | _[…]_ | _[…]_ | _[…]_ | _[…]_ | _[…]_ | _[…]_ | _[…]_ | |
| _[kolejne wiersze]_ | | | | | | | | | | |

**Najlepsza konfiguracja z siatki:** _[tag + ścieżka checkpointu]_

### 6.3. Analiza wyników predykcji

_[Krótki komentarz: czy AP/AUC są zadowalające względem losowego klasyfikatora (0.5 AUC); czy widać overfitting; wpływ `embed_dim` i `num_neighbors`; zgodność z H1.]_

**Wnioski (trening):** _[3–5 zdań]_

---

## 7. Wyniki interpretowalności (XAI)

> **Sekcja do uzupełnienia po uruchomieniu `explain_gnn.py` dla wybranych par.**

### 7.1. Metryki ilościowe XAI

Dla każdego przypadku z tabeli w §5.2:

| # | Score modelu | Score po maskowaniu top-k | Fidelity Δ | Węzły w podgrafie | Krawędzie w podgrafie | Plik wizualizacji |
|---|--------------|---------------------------|------------|-------------------|----------------------|-------------------|
| 1 | _[…]_ | _[…]_ | _[…]_ | _[…]_ | _[…]_ | `outputs/explanation_gnn.png` |
| 2 | _[…]_ | _[…]_ | _[…]_ | _[…]_ | _[…]_ | _[…]_ |
| 3 | _[…]_ | _[…]_ | _[…]_ | _[…]_ | _[…]_ | _[…]_ |

**Średnia / mediana fidelity** (opcjonalnie): _[…]_

### 7.2. Top-k krawędzi (przykład szczegółowy)

**Para:** _[nazwa leku]_ → _[nazwa skutku]_ (`compound=…`, `side-effect=…`, split=_test_)

| Rank | Waga maski | Źródło | Relacja | Cel |
|------|------------|--------|---------|-----|
| 1 | _[…]_ | _[…]_ | _[…]_ | _[…]_ |
| 2 | _[…]_ | _[…]_ | _[…]_ | _[…]_ |
| … | | | | |

_Wklej tabelę z stdout `explain_gnn.py` lub eksportu W&B._

### 7.3. Wizualizacje

| Rysunek | Opis |
|---------|------|
| Rys. 3 | Podgraf wyjaśnienia — przypadek 1 |
| Rys. 4 | Podgraf wyjaśnienia — przypadek 2 |

![Wyjaśnienie przypadek 1](../outputs/_PLACEHOLDER_case1.png)  
![Wyjaśnienie przypadek 2](../outputs/_PLACEHOLDER_case2.png)

### 7.4. Ocena jakościowa (sparsity, biologia, literatura)

| Kryterium | Ocena | Komentarz |
|-----------|-------|-----------|
| **Sparsity** | _[np. 15 krawędzi vs setki w podgrafie]_ | _[…]_ |
| **Fidelity** | _[wysoka/niska]_ | Czy usunięcie top-k istotnie obniża score? |
| **Zgodność z H2** | _[…]_ | Czy ścieżki mają 2–3 sensowne skoki (np. lek–gen–skutek)? |
| **Literatura** | _[…]_ | _[PubMed / DrugBank / opis znanych mechanizmów]_ |

**Wnioski (XAI):** _[3–5 zdań; odniesienie do H2]_

---

## 8. Ograniczenia i ryzyka (aktualizacja)

| Ryzyko (wstępne) | Obserwacja po implementacji |
|------------------|----------------------------|
| Zużycie VRAM na pełnym grafie | _[neighbor sampling rozwiązał / częściowo]_ |
| Niestabilność GNNExplainer | _[wrażliwość na `epochs`, `top_k`, `lr` explainera]_ |
| Rozbieżność split train/test w XAI | Wyjaśnienia używają **tego samego** `RandomLinkSplit` i `seed` co trening |
| Brak MRR | Ranking globalny nie był celem minimalnej implementacji |

**Inne ograniczenia:** _[np. tylko jedna warstwa interpretacji, brak walidacji przez eksperta medycznego, ang. nazwy węzłów]_

---

## 9. Wyniki

1. _[Wynik predykcji CcSE — jedno zdanie z liczbami z §6]_  
2. _[Wynik XAI — jedno zdanie z §7]_  
3. _[Odpowiedź na H1]_  
4. _[Odpowiedź na H2]_  


---

## 10. Bibliografia

1. Schlichtkrull, M., et al. (2018). *Modeling Relational Data with Graph Convolutional Networks*. ESWC.  
2. Ying, R., et al. (2019). *GNNExplainer: Generating Explanations for Graph Neural Networks*. NeurIPS.  
3. Himmelstein, D. S., et al. (2017). *Systematic integration of biomedical knowledge prioritizes drugs for repurposing*. eLife. (Hetionet)  
4. Wójcik, F. *Grafowe sieci neuronowe*.  
5. _[dodatkowe źródła użyte przy analizie jakościowej przypadków XAI]_
