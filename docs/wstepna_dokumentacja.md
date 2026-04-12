# Dokumentacja Wstępna Projektu: Zaawansowane Sieci Neuronowe

**Temat:** Interpretowalność relacyjnych sieci grafowych (RGCN) w zadaniu Drug Repurposing na grafie wiedzy Hetionet.  
**Autorzy:** Jakub Jażdżyk, Kajetan Rożej  
**Data:** 12.04.2026  

---

## 1. Wprowadzenie i Kontekst Projektu
Współczesna medycyna stoi przed wyzwaniem ogromnej ilości danych, których powiązania wykraczają poza możliwości percepcyjne człowieka. Jednym z najbardziej obiecujących kierunków jest **Drug Repurposing** - poszukiwanie nowych zastosowań dla leków już dopuszczonych do obrotu. Projekt ten ma na celu budowę systemu, który nie tylko wskaże potencjalne nowe terapie, ale przede wszystkim „tłumacząc się” ze swojej decyzji, wskaże biologiczne ścieżki, które do niej doprowadziły.

Głównym problemem badawczym jest tzw. „Black-Box effect” w modelach Deep Learning. W medycynie sama predykcja („ten lek leczy tę chorobę”) jest niewystarczająca - lekarz i badacz potrzebują uzasadnienia w postaci konkretnych interakcji genetycznych i białkowych.

## 2. Cel i Hipotezy Badawcze
Celem projektu jest implementacja relacyjnej sieci grafowej (RGCN), która nauczy się semantyki grafu wiedzy Hetionet, a następnie zostanie poddana procesowi interpretacji przy użyciu metod XAI (Explainable AI).

**Hipotezy:**
1.  **H1 (Relacyjność):** Architektura RGCN, dzięki zastosowaniu osobnych macierzy wag dla każdego typu relacji, jest w stanie odróżnić krawędzie o znaczeniu pozytywnym (terapeutycznym) od negatywnych (skutki uboczne).
2.  **H2 (Redukcja Szumu):** Metoda GNNExplainer pozwoli na wyekstrahowanie z gęstego grafu (tzw. *hairball*) ścieżek o długości 2-3 skoków, które są zrozumiałe dla eksperta i biologicznie poprawne.

## 3. Charakterystyka Zbioru Danych (Hetionet v1.0)
Projekt bazuje na Hetionet – wielowarstwowym grafie wiedzy integrującym dane z dziesiątek baz biomedycznych.
*   **Węzły (47 031):** Podzielone na 11 typów (Leki, Choroby, Geny, Ścieżki sygnałowe, Anatomia).
*   **Krawędzie (2 250 197):** Podzielone na 24 typy relacji (meta-edges).
*   **Kluczowa relacja (Target):** `CtD` (Compound–treats–Disease). Tę relację model będzie musiał zgadywać, analizując pozostałe 23 typy połączeń.

Poniżej: **przykładowy podgraf** (BFS, 2 skoki, limit węzłów) wokół losowej krawędzi CtD - wygenerowany skryptem `scripts/visualize_hetionet_subgraph.py` z przetworzonych danych. Kolory = typ węzła (Compound, Disease, Gene itd.); pełny graf jest zbyt gęsty, żeby pokazać go na jednym slajdzie.

![Fragment grafu Hetionet (`hetionet_subgraph.png`)](../figures/hetionet_subgraph.png)

## 4. Metodyka i Architektura Rozwiązania

### 4.1. Zadanie Link Prediction (Uczenie Samo-nadzorowane)
Model zostanie wytrenowany w paradygmacie uzupełniania brakujących ogniw. Proces ten przebiega w trzech krokach:
1.  **Maskowanie:** Część istniejących relacji "lek-leczy-chorobę" zostaje ukryta przed modelem.
2.  **Uczenie:** Model analizuje pozostałe 2 miliony krawędzi (np. jak dany lek wpływa na geny, a jak te geny korelują z chorobą), budując wewnętrzną reprezentację wiedzy.
3.  **Weryfikacja:** Model musi ocenić prawdopodobieństwo istnienia krawędzi, których nie widział. Wysoki wynik dla par nieistniejących w bazie danych będzie traktowany jako potencjalne odkrycie nowego zastosowania leku.

### 4.2. Encoder: Relational Graph Convolutional Network (RGCN)
Sercem modelu jest sieć RGCN. Jej działanie opiera się na mechanizmie **Message Passing**:
*   Węzły wymieniają się informacjami (embeddingami) ze swoimi sąsiadami.
*   W odróżnieniu od standardowych sieci grafowych, w RGCN każda wiadomość przechodzi przez filtr (macierz wag) przypisany do konkretnego typu relacji.
*   Dzięki temu model „rozumie”, że informacja płynąca od genu jest inna niż informacja płynąca od objawu chorobowego.

### 4.3. Decoder: DistMult
Jako dekoder zostanie wykorzystana funkcja punktacji **DistMult**, która oblicza iloczyn skalarny między embeddingami leku i choroby, uwzględniając relację terapeutyczną. Pozwala to na uzyskanie końcowego prawdopodobieństwa sukcesu leczenia.

## 5. Interpretowalność (Explainable AI)
Kluczowym elementem projektu jest moduł wyjaśniający oparty na **GNNExplainer**. Proces generowania wyjaśnienia dla wybranej predykcji będzie przebiegał następująco:
1.  **Optymalizacja maski:** GNNExplainer przeprowadza serię symulacji, nakładając maski na krawędzie w otoczeniu wybranego leku.
2.  **Analiza istotności:** Algorytm identyfikuje, które krawędzie są kluczowe dla końcowego wyniku. Jeśli usunięcie połączenia z „Genem X” powoduje drastyczny spadek pewności modelu, krawędź ta jest uznawana za ważną.
3.  **Wizualizacja ścieżki dowodowej:** Wynikiem działania modułu będzie czytelny podgraf, pokazujący np. łańcuch: *Lek A -> Wiąże Gen B -> Współwystępuje z Chorobą C*.

## 6. Ewaluacja Projektu
Skuteczność systemu zostanie oceniona na dwóch poziomach:
*   **Poziom AI (Predykcja):** Wykorzystanie metryk AUC-ROC oraz MRR (Mean Reciprocal Rank), aby sprawdzić, jak dobrze model odnajduje ukryte połączenia medyczne.
*   **Poziom XAI (Wyjaśnienie):** 
    *   **Fidelity (Wierność):** Czy wyjaśnienie faktycznie odzwierciedla logikę modelu?
    *   **Sparsity (Zwięzłość):** Czy wyjaśnienie jest na tyle krótkie, by człowiek mógł je przeanalizować?
    *   **Analiza Jakościowa:** Weryfikacja wybranych przypadków z literaturą medyczną.

## 7. Harmonogram i Kamienie Milowe
1.  **Faza 1 (Zakończona):** Przygotowanie środowiska i preprocessing danych Hetionet do formatu PyTorch Geometric.
2.  **Faza 2 (W toku):** Implementacja i trening modelu RGCN przy użyciu techniki *Neighbor Sampling* (obsługa dużego grafu).
3.  **Faza 3 (Planowana):** Integracja z GNNExplainer i budowa modułu wizualizacji grafów wyjaśniających.
4.  **Faza 4 (Planowana):** Eksperymenty badawcze, zebranie metryk i przygotowanie raportu końcowego.

## 8. Ryzyka Projektowe
*   **Złożoność obliczeniowa:** Pełny graf Hetionet jest duży, co może wymagać optymalizacji zużycia pamięci VRAM.
*   **Niestabilność wyjaśnień:** Metody oparte na maskowaniu mogą być wrażliwe na hiperparametry, co będzie wymagało starannego dostrojenia modułu XAI.

---