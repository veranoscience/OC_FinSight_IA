# FinSight — Plateforme de Prédiction & Analyse de Risque Financier

> Projet de fin d'études — Formation Data Scientist Machine Learning

FinSight est une plateforme Data Science complète qui analyse des actifs financiers (actions CAC40/S&P500 et métaux précieux) pour prédire leur tendance à J+30 et scorer leur risque, avec explicabilité SHAP et interface conversationnelle via un agent IA.

---

## Fonctionnalités

- **Prédiction de tendance** — classification XGBoost multiclasse : hausse / stable / baisse à J+30
- **Scoring de risque** — volatilité réalisée discrétisée : faible / moyen / élevé
- **Explicabilité** — SHAP waterfall plot pour chaque prédiction (pourquoi ce résultat ?)
- **RAG sur news** — recherche sémantique dans les actualités financières des 7 derniers jours
- **Agent conversationnel** — chatbot FinSight (Mistral + tool calling natif)
- **Dashboard interactif** — Streamlit avec 4 onglets : Prédiction, Graphiques, Actualités, Chatbot

---

## Actifs couverts

| Ticker | Actif |
|--------|-------|
| `MC.PA` | LVMH |
| `TTE.PA` | TotalEnergies |
| `AAPL` | Apple |
| `MSFT` | Microsoft |
| `GC=F` | Or (Gold) |
| `SI=F` | Argent (Silver) |
| `PL=F` | Platine (Platinum) |

---

## Architecture

```
finsight/
├── app/
│   └── streamlit_app.py        # Dashboard Streamlit 
├── data/
│   ├── raw/                    # Données brutes 
│   └── processed/              # Features CSV + modèles sauvegardés + index FAISS
├── notebooks/
│   ├── 01_eda.ipynb            # Analyse exploratoire des données
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb       # Walk-forward, baseline, GridSearch
│   ├── 04_shap_explainability.ipynb
│   └── 05_rag_pipeline.ipynb   # Ingestion news -> FAISS -> RAG
├── scripts/
│   └── train_pipeline.py       # Pipeline complet feature engineering + entraînement
├── src/
│   ├── config.py               
│   ├── data/
│   │   ├── collector.py        # yfinance + FRED API + NewsAPI
│   │   └── features.py         
│   ├── models/
│   │   ├── train.py            # Walk-forward CV + XGBoost + MLflow
│   │   ├── evaluate.py         # F1, ROC, matrice de confusion
│   │   └── predict.py          # Prédiction live + backtest
│   ├── explainability/
│   │   └── shap_plots.py       # Beeswarm, waterfall, bar global
│   ├── rag/
│   │   ├── ingest.py           # Chunking + embeddings + FAISS
│   │   └── retriever.py        # Recherche sémantique + génération RAG
│   └── agent/
│       └── finsight_agent.py   # Agent Mistral (tool calling natif)
├── tests/                     
├── mlflow/                  
└── pyproject.toml
```

---

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| Données financières | `yfinance` |
| Données macroéconomiques | `fredapi` (Fed Rate, CPI, US 10Y) |
| Actualités financières | `NewsAPI` |
| Indicateurs techniques | `ta` (Technical Analysis) |
| Modèles ML | `xgboost`, `scikit-learn` |
| Explicabilité | `shap` (TreeExplainer) |
| Tracking ML | `mlflow` |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2, dim=384) |
| Base vectorielle | `faiss-cpu` (IndexFlatIP, similarité cosinus) |
| LLM | `mistralai` (mistral-small-latest) |
| Interface | `streamlit` |
| Gestion d'environnement | `uv` |

---

## Installation

### Prérequis

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) installé

### 1. Cloner et installer les dépendances

```bash
git clone <url-du-repo>
cd finsight
uv sync
```

### 2. Configurer les variables d'environnement

Créer un fichier `.env` à la racine :

```env
MISTRAL_API_KEY=...     # https://console.mistral.ai (gratuit)
NEWS_API_KEY=...        # https://newsapi.org (gratuit)
FRED_API_KEY=...        # https://fred.stlouisfed.org/docs/api/api_key.html (gratuit)
MLFLOW_TRACKING_URI=./mlflow
```

---

## Lancement

### Pipeline complet (feature engineering + entraînement)

À exécuter une première fois pour générer les modèles :

```bash
uv run python scripts/train_pipeline.py
```

### Dashboard Streamlit

```bash
uv run streamlit run app/streamlit_app.py
```

### Tests

```bash
uv run python -m pytest tests/ -v
```

### MLflow UI (suivi des expériences)

```bash
uv run mlflow ui --backend-store-uri ./mlflow
```

Puis ouvrir [http://localhost:5000](http://localhost:5000).

---

## Modèles ML

### Méthodologie : Walk-forward cross-validation

La validation temporelle est au cœur du projet. On ne mélange jamais passé et futur :

```
|── Train ──|── Test ──|
             |── Train ──|── Test ──|
                          |── Train ──|── Test ──|
```

- **Données d'entraînement** : 2015–2022
- **Données de test** : 2023–2024 (5 splits walk-forward)
- **Taille minimale du train** : 504 jours (2 ans)

### Features utilisées 

**Indicateurs techniques (15)** : RSI-14, MACD, signal MACD, histogramme MACD, Bollinger %B, Bollinger width, EMA-20, EMA-50, croisement EMA, volume relatif, rendements 1j/5j/20j, volatilité historique 20j/60j.

**Macroéconomiques (3)** : Taux directeur Fed, CPI US, Taux 10 ans US.

---

## Pipeline RAG

Les actualités financières des 7 derniers jours sont indexées dans FAISS pour permettre une recherche sémantique :

```
NewsAPI -> texte brut -> chunking (800 chars, overlap 100)
       -> embeddings (all-MiniLM-L6-v2, dim=384)
       -> FAISS IndexFlatIP (cosine similarity)
       -> Retrieval top-5 -> Mistral (génération RAG)
```

---

## Agent conversationnel

L'agent FinSight utilise le **tool calling natif de Mistral** (pas LangChain Agents) avec une boucle ReAct manuelle. Il dispose de 3 outils :

| Outil | Description |
|-------|-------------|
| `predict_asset` | Prédiction tendance + risque pour un actif |
| `search_news` | Recherche sémantique dans les news récentes |
| `market_summary` | Rapport complet ML + news pour un actif |

---

## Disclaimer

> Cet outil est à but éducatif et ne constitue pas un conseil en investissement financier. Les prédictions sont issues de modèles statistiques et ne garantissent pas les performances futures.
