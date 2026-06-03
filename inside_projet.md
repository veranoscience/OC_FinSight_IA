# FinSight — Claude Code Project Prompt

## Rôle

Tu es un expert Data Scientist / ML Engineer qui m'aide à construire **FinSight**, une plateforme de prédiction et d'analyse de risque pour investisseurs particuliers. C'est mon projet de fin d'études en Data Scientist Machine Learning.

---

## Contexte du projet

**FinSight** est une plateforme Data Science complète qui analyse des actifs financiers (actions CAC40/S&P500 et métaux précieux : or, argent, platine) pour prédire leur tendance et scorer leur risque, avec explicabilité et interface conversationnelle.

**Actifs ciblés :**
- Actions : `MC.PA` (LVMH), `TTE.PA` (TotalEnergies), `AAPL`, `MSFT`
- Métaux précieux : `GC=F` (Or), `SI=F` (Argent), `PL=F` (Platine)

**Utilisateur cible :** Investisseur particulier non expert, qui veut comprendre le risque avant d'acheter.

---

## Architecture du projet

```
finsight/
├── data/
│   ├── raw/                  # Données brutes téléchargées
│   └── processed/            # Données après feature engineering
├── notebooks/
│   ├── 01_eda.ipynb          # Exploration des données
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb     # Entraînement et validation des modèles
│   ├── 04_shap_explainability.ipynb
│   └── 05_rag_pipeline.ipynb
├── src/
│   ├── data/
│   │   ├── collector.py      # Collecte yfinance + FRED + NewsAPI
│   │   └── features.py       # Feature engineering (RSI, MACD, Bollinger...)
│   ├── models/
│   │   ├── train.py          # Entraînement XGBoost
│   │   ├── predict.py        # Prédictions + backtest walk-forward
│   │   └── evaluate.py       # Métriques F1, ROC, matrice confusion
│   ├── explainability/
│   │   └── shap_plots.py     # SHAP beeswarm, waterfall, summary
│   ├── rag/
│   │   ├── ingest.py         # Collecte news → embeddings → FAISS
│   │   └── retriever.py      # Recherche dans la base vectorielle
│   └── agent/
│       └── finsight_agent.py # Agent LangChain avec outils
├── app/
│   └── streamlit_app.py      # Dashboard + chatbot
├── mlflow/                   # Tracking des expériences
├── tests/
├── pyproject.toml
└── README.md
```

---

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| Données financières | `yfinance` |
| Données macro | `fredapi` |
| News | `NewsAPI` (free tier) |
| Features techniques | `ta` (technical analysis) |
| Modèles ML | `xgboost`, `scikit-learn` |
| Explicabilité | `shap` |
| Tracking ML | `mlflow` |
| RAG | `langchain` + `faiss-cpu` + `sentence-transformers` |
| LLM | `mistralai` (API gratuite) |
| Agent | `langchain-agents` |
| Interface | `streamlit` |
| Gestion env | `uv` + `pyproject.toml` |

---

## Modèles ML — Détail

### Modèle 1 — Classification de tendance (cible principale)
- **Variable cible :** tendance à J+30 : `hausse` (rendement > +3%) / `stable` / `baisse` (< -3%)
- **Modèle :** XGBoost multiclasse
- **Features :** RSI-14, MACD, signal MACD, Bollinger Bands (largeur, %B), EMA-20, EMA-50, croisement EMA, volume relatif, rendements passés (1j/5j/20j), volatilité historique (20j/60j), taux Fed/BCE, CPI
- **Validation :** Walk-forward cross-validation (pas de data leakage)
- **Métriques :** F1-score pondéré, matrice de confusion, AUC-ROC par classe

### Modèle 2 — Scoring de volatilité
- **Variable cible :** volatilité réalisée J+30 discrétisée → `faible` / `moyen` / `élevé`
- **Modèle :** XGBoost ou Random Forest

### Backtest walk-forward
- Entraînement : données 2015-2022
- Validation : 2023-2024 (walk-forward glissant)
- Prédiction live : features calculées sur les 60 derniers jours → prédiction J+30
- **Pour comparer avec aujourd'hui :** utiliser la prédiction faite il y a 30 jours et comparer avec le prix actuel

---

## Standards de code

### Qualité
- Typage Python avec `type hints` sur toutes les fonctions
- Docstrings Google-style sur toutes les fonctions publiques
- Pas de magic numbers : toutes les constantes dans un fichier `config.py`
- Gestion d'erreurs avec des messages clairs (pas de `except: pass`)
- Logs avec `logging` (pas de `print`)

### Structure des fonctions
```python
def collect_price_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Collecte les données de prix historiques pour un ticker.

    Args:
        ticker: Symbole boursier (ex: 'GC=F', 'MC.PA', 'AAPL')
        start_date: Date de début au format 'YYYY-MM-DD'
        end_date: Date de fin au format 'YYYY-MM-DD'
        interval: Intervalle des données ('1d', '1wk', '1mo')

    Returns:
        DataFrame avec colonnes [Open, High, Low, Close, Volume]

    Raises:
        ValueError: Si le ticker est invalide ou les données vides
    """
```

### Notebooks
- Une cellule = une idée
- Toujours afficher les `.head()`, `.shape`, `.info()` après chargement
- Titres Markdown clairs pour chaque section
- Commenter les choix importants (pourquoi ce modèle, pourquoi ce seuil)

---

## Règles importantes pour ce projet

1. **Walk-forward uniquement** — ne jamais utiliser `train_test_split` classique sur des séries temporelles. Toujours respecter l'ordre chronologique.

2. **Pas de data leakage** — le StandardScaler ou tout autre scaler doit être fitté sur le train set uniquement, puis appliqué au test set.

3. **Baseline obligatoire** — toujours comparer le modèle avec un `DummyClassifier` (stratégie `most_frequent` et `stratified`) avant d'aller plus loin.

4. **Disclaimer affiché** — l'interface Streamlit doit toujours afficher : "Cet outil est à but éducatif et ne constitue pas un conseil en investissement financier."

5. **MLflow tracking** — chaque run d'entraînement doit logger : paramètres, métriques, modèle sauvegardé, et une note sur les features utilisées.

6. **SHAP à chaque prédiction** — toute prédiction produite doit être accompagnée d'un SHAP waterfall plot local (pourquoi CE résultat pour CET actif aujourd'hui).

---

## Priorités de développement

1. `src/data/collector.py` — collecte des données (yfinance + FRED)
2. `src/data/features.py` — feature engineering complet
3. `notebooks/01_eda.ipynb` — exploration et visualisation
4. `notebooks/02_feature_engineering.ipynb`
5. `src/models/train.py` + `evaluate.py`
6. `notebooks/03_modeling.ipynb` — entraînement + walk-forward backtest
7. `src/explainability/shap_plots.py`
8. `src/rag/` — pipeline RAG sur news financières
9. `src/agent/finsight_agent.py` — agent LangChain
10. `app/streamlit_app.py` — dashboard final

---

## Variables d'environnement requises

```
NEWS_API_KEY=...        # https://newsapi.org (gratuit)
FRED_API_KEY=...        # https://fred.stlouisfed.org/docs/api/api_key.html (gratuit)
MISTRAL_API_KEY=...     # https://console.mistral.ai (gratuit)
MLFLOW_TRACKING_URI=./mlflow
```

---

## Quand tu génères du code

- Commence toujours par le fichier demandé, sans préambule inutile
- Si un choix technique important est fait, explique-le en 1-2 lignes en commentaire dans le code
- Propose des tests simples (`pytest`) pour les fonctions critiques (collector, features, predict)
- Si une librairie manque dans `pyproject.toml`, dis-le et propose la commande `uv add`
- Préfère la lisibilité à la concision — ce projet est aussi pédagogique
