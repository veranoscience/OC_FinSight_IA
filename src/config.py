"""
Configuration centrale de FinSight
Toutes les constantes du projet sont définies ici pour assurer la cohérence et faciliter les modifications futures
"""

from pathlib import Path

#  Chemins 

ROOT_DIR = Path(__file__).parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "mlflow"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

#  Actifs financiers 

TICKERS_STOCKS = ["MC.PA", "TTE.PA", "AAPL", "MSFT"]
TICKERS_METALS = ["GC=F", "SI=F", "PL=F"]
ALL_TICKERS = TICKERS_STOCKS + TICKERS_METALS

TICKER_LABELS = {
    "MC.PA": "LVMH",
    "TTE.PA": "TotalEnergies",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GC=F": "Or (Gold)",
    "SI=F": "Argent (Silver)",
    "PL=F": "Platine (Platinum)",
}

#  Collecte des données 

TRAIN_START_DATE = "2015-01-01"
TRAIN_END_DATE = "2022-12-31"
TEST_START_DATE = "2023-01-01"
TEST_END_DATE = "2024-12-31"

DATA_INTERVAL = "1d"

# Nombre de jours pour la fenêtre de feature engineering en live
LIVE_LOOKBACK_DAYS = 60

#  Modèle — Classification de tendance 

# Horizon de prédiction (en jours ouvrés)
PREDICTION_HORIZON_DAYS = 30

# Seuils pour la variable cible
TREND_UP_THRESHOLD = 0.03      # +3% → hausse
TREND_DOWN_THRESHOLD = -0.03   # -3% → baisse
# Entre les deux → stable

TREND_LABELS = {0: "baisse", 1: "stable", 2: "hausse"}
TREND_COLORS = {"baisse": "#e74c3c", "stable": "#f39c12", "hausse": "#27ae60"}

#  Modèle — Scoring de volatilité 

# Quantiles pour discrétiser la volatilité réalisée J+30
VOLATILITY_QUANTILES = [0.0, 0.33, 0.67, 1.0]
VOLATILITY_LABELS = {0: "faible", 1: "moyen", 2: "élevé"}
VOLATILITY_COLORS = {"faible": "#27ae60", "moyen": "#f39c12", "élevé": "#e74c3c"}

#  Features techniques 

RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2
EMA_SHORT = 20
EMA_LONG = 50
VOLUME_AVG_WINDOW = 20
RETURNS_WINDOWS = [1, 5, 20]       # jours pour rendements passés
VOLATILITY_WINDOWS = [20, 60]      # jours pour volatilité historique

#  Features macroéconomiques (FRED) 

# Séries FRED utilisées
FRED_SERIES = {
    "fed_rate": "FEDFUNDS",        # Taux directeur Fed
    "cpi": "CPIAUCSL",             # Inflation CPI US
    "us_10y": "DGS10",             # Taux 10 ans US
}

#  Walk-forward validation 

# Nombre de splits pour la validation walk-forward
WF_N_SPLITS = 5
# Taille minimale du set d'entraînement (en jours)
WF_MIN_TRAIN_SIZE = 504            # ~2 ans de données quotidiennes

#  XGBoost — hyperparamètres par défaut 

XGBOOST_DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": -1,
}

#  RAG 

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = DATA_PROCESSED_DIR / "faiss_index"
NEWS_LOOKBACK_DAYS = 7             # Fenêtre de collecte des news
RAG_TOP_K = 5                      # Nombre de chunks récupérés

#  MLflow 

MLFLOW_EXPERIMENT_TREND = "finsight-trend-classification"
MLFLOW_EXPERIMENT_VOLATILITY = "finsight-volatility-scoring"

#  Interface 

DISCLAIMER_AMF = (
    "Cet outil est à but éducatif et ne constitue pas un conseil "
    "en investissement financier. Les prédictions sont issues de modèles "
    "statistiques et ne garantissent pas les performances futures."
)
