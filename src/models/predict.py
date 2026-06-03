"""
Prédictions live pour FinSight.

Ce fichier répond à une seule question :
  "Étant donné les données des 60 derniers jours, que va faire cet actif dans 30 jours ?"

Flux :
  1. Charger les données récentes (yfinance + FRED)
  2. Calculer les features sur la fenêtre glissante
  3. Appliquer le scaler (fitté sur le train, jamais re-fitté ici)
  4. Retourner classe prédite + probabilités
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import (
    DATA_PROCESSED_DIR,
    LIVE_LOOKBACK_DAYS,
    PREDICTION_HORIZON_DAYS,
    TICKER_LABELS,
    TREND_LABELS,
    VOLATILITY_LABELS,
)
from src.data.collector import collect_fred_data, collect_price_data
from src.data.features import build_feature_matrix, get_feature_names

logger = logging.getLogger(__name__)

# Chemins où train.py sauvegarde les modèles et scalers
MODELS_SAVE_DIR = DATA_PROCESSED_DIR / "models"


# ─── Sauvegarde / chargement des modèles ─────────────────────────────────────


def save_model(
    model: XGBClassifier,
    scaler: StandardScaler,
    ticker: str,
    model_type: str,
) -> None:
    """
    Sauvegarde le modèle et le scaler sur disque.

    On sauvegarde le scaler séparément du modèle MLflow parce qu'on en a
    besoin ici pour transformer les features live sans refaire de fit.

    Args:
        model: Modèle XGBClassifier entraîné.
        scaler: StandardScaler fitté sur le dernier split train.
        ticker: Symbole boursier (ex: 'AAPL').
        model_type: 'trend' ou 'volatility'.
    """
    MODELS_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_SAVE_DIR / f"{ticker}_{model_type}_model.json"
    scaler_path = MODELS_SAVE_DIR / f"{ticker}_{model_type}_scaler.pkl"

    model.save_model(str(model_path))
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    logger.info("Modèle sauvegardé : %s", model_path)
    logger.info("Scaler sauvegardé : %s", scaler_path)


def load_model(
    ticker: str,
    model_type: str,
) -> tuple[XGBClassifier, StandardScaler]:
    """
    Charge le modèle et le scaler depuis le disque.

    Args:
        ticker: Symbole boursier.
        model_type: 'trend' ou 'volatility'.

    Returns:
        Tuple (model, scaler).

    Raises:
        FileNotFoundError: Si le modèle n'a pas encore été entraîné.
    """
    model_path = MODELS_SAVE_DIR / f"{ticker}_{model_type}_model.json"
    scaler_path = MODELS_SAVE_DIR / f"{ticker}_{model_type}_scaler.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable pour {ticker} ({model_type}). "
            f"Lance d'abord train.py pour entraîner le modèle."
        )

    model = XGBClassifier()
    model.load_model(str(model_path))

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    logger.info("Modèle chargé : %s", model_path)
    return model, scaler


# ─── Prédiction live ──────────────────────────────────────────────────────────


def predict_live(
    ticker: str,
    model_type: str = "trend",
    lookback_days: int = LIVE_LOOKBACK_DAYS,
) -> dict:
    """
    Produit une prédiction live pour un ticker donné.

    Collecte les données des `lookback_days` derniers jours, calcule les
    features, et retourne la prédiction du modèle chargé depuis le disque.

    Args:
        ticker: Symbole boursier (ex: 'AAPL', 'MC.PA').
        model_type: 'trend' (hausse/stable/baisse) ou 'volatility' (faible/moyen/élevé).
        lookback_days: Nombre de jours de données à collecter pour les features.
                       Doit être > 60 pour que toutes les features soient calculables.

    Returns:
        Dictionnaire contenant :
        - ticker : symbole
        - label : nom lisible de l'actif
        - prediction_class : int (0, 1, 2)
        - prediction_label : str (ex: 'hausse')
        - probabilities : dict {label: probabilité}
        - current_price : dernier prix de clôture
        - prediction_date : date de la prédiction
        - horizon_days : horizon de prédiction en jours

    Raises:
        FileNotFoundError: Si le modèle n'a pas été entraîné.
        ValueError: Si les features ne peuvent pas être calculées.
    """
    logger.info("Prédiction live : %s (%s)", ticker, model_type)

    model, scaler = load_model(ticker, model_type)

    # On prend lookback_days + marge pour absorber les NaN de début de série
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.today() - pd.Timedelta(days=lookback_days + 90)).strftime("%Y-%m-%d")

    price_df = collect_price_data(ticker, start_date=start_date, end_date=end_date)
    macro_df = collect_fred_data(start_date=start_date, end_date=end_date)

    # Calcul des features sans targets (on prédit, on ne connaît pas le futur)
    feat_df = build_feature_matrix(price_df, macro_df=macro_df, add_targets=False)

    feature_cols = get_feature_names(include_macro=True)
    available = [c for c in feature_cols if c in feat_df.columns]
    feat_df = feat_df.dropna(subset=available)

    if feat_df.empty:
        raise ValueError(
            f"Impossible de calculer les features pour {ticker}. "
            f"Vérifiez que les données sont disponibles."
        )

    # On prend uniquement la dernière ligne = aujourd'hui
    X_live = feat_df[available].iloc[[-1]]
    X_live_scaled = scaler.transform(X_live)

    pred_class = int(model.predict(X_live_scaled)[0])
    pred_proba = model.predict_proba(X_live_scaled)[0]

    # Mapping classes → labels lisibles
    if model_type == "trend":
        labels_map = TREND_LABELS
    else:
        labels_map = VOLATILITY_LABELS

    probabilities = {
        labels_map[i]: round(float(pred_proba[i]), 4)
        for i in range(len(pred_proba))
    }

    result = {
        "ticker": ticker,
        "label": TICKER_LABELS.get(ticker, ticker),
        "model_type": model_type,
        "prediction_class": pred_class,
        "prediction_label": labels_map[pred_class],
        "probabilities": probabilities,
        "current_price": round(float(price_df["Close"].iloc[-1]), 2),
        "prediction_date": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "horizon_days": PREDICTION_HORIZON_DAYS,
        "features_used": X_live.to_dict(orient="records")[0],
    }

    logger.info(
        "Prédiction %s pour %s : %s (confiance %.1f%%)",
        model_type, ticker,
        result["prediction_label"],
        max(pred_proba) * 100,
    )
    return result


def predict_all_tickers(
    tickers: list[str],
    model_type: str = "trend",
) -> pd.DataFrame:
    """
    Produit une prédiction live pour une liste de tickers.

    Utile pour générer le tableau de bord Streamlit d'un coup.

    Args:
        tickers: Liste de symboles boursiers.
        model_type: 'trend' ou 'volatility'.

    Returns:
        DataFrame avec une ligne par ticker et les colonnes :
        [ticker, label, prediction_label, confidence, current_price, prediction_date].
    """
    rows = []
    for ticker in tickers:
        try:
            result = predict_live(ticker, model_type=model_type)
            rows.append({
                "ticker": result["ticker"],
                "actif": result["label"],
                "prédiction": result["prediction_label"],
                "confiance": f"{max(result['probabilities'].values()) * 100:.1f}%",
                "prix_actuel": result["current_price"],
                "date": result["prediction_date"],
            })
        except Exception as e:
            logger.warning("Prédiction échouée pour %s : %s", ticker, e)
            rows.append({"ticker": ticker, "actif": ticker, "prédiction": "N/A",
                         "confiance": "N/A", "prix_actuel": None, "date": None})

    return pd.DataFrame(rows)


# ─── Comparaison avec le passé ────────────────────────────────────────────────


def evaluate_past_prediction(
    ticker: str,
    model: XGBClassifier,
    scaler: StandardScaler,
    reference_date: str,
    model_type: str = "trend",
) -> dict:
    """
    Vérifie si la prédiction faite à `reference_date` était correcte.

    La prédiction faite à J est comparée au prix réel à J+30.
    Cela permet d'afficher dans le dashboard "notre prédiction du mois
    dernier était correcte/incorrecte".

    Args:
        ticker: Symbole boursier.
        model: Modèle XGBClassifier chargé.
        scaler: StandardScaler correspondant.
        reference_date: Date à laquelle la prédiction aurait été faite ('YYYY-MM-DD').
        model_type: 'trend' ou 'volatility'.

    Returns:
        Dictionnaire avec la prédiction faite, la réalité observée, et si c'était correct.
    """
    ref = pd.Timestamp(reference_date)
    future_date = (ref + pd.Timedelta(days=PREDICTION_HORIZON_DAYS + 10)).strftime("%Y-%m-%d")
    start_date = (ref - pd.Timedelta(days=120)).strftime("%Y-%m-%d")

    price_df = collect_price_data(ticker, start_date=start_date, end_date=future_date)
    macro_df = collect_fred_data(start_date=start_date, end_date=future_date)
    feat_df = build_feature_matrix(price_df, macro_df=macro_df, add_targets=False)

    feature_cols = get_feature_names(include_macro=True)
    available = [c for c in feature_cols if c in feat_df.columns]
    feat_df = feat_df.dropna(subset=available)

    # Features au moment de la prédiction (reference_date)
    row = feat_df[available].loc[:reference_date].iloc[[-1]]
    pred_class = int(model.predict(scaler.transform(row))[0])

    # Réalité : rendement entre reference_date et J+30
    prices_in_window = price_df["Close"].loc[reference_date:future_date]
    if len(prices_in_window) < 2:
        return {"error": "Pas assez de données pour évaluer la prédiction passée."}

    actual_return = (prices_in_window.iloc[-1] - prices_in_window.iloc[0]) / prices_in_window.iloc[0]

    from src.config import TREND_DOWN_THRESHOLD, TREND_UP_THRESHOLD
    if actual_return > TREND_UP_THRESHOLD:
        actual_class = 2
    elif actual_return < TREND_DOWN_THRESHOLD:
        actual_class = 0
    else:
        actual_class = 1

    labels_map = TREND_LABELS if model_type == "trend" else VOLATILITY_LABELS

    return {
        "ticker": ticker,
        "reference_date": reference_date,
        "predicted": labels_map[pred_class],
        "actual": labels_map[actual_class],
        "actual_return_pct": round(actual_return * 100, 2),
        "correct": pred_class == actual_class,
    }
