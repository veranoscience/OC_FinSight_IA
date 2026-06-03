"""
Pipeline d'entraînement complet FinSight.

Ce script orchestre les 3 étapes dans le bon ordre :
  1. Feature engineering — construit la matrice features+target pour chaque actif
  2. Entraînement — walk-forward CV sur chaque ticker (tendance + volatilité)
  3. Sauvegarde — modèle + scaler sur disque pour predict.py

Lancement : uv run python scripts/train_pipeline.py
"""

import logging
import sys
import time
from pathlib import Path

# Permet d'importer les modules src/ depuis n'importe où
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.config import ALL_TICKERS, TRAIN_START_DATE, TEST_END_DATE, DATA_PROCESSED_DIR
from src.data.collector import load_raw_prices, load_fred_data
from src.data.features import build_feature_matrix
from src.models.train import train_trend_model, train_volatility_model
from src.models.predict import save_model


def run_feature_engineering(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    Étape 1 : charge les données brutes et construit les matrices de features.

    Pour chaque ticker, on :
    - Charge les prix depuis data/raw/prices/ (déjà collectés)
    - Charge les données macro FRED (déjà collectées)
    - Calcule les 18 features + les 2 variables cibles
    - Sauvegarde le résultat dans data/processed/

    Pourquoi sauvegarder le résultat ?
    Recalculer les features à chaque fois prendrait du temps.
    On sauvegarde une fois et on recharge si besoin.
    """
    logger.info("=" * 55)
    logger.info("ÉTAPE 1 — Feature Engineering")
    logger.info("=" * 55)

    macro = load_fred_data()
    processed_dir = DATA_PROCESSED_DIR / "features"
    processed_dir.mkdir(parents=True, exist_ok=True)

    feature_matrices: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        logger.info("Features : %s", ticker)
        prices = load_raw_prices(ticker)

        # Filtre sur la période d'entraînement uniquement (2015-2024)
        # On garde tout le dataset ici — la séparation train/test se fait
        # dans walk_forward_splits(), pas ici.
        prices = prices.loc[TRAIN_START_DATE:TEST_END_DATE]
        macro_filtered = macro.loc[TRAIN_START_DATE:TEST_END_DATE]

        df_feat = build_feature_matrix(prices, macro_df=macro_filtered, add_targets=True)

        # Suppression des lignes sans cible valide (les 30 derniers jours)
        df_feat = df_feat.dropna(subset=["target_trend", "target_volatility"])

        save_path = processed_dir / f"{ticker.replace('=', '_')}_features.csv"
        df_feat.to_csv(save_path)

        feature_matrices[ticker] = df_feat
        logger.info(
            "  → %d lignes × %d colonnes | Sauvegardé : %s",
            df_feat.shape[0], df_feat.shape[1], save_path.name
        )

    return feature_matrices


def run_training(feature_matrices: dict[str, pd.DataFrame]) -> dict:
    """
    Étape 2 : entraînement XGBoost (tendance + volatilité) pour chaque ticker.

    Pour chaque ticker on entraîne 2 modèles :
    - Modèle de tendance : prédit hausse / stable / baisse à J+30
    - Modèle de volatilité : prédit niveau de risque faible / moyen / élevé

    Les résultats walk-forward sont collectés pour affichage final.
    """
    logger.info("=" * 55)
    logger.info("ÉTAPE 2 — Entraînement des modèles")
    logger.info("=" * 55)

    all_results = {}

    for ticker, df_feat in feature_matrices.items():
        logger.info("-" * 40)
        logger.info("Ticker : %s", ticker)
        logger.info("-" * 40)
        t0 = time.time()

        # --- Modèle tendance ---
        logger.info("[%s] Modèle TENDANCE...", ticker)
        model_trend, scaler_trend, wf_trend = train_trend_model(df_feat, ticker=ticker)
        save_model(model_trend, scaler_trend, ticker, model_type="trend")

        # --- Modèle volatilité ---
        logger.info("[%s] Modèle VOLATILITÉ...", ticker)
        model_vol, scaler_vol, wf_vol = train_volatility_model(df_feat, ticker=ticker)
        save_model(model_vol, scaler_vol, ticker, model_type="volatility")

        elapsed = time.time() - t0
        all_results[ticker] = {
            "trend": wf_trend,
            "volatility": wf_vol,
            "elapsed_s": round(elapsed, 1),
        }
        logger.info("[%s] Terminé en %.1fs", ticker, elapsed)

    return all_results


def print_summary(all_results: dict) -> None:
    """
    Étape 3 : affiche un tableau récapitulatif des performances walk-forward.

    Permet de voir d'un coup d'œil quel actif est le mieux prédit
    et si le modèle dépasse le baseline DummyClassifier.
    """
    logger.info("=" * 55)
    logger.info("RÉSUMÉ FINAL — Performances Walk-Forward")
    logger.info("=" * 55)

    header = f"{'Ticker':<10} {'F1 Tendance':>12} {'F1 Volatilité':>14} {'Temps':>8}"
    logger.info(header)
    logger.info("-" * 50)

    for ticker, res in all_results.items():
        f1_trend = res["trend"].get("mean_f1_weighted", 0)
        f1_vol = res["volatility"].get("mean_f1_weighted", 0)
        elapsed = res["elapsed_s"]
        logger.info(
            f"{ticker:<10} {f1_trend:>12.4f} {f1_vol:>14.4f} {elapsed:>7.1f}s"
        )

    logger.info("=" * 55)
    logger.info("Modèles sauvegardés dans data/processed/models/")
    logger.info("Runs MLflow disponibles : uv run mlflow ui")


if __name__ == "__main__":
    logger.info("FinSight — Pipeline d'entraînement")
    logger.info("Tickers : %s", ALL_TICKERS)

    # Étape 1 : feature engineering
    feature_matrices = run_feature_engineering(ALL_TICKERS)

    # Étape 2 : entraînement
    all_results = run_training(feature_matrices)

    # Étape 3 : résumé
    print_summary(all_results)
