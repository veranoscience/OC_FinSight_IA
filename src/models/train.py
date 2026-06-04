"""
Entraînement des modèles XGBoost avec walk-forward cross-validation
Règle absolue : jamais de train_test_split classique sur séries temporelles
"""

import logging
from typing import Any

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import (
    MLFLOW_EXPERIMENT_TREND,
    MLFLOW_EXPERIMENT_VOLATILITY,
    WF_MIN_TRAIN_SIZE,
    WF_N_SPLITS,
    XGBOOST_DEFAULT_PARAMS,
)
from src.data.features import get_feature_names
from src.models.evaluate import compute_metrics

logger = logging.getLogger(__name__)


#  Walk-forward cross-validation 


def walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int = WF_N_SPLITS,
    min_train_size: int = WF_MIN_TRAIN_SIZE,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Génère les splits temporels pour la validation walk-forward.

    Chaque split élargit progressivement le set d'entraînement.
    Le set de test ne se chevauche jamais avec le train.

    Args:
        df: DataFrame avec index DatetimeIndex trié.
        n_splits: Nombre de splits à générer (défaut 5).
        min_train_size: Taille minimale du set d'entraînement en lignes.

    Returns:
        Liste de tuples (train_index, test_index) — index DatetimeIndex.

    Raises:
        ValueError: Si le dataset est trop petit pour générer les splits.
    """
    n = len(df)
    if n < min_train_size + n_splits:
        raise ValueError(
            f"Dataset trop petit ({n} lignes) pour {n_splits} splits "
            f"avec min_train_size={min_train_size}."
        )

    total_test_size = n - min_train_size
    test_size = total_test_size // n_splits
    splits = []

    for i in range(n_splits):
        train_end = min_train_size + i * test_size
        test_end = train_end + test_size

        train_idx = df.index[:train_end]
        test_idx = df.index[train_end:test_end]

        if len(test_idx) == 0:
            break

        splits.append((train_idx, test_idx))
        logger.debug(
            "Split %d/%d | Train: %s → %s (%d) | Test: %s → %s (%d)",
            i + 1, n_splits,
            train_idx[0].date(), train_idx[-1].date(), len(train_idx),
            test_idx[0].date(), test_idx[-1].date(), len(test_idx),
        )

    return splits


#  Entraînement XGBoost 


def train_trend_model(
    df: pd.DataFrame,
    ticker: str,
    xgb_params: dict[str, Any] | None = None,
    include_macro: bool = True,
) -> tuple[XGBClassifier, StandardScaler, dict]:
    """
    Entraîne le modèle de classification de tendance (hausse/stable/baisse).

    Utilise la walk-forward cross-validation pour évaluer les performances,
    puis ré-entraîne un modèle final sur l'ensemble des données disponibles.
    Compare systématiquement avec un DummyClassifier baseline.

    Toutes les expériences sont loggées dans MLflow.

    Args:
        df: DataFrame avec features ET target_trend calculés.
            Doit avoir un index DatetimeIndex trié et ne pas contenir de NaN
            dans les colonnes de features.
        ticker: Symbole boursier (pour le nommage MLflow).
        xgb_params: Hyperparamètres XGBoost (défaut = XGBOOST_DEFAULT_PARAMS).
        include_macro: Si True, inclut les features macro FRED.

    Returns:
        Tuple (model_final, scaler, wf_results) où :
        - model_final : XGBClassifier entraîné sur toutes les données
        - scaler : StandardScaler fitté sur le dernier split train
        - wf_results : dictionnaire des métriques walk-forward

    Raises:
        ValueError: Si 'target_trend' est absent ou entièrement NaN.
    """
    if "target_trend" not in df.columns:
        raise ValueError("La colonne 'target_trend' est absente du DataFrame.")

    feature_cols = get_feature_names(include_macro=include_macro)
    available_features = [c for c in feature_cols if c in df.columns]

    # On ne garde que les lignes avec une cible valide
    df_clean = df[available_features + ["target_trend"]].dropna()

    if df_clean.empty:
        raise ValueError("Aucune ligne valide après suppression des NaN dans target_trend.")

    y = df_clean["target_trend"].astype(int)
    X = df_clean[available_features]

    if xgb_params is None:
        xgb_params = XGBOOST_DEFAULT_PARAMS.copy()

    logger.info("Entraînement tendance pour %s | %d samples | %d features",
                ticker, len(X), len(available_features))

    mlflow.set_experiment(MLFLOW_EXPERIMENT_TREND)

    wf_scores: list[dict] = []
    splits = walk_forward_splits(df_clean, n_splits=WF_N_SPLITS, min_train_size=WF_MIN_TRAIN_SIZE)

    with mlflow.start_run(run_name=f"{ticker}_trend_wf"):
        mlflow.log_params({**xgb_params, "ticker": ticker, "n_features": len(available_features)})
        mlflow.log_param("features", available_features)

        for i, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]

            # Le scaler est FITTÉ sur le train uniquement — pas de leakage
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = XGBClassifier(**xgb_params, verbosity=0)
            model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)

            y_pred = model.predict(X_test_scaled)
            metrics = compute_metrics(y_test, y_pred, prefix=f"split_{i+1}")
            wf_scores.append(metrics)

            logger.info(
                "  Split %d/%d | F1 pondéré: %.4f",
                i + 1, len(splits), metrics[f"split_{i+1}_f1_weighted"]
            )

        # Agrégation des métriques walk-forward
        wf_results = _aggregate_wf_scores(wf_scores)
        mlflow.log_metrics(wf_results)

        # Baseline DummyClassifier obligatoire
        _log_dummy_baseline(X, y, splits, mlflow)

        # Modèle final : entraîné sur tout le dataset
        scaler_final = StandardScaler()
        X_scaled_all = scaler_final.fit_transform(X)
        model_final = XGBClassifier(**xgb_params, verbosity=0)
        model_final.fit(X_scaled_all, y)

        mlflow.xgboost.log_model(model_final, artifact_path="model_trend")
        logger.info(
            "Run MLflow terminé | F1 pondéré moyen (WF): %.4f",
            wf_results["mean_f1_weighted"]
        )

    return model_final, scaler_final, wf_results


def train_volatility_model(
    df: pd.DataFrame,
    ticker: str,
    xgb_params: dict[str, Any] | None = None,
    include_macro: bool = True,
) -> tuple[XGBClassifier, StandardScaler, dict]:
    """
    Entraîne le modèle de scoring de volatilité (faible/moyen/élevé).

    Même logique que train_trend_model mais sur la cible 'target_volatility'.
    Les quantiles de discrétisation sont calculés sur le train set de chaque split
    pour éviter le data leakage.

    Args:
        df: DataFrame avec features ET target_volatility calculés.
        ticker: Symbole boursier.
        xgb_params: Hyperparamètres XGBoost.
        include_macro: Si True, inclut les features macro FRED.

    Returns:
        Tuple (model_final, scaler, wf_results).

    Raises:
        ValueError: Si 'target_volatility' est absent ou entièrement NaN.
    """
    if "target_volatility" not in df.columns:
        raise ValueError("La colonne 'target_volatility' est absente du DataFrame.")

    feature_cols = get_feature_names(include_macro=include_macro)
    available_features = [c for c in feature_cols if c in df.columns]

    df_clean = df[available_features + ["target_volatility"]].dropna()

    if df_clean.empty:
        raise ValueError("Aucune ligne valide après suppression des NaN dans target_volatility.")

    y = df_clean["target_volatility"].astype(int)
    X = df_clean[available_features]

    params = xgb_params or XGBOOST_DEFAULT_PARAMS.copy()

    logger.info("Entraînement volatilité pour %s | %d samples", ticker, len(X))

    mlflow.set_experiment(MLFLOW_EXPERIMENT_VOLATILITY)
    wf_scores: list[dict] = []
    splits = walk_forward_splits(df_clean, n_splits=WF_N_SPLITS, min_train_size=WF_MIN_TRAIN_SIZE)

    with mlflow.start_run(run_name=f"{ticker}_volatility_wf"):
        mlflow.log_params({**params, "ticker": ticker})

        for i, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = XGBClassifier(**params, verbosity=0)
            model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)

            y_pred = model.predict(X_test_scaled)
            metrics = compute_metrics(y_test, y_pred, prefix=f"split_{i+1}")
            wf_scores.append(metrics)

        wf_results = _aggregate_wf_scores(wf_scores)
        mlflow.log_metrics(wf_results)
        _log_dummy_baseline(X, y, splits, mlflow)

        scaler_final = StandardScaler()
        X_scaled_all = scaler_final.fit_transform(X)
        model_final = XGBClassifier(**params, verbosity=0)
        model_final.fit(X_scaled_all, y)

        mlflow.xgboost.log_model(model_final, artifact_path="model_volatility")

    return model_final, scaler_final, wf_results


#  Helpers  


def _aggregate_wf_scores(wf_scores: list[dict]) -> dict[str, float]:
    """Calcule la moyenne et l'écart-type des métriques sur tous les splits."""
    all_keys = set().union(*[s.keys() for s in wf_scores])
    # Extraire les noms de métriques sans le préfixe 'split_N_'
    metric_names: set[str] = set()
    for key in all_keys:
        parts = key.split("_", 2)
        if len(parts) == 3:
            metric_names.add(parts[2])

    results: dict[str, float] = {}
    for metric in metric_names:
        values = []
        for i, s in enumerate(wf_scores):
            key = f"split_{i+1}_{metric}"
            if key in s:
                values.append(s[key])
        if values:
            results[f"mean_{metric}"] = float(np.mean(values))
            results[f"std_{metric}"] = float(np.std(values))

    return results


def _log_dummy_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple],
    mlflow_client: Any,
) -> None:
    """
    Évalue et logge les performances du DummyClassifier (most_frequent et stratified).
    Obligatoire pour valider que XGBoost apporte une vraie valeur ajoutée.
    """
    for strategy in ["most_frequent", "stratified"]:
        scores = []
        for train_idx, test_idx in splits:
            dummy = DummyClassifier(strategy=strategy, random_state=42)
            dummy.fit(X.loc[train_idx], y.loc[train_idx])
            y_pred = dummy.predict(X.loc[test_idx])
            f1 = f1_score(y.loc[test_idx], y_pred, average="weighted", zero_division=0)
            scores.append(f1)
        mlflow_client.log_metric(f"dummy_{strategy}_f1_weighted", float(np.mean(scores)))
        logger.info("Baseline DummyClassifier (%s) | F1 pondéré: %.4f", strategy, np.mean(scores))
