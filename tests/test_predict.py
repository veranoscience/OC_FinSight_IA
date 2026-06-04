"""
Tests unitaires — Pipeline de prédiction ML.

On teste :
  - save_model / load_model : aller-retour sans perte
  - predict_live : retourne bien les clés attendues
  - predict_all_tickers : DataFrame avec les bonnes colonnes
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


#  Helpers 


def make_dummy_model(n_features: int = 5) -> tuple[XGBClassifier, StandardScaler]:
    """Crée un petit XGBClassifier entraîné et un scaler fitté."""
    X = np.random.rand(120, n_features).astype(np.float32)
    y = np.random.randint(0, 3, 120)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBClassifier(
        n_estimators=5,
        max_depth=2,
        num_class=3,
        objective="multi:softprob",
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_scaled, y)
    return model, scaler


#  Tests save/load 


def test_save_and_load_model_roundtrip():
    """save_model puis load_model doit retourner un modèle fonctionnel."""
    from src.models.predict import load_model, save_model

    model, scaler = make_dummy_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        # On patche DATA_PROCESSED_DIR pour pointer vers le dossier temporaire
        with patch("src.models.predict.DATA_PROCESSED_DIR", Path(tmpdir)):
            save_model(model, scaler, ticker="TEST", model_type="trend")
            loaded_model, loaded_scaler = load_model("TEST", "trend")

    # Le modèle rechargé doit pouvoir prédire
    X_test = np.random.rand(1, 5).astype(np.float32)
    X_scaled = loaded_scaler.transform(X_test)
    pred = loaded_model.predict(X_scaled)
    assert pred.shape == (1,)
    assert pred[0] in [0, 1, 2]


def test_load_model_raises_if_missing():
    """load_model lève FileNotFoundError si le modèle n'existe pas."""
    from src.models.predict import load_model

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.models.predict.DATA_PROCESSED_DIR", Path(tmpdir)):
            with pytest.raises(FileNotFoundError):
                load_model("TICKER_INEXISTANT", "trend")


#  Tests predict_live 


def _make_price_df(n: int = 80) -> pd.DataFrame:
    """DataFrame de prix synthétique pour les tests predict_live."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "Close": close, "High": close + 1,
            "Low": close - 1, "Volume": np.ones(n) * 1e6, "Open": close - 0.3,
        },
        index=dates,
    )


def _predict_live_mocks(model, scaler):
    """Contexte de mocks commun pour predict_live : prix + FRED vide + modèle."""
    return (
        patch("src.models.predict.load_model", return_value=(model, scaler)),
        patch("src.models.predict.collect_price_data", return_value=_make_price_df()),
        # collect_fred_data retourne None → build_feature_matrix ignore le bloc macro
        patch("src.models.predict.collect_fred_data", return_value=None),
    )


def test_predict_live_returns_expected_keys():
    """predict_live doit retourner un dict avec les clés obligatoires."""
    from src.models.predict import predict_live

    # 15 features techniques sans macro (build_feature_matrix sans macro_df)
    model, scaler = make_dummy_model(n_features=15)

    m1, m2, m3 = _predict_live_mocks(model, scaler)
    with m1, m2, m3:
        result = predict_live("AAPL", model_type="trend", lookback_days=60)

    required_keys = {"ticker", "model_type", "prediction_class", "prediction_label",
                     "probabilities", "current_price", "prediction_date"}
    assert required_keys.issubset(set(result.keys()))


def test_predict_live_prediction_in_valid_range():
    """La prédiction doit être 0, 1 ou 2."""
    from src.models.predict import predict_live

    model, scaler = make_dummy_model(n_features=15)

    m1, m2, m3 = _predict_live_mocks(model, scaler)
    with m1, m2, m3:
        result = predict_live("AAPL", model_type="trend")

    assert result["prediction_class"] in [0, 1, 2]


def test_predict_live_probabilities_sum_to_one():
    """Les probabilités de prédiction doivent sommer à 1."""
    from src.models.predict import predict_live

    model, scaler = make_dummy_model(n_features=15)

    m1, m2, m3 = _predict_live_mocks(model, scaler)
    with m1, m2, m3:
        result = predict_live("AAPL", model_type="trend")

    total = sum(result["probabilities"].values())
    assert abs(total - 1.0) < 1e-5
