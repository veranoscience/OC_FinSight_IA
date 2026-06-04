"""
Tests unitaires — Feature Engineering.

On teste les fonctions critiques de src/data/features.py :
  - Les indicateurs techniques produisent les bonnes colonnes
  - Les targets sont correctement construites (horizon, NaN finaux)
  - build_feature_matrix retourne le bon nombre de colonnes
"""

import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    add_bollinger_bands,
    add_ema,
    add_historical_volatility,
    add_macd,
    add_returns,
    add_rsi,
    add_trend_target,
    add_volatility_target,
    add_volume_features,
    build_feature_matrix,
    get_feature_names,
)


#  Fixtures 


def make_price_df(n: int = 300) -> pd.DataFrame:
    """Crée un DataFrame de prix synthétique avec suffisamment de lignes."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high  = close + np.random.uniform(0.5, 2.0, n)
    low   = close - np.random.uniform(0.5, 2.0, n)
    volume = np.random.randint(1_000_000, 10_000_000, n).astype(float)

    return pd.DataFrame(
        {"Close": close, "High": high, "Low": low, "Volume": volume},
        index=dates,
    )


#  Tests des indicateurs 


def test_rsi_column_created():
    df = make_price_df()
    result = add_rsi(df.copy())
    assert "rsi" in result.columns


def test_rsi_range():
    """RSI est toujours entre 0 et 100 (sur les valeurs non-NaN)."""
    df = make_price_df()
    result = add_rsi(df.copy())
    valid = result["rsi"].dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_macd_columns_created():
    df = make_price_df()
    result = add_macd(df.copy())
    for col in ["macd", "macd_signal", "macd_diff"]:
        assert col in result.columns, f"Colonne manquante : {col}"


def test_bollinger_columns_created():
    df = make_price_df()
    result = add_bollinger_bands(df.copy())
    for col in ["bb_pct_b", "bb_width"]:
        assert col in result.columns, f"Colonne manquante : {col}"


def test_ema_columns_created():
    df = make_price_df()
    result = add_ema(df.copy(), short=20, long=50)
    assert "ema_short" in result.columns
    assert "ema_long" in result.columns


def test_volume_ratio_column():
    df = make_price_df()
    result = add_volume_features(df.copy())
    assert "volume_ratio" in result.columns


def test_returns_columns():
    df = make_price_df()
    result = add_returns(df.copy())
    assert "return_1d" in result.columns
    assert "return_5d" in result.columns


def test_historical_volatility_columns():
    df = make_price_df()
    result = add_historical_volatility(df.copy())
    assert "volatility_20d" in result.columns
    assert "volatility_60d" in result.columns


#  Tests des targets 


def test_trend_target_classes():
    """La cible de tendance ne contient que 3 valeurs : 0, 1, 2."""
    df = make_price_df(400)
    result = add_trend_target(df.copy(), horizon=30)
    valid = result["target_trend"].dropna()
    assert set(valid.unique()).issubset({0, 1, 2})


def test_trend_target_last_rows_nan():
    """Les 30 dernières lignes de target_trend doivent être NaN (horizon futur)."""
    horizon = 30
    df = make_price_df(300)
    result = add_trend_target(df.copy(), horizon=horizon)
    assert result["target_trend"].iloc[-horizon:].isna().all()


def test_volatility_target_classes():
    """La cible de volatilité ne contient que 3 valeurs : 0, 1, 2."""
    df = make_price_df(400)
    result = add_volatility_target(df.copy(), horizon=30)
    valid = result["target_volatility"].dropna()
    assert set(valid.unique()).issubset({0, 1, 2})


def test_volatility_target_balanced_terciles():
    """Les terciles doivent être à peu près équilibrés (±10 points de % par classe)."""
    df = make_price_df(2000)
    result = add_volatility_target(df.copy(), horizon=30)
    counts = result["target_volatility"].value_counts(normalize=True)
    for cls in [0, 1, 2]:
        assert abs(counts.get(cls, 0) - 1 / 3) < 0.10


#  Test build_feature_matrix 


def test_build_feature_matrix_columns():
    """build_feature_matrix crée au minimum les 15 features techniques."""
    df = make_price_df(400)
    result = build_feature_matrix(df, macro_df=None, add_targets=False)
    expected_features = get_feature_names(include_macro=False)
    for feat in expected_features:
        assert feat in result.columns, f"Feature manquante : {feat}"


def test_build_feature_matrix_no_leakage():
    """Les noms de features (ce que le modèle utilise) ne doivent pas inclure les colonnes brutes."""
    names = get_feature_names(include_macro=False)
    assert "High" not in names
    assert "Low" not in names
    assert "Open" not in names
    assert "Volume" not in names


def test_get_feature_names_count():
    """Sans macro : 15 features. Avec macro : 15 + 3 = 18."""
    names_no_macro   = get_feature_names(include_macro=False)
    names_with_macro = get_feature_names(include_macro=True)
    assert len(names_no_macro)   == 15
    assert len(names_with_macro) == 18
