"""
Feature engineering pour FinSight.
Calcule tous les indicateurs techniques et assemble la matrice de features + target.
"""

import logging

import numpy as np
import pandas as pd
import ta

from src.config import (
    BOLLINGER_STD,
    BOLLINGER_WINDOW,
    EMA_LONG,
    EMA_SHORT,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    PREDICTION_HORIZON_DAYS,
    RSI_WINDOW,
    RETURNS_WINDOWS,
    TREND_DOWN_THRESHOLD,
    TREND_UP_THRESHOLD,
    VOLATILITY_WINDOWS,
    VOLUME_AVG_WINDOW,
)

logger = logging.getLogger(__name__)


# ─── Indicateurs techniques ───────────────────────────────────────────────────


def add_rsi(df: pd.DataFrame, window: int = RSI_WINDOW) -> pd.DataFrame:
    """
    Ajoute le RSI (Relative Strength Index).

    Args:
        df: DataFrame avec colonne 'Close'.
        window: Fenêtre de calcul (défaut 14).

    Returns:
        DataFrame avec colonne 'rsi' ajoutée.
    """
    df = df.copy()
    df["rsi"] = ta.momentum.RSIIndicator(close=df["Close"], window=window).rsi()
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    """
    Ajoute MACD, signal MACD et histogramme.

    Args:
        df: DataFrame avec colonne 'Close'.
        fast: Fenêtre EMA rapide (défaut 12).
        slow: Fenêtre EMA lente (défaut 26).
        signal: Fenêtre signal (défaut 9).

    Returns:
        DataFrame avec colonnes ['macd', 'macd_signal', 'macd_diff'].
    """
    df = df.copy()
    macd_indicator = ta.trend.MACD(
        close=df["Close"],
        window_fast=fast,
        window_slow=slow,
        window_sign=signal,
    )
    df["macd"] = macd_indicator.macd()
    df["macd_signal"] = macd_indicator.macd_signal()
    df["macd_diff"] = macd_indicator.macd_diff()
    return df


def add_bollinger_bands(
    df: pd.DataFrame,
    window: int = BOLLINGER_WINDOW,
    std: float = BOLLINGER_STD,
) -> pd.DataFrame:
    """
    Ajoute les bandes de Bollinger : largeur et position (%B).

    Args:
        df: DataFrame avec colonne 'Close'.
        window: Fenêtre glissante (défaut 20).
        std: Nombre d'écarts-types (défaut 2).

    Returns:
        DataFrame avec colonnes ['bb_width', 'bb_pct_b'].
    """
    df = df.copy()
    bb = ta.volatility.BollingerBands(close=df["Close"], window=window, window_dev=std)
    df["bb_width"] = bb.bollinger_wband()    # largeur normalisée (hband-lband)/mavg
    df["bb_pct_b"] = bb.bollinger_pband()    # position du prix dans les bandes [0,1]
    return df


def add_ema(
    df: pd.DataFrame,
    short: int = EMA_SHORT,
    long: int = EMA_LONG,
) -> pd.DataFrame:
    """
    Ajoute les EMA courte et longue, plus le signal de croisement.

    Le croisement EMA (ema_cross) vaut +1 si l'EMA courte est au-dessus de
    l'EMA longue (tendance haussière), -1 sinon.

    Args:
        df: DataFrame avec colonne 'Close'.
        short: Fenêtre EMA courte (défaut 20).
        long: Fenêtre EMA longue (défaut 50).

    Returns:
        DataFrame avec colonnes ['ema_short', 'ema_long', 'ema_cross'].
    """
    df = df.copy()
    df["ema_short"] = ta.trend.EMAIndicator(close=df["Close"], window=short).ema_indicator()
    df["ema_long"] = ta.trend.EMAIndicator(close=df["Close"], window=long).ema_indicator()
    df["ema_cross"] = (df["ema_short"] > df["ema_long"]).astype(int) * 2 - 1
    return df


def add_volume_features(df: pd.DataFrame, window: int = VOLUME_AVG_WINDOW) -> pd.DataFrame:
    """
    Ajoute le volume relatif (volume du jour / moyenne mobile du volume).

    Un ratio > 1 indique un volume anormalement élevé, souvent associé
    à des mouvements de prix significatifs.

    Args:
        df: DataFrame avec colonne 'Volume'.
        window: Fenêtre de la moyenne mobile du volume (défaut 20).

    Returns:
        DataFrame avec colonne 'volume_ratio'.
    """
    df = df.copy()
    vol_avg = df["Volume"].rolling(window=window, min_periods=1).mean()
    df["volume_ratio"] = df["Volume"] / vol_avg.replace(0, np.nan)
    return df


def add_returns(df: pd.DataFrame, windows: list[int] = RETURNS_WINDOWS) -> pd.DataFrame:
    """
    Ajoute les rendements passés sur différentes fenêtres.

    Args:
        df: DataFrame avec colonne 'Close'.
        windows: Liste de fenêtres en jours (défaut [1, 5, 20]).

    Returns:
        DataFrame avec colonnes ['return_1d', 'return_5d', 'return_20d'].
    """
    df = df.copy()
    for w in windows:
        df[f"return_{w}d"] = df["Close"].pct_change(periods=w)
    return df


def add_historical_volatility(
    df: pd.DataFrame,
    windows: list[int] = VOLATILITY_WINDOWS,
) -> pd.DataFrame:
    """
    Ajoute la volatilité historique (écart-type annualisé des rendements log).

    Args:
        df: DataFrame avec colonne 'Close'.
        windows: Liste de fenêtres en jours (défaut [20, 60]).

    Returns:
        DataFrame avec colonnes ['volatility_20d', 'volatility_60d'].
    """
    df = df.copy()
    log_returns = np.log(df["Close"] / df["Close"].shift(1))
    for w in windows:
        # Annualisation avec √252 (jours de trading)
        df[f"volatility_{w}d"] = log_returns.rolling(window=w).std() * np.sqrt(252)
    return df


# ─── Variable cible ───────────────────────────────────────────────────────────


def add_trend_target(
    df: pd.DataFrame,
    horizon: int = PREDICTION_HORIZON_DAYS,
    up_threshold: float = TREND_UP_THRESHOLD,
    down_threshold: float = TREND_DOWN_THRESHOLD,
) -> pd.DataFrame:
    """
    Calcule la variable cible de tendance à J+horizon.

    La cible est basée sur le rendement futur : si > up_threshold → 2 (hausse),
    si < down_threshold → 0 (baisse), sinon → 1 (stable).

    IMPORTANT : cette fonction crée une target basée sur des prix futurs.
    Elle doit être appelée AVANT la séparation train/test pour que les NaN
    de fin de série soient correctement gérés.

    Args:
        df: DataFrame avec colonne 'Close'.
        horizon: Nombre de jours ouvrés dans le futur (défaut 30).
        up_threshold: Seuil de hausse en rendement relatif (défaut 0.03 = +3%).
        down_threshold: Seuil de baisse (défaut -0.03 = -3%).

    Returns:
        DataFrame avec colonne 'target_trend' (0=baisse, 1=stable, 2=hausse).
        Les dernières `horizon` lignes auront NaN (pas de futur disponible).
    """
    df = df.copy()
    future_return = df["Close"].pct_change(periods=horizon).shift(-horizon)

    conditions = [
        future_return > up_threshold,
        future_return < down_threshold,
    ]
    choices = [2, 0]  # 2=hausse, 0=baisse, 1=stable par défaut

    df["target_trend"] = np.select(conditions, choices, default=1)

    # Les lignes sans future disponible → NaN (pas d'int possible → float)
    df.loc[df.index[-horizon:], "target_trend"] = np.nan

    return df


def add_volatility_target(
    df: pd.DataFrame,
    horizon: int = PREDICTION_HORIZON_DAYS,
) -> pd.DataFrame:
    """
    Calcule la variable cible de volatilité à J+horizon, discrétisée en 3 niveaux.

    La volatilité réalisée future est calculée comme l'écart-type des rendements
    log sur les `horizon` prochains jours, annualisée.
    Elle est ensuite discrétisée en 3 niveaux via les terciles (quantiles 0/33/67/100).

    Args:
        df: DataFrame avec colonne 'Close'.
        horizon: Fenêtre de volatilité future en jours (défaut 30).

    Returns:
        DataFrame avec colonne 'target_volatility' (0=faible, 1=moyen, 2=élevé).
    """
    df = df.copy()
    log_returns = np.log(df["Close"] / df["Close"].shift(1))

    # Volatilité réalisée future : rolling std sur les prochains `horizon` jours
    # Shift(-horizon) pour aligner avec la date courante
    future_vol = (
        log_returns
        .rolling(window=horizon)
        .std()
        .shift(-horizon)
        * np.sqrt(252)
    )

    # Discrétisation en terciles (calculés sur tout le dataset — pas de leakage ici
    # car les quantiles sont recalculés sur le train set dans train.py)
    quantiles = future_vol.quantile([0.33, 0.67])
    df["target_volatility"] = pd.cut(
        future_vol,
        bins=[-np.inf, quantiles[0.33], quantiles[0.67], np.inf],
        labels=[0, 1, 2],
    ).astype(float)

    return df


# ─── Pipeline complet ─────────────────────────────────────────────────────────


def build_feature_matrix(
    price_df: pd.DataFrame,
    macro_df: pd.DataFrame | None = None,
    add_targets: bool = True,
) -> pd.DataFrame:
    """
    Construit la matrice de features complète à partir des données brutes.

    Applique séquentiellement tous les indicateurs techniques, fusionne les
    données macro, puis calcule les variables cibles si demandé.

    Args:
        price_df: DataFrame de prix avec colonnes [Open, High, Low, Close, Volume].
        macro_df: DataFrame de données macro FRED (optionnel). Si fourni, les
                  séries sont jointes par date avec forward-fill.
        add_targets: Si True, ajoute les colonnes 'target_trend' et 'target_volatility'.

    Returns:
        DataFrame avec toutes les features et (si add_targets=True) les targets.
        Les lignes avec NaN dans les features sont supprimées.
    """
    logger.info("Calcul des features techniques...")
    df = price_df.copy()

    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_ema(df)
    df = add_volume_features(df)
    df = add_returns(df)
    df = add_historical_volatility(df)

    if macro_df is not None:
        logger.info("Fusion des données macro FRED...")
        # Reindex sur l'index des prix (journalier), forward-fill pour les jours sans données
        macro_aligned = macro_df.reindex(df.index, method="ffill")
        df = df.join(macro_aligned, how="left")

    if add_targets:
        logger.info("Calcul des variables cibles...")
        df = add_trend_target(df)
        df = add_volatility_target(df)

    # Suppression des lignes avec NaN dans les features (pas dans les targets)
    # On filtre sur les colonnes qui existent réellement (macro optionnel)
    feature_cols = [c for c in get_feature_names() if c in df.columns]
    initial_len = len(df)
    df = df.dropna(subset=feature_cols)
    dropped = initial_len - len(df)
    if dropped > 0:
        logger.info("  → %d lignes supprimées (NaN dans les features)", dropped)

    logger.info("Matrice de features : %d lignes × %d colonnes", len(df), len(df.columns))
    return df


def get_feature_names(include_macro: bool = True) -> list[str]:
    """
    Retourne la liste ordonnée des noms de features utilisées par les modèles.

    Args:
        include_macro: Si True, inclut les features macro FRED.

    Returns:
        Liste de noms de colonnes.
    """
    technical_features = [
        "rsi",
        "macd",
        "macd_signal",
        "macd_diff",
        "bb_width",
        "bb_pct_b",
        "ema_short",
        "ema_long",
        "ema_cross",
        "volume_ratio",
        "return_1d",
        "return_5d",
        "return_20d",
        "volatility_20d",
        "volatility_60d",
    ]

    macro_features = [
        "fed_rate",
        "cpi",
        "us_10y",
    ]

    if include_macro:
        return technical_features + macro_features
    return technical_features
