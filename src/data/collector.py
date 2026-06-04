"""
Collecte des données financières et macroéconomiques pour FinSight
Sources : yfinance (prix/volumes), FRED (macro), NewsAPI (articles)
"""

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from src.config import (
    ALL_TICKERS,
    DATA_RAW_DIR,
    FRED_SERIES,
    NEWS_LOOKBACK_DAYS,
    TRAIN_START_DATE,
    TICKER_LABELS,
)

load_dotenv()

logger = logging.getLogger(__name__)


# Prix et volumes (yfinance)


def collect_price_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Collecte les données de prix historiques pour un ticker

    Args:
        ticker: Symbole boursier (ex: 'GC=F', 'MC.PA', 'AAPL')
        start_date: Date de début au format 'YYYY-MM-DD'
        end_date: Date de fin au format 'YYYY-MM-DD'
        interval: Intervalle des données ('1d', '1wk', '1mo')

    Returns:
        DataFrame avec colonnes [Open, High, Low, Close, Volume] et index DatetimeIndex.

    Raises:
        ValueError: Si le ticker est invalide ou les données sont vides.
    """
    logger.info("Collecte des prix : %s [%s -> %s]", ticker, start_date, end_date)

    try:
        raw = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        raise ValueError(f"Échec du téléchargement pour {ticker} : {e}") from e

    if raw.empty:
        raise ValueError(
            f"Aucune donnée retournée pour {ticker} entre {start_date} et {end_date}."
        )

    # yfinance peut retourner un MultiIndex si un seul ticker — on aplatit
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.dropna(subset=["Close"])

    logger.info("  -> %d lignes collectées pour %s", len(df), ticker)
    return df


def collect_all_tickers(
    tickers: list[str],
    start_date: str,
    end_date: str,
    save_raw: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Collecte les données de prix pour une liste de tickers.

    Args:
        tickers: Liste de symboles boursiers.
        start_date: Date de début au format 'YYYY-MM-DD'.
        end_date: Date de fin au format 'YYYY-MM-DD'.
        save_raw: Si True, sauvegarde chaque DataFrame en CSV dans data/raw/.

    Returns:
        Dictionnaire {ticker: DataFrame}.

    Raises:
        RuntimeError: Si aucun ticker n'a pu être collecté.
    """
    results: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for ticker in tickers:
        try:
            df = collect_price_data(ticker, start_date, end_date)
            results[ticker] = df

            if save_raw:
                _save_raw_csv(df, ticker, "prices")

            # Pause légère pour éviter le rate limiting yfinance
            time.sleep(0.5)

        except ValueError as e:
            logger.warning("Ticker ignoré : %s", e)
            failed.append(ticker)

    if not results:
        raise RuntimeError(
            f"Aucune donnée collectée. Tickers échoués : {failed}"
        )

    if failed:
        logger.warning("Tickers non collectés : %s", failed)

    return results


#  Données macroéconomiques (FRED) 


def collect_fred_data(
    series: dict[str, str] | None = None,
    start_date: str = TRAIN_START_DATE,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Collecte les séries macroéconomiques depuis FRED.

    Args:
        series: Dictionnaire {nom_colonne: series_id_fred}.
                Si None, utilise les séries définies dans config.py.
        start_date: Date de début au format 'YYYY-MM-DD'.
        end_date: Date de fin (défaut = aujourd'hui).

    Returns:
        DataFrame avec les séries macro en colonnes, index DatetimeIndex.
        Les valeurs manquantes sont propagées forward (forward-fill) — adapté
        aux séries FRED qui sont souvent mensuelles.

    Raises:
        ImportError: Si fredapi n'est pas installé.
        ValueError: Si la clé API FRED est manquante ou une série est invalide.
    """
    try:
        from fredapi import Fred
    except ImportError as e:
        raise ImportError("Installez fredapi : uv add fredapi") from e

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED_API_KEY manquante dans .env")

    if series is None:
        series = FRED_SERIES

    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    fred = Fred(api_key=api_key)
    frames: dict[str, pd.Series] = {}

    for col_name, series_id in series.items():
        logger.info("Collecte FRED : %s (%s)", col_name, series_id)
        try:
            s = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
            s.name = col_name
            frames[col_name] = s
        except Exception as e:
            raise ValueError(f"Erreur FRED pour la série {series_id} : {e}") from e

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    # Forward-fill : les séries FRED sont mensuelles, on propage aux jours manquants
    df = df.ffill()

    logger.info("Données FRED collectées : %d lignes, %d séries", len(df), len(df.columns))

    _save_raw_csv(df, "macro", "fred")
    return df


# News (NewsAPI) 


def collect_news(
    ticker: str,
    lookback_days: int = NEWS_LOOKBACK_DAYS,
    language: str = "en",
    max_articles: int = 100,
) -> list[dict]:
    """
    Collecte les articles de news récents pour un ticker via NewsAPI.

    Args:
        ticker: Symbole boursier (utilisé pour générer les mots-clés de recherche).
        lookback_days: Nombre de jours en arrière pour la collecte.
        language: Langue des articles ('en' ou 'fr').
        max_articles: Nombre maximum d'articles à récupérer.

    Returns:
        Liste de dictionnaires avec les champs :
        [title, description, content, url, publishedAt, source].

    Raises:
        ValueError: Si la clé API NewsAPI est manquante ou invalide.
    """
    try:
        from newsapi import NewsApiClient
    except ImportError as e:
        raise ImportError("Installez newsapi-python : uv add newsapi-python") from e

    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        raise ValueError("NEWS_API_KEY manquante dans .env")

    newsapi = NewsApiClient(api_key=api_key)

    from_date = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    to_date = datetime.today().strftime("%Y-%m-%d")

    # Construit la requête en utilisant le label lisible du ticker
    query_term = TICKER_LABELS.get(ticker, ticker)
    logger.info("Collecte news pour '%s' (%s) depuis %s", ticker, query_term, from_date)

    try:
        response = newsapi.get_everything(
            q=query_term,
            from_param=from_date,
            to=to_date,
            language=language,
            sort_by="relevancy",
            page_size=min(max_articles, 100),
        )
    except Exception as e:
        raise ValueError(f"Erreur NewsAPI pour {ticker} : {e}") from e

    articles = response.get("articles", [])
    logger.info("  -> %d articles collectés pour %s", len(articles), ticker)

    cleaned = [
        {
            "ticker": ticker,
            "title": a.get("title", ""),
            "description": a.get("description", ""),
            "content": a.get("content", ""),
            "url": a.get("url", ""),
            "published_at": a.get("publishedAt", ""),
            "source": a.get("source", {}).get("name", ""),
        }
        for a in articles
        if a.get("title")  # filtre les articles sans titre
    ]

    return cleaned


# Utilitaires 


def _save_raw_csv(df: pd.DataFrame, name: str, subfolder: str = "") -> Path:
    """Sauvegarde un DataFrame en CSV dans data/raw/."""
    folder = DATA_RAW_DIR / subfolder if subfolder else DATA_RAW_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.csv"
    df.to_csv(path)
    logger.info("Sauvegardé : %s", path)
    return path


def load_raw_prices(ticker: str) -> pd.DataFrame:
    """
    Charge les données de prix brutes depuis data/raw/prices/.

    Args:
        ticker: Symbole boursier.

    Returns:
        DataFrame avec index DatetimeIndex.

    Raises:
        FileNotFoundError: Si le fichier CSV n'existe pas (collecter d'abord).
    """
    path = DATA_RAW_DIR / "prices" / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Données introuvables pour {ticker}. "
            f"Lancez d'abord collect_all_tickers()."
        )
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df


def load_fred_data() -> pd.DataFrame:
    """
    Charge les données macro FRED depuis data/raw/fred/.

    Returns:
        DataFrame avec index DatetimeIndex.

    Raises:
        FileNotFoundError: Si le fichier CSV n'existe pas.
    """
    path = DATA_RAW_DIR / "fred" / "macro.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Données FRED introuvables. Lancez d'abord collect_fred_data()."
        )
    return pd.read_csv(path, index_col="Date", parse_dates=True)
