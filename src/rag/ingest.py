"""
Ingestion des news dans la base vectorielle FAISS.

Pipeline :
  1. Charger les articles JSON (collectés par collector.py)
  2. Nettoyer et concaténer titre + description + contenu
  3. Découper en chunks (textes courts pour l'embedding)
  4. Calculer les embeddings avec sentence-transformers
  5. Stocker dans un index FAISS + métadonnées dans un fichier JSON

Pourquoi FAISS ?
  FAISS (Facebook AI Similarity Search) est une bibliothèque optimisée
  pour la recherche de vecteurs similaires dans de grandes collections.
  C'est le standard pour les pipelines RAG locaux sans serveur externe.
"""

import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    ALL_TICKERS,
    DATA_RAW_DIR,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    TICKER_LABELS,
)

logger = logging.getLogger(__name__)

# Chemin du fichier de métadonnées (textes et infos des chunks)
METADATA_PATH = FAISS_INDEX_PATH.parent / "faiss_metadata.json"

# Taille des chunks en caractères — assez court pour rester dans la fenêtre
# de contexte du modèle d'embedding (512 tokens ≈ ~2000 caractères)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100  # chevauchement pour ne pas couper une idée en deux


# ─── Nettoyage et préparation des articles ────────────────────────────────────


def load_news_json(ticker: str) -> list[dict]:
    """
    Charge les articles JSON d'un ticker depuis data/raw/news/.

    Args:
        ticker: Symbole boursier (ex: 'AAPL', 'GC=F').

    Returns:
        Liste de dictionnaires d'articles. Liste vide si le fichier n'existe pas.
    """
    filename = ticker.replace("=", "_") + ".json"
    path = DATA_RAW_DIR / "news" / filename

    if not path.exists():
        logger.warning("Fichier news introuvable : %s", path)
        return []

    with open(path, encoding="utf-8") as f:
        articles = json.load(f)

    logger.info("Chargé : %d articles pour %s", len(articles), ticker)
    return articles


def build_article_text(article: dict) -> str:
    """
    Construit un texte unique à partir des champs d'un article.

    On concatène titre + description + contenu car les embeddings sont
    plus riches avec le contexte complet. Les champs manquants sont ignorés.

    Args:
        article: Dictionnaire avec les champs title, description, content, etc.

    Returns:
        Texte nettoyé, prêt pour le chunking.
    """
    parts = []

    if article.get("title"):
        parts.append(article["title"].strip())
    if article.get("description") and article["description"] != article.get("title"):
        parts.append(article["description"].strip())
    if article.get("content"):
        # NewsAPI tronque le contenu à ~200 caractères avec "[+N chars]" — on enlève ça
        content = article["content"].split("[+")[0].strip()
        if content:
            parts.append(content)

    return " | ".join(parts)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Découpe un texte en chunks avec chevauchement.

    Le chevauchement (overlap) évite de couper une phrase importante
    pile à la frontière entre deux chunks.

    Args:
        text: Texte à découper.
        chunk_size: Taille maximale d'un chunk en caractères.
        overlap: Nombre de caractères partagés entre chunks consécutifs.

    Returns:
        Liste de chunks textuels.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def prepare_corpus(tickers: list[str] = ALL_TICKERS) -> tuple[list[str], list[dict]]:
    """
    Prépare le corpus complet à partir des articles de tous les tickers.

    Pour chaque ticker, charge les articles, construit le texte, découpe
    en chunks. Les métadonnées (ticker, date, source, url) sont conservées
    pour chaque chunk afin de pouvoir les afficher dans le dashboard.

    Args:
        tickers: Liste de tickers à inclure.

    Returns:
        Tuple (chunks, metadata) où :
        - chunks : liste de textes (un élément par chunk)
        - metadata : liste de dicts avec {ticker, label, date, source, url}
    """
    all_chunks: list[str] = []
    all_metadata: list[dict] = []

    for ticker in tickers:
        articles = load_news_json(ticker)
        label = TICKER_LABELS.get(ticker, ticker)

        for article in articles:
            text = build_article_text(article)
            if not text or len(text) < 30:
                continue

            chunks = chunk_text(text)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({
                    "ticker": ticker,
                    "label": label,
                    "published_at": article.get("published_at", ""),
                    "source": article.get("source", ""),
                    "url": article.get("url", ""),
                    "title": article.get("title", ""),
                })

    logger.info(
        "Corpus préparé : %d chunks pour %d tickers",
        len(all_chunks), len(tickers)
    )
    return all_chunks, all_metadata


# ─── Embeddings et index FAISS ────────────────────────────────────────────────


def build_faiss_index(
    tickers: list[str] = ALL_TICKERS,
    model_name: str = EMBEDDING_MODEL,
    save: bool = True,
) -> tuple[faiss.Index, list[str], list[dict]]:
    """
    Construit l'index FAISS à partir des articles de news.

    Étapes :
    1. Préparer le corpus (chunks + métadonnées)
    2. Calculer les embeddings avec sentence-transformers
    3. Normaliser les vecteurs (pour utiliser la similarité cosinus)
    4. Créer l'index FAISS et y ajouter les vecteurs
    5. Sauvegarder l'index et les métadonnées sur disque

    Pourquoi normaliser ?
    FAISS avec IndexFlatIP (Inner Product) sur des vecteurs normalisés
    est équivalent à la similarité cosinus, qui est la métrique standard
    pour comparer des textes.

    Args:
        tickers: Liste de tickers à inclure.
        model_name: Modèle sentence-transformers à utiliser.
        save: Si True, sauvegarde l'index sur disque.

    Returns:
        Tuple (index, chunks, metadata).

    Raises:
        ValueError: Si le corpus est vide.
    """
    logger.info("Chargement du modèle d'embedding : %s", model_name)
    encoder = SentenceTransformer(model_name)

    chunks, metadata = prepare_corpus(tickers)

    if not chunks:
        raise ValueError(
            "Corpus vide — aucun article trouvé. "
            "Lance d'abord collect_news() pour tous les tickers."
        )

    logger.info("Calcul des embeddings pour %d chunks...", len(chunks))
    embeddings = encoder.encode(
        chunks,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # normalisation pour similarité cosinus
    )

    # Dimension des vecteurs (384 pour all-MiniLM-L6-v2)
    dim = embeddings.shape[1]
    logger.info("Dimension des embeddings : %d", dim)

    # IndexFlatIP = recherche exacte par produit scalaire (= cosinus sur normalisés)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    logger.info("Index FAISS créé : %d vecteurs", index.ntotal)

    if save:
        _save_index(index, chunks, metadata)

    return index, chunks, metadata


def _save_index(
    index: faiss.Index,
    chunks: list[str],
    metadata: list[dict],
) -> None:
    """Sauvegarde l'index FAISS et les métadonnées sur disque."""
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    logger.info("Index FAISS sauvegardé : %s", FAISS_INDEX_PATH)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "metadata": metadata}, f, ensure_ascii=False, indent=2)
    logger.info("Métadonnées sauvegardées : %s", METADATA_PATH)


def load_faiss_index() -> tuple[faiss.Index, list[str], list[dict]]:
    """
    Charge l'index FAISS et les métadonnées depuis le disque.

    Returns:
        Tuple (index, chunks, metadata).

    Raises:
        FileNotFoundError: Si l'index n'a pas encore été créé.
    """
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Index FAISS introuvable : {FAISS_INDEX_PATH}. "
            "Lance d'abord build_faiss_index()."
        )

    index = faiss.read_index(str(FAISS_INDEX_PATH))

    with open(METADATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    logger.info("Index FAISS chargé : %d vecteurs", index.ntotal)
    return index, data["chunks"], data["metadata"]
