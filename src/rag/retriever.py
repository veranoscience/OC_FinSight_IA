"""
Recherche dans la base vectorielle FAISS et génération de réponses via Mistral.

Pipeline RAG complet :
  1. Encoder la question de l'utilisateur en vecteur
  2. Rechercher les k chunks les plus proches dans FAISS (similarité cosinus)
  3. Construire un prompt avec les chunks comme contexte
  4. Envoyer à Mistral pour générer une réponse en langage naturel

Pourquoi RAG plutôt que juste Mistral ?
  Mistral seul ne connaît pas les news des 7 derniers jours. Le RAG lui
  fournit ce contexte récent directement dans le prompt — il peut alors
  répondre avec des informations actualisées.
"""

import logging
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, RAG_TOP_K, TICKER_LABELS

logger = logging.getLogger(__name__)

# Prompt système pour l'agent Mistral — définit son comportement
SYSTEM_PROMPT = """Tu es FinSight, un assistant d'analyse financière.
Tu réponds en français, de façon claire et concise.
Tu bases tes réponses UNIQUEMENT sur les articles de news fournis dans le contexte.
Si le contexte ne contient pas l'information demandée, dis-le clairement.
Tu termines toujours par rappeler : les prédictions sont à titre informatif uniquement."""


# ─── Retrieval (recherche dans FAISS) ────────────────────────────────────────


class FinSightRetriever:
    """
    Moteur de recherche sémantique sur les news financières.

    Charge le modèle d'embedding une seule fois à l'initialisation
    pour éviter de le recharger à chaque requête (lent).

    Attributes:
        encoder: Modèle SentenceTransformer pour encoder les questions.
        index: Index FAISS avec les embeddings des chunks.
        chunks: Textes des chunks indexés.
        metadata: Métadonnées associées à chaque chunk.
    """

    def __init__(
        self,
        index: faiss.Index,
        chunks: list[str],
        metadata: list[dict],
        model_name: str = EMBEDDING_MODEL,
    ) -> None:
        """
        Initialise le retriever avec un index FAISS déjà chargé.

        Args:
            index: Index FAISS.
            chunks: Textes des chunks.
            metadata: Métadonnées des chunks.
            model_name: Modèle d'embedding (doit être le même que celui utilisé pour l'ingestion).
        """
        logger.info("Chargement du modèle d'embedding pour le retriever...")
        self.encoder = SentenceTransformer(model_name)
        self.index = index
        self.chunks = chunks
        self.metadata = metadata
        logger.info("Retriever prêt — %d chunks indexés", len(chunks))

    def search(
        self,
        query: str,
        top_k: int = RAG_TOP_K,
        ticker_filter: str | None = None,
    ) -> list[dict]:
        """
        Recherche les chunks les plus pertinents pour une question.

        Encode la question, cherche les k voisins les plus proches dans FAISS,
        et retourne les chunks avec leurs métadonnées et scores.

        Args:
            query: Question ou requête en langage naturel.
            top_k: Nombre de chunks à retourner.
            ticker_filter: Si fourni, filtre les résultats sur ce ticker uniquement.

        Returns:
            Liste de dicts avec {chunk, score, ticker, label, title, source, url, date}.
            Triée par score décroissant (plus pertinent en premier).
        """
        logger.info("Recherche : '%s' (top_k=%d)", query[:80], top_k)

        query_vec = self.encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # On cherche plus de résultats si un filtre ticker est appliqué
        k_search = top_k * 5 if ticker_filter else top_k
        k_search = min(k_search, self.index.ntotal)

        scores, indices = self.index.search(query_vec, k_search)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            meta = self.metadata[idx]

            if ticker_filter and meta.get("ticker") != ticker_filter:
                continue

            results.append({
                "chunk": self.chunks[idx],
                "score": float(score),
                "ticker": meta.get("ticker", ""),
                "label": meta.get("label", ""),
                "title": meta.get("title", ""),
                "source": meta.get("source", ""),
                "url": meta.get("url", ""),
                "published_at": meta.get("published_at", ""),
            })

            if len(results) >= top_k:
                break

        logger.info("  → %d chunks trouvés", len(results))
        return results


# ─── Génération de réponse (RAG complet) ─────────────────────────────────────


def build_rag_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Construit le prompt RAG à envoyer à Mistral.

    Le prompt suit la structure standard RAG :
      SYSTEM : rôle et règles de comportement
      CONTEXT : chunks récupérés (les articles pertinents)
      QUESTION : la question de l'utilisateur

    Cette séparation claire aide Mistral à distinguer ce qu'il sait
    de ce qu'on lui fournit comme contexte.

    Args:
        query: Question de l'utilisateur.
        retrieved_chunks: Chunks récupérés par le retriever.

    Returns:
        Prompt formaté prêt à envoyer à l'API Mistral.
    """
    if not retrieved_chunks:
        context = "Aucune news récente disponible pour cette requête."
    else:
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source_info = f"[{chunk['label']} — {chunk['source']} — {chunk['published_at'][:10]}]"
            context_parts.append(f"Article {i} {source_info} :\n{chunk['chunk']}")
        context = "\n\n".join(context_parts)

    return f"""CONTEXTE (articles récents) :
{context}

QUESTION : {query}

Réponds en te basant uniquement sur les articles ci-dessus."""


def generate_rag_response(
    query: str,
    retriever: FinSightRetriever,
    ticker_filter: str | None = None,
    top_k: int = RAG_TOP_K,
    max_tokens: int = 512,
) -> dict:
    """
    Pipeline RAG complet : retrieve → augment → generate.

    Args:
        query: Question de l'utilisateur en langage naturel.
        retriever: Instance de FinSightRetriever déjà initialisée.
        ticker_filter: Filtrer les news sur un actif spécifique.
        top_k: Nombre de chunks à récupérer.
        max_tokens: Longueur maximale de la réponse Mistral.

    Returns:
        Dictionnaire avec :
        - response : texte de la réponse Mistral
        - sources : liste des chunks utilisés (pour affichage dans dashboard)
        - query : la question originale

    Raises:
        ValueError: Si la clé API Mistral est manquante.
    """
    try:
        from mistralai.client import Mistral
    except ImportError as e:
        raise ImportError("Installez mistralai : uv add mistralai") from e

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY manquante dans .env")

    # Étape 1 : Retrieve — chercher les chunks pertinents
    retrieved = retriever.search(query, top_k=top_k, ticker_filter=ticker_filter)

    # Étape 2 : Augment — construire le prompt avec le contexte
    user_prompt = build_rag_prompt(query, retrieved)

    # Étape 3 : Generate — envoyer à Mistral
    client = Mistral(api_key=api_key)

    logger.info("Envoi à Mistral (mistral-small-latest)...")
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.3,  # faible température = réponses plus factuelles
    )

    answer = response.choices[0].message.content

    logger.info("Réponse Mistral reçue (%d caractères)", len(answer))

    return {
        "query": query,
        "response": answer,
        "sources": retrieved,
        "n_chunks_used": len(retrieved),
    }


def summarize_ticker_news(
    ticker: str,
    retriever: FinSightRetriever,
    max_tokens: int = 400,
) -> dict:
    """
    Génère un résumé automatique des news récentes pour un actif.

    Appelé par le dashboard Streamlit pour afficher le contexte news
    à côté de la prédiction ML.

    Args:
        ticker: Symbole boursier.
        retriever: Instance de FinSightRetriever.
        max_tokens: Longueur du résumé.

    Returns:
        Dictionnaire avec response et sources.
    """
    label = TICKER_LABELS.get(ticker, ticker)
    query = (
        f"Quelles sont les informations importantes des dernières nouvelles "
        f"concernant {label} ({ticker}) ? "
        f"Résume les points clés en 3-4 phrases."
    )

    return generate_rag_response(
        query=query,
        retriever=retriever,
        ticker_filter=ticker,
        top_k=RAG_TOP_K,
        max_tokens=max_tokens,
    )
