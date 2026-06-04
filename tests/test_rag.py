"""
Tests unitaires — Pipeline RAG.

On teste :
  - chunk_text : longueur des chunks, recouvrement
  - build_article_text : nettoyage du texte "[+N chars]"
  - build_rag_prompt : structure du prompt avec et sans chunks
  - FinSightRetriever.search : retourne les bons champs
"""

import numpy as np
import pytest

from src.rag.ingest import build_article_text, chunk_text
from src.rag.retriever import FinSightRetriever, build_rag_prompt


#  Tests chunk_text 


def test_chunk_text_basic():
    """Un texte de 1000 chars avec chunk_size=800, overlap=100 → 2 chunks."""
    text = "A" * 1000
    chunks = chunk_text(text, chunk_size=800, overlap=100)
    assert len(chunks) >= 1


def test_chunk_text_no_empty_chunks():
    """chunk_text ne doit pas retourner de chunks vides."""
    text = "Ceci est un article financier sur Apple Inc. " * 30
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert all(len(c) > 0 for c in chunks)


def test_chunk_text_short_text_single_chunk():
    """Un texte court (< chunk_size) doit retourner un seul chunk."""
    text = "Texte court."
    chunks = chunk_text(text, chunk_size=800, overlap=100)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_overlap():
    """Avec recouvrement, le début du chunk N+1 doit chevaucher la fin du chunk N."""
    text = "ABCDEFGHIJ" * 50  # 500 chars
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    if len(chunks) >= 2:
        # La fin du premier chunk doit apparaître dans le début du second
        end_of_first = chunks[0][-50:]
        start_of_second = chunks[1][:50]
        assert end_of_first == start_of_second


#  Tests build_article_text 


def test_build_article_text_removes_truncation_marker():
    """build_article_text doit supprimer les marqueurs '[+N chars]'."""
    article = {
        "title": "Apple earnings",
        "description": "Apple reported strong results.",
        "content": "Revenue grew 15%... [+1500 chars]",
    }
    text = build_article_text(article)
    assert "[+" not in text
    assert "chars]" not in text


def test_build_article_text_concatenates_fields():
    """build_article_text doit contenir le titre ET la description."""
    article = {
        "title": "Tesla stock rises",
        "description": "Tesla shares up 5%.",
        "content": "Analysts say...",
    }
    text = build_article_text(article)
    assert "Tesla stock rises" in text
    assert "Tesla shares up 5%" in text


def test_build_article_text_handles_missing_fields():
    """build_article_text ne doit pas crasher si des champs sont None."""
    article = {"title": "Gold falls", "description": None, "content": None}
    text = build_article_text(article)
    assert "Gold falls" in text


#  Tests build_rag_prompt 


def test_build_rag_prompt_contains_question():
    query = "Quelle est la tendance d'Apple ?"
    prompt = build_rag_prompt(query, [])
    assert query in prompt


def test_build_rag_prompt_no_chunks_fallback():
    """Sans chunks, le prompt doit indiquer l'absence de news."""
    prompt = build_rag_prompt("Question ?", [])
    assert "Aucune news" in prompt or "aucune" in prompt.lower()


def test_build_rag_prompt_includes_chunk_content():
    chunks = [
        {
            "chunk": "Apple a annoncé des résultats records.",
            "label": "Apple", "source": "Reuters",
            "published_at": "2024-03-15T10:00:00Z",
        }
    ]
    prompt = build_rag_prompt("Résultats Apple ?", chunks)
    assert "Apple a annoncé des résultats records." in prompt


#  Tests FinSightRetriever 


def make_mock_retriever() -> FinSightRetriever:
    """
    Crée un retriever avec un index FAISS synthétique de 10 vecteurs.
    On contourne le chargement du vrai modèle d'embedding.
    """
    import faiss
    from unittest.mock import MagicMock, patch

    dim = 384
    n   = 10

    # Index FAISS avec des vecteurs aléatoires normalisés
    index = faiss.IndexFlatIP(dim)
    vecs  = np.random.rand(n, dim).astype(np.float32)
    faiss.normalize_L2(vecs)
    index.add(vecs)

    chunks = [f"Chunk numéro {i} sur Apple." for i in range(n)]
    metadata = [
        {"ticker": "AAPL", "label": "Apple", "title": f"Article {i}",
         "source": "Reuters", "url": "", "published_at": "2024-03-01T00:00:00Z"}
        for i in range(n)
    ]

    # On patch SentenceTransformer pour éviter de charger le modèle
    with patch("src.rag.retriever.SentenceTransformer") as MockST:
        mock_encoder = MagicMock()
        # L'encoder retourne un vecteur aléatoire normalisé
        mock_encoder.encode.return_value = vecs[:1]
        MockST.return_value = mock_encoder
        retriever = FinSightRetriever(index, chunks, metadata)
        retriever.encoder = mock_encoder  # Assigner le mock après init

    return retriever


def test_retriever_search_returns_list():
    retriever = make_mock_retriever()
    results = retriever.search("Apple earnings", top_k=3)
    assert isinstance(results, list)


def test_retriever_search_result_has_required_keys():
    retriever = make_mock_retriever()
    results = retriever.search("Apple news", top_k=3)
    if results:
        required = {"chunk", "score", "ticker", "label", "title", "source"}
        assert required.issubset(set(results[0].keys()))


def test_retriever_search_ticker_filter():
    """Avec ticker_filter='MSFT', aucun résultat 'AAPL' ne doit apparaître."""
    retriever = make_mock_retriever()
    results = retriever.search("news", top_k=5, ticker_filter="MSFT")
    for r in results:
        assert r["ticker"] == "MSFT"


def test_retriever_search_respects_top_k():
    retriever = make_mock_retriever()
    results = retriever.search("Apple", top_k=3)
    assert len(results) <= 3
