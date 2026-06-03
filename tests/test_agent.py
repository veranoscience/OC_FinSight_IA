"""
Tests unitaires — Agent FinSight.

On teste les tool functions isolément, sans appel API réel :
  - _predict_asset : retourne la bonne structure
  - _search_news   : retourne la bonne structure
  - _market_summary: combine les deux
  - FinSightAgent  : gestion de l'historique
"""

from unittest.mock import MagicMock, patch

import pytest


# ─── Tests _predict_asset ─────────────────────────────────────────────────────


def test_predict_asset_unknown_ticker():
    """Un ticker non disponible doit retourner un message d'erreur clair."""
    from src.agent.finsight_agent import _predict_asset, _components

    # On simule un état où seul AAPL est disponible
    _components["models_trend"] = {"AAPL": MagicMock()}

    result = _predict_asset("TICKER_INCONNU")
    assert "non disponible" in result.lower() or "unavailable" in result.lower() or "AAPL" in result


def test_predict_asset_calls_predict_live():
    """_predict_asset doit appeler predict_live deux fois (trend + volatility)."""
    from src.agent import finsight_agent
    from src.agent.finsight_agent import _components

    _components["models_trend"] = {"AAPL": MagicMock()}
    _components["models_vol"]   = {"AAPL": MagicMock()}

    mock_result = {
        "current_price": 185.5,
        "prediction_label": "hausse",
        "probabilities": {"baisse": 0.2, "stable": 0.3, "hausse": 0.5},
    }

    # predict_live est importé localement dans _predict_asset → patch le module source
    with patch("src.models.predict.predict_live", return_value=mock_result) as mock_pl:
        result = finsight_agent._predict_asset("AAPL")

    # predict_live appelé 2 fois : une pour trend, une pour volatility
    assert mock_pl.call_count == 2
    assert "AAPL" in result
    assert "hausse" in result.lower()


def test_predict_asset_handles_error():
    """_predict_asset doit retourner un message d'erreur si predict_live plante."""
    from src.agent import finsight_agent
    from src.agent.finsight_agent import _components

    _components["models_trend"] = {"AAPL": MagicMock()}
    _components["models_vol"]   = {"AAPL": MagicMock()}

    # predict_live est importé localement dans _predict_asset → patch le module source
    with patch("src.models.predict.predict_live", side_effect=RuntimeError("boom")):
        result = finsight_agent._predict_asset("AAPL")

    assert "erreur" in result.lower() or "error" in result.lower()


# ─── Tests _search_news ───────────────────────────────────────────────────────


def test_search_news_no_retriever():
    """Sans retriever initialisé, _search_news doit retourner un message clair."""
    from src.agent.finsight_agent import _search_news, _components

    _components.pop("retriever", None)
    result = _search_news("Apple news")
    assert "non initialisé" in result.lower() or "rag" in result.lower()


def test_search_news_calls_generate_rag_response():
    """_search_news doit appeler generate_rag_response avec la bonne query."""
    from src.agent import finsight_agent
    from src.agent.finsight_agent import _components

    mock_retriever = MagicMock()
    _components["retriever"] = mock_retriever

    mock_rag_result = {
        "response": "Apple a annoncé de bons résultats.",
        "sources": [
            {"label": "Apple", "title": "Apple Q1 2024",
             "source": "Reuters", "url": ""}
        ],
    }

    # generate_rag_response est importé localement dans _search_news → patch le module source
    with patch("src.rag.retriever.generate_rag_response",
               return_value=mock_rag_result) as mock_rag:
        result = finsight_agent._search_news("Apple résultats")

    mock_rag.assert_called_once()
    assert "Apple a annoncé" in result


# ─── Tests FinSightAgent (historique) ────────────────────────────────────────


def test_agent_history_grows():
    """Après un chat, l'historique doit contenir user + assistant."""
    from src.agent.finsight_agent import FinSightAgent

    # Mock complet du client Mistral
    mock_response = MagicMock()
    mock_response.choices[0].message.tool_calls = None
    mock_response.choices[0].message.content = "Voici ma réponse."

    with patch("src.agent.finsight_agent.Mistral") as MockMistral:
        mock_client = MagicMock()
        mock_client.chat.complete.return_value = mock_response
        MockMistral.return_value = mock_client

        agent = FinSightAgent()
        agent.chat("Bonjour")

    assert len(agent.history) == 2
    assert agent.history[0]["role"] == "user"
    assert agent.history[1]["role"] == "assistant"


def test_agent_reset_clears_history():
    """reset() doit vider l'historique de conversation."""
    from src.agent.finsight_agent import FinSightAgent

    mock_response = MagicMock()
    mock_response.choices[0].message.tool_calls = None
    mock_response.choices[0].message.content = "Réponse."

    with patch("src.agent.finsight_agent.Mistral") as MockMistral:
        mock_client = MagicMock()
        mock_client.chat.complete.return_value = mock_response
        MockMistral.return_value = mock_client

        agent = FinSightAgent()
        agent.chat("Question 1")
        agent.chat("Question 2")
        assert len(agent.history) == 4

        agent.reset()
        assert agent.history == []


def test_agent_raises_without_api_key():
    """FinSightAgent doit lever ValueError si MISTRAL_API_KEY est absente."""
    from src.agent.finsight_agent import FinSightAgent

    with patch.dict("os.environ", {}, clear=True):
        # Retire la clé API de l'environnement
        import os
        os.environ.pop("MISTRAL_API_KEY", None)

        with pytest.raises((ValueError, Exception)):
            FinSightAgent()
