"""
Agent conversationnel FinSight — Mistral tool calling natif.

L'agent dispose de 3 outils et décide lui-même lequel appeler
selon la question de l'utilisateur :

  Tool 1 — predict_asset   : prédiction ML (tendance + volatilité)
  Tool 2 — search_news     : recherche sémantique dans les news (RAG)
  Tool 3 — market_summary  : rapport complet sur un actif (ML + news)

Pourquoi l'API Mistral directement plutôt que LangChain Agents ?
  LangChain Agents subit des changements d'API majeurs entre versions.
  L'API Mistral supporte nativement le tool calling — plus stable,
  plus lisible, et sans couche d'abstraction inutile pour ce projet.
"""

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from mistralai.client import Mistral

from src.config import ALL_TICKERS, DISCLAIMER_AMF, TICKER_LABELS

load_dotenv()
logger = logging.getLogger(__name__)

# ─── Définitions des outils (format JSON Schema pour Mistral) ────────────────

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "predict_asset",
            "description": (
                "Prédit la tendance (hausse/stable/baisse) et le niveau de risque "
                "(faible/moyen/élevé) d'un actif financier à J+30. "
                "Utilise ce tool pour toute question sur la prédiction ou le risque d'un actif."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Symbole boursier (ex: 'AAPL', 'MC.PA', 'GC=F')",
                    }
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": (
                "Recherche dans les news financières récentes (7 derniers jours). "
                "Utilise ce tool pour les questions sur l'actualité d'une entreprise ou d'un marché."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Question ou mots-clés de recherche",
                    },
                    "ticker": {
                        "type": "string",
                        "description": "Ticker optionnel pour filtrer les news (ex: 'AAPL')",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "market_summary",
            "description": (
                "Génère un rapport complet sur un actif : prédiction ML + news récentes. "
                "Utilise ce tool quand l'utilisateur veut une analyse globale avant d'investir."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Symbole boursier (ex: 'AAPL', 'MC.PA', 'GC=F')",
                    }
                },
                "required": ["ticker"],
            },
        },
    },
]

SYSTEM_PROMPT = f"""Tu es FinSight, un assistant d'analyse financière intelligent.
Tu réponds TOUJOURS en français, de façon claire et structurée.

Actifs disponibles : {', '.join([f"{t} ({TICKER_LABELS[t]})" for t in ALL_TICKERS])}

Utilise les outils disponibles pour répondre — ne devine pas.
Si l'utilisateur mentionne un actif non disponible, propose les alternatives.

{DISCLAIMER_AMF}"""


# ─── Initialisation des composants partagés ───────────────────────────────────

_components: dict[str, Any] = {}


def init_agent_components() -> None:
    """
    Charge les modèles ML et le retriever RAG en mémoire.
    À appeler une seule fois au démarrage de l'application.
    """
    from src.models.predict import load_model
    from src.rag.ingest import load_faiss_index
    from src.rag.retriever import FinSightRetriever

    logger.info("Initialisation des composants de l'agent...")

    models_trend, scalers_trend = {}, {}
    models_vol,   scalers_vol   = {}, {}

    for ticker in ALL_TICKERS:
        try:
            models_trend[ticker], scalers_trend[ticker] = load_model(ticker, "trend")
            models_vol[ticker],   scalers_vol[ticker]   = load_model(ticker, "volatility")
        except FileNotFoundError:
            logger.warning("Modèle non trouvé pour %s", ticker)

    index, chunks, metadata = load_faiss_index()
    retriever = FinSightRetriever(index, chunks, metadata)

    _components.update({
        "models_trend":  models_trend,
        "scalers_trend": scalers_trend,
        "models_vol":    models_vol,
        "scalers_vol":   scalers_vol,
        "retriever":     retriever,
    })

    logger.info("Agent prêt — %d modèles, %d chunks RAG", len(models_trend), len(chunks))


# ─── Implémentation des outils ────────────────────────────────────────────────


def _predict_asset(ticker: str) -> str:
    """Appelle predict_live pour les deux modèles et formate la réponse."""
    ticker = ticker.strip().upper()

    if ticker not in _components.get("models_trend", {}):
        available = list(_components.get("models_trend", {}).keys())
        return f"Ticker '{ticker}' non disponible. Actifs : {', '.join(available)}"

    try:
        from src.models.predict import predict_live
        r_trend = predict_live(ticker, model_type="trend")
        r_vol   = predict_live(ticker, model_type="volatility")
    except Exception as e:
        return f"Erreur de prédiction pour {ticker} : {e}"

    label = TICKER_LABELS.get(ticker, ticker)
    p_t   = r_trend["probabilities"]
    p_v   = r_vol["probabilities"]

    return (
        f"Prédiction FinSight — {label} ({ticker})\n"
        f"Prix actuel : {r_trend['current_price']}\n"
        f"Tendance J+30 : {r_trend['prediction_label'].upper()} "
        f"(baisse {p_t.get('baisse',0):.0%} / stable {p_t.get('stable',0):.0%} / hausse {p_t.get('hausse',0):.0%})\n"
        f"Risque : {r_vol['prediction_label'].upper()} "
        f"(faible {p_v.get('faible',0):.0%} / moyen {p_v.get('moyen',0):.0%} / élevé {p_v.get('élevé',0):.0%})\n"
        f"⚠️ {DISCLAIMER_AMF}"
    )


def _search_news(query: str, ticker: str = "") -> str:
    """Recherche dans FAISS et génère une réponse RAG avec Mistral."""
    retriever = _components.get("retriever")
    if retriever is None:
        return "RAG non initialisé. Appelez init_agent_components()."

    ticker_filter = ticker.strip().upper() if ticker else None
    if ticker_filter and ticker_filter not in TICKER_LABELS:
        ticker_filter = None

    try:
        from src.rag.retriever import generate_rag_response
        result = generate_rag_response(
            query=query,
            retriever=retriever,
            ticker_filter=ticker_filter,
            top_k=5,
        )
    except Exception as e:
        return f"Erreur RAG : {e}"

    sources = "\n".join([
        f"  • [{s['label']}] {s['title'][:60]} ({s['source']})"
        for s in result["sources"][:3]
    ])
    return f"{result['response']}\n\nSources :\n{sources}"


def _market_summary(ticker: str) -> str:
    """Combine prédiction ML et résumé news pour un actif."""
    ticker = ticker.strip().upper()
    label  = TICKER_LABELS.get(ticker, ticker)

    ml_section   = _predict_asset(ticker)
    retriever    = _components.get("retriever")
    news_section = "RAG non disponible."

    if retriever:
        try:
            from src.rag.retriever import summarize_ticker_news
            news_result  = summarize_ticker_news(ticker, retriever)
            news_section = news_result["response"]
        except Exception as e:
            news_section = f"Impossible de récupérer les news : {e}"

    return (
        f"{'='*50}\n"
        f"ANALYSE COMPLÈTE — {label} ({ticker})\n"
        f"{'='*50}\n\n"
        f"{ml_section}\n\n"
        f"--- ACTUALITÉS RÉCENTES ---\n{news_section}"
    )


# Registre des fonctions — associe nom de tool → fonction Python
TOOL_FUNCTIONS = {
    "predict_asset":  _predict_asset,
    "search_news":    _search_news,
    "market_summary": _market_summary,
}


# ─── Boucle agent (ReAct loop manuelle) ──────────────────────────────────────


class FinSightAgent:
    """
    Agent conversationnel avec mémoire de conversation et tool calling Mistral.

    À chaque tour :
    1. Envoie le message + historique à Mistral
    2. Si Mistral demande un tool → l'exécute et renvoie le résultat
    3. Mistral génère la réponse finale à partir du résultat du tool

    Attributes:
        client: Client Mistral API.
        history: Historique de la conversation (liste de messages).
    """

    def __init__(self) -> None:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY manquante dans .env")
        self.client  = Mistral(api_key=api_key)
        self.history: list[dict] = []

    def chat(self, user_message: str, max_tool_calls: int = 3) -> str:
        """
        Envoie un message et retourne la réponse de l'agent.

        Args:
            user_message: Message de l'utilisateur en langage naturel.
            max_tool_calls: Nombre maximum de tools appelés par tour (sécurité anti-boucle).

        Returns:
            Réponse textuelle de l'agent.
        """
        self.history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

        # Boucle ReAct : Mistral peut appeler plusieurs tools en séquence
        for _ in range(max_tool_calls):
            response = self.client.chat.complete(
                model="mistral-small-latest",
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.2,
            )

            msg = response.choices[0].message

            # Pas d'appel de tool → réponse finale
            if not msg.tool_calls:
                answer = msg.content
                self.history.append({"role": "assistant", "content": answer})
                return answer

            # Exécution des tools demandés par Mistral
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]})

            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                logger.info("Tool appelé : %s(%s)", fn_name, fn_args)

                fn = TOOL_FUNCTIONS.get(fn_name)
                if fn is None:
                    tool_result = f"Tool inconnu : {fn_name}"
                else:
                    try:
                        tool_result = fn(**fn_args)
                    except Exception as e:
                        tool_result = f"Erreur dans {fn_name} : {e}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })

        # Sécurité : si max_tool_calls atteint, réponse avec le dernier contexte
        final = self.client.chat.complete(
            model="mistral-small-latest",
            messages=messages,
            temperature=0.2,
        )
        answer = final.choices[0].message.content
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def reset(self) -> None:
        """Réinitialise l'historique de la conversation."""
        self.history = []
        logger.info("Historique de conversation réinitialisé.")
