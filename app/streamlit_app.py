"""
FinSight — Dashboard Streamlit.

Interface utilisateur complète :
  Onglet 1 — Prédiction : tendance J+30, scoring risque, SHAP waterfall
  Onglet 2 — Graphiques : prix, RSI, MACD, Bollinger
  Onglet 3 — News       : résumé RAG des actualités récentes
  Onglet 4 — Chatbot    : agent conversationnel FinSight

Lancement : uv run streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.config import (
    ALL_TICKERS,
    DISCLAIMER_AMF,
    TICKER_LABELS,
    TREND_COLORS,
    VOLATILITY_COLORS,
    DATA_PROCESSED_DIR,
)
from src.data.features import get_feature_names

# ─── Configuration de la page ────────────────────────────────────────────────

st.set_page_config(
    page_title="FinSight — Analyse de Risque",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Chargement des composants (en cache — chargés une seule fois) ───────────

@st.cache_resource(show_spinner="Chargement des modèles ML...")
def load_models():
    """Charge tous les modèles et scalers en mémoire."""
    from src.models.predict import load_model
    models, scalers = {}, {}
    for ticker in ALL_TICKERS:
        for mtype in ("trend", "volatility"):
            try:
                models[f"{ticker}_{mtype}"], scalers[f"{ticker}_{mtype}"] = load_model(ticker, mtype)
            except FileNotFoundError:
                pass
    return models, scalers


@st.cache_resource(show_spinner="Chargement de l'index RAG...")
def load_rag():
    """Charge l'index FAISS et initialise le retriever."""
    from src.rag.ingest import load_faiss_index
    from src.rag.retriever import FinSightRetriever
    try:
        index, chunks, metadata = load_faiss_index()
        return FinSightRetriever(index, chunks, metadata)
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner="Initialisation de l'agent...")
def load_agent():
    """Initialise l'agent conversationnel."""
    from src.agent.finsight_agent import FinSightAgent, init_agent_components
    init_agent_components()
    return FinSightAgent()


@st.cache_data(show_spinner="Chargement des données historiques...")
def load_features(ticker: str) -> pd.DataFrame:
    """Charge la matrice de features depuis data/processed/ (utilisé pour les graphiques)."""
    path = DATA_PROCESSED_DIR / "features" / f"{ticker.replace('=', '_')}_features.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col="Date", parse_dates=True)


@st.cache_data(ttl=3600, show_spinner="Récupération des données en temps réel...")
def load_live_features(ticker: str) -> pd.DataFrame:
    """
    Récupère les prix via yfinance et calcule les features sur les 150 derniers jours.
    Mis en cache 1h pour éviter des appels répétés à yfinance.
    C'est cette fonction qui fournit les données ACTUELLES pour la prédiction.
    """
    from src.data.collector import collect_fred_data, collect_price_data
    from src.data.features import build_feature_matrix

    end_date   = pd.Timestamp.today().strftime("%Y-%m-%d")
    # +90 jours de marge pour absorber les NaN de début de série (RSI, MACD...)
    start_date = (pd.Timestamp.today() - pd.Timedelta(days=150 + 90)).strftime("%Y-%m-%d")

    try:
        price_df = collect_price_data(ticker, start_date=start_date, end_date=end_date)
        # Les modèles ont été entraînés avec les 3 features macro FRED
        # → il faut les inclure ici aussi, sinon le scaler (fitté sur 18 features) plante
        try:
            macro_df = collect_fred_data(start_date=start_date, end_date=end_date)
        except Exception:
            macro_df = None
        feat_df = build_feature_matrix(price_df, macro_df=macro_df, add_targets=False)
        return feat_df
    except Exception:
        return pd.DataFrame()


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title(" FinSight")
    st.caption("Analyse de Risque pour Investisseurs Particuliers")
    st.divider()

    ticker = st.selectbox(
        "Actif analysé",
        options=ALL_TICKERS,
        format_func=lambda t: f"{TICKER_LABELS[t]} ({t})",
        index=0,
    )

    st.divider()
    st.warning(DISCLAIMER_AMF, icon="⚠️")
    st.divider()
    st.caption("Projet de fin d'études — Data Scientist ML")
    st.caption("Formation : Data Scientist Machine Learning")

#  Chargement 

models, scalers = load_models()
retriever = load_rag()
df_feat   = load_features(ticker)      # données historiques CSV → graphiques (Tab 2)
df_live   = load_live_features(ticker) # données fraîches yfinance → prédiction (Tab 1)
label     = TICKER_LABELS[ticker]

#  Onglets 

tab1, tab2, tab3, tab4 = st.tabs([
    " Prédiction",
    " Graphiques",
    " Actualités",
    " Chatbot",
])


# ════════════════════════════════════════════════════════════════
# ONGLET 1 — PRÉDICTION
# ════════════════════════════════════════════════════════════════

with tab1:
    st.header(f"Projection financière — {label} ({ticker})")

    model_trend_key = f"{ticker}_trend"
    model_vol_key   = f"{ticker}_volatility"

    if model_trend_key not in models:
        st.error(f"Modèle non trouvé pour {ticker}. Lance le pipeline d'entraînement.")
        st.stop()

    if df_live.empty:
        st.error("Impossible de récupérer les données en temps réel. Vérifiez votre connexion.")
        st.stop()

    # ── Préparation des features live (données d'aujourd'hui) ──
    feature_cols = get_feature_names(include_macro=True)
    available    = [c for c in feature_cols if c in df_live.columns]
    df_live_clean = df_live[available].dropna()

    if df_live_clean.empty:
        st.error("Pas assez de données pour calculer les features.")
        st.stop()

    # Dernière ligne = données les plus récentes disponibles (aujourd'hui ou hier si marché fermé)
    X_live       = df_live_clean.iloc[[-1]]
    X_live_trend = scalers[model_trend_key].transform(X_live)
    X_live_vol   = scalers[model_vol_key].transform(X_live)

    pred_trend_class = int(models[model_trend_key].predict(X_live_trend)[0])
    pred_trend_proba = models[model_trend_key].predict_proba(X_live_trend)[0]
    pred_vol_class   = int(models[model_vol_key].predict(X_live_vol)[0])
    pred_vol_proba   = models[model_vol_key].predict_proba(X_live_vol)[0]

    trend_labels = {0: "Baisse", 1: "Stable", 2: "Hausse"}
    vol_labels   = {0: "Faible", 1: "Moyen", 2: "Élevé"}
    trend_label  = trend_labels[pred_trend_class]
    vol_label    = vol_labels[pred_vol_class]
    trend_color  = {"Baisse": "#e74c3c", "Stable": "#f39c12", "Hausse": "#27ae60"}[trend_label]
    vol_color    = {"Faible": "#27ae60", "Moyen": "#f39c12", "Élevé": "#e74c3c"}[vol_label]

    last_price = df_live["Close"].iloc[-1] if "Close" in df_live.columns else None
    last_date  = df_live_clean.index[-1].strftime("%d/%m/%Y")

    # ── Métriques principales ──
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div style='background:{trend_color}22; border-left:4px solid {trend_color};
             padding:16px; border-radius:8px; text-align:center;'>
            <div style='font-size:0.9em; color:gray;'>Tendance J+30</div>
            <div style='font-size:2.2em; font-weight:bold; color:{trend_color};'>{trend_label}</div>
            <div style='font-size:0.85em;'>Confiance : {max(pred_trend_proba):.0%}
            <span title="Probabilité assignée à la classe prédite par le modèle XGBoost"></span></div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background:{vol_color}22; border-left:4px solid {vol_color};
             padding:16px; border-radius:8px; text-align:center;'>
            <div style='font-size:0.9em; color:gray;'>Risque (volatilité)</div>
            <div style='font-size:2.2em; font-weight:bold; color:{vol_color};'>{vol_label}</div>
            <div style='font-size:0.85em;'>Confiance : {max(pred_vol_proba):.0%}
            <span title="Probabilité assignée à la classe prédite par le modèle XGBoost"></span></div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        price_str = f"{last_price:.2f}" if last_price is not None else "N/A"
        st.markdown(f"""
        <div style='background:#f8f9fa; border-left:4px solid #6c757d;
             padding:16px; border-radius:8px; text-align:center;'>
            <div style='font-size:0.9em; color:gray;'>Dernière observation</div>
            <div style='font-size:2.2em; font-weight:bold;'>{price_str}</div>
            <div style='font-size:0.85em; color:gray;'>{last_date}</div>
        </div>
        """, unsafe_allow_html=True)

    st.caption("La confiance = probabilité max parmi les 3 classes (hausse/stable/baisse). "
               "Ce n'est pas une certitude — c'est la classe que le modèle juge la plus probable.")

    st.divider()

    # ── Probabilités par classe ──
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Probabilités — Tendance")
        fig_trend = go.Figure(go.Bar(
            x=["Baisse", "Stable", "Hausse"],
            y=[p * 100 for p in pred_trend_proba],
            marker_color=["#e74c3c", "#f39c12", "#27ae60"],
            text=[f"{p:.1%}" for p in pred_trend_proba],
            textposition="outside",
        ))
        fig_trend.update_layout(
            yaxis_title="Probabilité (%)", yaxis_range=[0, 105],
            height=300, margin=dict(t=20, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_b:
        st.subheader("Probabilités — Risque")
        fig_vol = go.Figure(go.Bar(
            x=["Faible", "Moyen", "Élevé"],
            y=[p * 100 for p in pred_vol_proba],
            marker_color=["#27ae60", "#f39c12", "#e74c3c"],
            text=[f"{p:.1%}" for p in pred_vol_proba],
            textposition="outside",
        ))
        fig_vol.update_layout(
            yaxis_title="Probabilité (%)", yaxis_range=[0, 105],
            height=300, margin=dict(t=20, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    st.divider()

    # ── SHAP Waterfall ──
    st.subheader("Explication SHAP — Pourquoi cette prédiction ?")
    st.caption("Chaque barre montre la contribution d'une feature à la prédiction. "
               "Rouge = pousse vers la classe prédite. Bleu = tire à l'encontre.")

    if st.button("Générer l'explication SHAP", type="primary"):
        with st.spinner("Calcul des valeurs SHAP..."):
            try:
                from src.explainability.shap_plots import explain_prediction
                # On utilise df_live_clean : données réelles d'aujourd'hui
                # sample_idx=-1 = dernière ligne = le jour qu'on vient de prédire
                X_background = df_live_clean.iloc[:min(len(df_live_clean), 300)]
                fig_shap, shap_summary = explain_prediction(
                    model=models[model_trend_key],
                    scaler=scalers[model_trend_key],
                    X=X_background,
                    feature_names=available,
                    sample_idx=-1,
                    ticker=ticker,
                    model_type="trend",
                    prediction_date=last_date,
                )
                st.pyplot(fig_shap, use_container_width=True)
                plt.close()

                # Top contributions numériques
                st.subheader("Top contributions (valeurs SHAP)")
                shap_df = pd.DataFrame(
                    list(shap_summary.items())[:10],
                    columns=["Feature", "Contribution SHAP"]
                )
                shap_df["Impact"] = shap_df["Contribution SHAP"].apply(
                    lambda x: "↑ haussier" if x > 0 else "↓ baissier"
                )
                st.dataframe(shap_df.style.background_gradient(
                    subset=["Contribution SHAP"], cmap="RdYlGn"
                ), use_container_width=True)

            except Exception as e:
                st.error(f"Erreur SHAP : {e}")


# ════════════════════════════════════════════════════════════════
# ONGLET 2 — GRAPHIQUES
# ════════════════════════════════════════════════════════════════

with tab2:
    st.header(f"Analyse Technique — {label} ({ticker})")

    if df_feat.empty:
        st.error("Données non disponibles.")
        st.stop()

    # Sélection de la période
    period = st.select_slider(
        "Période d'affichage",
        options=["6 mois", "1 an", "2 ans", "5 ans", "Tout"],
        value="2 ans",
    )
    period_map = {"6 mois": 126, "1 an": 252, "2 ans": 504, "5 ans": 1260, "Tout": len(df_feat)}
    df_plot = df_feat.iloc[-period_map[period]:]

    # ── Prix + EMA ──
    st.subheader("Prix de clôture et moyennes mobiles")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot["Close"],
        name="Prix", line=dict(color="#2c3e50", width=1.5)
    ))
    if "ema_short" in df_plot.columns:
        fig_price.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["ema_short"],
            name="EMA 20j", line=dict(color="#3498db", width=1, dash="dot")
        ))
    if "ema_long" in df_plot.columns:
        fig_price.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["ema_long"],
            name="EMA 50j", line=dict(color="#e67e22", width=1, dash="dot")
        ))
    fig_price.update_layout(height=380, margin=dict(t=20, b=20),
                             legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig_price, use_container_width=True)

    # ── RSI ──
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("RSI (14j)")
        if "rsi" in df_plot.columns:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot["rsi"],
                name="RSI", line=dict(color="purple", width=1.5)
            ))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",
                              annotation_text="Suracheté (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green",
                              annotation_text="Survendu (30)")
            fig_rsi.add_hrect(y0=30, y1=70, fillcolor="gray", opacity=0.05)
            fig_rsi.update_layout(height=280, margin=dict(t=10, b=10),
                                  yaxis_range=[0, 100])
            st.plotly_chart(fig_rsi, use_container_width=True)

    # ── MACD ──
    with col2:
        st.subheader("MACD")
        if "macd" in df_plot.columns:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot["macd"],
                name="MACD", line=dict(color="blue", width=1.5)
            ))
            fig_macd.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot["macd_signal"],
                name="Signal", line=dict(color="orange", width=1.5)
            ))
            if "macd_diff" in df_plot.columns:
                colors = ["#27ae60" if v >= 0 else "#e74c3c"
                          for v in df_plot["macd_diff"]]
                fig_macd.add_trace(go.Bar(
                    x=df_plot.index, y=df_plot["macd_diff"],
                    name="Histogramme", marker_color=colors, opacity=0.6
                ))
            fig_macd.update_layout(height=280, margin=dict(t=10, b=10))
            st.plotly_chart(fig_macd, use_container_width=True)

    # ── Bandes de Bollinger ──
    st.subheader("Bandes de Bollinger (20j, ±2σ)")
    if "bb_pct_b" in df_plot.columns:
        import ta
        bb = ta.volatility.BollingerBands(
            close=df_plot["Close"], window=20, window_dev=2
        )
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(
            x=df_plot.index, y=bb.bollinger_hband(),
            name="Bande haute", line=dict(color="gray", dash="dot", width=1)
        ))
        fig_bb.add_trace(go.Scatter(
            x=df_plot.index, y=bb.bollinger_lband(),
            name="Bande basse", line=dict(color="gray", dash="dot", width=1),
            fill="tonexty", fillcolor="rgba(128,128,128,0.07)"
        ))
        fig_bb.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["Close"],
            name="Prix", line=dict(color="#2c3e50", width=1.5)
        ))
        fig_bb.add_trace(go.Scatter(
            x=df_plot.index, y=bb.bollinger_mavg(),
            name="Moyenne 20j", line=dict(color="orange", width=1, dash="dot")
        ))
        fig_bb.update_layout(height=320, margin=dict(t=10, b=10),
                              legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig_bb, use_container_width=True)

    # ── Volatilité historique ──
    st.subheader("Volatilité historique (annualisée)")
    col3, col4 = st.columns(2)
    with col3:
        if "volatility_20d" in df_plot.columns:
            fig_vol2 = go.Figure()
            fig_vol2.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot["volatility_20d"],
                name="Vol 20j", line=dict(color="purple", width=1.5)
            ))
            if "volatility_60d" in df_plot.columns:
                fig_vol2.add_trace(go.Scatter(
                    x=df_plot.index, y=df_plot["volatility_60d"],
                    name="Vol 60j", line=dict(color="mediumpurple", width=2)
                ))
            fig_vol2.update_layout(height=280, margin=dict(t=10, b=10))
            st.plotly_chart(fig_vol2, use_container_width=True)

    with col4:
        if "volume_ratio" in df_plot.columns:
            st.subheader("Volume relatif (vs moyenne 20j)")
            fig_vol3 = go.Figure(go.Bar(
                x=df_plot.index, y=df_plot["volume_ratio"],
                marker_color=["#e74c3c" if v > 2 else "#3498db"
                               for v in df_plot["volume_ratio"]],
                name="Volume relatif"
            ))
            fig_vol3.add_hline(y=1, line_dash="dash", line_color="black",
                               annotation_text="Moyenne")
            fig_vol3.update_layout(height=280, margin=dict(t=30, b=10))
            st.plotly_chart(fig_vol3, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# ONGLET 3 — ACTUALITÉS
# ════════════════════════════════════════════════════════════════

with tab3:
    st.header(f"Actualités Récentes — {label} ({ticker})")

    if retriever is None:
        st.warning("Index RAG non disponible. Lance `notebooks/05_rag_pipeline.ipynb` pour construire l'index.")
    else:
        col_a, col_b = st.columns([2, 1])

        with col_a:
            st.subheader("Résumé des dernières news")
            if st.button("Générer le résumé", type="primary"):
                with st.spinner("Analyse des news avec Mistral..."):
                    try:
                        from src.rag.retriever import summarize_ticker_news
                        result = summarize_ticker_news(ticker, retriever)
                        st.info(result["response"])

                        st.caption(f"Basé sur {result['n_chunks_used']} articles analysés")
                        if result["sources"]:
                            with st.expander("Sources utilisées"):
                                for s in result["sources"]:
                                    st.markdown(
                                        f"**{s['label']}** — {s['title']}\n\n"
                                        f"*{s['source']} — {s['published_at'][:10]}*"
                                        + (f"\n\n[Lire l'article]({s['url']})" if s.get("url") else "")
                                    )
                    except Exception as e:
                        st.error(f"Erreur : {e}")

        with col_b:
            st.subheader("Recherche personnalisée")
            custom_query = st.text_area(
                "Posez votre question sur cet actif",
                placeholder=f"Ex: What is the latest news about {label}?",
                height=100,
            )
            if st.button("Rechercher") and custom_query:
                with st.spinner("Recherche en cours..."):
                    try:
                        from src.rag.retriever import generate_rag_response
                        result = generate_rag_response(
                            query=custom_query,
                            retriever=retriever,
                            ticker_filter=ticker,
                            top_k=5,
                        )
                        st.info(result["response"])
                    except Exception as e:
                        st.error(f"Erreur : {e}")


# ════════════════════════════════════════════════════════════════
# ONGLET 4 — CHATBOT
# ════════════════════════════════════════════════════════════════

with tab4:
    st.header(" Agent FinSight")
    st.caption(
        "Posez vos questions en langage naturel. L'agent utilise les modèles ML "
        "et les news récentes pour répondre."
    )

    # Initialisation de la session
    if "agent" not in st.session_state:
        with st.spinner("Initialisation de l'agent..."):
            try:
                st.session_state.agent = load_agent()
                st.session_state.messages = []
            except Exception as e:
                st.error(f"Erreur initialisation agent : {e}")
                st.stop()

    # Exemples de questions
    st.subheader("Questions suggérées")
    suggestions = [
        f"Quelle est la prédiction pour {ticker} ?",
        f"Quelles sont les dernières nouvelles sur {label} ?",
        f"Fais-moi un résumé complet de {label}.",
        "Quels actifs ont le meilleur potentiel en ce moment ?",
    ]
    cols = st.columns(len(suggestions))
    for col, suggestion in zip(cols, suggestions):
        if col.button(suggestion, use_container_width=True):
            st.session_state.pending_message = suggestion

    st.divider()

    # Affichage de l'historique
    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Traitement du message en attente (depuis les boutons suggestions)
    if "pending_message" in st.session_state:
        user_input = st.session_state.pop("pending_message")
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("L'agent réfléchit..."):
                response = st.session_state.agent.chat(user_input)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Input utilisateur
    if user_input := st.chat_input("Posez votre question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("L'agent réfléchit..."):
                response = st.session_state.agent.chat(user_input)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Bouton reset
    if st.session_state.get("messages"):
        if st.button("Réinitialiser la conversation", type="secondary"):
            st.session_state.messages = []
            st.session_state.agent.reset()
            st.rerun()
