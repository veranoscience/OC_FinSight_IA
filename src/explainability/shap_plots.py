"""
Explicabilité SHAP pour FinSight

Deux niveaux d'explication :
  - Global  : comportement général du modèle sur tout le dataset (beeswarm, bar)
  - Local   : pourquoi CETTE prédiction pour CET actif aujourd'hui (waterfall)

Le waterfall local est obligatoire à chaque prédiction
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import TREND_LABELS, VOLATILITY_LABELS

logger = logging.getLogger(__name__)


#  Calcul des valeurs SHAP 


def compute_shap_values(
    model: XGBClassifier,
    X_scaled: np.ndarray,
    feature_names: list[str],
) -> shap.Explanation:
    """
    Calcule les valeurs SHAP pour un ensemble de données.

    On utilise TreeExplainer — l'explainer dédié aux arbres de décision
    (XGBoost, Random Forest). Il est exact (pas d'approximation) et rapide.

    Pour une classification multiclasse (3 classes), SHAP retourne
    une matrice de shape (n_samples, n_features, n_classes).
    Chaque valeur indique la contribution d'une feature au score de chaque classe.

    Args:
        model: Modèle XGBClassifier entraîné.
        X_scaled: Données scalées (numpy array), shape (n_samples, n_features).
        feature_names: Noms des features dans l'ordre des colonnes.

    Returns:
        shap.Explanation avec .values de shape (n_samples, n_features, n_classes)
        et .feature_names renseigné.
    """
    logger.info("Calcul des valeurs SHAP (%d samples)...", X_scaled.shape[0])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_scaled)

    # Attache les noms de features pour les plots
    shap_values.feature_names = feature_names

    logger.info("SHAP calculé : shape %s", str(shap_values.values.shape))
    return shap_values


def get_shap_for_class(
    shap_values: shap.Explanation,
    class_idx: int,
) -> shap.Explanation:
    """
    Extrait les valeurs SHAP pour une classe spécifique.

    En multiclasse, on travaille souvent sur la classe prédite.
    Cette fonction retourne un Explanation 2D (n_samples × n_features)
    pour la classe demandée.

    Args:
        shap_values: Explanation multiclasse (shape n×f×c).
        class_idx: Index de la classe (0=baisse, 1=stable, 2=hausse pour tendance).

    Returns:
        Explanation 2D pour la classe class_idx.
    """
    return shap_values[:, :, class_idx]


#  Plots globaux 


def plot_beeswarm(
    shap_values: shap.Explanation,
    class_idx: int,
    class_label: str,
    max_display: int = 15,
    title: str | None = None,
) -> plt.Figure:
    """
    Beeswarm plot — explication globale du modèle pour une classe.

    Chaque point = une observation du dataset.
    Position horizontale = impact de la feature (valeur SHAP).
    Couleur = valeur de la feature (rouge = haute, bleu = basse).

    Ce plot répond à : "Quelles features ont le plus d'impact sur la
    prédiction de cette classe, et dans quel sens ?"

    Args:
        shap_values: Explanation multiclasse (shape n×f×c).
        class_idx: Classe à visualiser.
        class_label: Nom lisible de la classe (ex: 'hausse').
        max_display: Nombre maximum de features à afficher.
        title: Titre du graphe (optionnel).

    Returns:
        Figure matplotlib.
    """
    sv_class = get_shap_for_class(shap_values, class_idx)

    fig = plt.figure(figsize=(10, max_display * 0.4 + 2))
    shap.plots.beeswarm(sv_class, max_display=max_display, show=False)

    if title:
        plt.title(title, fontsize=12, pad=15)
    else:
        plt.title(f'Impact des features — classe "{class_label}"', fontsize=12, pad=15)

    plt.tight_layout()
    return fig


def plot_bar_global(
    shap_values: shap.Explanation,
    class_idx: int,
    class_label: str,
    max_display: int = 15,
) -> plt.Figure:
    """
    Bar plot — importance globale des features (moyenne |SHAP|).

    Plus simple que le beeswarm, il montre juste le classement des features
    par importance moyenne sur tout le dataset.

    Args:
        shap_values: Explanation multiclasse.
        class_idx: Classe à visualiser.
        class_label: Nom lisible de la classe.
        max_display: Nombre de features à afficher.

    Returns:
        Figure matplotlib.
    """
    sv_class = get_shap_for_class(shap_values, class_idx)

    fig = plt.figure(figsize=(9, max_display * 0.4 + 2))
    shap.plots.bar(sv_class, max_display=max_display, show=False)
    plt.title(f'Importance globale (|SHAP| moyen) — classe "{class_label}"', fontsize=11)
    plt.tight_layout()
    return fig


#  Plot local (prédiction individuelle) 


def plot_waterfall(
    shap_values: shap.Explanation,
    sample_idx: int,
    class_idx: int,
    class_label: str,
    ticker: str,
    prediction_date: str | None = None,
    max_display: int = 12,
) -> plt.Figure:
    """
    Waterfall plot — explication d'une prédiction individuelle.

    C'est le plot obligatoire à chaque prédiction 
    Il montre, pour une observation précise :
    - La valeur de base (moyenne des prédictions sur le dataset)
    - La contribution de chaque feature (positive = pousse vers cette classe)
    - Le score final prédit pour cette classe

    Exemple de lecture :
      Base = 0.33 (probabilité moyenne de "hausse")
      + RSI élevé     , +0.12
      + MACD positif  , +0.08
      - Volume faible , -0.05
      = Score final   = 0.48 (la classe "hausse" est prédite)

    Args:
        shap_values: Explanation multiclasse (shape n×f×c).
        sample_idx: Index de l'observation à expliquer (0 = dernière si live).
        class_idx: Classe prédite à expliquer.
        class_label: Nom lisible de la classe.
        ticker: Symbole boursier (pour le titre).
        prediction_date: Date de la prédiction (pour le titre).
        max_display: Nombre de features à afficher.

    Returns:
        Figure matplotlib.
    """
    sv_class = get_shap_for_class(shap_values, class_idx)

    fig = plt.figure(figsize=(10, max_display * 0.45 + 2))
    shap.plots.waterfall(sv_class[sample_idx], max_display=max_display, show=False)

    date_str = f" — {prediction_date}" if prediction_date else ""
    plt.title(
        f'Explication locale — {ticker}{date_str}\n'
        f'Classe prédite : "{class_label}"',
        fontsize=11, pad=15
    )
    plt.tight_layout()
    return fig


def plot_force(
    shap_values: shap.Explanation,
    sample_idx: int,
    class_idx: int,
    class_label: str,
) -> None:
    """
    Force plot — version interactive du waterfall 


    Args:
        shap_values: Explanation multiclasse.
        sample_idx: Index de l'observation.
        class_idx: Classe à expliquer.
        class_label: Nom lisible.
    """
    sv_class = get_shap_for_class(shap_values, class_idx)
    shap.initjs()
    return shap.plots.force(sv_class[sample_idx], show=True)


#  Pipeline complet d'explication 


def explain_prediction(
    model: XGBClassifier,
    scaler: StandardScaler,
    X: pd.DataFrame,
    feature_names: list[str],
    sample_idx: int = -1,
    ticker: str = "",
    model_type: str = "trend",
    prediction_date: str | None = None,
) -> tuple[plt.Figure, dict]:
    """
    Pipeline complet : calcule SHAP et génère le waterfall pour une prédiction.

    Fonction principale appelée par le dashboard Streamlit et predict.py.
    Elle encapsule tout le workflow SHAP en un seul appel.

    Args:
        model: Modèle XGBClassifier entraîné.
        scaler: StandardScaler fitté sur le train.
        X: DataFrame des features (non scalées).
        feature_names: Noms des features.
        sample_idx: Index de la ligne à expliquer (-1 = dernière ligne).
        ticker: Symbole boursier.
        model_type: 'trend' ou 'volatility'.
        prediction_date: Date de la prédiction.

    Returns:
        Tuple (fig_waterfall, shap_summary) où shap_summary est un dict
        {feature: shap_value} pour la classe prédite, trié par impact absolu.
    """
    X_scaled = scaler.transform(X)

    # Calcul sur un sous-ensemble pour la rapidité (max 500 samples pour global)
    n_bg = min(len(X), 500)
    X_background = X_scaled[:n_bg]

    shap_values = compute_shap_values(model, X_background, feature_names)

    # Prédiction sur la ligne demandée
    if sample_idx == -1:
        sample_idx = n_bg - 1

    pred_class = int(model.predict(X_scaled[sample_idx:sample_idx+1])[0])

    if model_type == "trend":
        labels_map = TREND_LABELS
    else:
        labels_map = VOLATILITY_LABELS

    class_label = labels_map[pred_class]

    fig = plot_waterfall(
        shap_values,
        sample_idx=sample_idx,
        class_idx=pred_class,
        class_label=class_label,
        ticker=ticker,
        prediction_date=prediction_date,
    )

    # Résumé des contributions SHAP pour la classe prédite
    sv_class = get_shap_for_class(shap_values, pred_class)
    shap_row  = sv_class.values[sample_idx]
    shap_summary = dict(
        sorted(
            zip(feature_names, shap_row),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
    )

    logger.info(
        "Explication SHAP générée pour %s — classe '%s'",
        ticker, class_label
    )
    return fig, shap_summary
