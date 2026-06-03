"""
Métriques d'évaluation des modèles FinSight.
F1-score pondéré, matrice de confusion, AUC-ROC par classe.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """
    Calcule les métriques de classification pour un split.

    Args:
        y_true: Labels vrais (entiers 0, 1, 2).
        y_pred: Prédictions du modèle.
        prefix: Préfixe pour les clés du dictionnaire (ex: 'split_1').

    Returns:
        Dictionnaire avec les métriques :
        - f1_weighted : F1-score pondéré (métrique principale)
        - f1_macro : F1-score macro
        - f1_per_class : F1 par classe (dict)
    """
    sep = "_" if prefix else ""
    key = lambda name: f"{prefix}{sep}{name}"  # noqa: E731

    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    metrics = {
        key("f1_weighted"): float(f1_weighted),
        key("f1_macro"): float(f1_macro),
    }
    class_names = ["baisse", "stable", "hausse"]
    for i, cls in enumerate(class_names):
        if i < len(f1_per_class):
            metrics[key(f"f1_{cls}")] = float(f1_per_class[i])

    return metrics


def print_classification_report(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str] | None = None,
    title: str = "Rapport de classification",
) -> None:
    """
    Affiche le rapport de classification sklearn avec un titre.

    Args:
        y_true: Labels vrais.
        y_pred: Prédictions.
        target_names: Noms des classes (défaut : ['baisse', 'stable', 'hausse']).
        title: Titre affiché avant le rapport.
    """
    if target_names is None:
        target_names = ["baisse", "stable", "hausse"]

    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))


def plot_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Matrice de confusion",
    normalize: str | None = "true",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Affiche la matrice de confusion normalisée.

    Args:
        y_true: Labels vrais.
        y_pred: Prédictions.
        class_names: Noms des classes.
        title: Titre du graphe.
        normalize: Normalisation ('true', 'pred', 'all', None).
        ax: Axes matplotlib existant (optionnel).

    Returns:
        Axes matplotlib avec la matrice.
    """
    if class_names is None:
        class_names = ["baisse", "stable", "hausse"]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        normalize=normalize,
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(title)
    return ax


def plot_roc_curves(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Courbes ROC (one-vs-rest)",
) -> plt.Figure:
    """
    Trace les courbes ROC en mode one-vs-rest pour chaque classe.

    Args:
        y_true: Labels vrais.
        y_proba: Probabilités prédites (shape: n_samples × n_classes).
        class_names: Noms des classes.
        title: Titre du graphe.

    Returns:
        Figure matplotlib.
    """
    if class_names is None:
        class_names = ["baisse", "stable", "hausse"]

    n_classes = len(class_names)
    colors = ["#e74c3c", "#f39c12", "#27ae60"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Aléatoire (AUC=0.5)")

    y_true_arr = np.array(y_true)

    for i, (cls, color) in enumerate(zip(class_names, colors)):
        y_bin = (y_true_arr == i).astype(int)
        if y_bin.sum() == 0:
            continue
        try:
            fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
            auc = roc_auc_score(y_bin, y_proba[:, i])
            ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{cls} (AUC={auc:.3f})")
        except Exception as e:
            logger.warning("Impossible de tracer ROC pour la classe %s : %s", cls, e)

    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def plot_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 20,
    title: str = "Importance des features (XGBoost gain)",
) -> plt.Figure:
    """
    Affiche les top N features par importance (gain moyen).

    Args:
        model: Modèle XGBClassifier entraîné.
        feature_names: Noms des features dans l'ordre.
        top_n: Nombre de features à afficher.
        title: Titre du graphe.

    Returns:
        Figure matplotlib.
    """
    importance = model.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importance})
    fi_df = fi_df.sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, top_n * 0.35 + 1))
    bars = ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color="steelblue")
    ax.set_xlabel("Importance (gain moyen)")
    ax.set_title(title)

    # Annotation des valeurs
    for bar, val in zip(bars, fi_df["importance"][::-1]):
        ax.text(
            bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=8
        )

    plt.tight_layout()
    return fig


def summarize_walk_forward(wf_results: dict[str, float]) -> None:
    """
    Affiche un résumé lisible des résultats walk-forward.

    Args:
        wf_results: Dictionnaire retourné par train.py (mean_*, std_*).
    """
    print("\n" + "═" * 55)
    print("  RÉSULTATS WALK-FORWARD CROSS-VALIDATION")
    print("═" * 55)
    for key, val in sorted(wf_results.items()):
        if key.startswith("mean_"):
            metric = key[5:]
            std_key = f"std_{metric}"
            std_val = wf_results.get(std_key, 0)
            print(f"  {metric:30s}: {val:.4f} ± {std_val:.4f}")
    print("═" * 55 + "\n")
