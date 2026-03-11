"""Evaluation utility scaffolding for binary loan-approval models."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def _predict_scores(model: Any, X_test: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        min_value = np.min(scores)
        max_value = np.max(scores)
        if np.isclose(max_value, min_value):
            return np.full_like(scores, 0.5, dtype=float)
        return (scores - min_value) / (max_value - min_value)
    predictions = model.predict(X_test)
    return predictions.astype(float)


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    """Evaluate a model on test data and return metrics and plots."""
    y_pred = model.predict(X_test)
    y_scores = _predict_scores(model, X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_scores)),
        "pr_auc": float(average_precision_score(y_test, y_scores)),
        "brier_score": float(compute_brier_score(y_test, y_scores)),
    }

    metrics["confusion_matrix_figure"] = plot_confusion_matrix(y_test, y_pred)
    metrics["roc_curve_figure"] = plot_roc_curve(y_test, y_scores)
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    """Plot and return confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray) -> plt.Figure:
    """Plot and return ROC curve figure."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc_score:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def compute_brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Compute Brier score for probabilistic predictions."""
    return float(brier_score_loss(y_true, y_proba))
