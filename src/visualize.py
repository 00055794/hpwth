"""
Visualisation helpers for house prices prediction.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_price_distribution(
    y: pd.Series | np.ndarray,
    title: str = "House Price Distribution",
    output_path: str | None = None,
) -> None:
    """
    Plot the distribution of house prices.

    Parameters
    ----------
    y : array-like
        Target price values.
    title : str
        Plot title.
    output_path : str or None
        If provided, save the figure to this path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(y, kde=True, ax=axes[0], color="steelblue")
    axes[0].set_title(title)
    axes[0].set_xlabel("Price ($)")
    axes[0].set_ylabel("Count")

    sns.histplot(np.log1p(y), kde=True, ax=axes[1], color="darkorange")
    axes[1].set_title(f"Log-transformed {title}")
    axes[1].set_xlabel("log(Price + 1)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_path: str | None = None,
) -> None:
    """
    Plot a correlation heatmap for numeric features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with numeric features.
    output_path : str or None
        If provided, save the figure to this path.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 15,
    output_path: str | None = None,
) -> None:
    """
    Plot feature importances for tree-based models.

    Parameters
    ----------
    model :
        A fitted estimator with a ``feature_importances_`` attribute.
    feature_names : list[str]
        Names of the features.
    top_n : int
        Number of top features to display.
    output_path : str or None
        If provided, save the figure to this path.
    """
    if not hasattr(model, "feature_importances_"):
        print(f"Model {type(model).__name__} does not support feature importances.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in indices]
    values = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(names[::-1], values[::-1], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances – {type(model).__name__}")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_predictions_vs_actual(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    output_path: str | None = None,
) -> None:
    """
    Scatter plot of predicted versus actual prices.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    model_name : str
        Name of the model for the plot title.
    output_path : str or None
        If provided, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=15, color="steelblue")
    lims = [
        min(np.min(y_true), np.min(y_pred)),
        max(np.max(y_true), np.max(y_pred)),
    ]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.set_title(f"Predicted vs Actual – {model_name}")
    ax.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    output_path: str | None = None,
) -> None:
    """
    Plot residuals (prediction errors) against predicted values.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    model_name : str
        Name of the model for the plot title.
    output_path : str or None
        If provided, save the figure to this path.
    """
    residuals = np.asarray(y_true) - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(y_pred, residuals, alpha=0.3, s=15, color="darkorange")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Predicted Price ($)")
    axes[0].set_ylabel("Residual ($)")
    axes[0].set_title(f"Residuals vs Fitted – {model_name}")

    sns.histplot(residuals, kde=True, ax=axes[1], color="darkorange")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_title(f"Residual Distribution – {model_name}")
    axes[1].set_xlabel("Residual ($)")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(
    results: dict[str, dict[str, float]],
    metric: str = "r2",
    output_path: str | None = None,
) -> None:
    """
    Bar chart comparing multiple models on a given metric.

    Parameters
    ----------
    results : dict[str, dict[str, float]]
        Mapping of model name to metrics dict (as returned by
        :func:`src.model.evaluate`).
    metric : str
        Metric key to compare (``mae``, ``rmse``, ``r2``, or ``mape``).
    output_path : str or None
        If provided, save the figure to this path.
    """
    names = list(results.keys())
    values = [results[n][metric] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    # Higher is better for R²; lower is better for error metrics.
    if metric == "r2":
        best = max(values)
        colors = ["steelblue" if v == best else "lightsteelblue" for v in values]
    else:
        best = min(values)
        colors = ["steelblue" if v == best else "lightsteelblue" for v in values]
    ax.bar(names, values, color=colors)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Model Comparison – {metric.upper()}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
