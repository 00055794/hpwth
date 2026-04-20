"""
Model definitions and training utilities for house prices prediction.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def get_models() -> dict:
    """
    Return a dictionary of named regression models.

    Returns
    -------
    dict
        Mapping of model name to unfitted estimator instance.
    """
    models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=10.0),
        "lasso": Lasso(alpha=100.0, max_iter=10000),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42,
        ),
    }

    # Optional: add XGBoost / LightGBM if available
    try:
        from xgboost import XGBRegressor  # noqa: PLC0415

        models["xgboost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMRegressor  # noqa: PLC0415

        models["lightgbm"] = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
    except ImportError:
        pass

    return models


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
) -> dict[str, float]:
    """
    Compute regression evaluation metrics for a fitted model.

    Parameters
    ----------
    model :
        A fitted scikit-learn-compatible estimator.
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        True target values.

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``mae``, ``rmse``, ``r2``, ``mape``.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    # Mean absolute percentage error (avoid division by zero)
    y_test_arr = np.asarray(y_test, dtype=float)
    mask = y_test_arr != 0
    mape = np.mean(np.abs((y_test_arr[mask] - y_pred[mask]) / y_test_arr[mask])) * 100
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def cross_validate(
    model,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    cv: int = 5,
) -> dict[str, float]:
    """
    Run k-fold cross-validation and return mean / std of R² scores.

    Parameters
    ----------
    model :
        An unfitted scikit-learn-compatible estimator.
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``cv_r2_mean`` and ``cv_r2_std``.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)
    return {"cv_r2_mean": scores.mean(), "cv_r2_std": scores.std()}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model, filepath: str) -> None:
    """
    Persist a fitted model to disk using joblib.

    Parameters
    ----------
    model :
        Fitted estimator to save.
    filepath : str
        Destination file path (e.g. ``models/rf.pkl``).
    """
    joblib.dump(model, filepath)


def load_model(filepath: str):
    """
    Load a persisted model from disk.

    Parameters
    ----------
    filepath : str
        Path to the saved model file.

    Returns
    -------
    object
        The loaded estimator.
    """
    return joblib.load(filepath)
