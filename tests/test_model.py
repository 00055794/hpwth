"""
Unit tests for the model module.
"""

import sys
import os
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import (
    generate_sample_data,
    get_train_test_split,
    preprocess,
    scale_features,
    split_features_target,
)
from src.model import evaluate, get_models, load_model, save_model


@pytest.fixture(scope="module")
def prepared_data():
    """Return scaled train/test arrays for reuse across tests."""
    df = preprocess(generate_sample_data(n_samples=300, random_state=1))
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size=0.2)
    X_tr_s, X_te_s, _ = scale_features(X_train, X_test)
    return X_tr_s, X_te_s, y_train, y_test


class TestGetModels:
    def test_returns_dict(self):
        models = get_models()
        assert isinstance(models, dict)
        assert len(models) >= 5

    def test_standard_models_present(self):
        models = get_models()
        for name in ("linear_regression", "ridge", "lasso", "random_forest", "gradient_boosting"):
            assert name in models

    def test_models_have_fit_predict(self):
        for name, model in get_models().items():
            assert hasattr(model, "fit"), f"{name} has no fit()"
            assert hasattr(model, "predict"), f"{name} has no predict()"


class TestEvaluate:
    def test_metrics_keys(self, prepared_data):
        X_tr, X_te, y_tr, y_te = prepared_data
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(X_tr, y_tr)
        metrics = evaluate(model, X_te, y_te)
        assert set(metrics.keys()) == {"mae", "rmse", "r2", "mape"}

    def test_r2_range(self, prepared_data):
        X_tr, X_te, y_tr, y_te = prepared_data
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=20, random_state=42).fit(X_tr, y_tr)
        metrics = evaluate(model, X_te, y_te)
        assert -1.0 <= metrics["r2"] <= 1.0

    def test_mae_non_negative(self, prepared_data):
        X_tr, X_te, y_tr, y_te = prepared_data
        from sklearn.linear_model import Ridge
        model = Ridge().fit(X_tr, y_tr)
        metrics = evaluate(model, X_te, y_te)
        assert metrics["mae"] >= 0

    def test_rmse_non_negative(self, prepared_data):
        X_tr, X_te, y_tr, y_te = prepared_data
        from sklearn.linear_model import Ridge
        model = Ridge().fit(X_tr, y_tr)
        metrics = evaluate(model, X_te, y_te)
        assert metrics["rmse"] >= 0


class TestSaveLoadModel:
    def test_save_and_load(self, prepared_data):
        X_tr, X_te, y_tr, y_te = prepared_data
        from sklearn.linear_model import Ridge
        model = Ridge().fit(X_tr, y_tr)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            save_model(model, path)
            loaded = load_model(path)
            np.testing.assert_array_almost_equal(
                model.predict(X_te), loaded.predict(X_te)
            )
        finally:
            os.unlink(path)
