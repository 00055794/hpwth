"""
Unit tests for the data preprocessing module.
"""

import numpy as np
import pandas as pd
import pytest

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import (
    generate_sample_data,
    get_train_test_split,
    preprocess,
    scale_features,
    split_features_target,
)


class TestGenerateSampleData:
    def test_returns_dataframe(self):
        df = generate_sample_data(n_samples=100)
        assert isinstance(df, pd.DataFrame)

    def test_shape(self):
        df = generate_sample_data(n_samples=200)
        assert df.shape[0] == 200

    def test_required_columns(self):
        df = generate_sample_data(n_samples=50)
        required = {
            "sqft_living", "sqft_lot", "bedrooms", "bathrooms",
            "floors", "condition", "grade", "yr_built", "yr_renovated",
            "waterfront", "view", "zipcode", "price",
        }
        assert required.issubset(set(df.columns))

    def test_price_positive(self):
        df = generate_sample_data(n_samples=100)
        assert (df["price"] > 0).all()

    def test_reproducibility(self):
        df1 = generate_sample_data(n_samples=50, random_state=0)
        df2 = generate_sample_data(n_samples=50, random_state=0)
        pd.testing.assert_frame_equal(df1, df2)


class TestPreprocess:
    @pytest.fixture
    def raw_df(self):
        return generate_sample_data(n_samples=100)

    def test_no_duplicates_removed(self, raw_df):
        df = preprocess(raw_df)
        assert df.duplicated().sum() == 0

    def test_derived_features_added(self, raw_df):
        df = preprocess(raw_df)
        assert "house_age" in df.columns
        assert "was_renovated" in df.columns
        assert "sqft_living_log" in df.columns

    def test_house_age_non_negative(self, raw_df):
        df = preprocess(raw_df)
        assert (df["house_age"] >= 0).all()

    def test_was_renovated_binary(self, raw_df):
        df = preprocess(raw_df)
        assert set(df["was_renovated"].unique()).issubset({0, 1})

    def test_fills_missing_values(self):
        df = generate_sample_data(n_samples=50)
        df.loc[df.index[:5], "sqft_living"] = np.nan
        df_proc = preprocess(df)
        assert df_proc["sqft_living"].isnull().sum() == 0

    def test_zipcode_encoded(self, raw_df):
        df = preprocess(raw_df)
        assert df["zipcode"].dtype in [np.int32, np.int64, int]


class TestSplitFeaturesTarget:
    @pytest.fixture
    def processed_df(self):
        return preprocess(generate_sample_data(n_samples=100))

    def test_returns_x_y(self, processed_df):
        X, y = split_features_target(processed_df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_price_not_in_features(self, processed_df):
        X, _ = split_features_target(processed_df)
        assert "price" not in X.columns

    def test_lengths_match(self, processed_df):
        X, y = split_features_target(processed_df)
        assert len(X) == len(y)

    def test_no_price_per_sqft_leakage(self, processed_df):
        X, _ = split_features_target(processed_df)
        assert "price_per_sqft" not in X.columns


class TestTrainTestSplit:
    def test_split_sizes(self):
        df = preprocess(generate_sample_data(n_samples=100))
        X, y = split_features_target(df)
        X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size=0.2)
        assert len(X_train) == 80
        assert len(X_test) == 20

    def test_no_overlap(self):
        df = preprocess(generate_sample_data(n_samples=100))
        X, y = split_features_target(df)
        X_train, X_test, _, _ = get_train_test_split(X, y, test_size=0.2)
        assert len(set(X_train.index) & set(X_test.index)) == 0


class TestScaleFeatures:
    def test_shape_preserved(self):
        df = preprocess(generate_sample_data(n_samples=100))
        X, y = split_features_target(df)
        X_train, X_test, _, _ = get_train_test_split(X, y, test_size=0.2)
        X_tr_s, X_te_s, scaler = scale_features(X_train, X_test)
        assert X_tr_s.shape == X_train.shape
        assert X_te_s.shape == X_test.shape

    def test_scaler_returned(self):
        from sklearn.preprocessing import StandardScaler
        df = preprocess(generate_sample_data(n_samples=100))
        X, y = split_features_target(df)
        X_train, X_test, _, _ = get_train_test_split(X, y, test_size=0.2)
        _, _, scaler = scale_features(X_train, X_test)
        assert isinstance(scaler, StandardScaler)
