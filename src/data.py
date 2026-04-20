"""
Data loading and preprocessing for house prices prediction.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def generate_sample_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic house prices dataset.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with house features and prices.
    """
    rng = np.random.default_rng(random_state)

    # House characteristics
    sqft_living = rng.integers(500, 5000, n_samples)
    sqft_lot = rng.integers(1000, 20000, n_samples)
    bedrooms = rng.integers(1, 7, n_samples)
    bathrooms = rng.integers(1, 5, n_samples)
    floors = rng.choice([1, 1.5, 2, 2.5, 3], n_samples)
    condition = rng.integers(1, 6, n_samples)
    grade = rng.integers(3, 13, n_samples)
    yr_built = rng.integers(1900, 2023, n_samples)
    yr_renovated = rng.choice(
        np.concatenate([[0] * 700, rng.integers(1970, 2023, 300)]),
        n_samples,
        replace=False if n_samples <= 1000 else True,
    )
    waterfront = rng.choice([0, 1], n_samples, p=[0.95, 0.05])
    view = rng.integers(0, 5, n_samples)
    zipcode = rng.choice(
        [98001, 98002, 98003, 98004, 98005, 98006, 98007, 98008, 98010, 98011],
        n_samples,
    )

    # Price formula with noise
    price = (
        sqft_living * 150
        + sqft_lot * 2
        + bedrooms * 10000
        + bathrooms * 15000
        + floors * 8000
        + condition * 5000
        + grade * 20000
        + (2023 - yr_built) * -200
        + waterfront * 200000
        + view * 10000
        + rng.normal(0, 30000, n_samples)
    )
    price = np.clip(price, 50000, 3000000)

    df = pd.DataFrame(
        {
            "sqft_living": sqft_living,
            "sqft_lot": sqft_lot,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "floors": floors,
            "condition": condition,
            "grade": grade,
            "yr_built": yr_built,
            "yr_renovated": yr_renovated,
            "waterfront": waterfront,
            "view": view,
            "zipcode": zipcode,
            "price": price.astype(int),
        }
    )
    return df


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load house price data from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    return pd.read_csv(filepath)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the house price DataFrame.

    Steps
    -----
    - Drop duplicate rows.
    - Fill missing numeric values with column medians.
    - Encode ``zipcode`` as a categorical label.
    - Add derived features: ``house_age``, ``was_renovated``,
      ``sqft_living_log``, ``price_per_sqft`` (when price is present).

    Parameters
    ----------
    df : pd.DataFrame
        Raw house price DataFrame.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame.
    """
    df = df.drop_duplicates().copy()

    # Fill missing numeric values with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Encode zipcode as integer label
    if "zipcode" in df.columns:
        le = LabelEncoder()
        df["zipcode"] = le.fit_transform(df["zipcode"].astype(str))

    # Derived features
    if "yr_built" in df.columns:
        df["house_age"] = 2023 - df["yr_built"]

    if "yr_renovated" in df.columns:
        df["was_renovated"] = (df["yr_renovated"] > 0).astype(int)

    if "sqft_living" in df.columns:
        df["sqft_living_log"] = np.log1p(df["sqft_living"])

    if "price" in df.columns and "sqft_living" in df.columns:
        df["price_per_sqft"] = df["price"] / df["sqft_living"].replace(0, np.nan)

    return df


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "price",
    drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into feature matrix X and target vector y.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame.
    target_col : str
        Name of the target column.
    drop_cols : list[str] or None
        Extra columns to drop from features.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (X, y) pair.
    """
    cols_to_drop = [target_col]
    if drop_cols:
        cols_to_drop.extend(drop_cols)
    # Drop derived target-leakage column if present
    if "price_per_sqft" in df.columns and "price_per_sqft" not in (drop_cols or []):
        cols_to_drop.append("price_per_sqft")
    X = df.drop(columns=cols_to_drop, errors="ignore")
    y = df[target_col]
    return X, y


def get_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and test sets.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    test_size : float
        Proportion of data to include in the test split.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Standardise features using a scaler fit on the training set.

    Column names are preserved in the returned DataFrames so that
    estimators relying on feature names (e.g. LightGBM) work correctly.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    X_test : pd.DataFrame
        Test feature matrix.

    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled, scaler
