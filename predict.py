"""
Command-line prediction script: predict house prices for new data.

Usage
-----
    python predict.py --model models/random_forest.pkl --input new_houses.csv

The input CSV must contain the same raw columns used during training.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.data import preprocess, split_features_target
from src.model import load_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict house prices with a saved model.")
    parser.add_argument("--model", required=True, help="Path to the saved .pkl model file.")
    parser.add_argument("--input", required=True, help="Path to the input CSV file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write predictions CSV. Defaults to <input>_predictions.csv.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not os.path.exists(args.model):
        sys.exit(f"Model file not found: {args.model}")
    if not os.path.exists(args.input):
        sys.exit(f"Input file not found: {args.input}")

    print(f"Loading model from {args.model} …")
    model = load_model(args.model)

    print(f"Loading input data from {args.input} …")
    df_raw = pd.read_csv(args.input)
    print(f"Input shape: {df_raw.shape}")

    # Preprocess (drop price column if present so it is not used as a feature)
    if "price" in df_raw.columns:
        df_raw = df_raw.drop(columns=["price"])

    df = preprocess(df_raw)

    # Drop derived price-related columns that are not in training data
    drop_cols = [c for c in ["price_per_sqft"] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")

    print(f"Predicting {len(X)} samples …")
    predictions = model.predict(X)
    predictions = np.clip(predictions, 0, None)  # prices cannot be negative

    output_df = df_raw.copy()
    output_df["predicted_price"] = predictions.astype(int)

    out_path = args.output or args.input.replace(".csv", "_predictions.csv")
    output_df.to_csv(out_path, index=False)
    print(f"Predictions saved to {out_path}")
    print(f"\nPrediction stats:")
    print(f"  Min:  ${predictions.min():>12,.0f}")
    print(f"  Max:  ${predictions.max():>12,.0f}")
    print(f"  Mean: ${predictions.mean():>12,.0f}")


if __name__ == "__main__":
    main()
