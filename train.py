"""
Command-line entry point: train and evaluate all house-price models.

Usage
-----
    python train.py [--data PATH] [--output-dir DIR] [--no-plots]

If ``--data`` is omitted the script generates a synthetic dataset.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(__file__))

from src.data import (
    generate_sample_data,
    get_train_test_split,
    load_data,
    preprocess,
    scale_features,
    split_features_target,
)
from src.model import cross_validate, evaluate, get_models, save_model
from src.visualize import (
    plot_feature_importance,
    plot_model_comparison,
    plot_predictions_vs_actual,
    plot_price_distribution,
    plot_residuals,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train house price prediction models.")
    parser.add_argument(
        "--data",
        default=None,
        help="Path to a CSV file with house data. If omitted, synthetic data is used.",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to save trained models and plots (default: models).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if args.data:
        print(f"Loading data from {args.data} …")
        df_raw = load_data(args.data)
    else:
        print("No data file specified – generating synthetic dataset (1 000 samples).")
        df_raw = generate_sample_data(n_samples=1000)

    print(f"Dataset shape: {df_raw.shape}")

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    df = preprocess(df_raw)
    X, y = split_features_target(df)
    print(f"Features: {list(X.columns)}")
    print(f"Target (price) – min: {y.min():,.0f}  max: {y.max():,.0f}  "
          f"mean: {y.mean():,.0f}")

    # ------------------------------------------------------------------
    # 3. Train / test split + scaling
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # ------------------------------------------------------------------
    # 4. Price distribution plot
    # ------------------------------------------------------------------
    if not args.no_plots:
        plot_price_distribution(
            y,
            output_path=os.path.join(args.output_dir, "price_distribution.png"),
        )
        print("Saved price_distribution.png")

    # ------------------------------------------------------------------
    # 5. Train, evaluate and optionally cross-validate each model
    # ------------------------------------------------------------------
    models = get_models()
    results: dict[str, dict[str, float]] = {}
    feature_names = list(X.columns)

    for name, model in models.items():
        print(f"\n{'─' * 55}")
        print(f"  Training {name} …")

        model.fit(X_train_scaled, y_train)
        metrics = evaluate(model, X_test_scaled, y_test)
        results[name] = metrics

        print(
            f"  MAE:  {metrics['mae']:>12,.2f}  |  "
            f"RMSE: {metrics['rmse']:>12,.2f}  |  "
            f"R²:   {metrics['r2']:.4f}  |  "
            f"MAPE: {metrics['mape']:.2f}%"
        )

        if args.cv > 1:
            cv_metrics = cross_validate(model, X_train_scaled, y_train, cv=args.cv)
            print(
                f"  CV R² (mean ± std): "
                f"{cv_metrics['cv_r2_mean']:.4f} ± {cv_metrics['cv_r2_std']:.4f}"
            )

        # Save model
        model_path = os.path.join(args.output_dir, f"{name}.pkl")
        save_model(model, model_path)
        print(f"  Saved to {model_path}")

        # Plots
        if not args.no_plots:
            y_pred = model.predict(X_test_scaled)
            plot_predictions_vs_actual(
                y_test,
                y_pred,
                model_name=name,
                output_path=os.path.join(args.output_dir, f"{name}_pred_vs_actual.png"),
            )
            plot_residuals(
                y_test,
                y_pred,
                model_name=name,
                output_path=os.path.join(args.output_dir, f"{name}_residuals.png"),
            )
            plot_feature_importance(
                model,
                feature_names=feature_names,
                output_path=os.path.join(
                    args.output_dir, f"{name}_feature_importance.png"
                ),
            )

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print(f"\n{'═' * 55}")
    print("  Model Comparison Summary")
    print(f"{'─' * 55}")
    summary = pd.DataFrame(results).T[["mae", "rmse", "r2", "mape"]]
    summary.index.name = "model"
    print(summary.to_string(float_format=lambda x: f"{x:,.4f}"))

    best_model = summary["r2"].idxmax()
    print(f"\n  ✓ Best model by R²: {best_model}  (R² = {summary.loc[best_model, 'r2']:.4f})")

    if not args.no_plots:
        plot_model_comparison(
            results,
            metric="r2",
            output_path=os.path.join(args.output_dir, "model_comparison_r2.png"),
        )
        plot_model_comparison(
            results,
            metric="rmse",
            output_path=os.path.join(args.output_dir, "model_comparison_rmse.png"),
        )
        print(f"\nAll plots saved to '{args.output_dir}/'.")

    # Save summary CSV
    summary_path = os.path.join(args.output_dir, "results_summary.csv")
    summary.to_csv(summary_path)
    print(f"Results summary saved to {summary_path}")


if __name__ == "__main__":
    main()
