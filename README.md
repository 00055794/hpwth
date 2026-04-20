# hpwth – House Prices Prediction

A Python machine-learning project that trains several regression models to
predict house prices.  It ships with a synthetic data generator so you can
run everything out of the box, or plug in your own CSV dataset.

---

## Project structure

```
hpwth/
├── src/
│   ├── data.py        # Data loading, preprocessing, train/test split
│   ├── model.py       # Model definitions, evaluation, save/load helpers
│   └── visualize.py   # Plotting utilities
├── tests/
│   ├── test_data.py   # Unit tests for src/data.py
│   └── test_model.py  # Unit tests for src/model.py
├── train.py           # CLI: train all models and save results
├── predict.py         # CLI: run predictions with a saved model
├── requirements.txt
└── README.md
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train models

```bash
# Use built-in synthetic data
python train.py

# Use your own CSV dataset (must contain a `price` column)
python train.py --data data/houses.csv
```

Trained models are saved in `models/` along with evaluation plots and a
`results_summary.csv`.

Available CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--data PATH` | *(synthetic)* | Path to a CSV file |
| `--output-dir DIR` | `models` | Where to save models & plots |
| `--no-plots` | `False` | Skip generating plots |
| `--cv N` | `5` | Cross-validation folds |

### 3. Predict on new data

```bash
python predict.py --model models/random_forest.pkl --input new_houses.csv
```

Predictions are appended as a `predicted_price` column and written to
`new_houses_predictions.csv` (or the path you specify with `--output`).

---

## Models

| Model | Library |
|-------|---------|
| Linear Regression | scikit-learn |
| Ridge Regression | scikit-learn |
| Lasso Regression | scikit-learn |
| Random Forest | scikit-learn |
| Gradient Boosting | scikit-learn |
| XGBoost | xgboost *(optional)* |
| LightGBM | lightgbm *(optional)* |

---

## Dataset features

The synthetic generator (and the expected CSV schema) uses these columns:

| Column | Description |
|--------|-------------|
| `sqft_living` | Interior living area (sq ft) |
| `sqft_lot` | Lot area (sq ft) |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `floors` | Number of floors |
| `condition` | Overall condition (1–5) |
| `grade` | Construction grade (3–12) |
| `yr_built` | Year built |
| `yr_renovated` | Year renovated (0 = never) |
| `waterfront` | Waterfront property (0/1) |
| `view` | View quality (0–4) |
| `zipcode` | ZIP code |
| `price` | **Target** – sale price ($) |

---

## Running tests

```bash
python -m pytest tests/ -v
```
