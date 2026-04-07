"""
Neural Network inference.
Loads model artifacts from nn_model/ and predicts house price in KZT.

Architecture: Linear(input_dim→64) → ReLU → Linear(64→16) → ReLU → Linear(16→1)
Target was trained on PRICE_ln = ln(price_kzt), so final output = exp(inverse_scaler(model(X)))
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import load

MODEL_DIR = Path(__file__).resolve().parent / "nn_model"


class HousePriceNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 16),        nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NNInference:
    """Encapsulates the loaded model + scalers for single-call inference."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        with open(model_dir / "feature_list.json", "r", encoding="utf-8") as f:
            self.feature_list: list = json.load(f)

        self.scaler_X = load(model_dir / "scaler_X.joblib")
        self.scaler_y = load(model_dir / "scaler_y.joblib")

        self.model = HousePriceNN(input_dim=len(self.feature_list))
        state = torch.load(model_dir / "model.pt", map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

        with open(model_dir / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata: dict = json.load(f)

        print(f"✔ NNInference: {len(self.feature_list)} features, "
              f"MAPE={self.metadata.get('MAPE', '?')}%")

    def predict_kzt(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict house price in KZT.

        Args:
            features_df: DataFrame with columns = feature_list (in any order;
                         will be reordered internally).

        Returns:
            np.ndarray of price predictions in KZT.
        """
        # Ensure correct column order and fill any missing with 0
        df = features_df.copy()
        for col in self.feature_list:
            if col not in df.columns:
                df[col] = 0.0
        X = df[self.feature_list].astype(float).values

        Xs = self.scaler_X.transform(X)
        with torch.no_grad():
            y_scaled = self.model(torch.tensor(Xs, dtype=torch.float32)).numpy()

        y_ln = self.scaler_y.inverse_transform(y_scaled).flatten()
        return np.exp(y_ln)
