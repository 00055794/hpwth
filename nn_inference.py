"""
Ensemble inference: LightGBM (86%) + Neural Network (14%) + Ridge meta-learner.

Pipeline:
  scaler_X.transform(X) -> LGB .predict()  -> lgb_log
                        -> NN  .forward()  -> scaler_y.inverse_transform() -> nn_log
  ridge_meta.predict([[lgb_log, nn_log]]) -> ensemble_log
  np.exp(ensemble_log) x price_index_current -> nominal KZT/m2 (current quarter)

Target was trained as log(asking_price_real) where:
  asking_price_real = price_per_sqm / price_index[listing_quarter]   (base 2025Q4=1.0)
Back-transform to nominal: np.exp(pred) x price_index_current   (Cell 184 methodology)

NN architecture (matches training notebook, old run saved to model.pt):
  Linear(N->64) -> ReLU -> Linear(64->16) -> ReLU -> Linear(16->1)   [N = len(feature_list.json)]
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
    """Architecture from training notebook -- must match model.pt state_dict keys."""
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
    """LightGBM + NN + Ridge ensemble inference."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        with open(model_dir / "feature_list.json", "r", encoding="utf-8") as f:
            self.feature_list: list = json.load(f)

        # Shared input scaler (QuantileTransformer)
        self.scaler_X = load(model_dir / "scaler_X.joblib")
        # Output scaler for NN branch (StandardScaler on log-price; v2 replaced MinMaxScaler)
        self.scaler_y = load(model_dir / "scaler_y.joblib")

        # Neural Network
        self.nn_model = HousePriceNN(input_dim=len(self.feature_list))
        state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
        # Handle state_dict packed under "net." prefix or bare
        try:
            self.nn_model.load_state_dict(state)
        except RuntimeError:
            # Try remapping if saved with named layers
            remapped = {}
            layer_map = {"fc1": "net.0", "fc2": "net.2", "fc3": "net.4"}
            for k, v in state.items():
                for src, dst in layer_map.items():
                    if k.startswith(src):
                        k = k.replace(src, dst, 1)
                        break
                remapped[k] = v
            self.nn_model.load_state_dict(remapped)
        self.nn_model.eval()

        # LightGBM booster
        lgb_path = model_dir / "lgb_model.txt"
        if lgb_path.exists():
            import lightgbm as lgb
            self.lgb_model = lgb.Booster(model_file=str(lgb_path))
            print(f"LGB model loaded from {lgb_path.name}")
        else:
            self.lgb_model = None
            print("lgb_model.txt not found -- NN-only mode")

        # Ridge meta-learner
        ridge_path = model_dir / "ridge_meta.joblib"
        if ridge_path.exists():
            self.ridge_meta = load(ridge_path)
            print(f"Ridge meta loaded coef={self.ridge_meta.coef_}")
        else:
            self.ridge_meta = None
            print("ridge_meta.joblib not found -- NN-only mode")

        try:
            with open(model_dir / "metadata.json", "r", encoding="utf-8") as f:
                self.metadata: dict = json.load(f)
        except Exception:
            self.metadata = {"stat_feature_medians": {}}
            print("metadata.json unreadable -- using empty stat medians")

        mape = self.metadata.get("test_mape_pct", self.metadata.get("MAPE", "?"))
        print(f"Ensemble ready: {len(self.feature_list)} features, test MAPE={mape}%")

    def _prepare_X(self, features_df: pd.DataFrame) -> np.ndarray:
        df = features_df.copy()
        for col in self.feature_list:
            if col not in df.columns:
                df[col] = float(self.metadata.get("stat_feature_medians", {}).get(col, 0.0))
        stat_medians = self.metadata.get("stat_feature_medians", {})
        for col in stat_medians:
            if col in df.columns:
                df[col] = df[col].fillna(float(stat_medians[col]))
        return df[self.feature_list].astype(float).fillna(0.0).values

    def predict_kzt(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Returns nominal price per sqm in current-quarter KZT.
        Step 1: model output -> log(asking_price_real) in 2025Q4 KZT
        Step 2: multiply by price_index_current -> nominal KZT/m2 for today
        (Cell 184: "For nominal display: np.exp(pred) x price_index[current_quarter]")
        """
        X_raw = self._prepare_X(features_df)
        X_sc  = self.scaler_X.transform(X_raw)

        # NN branch: scaled output -> inverse scaler -> log-price
        with torch.no_grad():
            y_nn_sc = self.nn_model(
                torch.tensor(X_sc, dtype=torch.float32)).numpy()
        nn_log = self.scaler_y.inverse_transform(y_nn_sc.reshape(-1, 1)).flatten()

        # LGB branch: log-price directly
        if self.lgb_model is not None:
            lgb_log = self.lgb_model.predict(X_sc)
        else:
            lgb_log = nn_log.copy()

        # Ridge ensemble
        if self.ridge_meta is not None:
            stack   = np.column_stack([lgb_log, nn_log])
            ens_log = self.ridge_meta.predict(stack)
        else:
            ens_log = lgb_log

        # Convert real (2025Q4) price to nominal current price
        # price_index_current is the BMN price level at current date vs 2025Q4 base
        if "price_index_current" in features_df.columns:
            price_index = float(features_df["price_index_current"].iloc[0])
        else:
            price_index = 1.0

        return np.exp(ens_log) * price_index
