"""
PHASE 0 — STEP 0.1: Save All Artifacts from the Notebook
=========================================================
Run this script INSIDE the Jupyter notebook (after training) to save:
  - nn_model/model.pt
  - nn_model/scaler_X.joblib
  - nn_model/scaler_y.joblib
  - nn_model/feature_list.json
  - nn_model/cat_mappings.json
  - nn_model/metadata.json
  - data/segment_code_map.json

USAGE — run this cell at the end of your training notebook:
    %run scripts/save_artifacts.py
  OR paste the code directly into a notebook cell.

REQUIRED notebook variables:
    model          — trained HousePriceNN instance (torch.nn.Module)
    scaler_X       — fitted MinMaxScaler for X
    scaler_y       — fitted MinMaxScaler for y (ln price)
    X_train        — pd.DataFrame with training features (column order = truth)
    mape, mae, mse, rmse, r2  — evaluation metrics (floats)
    segment_encoder — dict {segment_id: int_code}  (from LabelEncoder.fit)

ACCURACY CHECKLIST:
  1. feature_list.json order = X_train.columns  ← guaranteed here
  2. scaler_X fitted only on X_train            ← your responsibility
  3. scaler_y fitted on PRICE_ln (ln price)     ← your responsibility
  4. REGION_GRID uses region_grid_encoder.json  ← same file in data/
  5. segment_code uses segment_encoder dict     ← saved as segment_code_map.json
  6. Stat features match training col names     ← from X_train.columns
  7. OSM distances in km (not metres)           ← precompute_distances.py
  8. Distance grid step = GRID_STEP = 0.01      ← precompute_distances.py
  9. NN arch = Input→64→ReLU→16→ReLU→1          ← model definition
 10. Inference: exp(scaler_y.inverse(model(scaler_X.transform(X))))
"""

import json
import torch
import numpy as np
from pathlib import Path
from joblib import dump

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "nn_model"
DATA_DIR  = BASE_DIR / "data"

MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# ── 1. Feature list (CRITICAL: must match X_train.columns order) ──────────────
feature_list = list(X_train.columns)
print(f"Features ({len(feature_list)}): {feature_list}")

# Sanity check expected count: 11 user inputs + REGION_GRID + segment_code
# + N stat features + 6 OSM distances = 47 total (or 13 without stat/osm)
expected_min = 13
if len(feature_list) < expected_min:
    print(f"⚠️  WARNING: only {len(feature_list)} features found (expected ≥{expected_min})")
    print("   Make sure stat features and OSM distances are in X_train!")

# ── 2. Save model weights ──────────────────────────────────────────────────────
torch.save(model.state_dict(), MODEL_DIR / "model.pt")
print(f"✅ Saved model.pt  (input_dim={len(feature_list)})")

# ── 3. Save scalers ───────────────────────────────────────────────────────────
dump(scaler_X, MODEL_DIR / "scaler_X.joblib")
dump(scaler_y, MODEL_DIR / "scaler_y.joblib")
print("✅ Saved scaler_X.joblib, scaler_y.joblib")

# ── 4. Save feature list ──────────────────────────────────────────────────────
with open(MODEL_DIR / "feature_list.json", "w", encoding="utf-8") as f:
    json.dump(feature_list, f, indent=2)
print("✅ Saved feature_list.json")

# ── 5. Save categorical mappings ─────────────────────────────────────────────
# cat_mappings stores any label-encoded mappings used before training
# If you used numeric codes directly (1,2,3), this can be empty {}
cat_maps = {}
# Add FURNITURE, CONDITION, MATERIAL mappings if they were string-encoded
# Example: cat_maps["FURNITURE"] = {"No": 1, "Partial": 2, "Full": 3}
with open(MODEL_DIR / "cat_mappings.json", "w", encoding="utf-8") as f:
    json.dump(cat_maps, f, indent=2)
print("✅ Saved cat_mappings.json")

# ── 6. Save training metrics ──────────────────────────────────────────────────
metadata = {
    "input_dim":    len(feature_list),
    "architecture": "Linear(input→64)→ReLU→Linear(64→16)→ReLU→Linear(16→1)",
    "target":       "PRICE_ln (log of price in KZT)",
    "MAPE":         float(mape),
    "MAE":          float(mae),
    "MSE":          float(mse),
    "RMSE":         float(rmse),
    "R2":           float(r2),
    "n_features":   len(feature_list),
}
with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(f"✅ Saved metadata.json  (MAPE={mape:.4f}%, R²={r2:.4f})")

# ── 7. Save segment code map ──────────────────────────────────────────────────
# segment_encoder must be a dict: {segment_id_value: int_code}
# In the notebook this was built as:
#   segment_ids = sorted(segments_gdf['segment_id'].dropna().unique())
#   segment_encoder = {seg_id: code for code, seg_id in enumerate(segment_ids)}
if 'segment_encoder' in dir():
    seg_map = {str(k): int(v) for k, v in segment_encoder.items()}
    with open(DATA_DIR / "segment_code_map.json", "w", encoding="utf-8") as f:
        json.dump(seg_map, f, indent=2)
    print(f"✅ Saved segment_code_map.json  ({len(seg_map)} segments)")
else:
    print("⚠️  segment_encoder not found in notebook scope — skipping segment_code_map.json")
    print("   Spatial join at runtime will compute segment_code on-the-fly.")

print("\n" + "="*60)
print("DONE — artifacts saved. Next steps:")
print(f"  1. git add nn_model/ data/segment_code_map.json")
print(f"  2. Run: python scripts/precompute_distances.py")
print(f"  3. git add data/distance_grid.parquet")
print(f"  4. git commit && git push")
print("="*60)
