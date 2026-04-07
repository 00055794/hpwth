"""
FastAPI backend for KZ Real Estate Price Estimator.
Serves Jinja2 HTML with Yandex Maps and handles NN model predictions.

Architecture (flat modules at project root):
  feature_pipeline.py  – assembles all features (user inputs + REGION_GRID +
                          segment_code + stat + OSM distances)
  nn_inference.py      – loads nn_model/ artifacts and predicts price in KZT

The model receives only the features listed in nn_model/feature_list.json
(currently 13; upgrades to 47 after running scripts/save_artifacts.py).
"""
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import uvicorn

BASE_DIR   = Path(__file__).resolve().parent
YANDEX_KEY = os.getenv("YANDEX_MAPS_API_KEY", "")

# ── Global resources (loaded once at startup) ─────────────────────────────────
_pipeline = None
_nn       = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _nn

    from feature_pipeline import FeaturePipeline
    from nn_inference import NNInference

    print("Loading FeaturePipeline …")
    _pipeline = FeaturePipeline()

    print("Loading NNInference …")
    _nn = NNInference()

    print("✅  All resources loaded — app is ready")
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="KZ House Price Estimator", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ── Request schema ────────────────────────────────────────────────────────────
class PredictionInput(BaseModel):
    ROOMS:        int   = Field(..., ge=1,    le=20,   description="Number of rooms")
    LONGITUDE:    float = Field(..., ge=46.0, le=87.0, description="Property longitude (WGS-84)")
    LATITUDE:     float = Field(..., ge=40.0, le=55.0, description="Property latitude (WGS-84)")
    TOTAL_AREA:   float = Field(..., gt=0,    le=1000, description="Total area m²")
    FLOOR:        int   = Field(..., ge=1,    le=100,  description="Floor number")
    TOTAL_FLOORS: int   = Field(..., ge=1,    le=100,  description="Total floors in building")
    FURNITURE:    int   = Field(..., ge=1,    le=3,    description="1=No  2=Partial  3=Full")
    CONDITION:    int   = Field(..., ge=1,    le=5,    description="1=Poor … 5=Perfect")
    CEILING:      float = Field(..., ge=1.5,  le=10.0, description="Ceiling height in metres")
    MATERIAL:     int   = Field(..., ge=1,    le=4,    description="1=Panel 2=Brick 3=Monolith 4=Other")
    YEAR:         int   = Field(..., ge=1900, le=2030, description="Year built")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "yandex_key": YANDEX_KEY},
    )


@app.post("/predict")
async def predict(data: PredictionInput):
    if _pipeline is None or _nn is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet — retry in a moment.")

    try:
        user_input = data.model_dump()
        lat = data.LATITUDE
        lon = data.LONGITUDE

        # Assemble model-ready feature matrix
        features_df = _pipeline.assemble(user_input)

        # Extract derived codes for display
        region_grid_code = int(features_df["REGION_GRID"].iloc[0]) \
            if "REGION_GRID" in features_df.columns else -1
        segment_code_val = int(features_df["segment_code"].iloc[0]) \
            if "segment_code" in features_df.columns else -1

        # NN prediction → KZT
        price_kzt = float(_nn.predict_kzt(features_df)[0])

        # Display-only info (region name, distances, stat summary for UI)
        display = _pipeline.get_display_info(lat, lon)

        return {
            "success":       True,
            "price_kzt":     round(price_kzt, 0),
            "price_per_sqm": round(price_kzt / data.TOTAL_AREA, 0),
            "region_grid":   region_grid_code,
            "segment_code":  segment_code_val,
            "distances":     display["distances"],
            "stat":          display["stat"],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _nn is not None}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
