"""
FastAPI backend for KZ Real Estate Price Estimator.
Serves Jinja2 HTML with Yandex Maps and handles NN model predictions.

Architecture (flat modules at project root):
  feature_pipeline.py  - assembles all features (user inputs + REGION_GRID +
                          segment_code + stat + OSM distances)
  nn_inference.py      - loads nn_model/ artifacts and predicts price in KZT

The model receives only the features listed in nn_model/feature_list.json
(currently 13; upgrades to 47 after running scripts/save_artifacts.py).
"""
import io
import math
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import pandas as pd
import uvicorn

BASE_DIR   = Path(__file__).resolve().parent
YANDEX_KEY = os.getenv("YANDEX_MAPS_API_KEY", "")
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

# -- Global resources (loaded once at startup) ---------------------------------
_pipeline = None
_nn       = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _nn

    from feature_pipeline import FeaturePipeline
    from nn_inference import NNInference

    print("Loading FeaturePipeline ...")
    _pipeline = FeaturePipeline()

    print("Loading NNInference ...")
    _nn = NNInference()

    print("[OK]  All resources loaded -- app is ready")
    yield


# -- App -----------------------------------------------------------------------
app = FastAPI(title="KZ House Price Estimator", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# -- Request schema ------------------------------------------------------------
class PredictionInput(BaseModel):
    ROOMS:        int   = Field(..., ge=1,    le=20,   description="Number of rooms")
    LONGITUDE:    float = Field(..., ge=46.0, le=87.0, description="Property longitude (WGS-84)")
    LATITUDE:     float = Field(..., ge=40.0, le=55.0, description="Property latitude (WGS-84)")
    TOTAL_AREA:   float = Field(..., gt=0,    le=1000, description="Total area m2")
    FLOOR:        int   = Field(..., ge=1,    le=100,  description="Floor number")
    TOTAL_FLOORS: int   = Field(..., ge=1,    le=100,  description="Total floors in building")
    FURNITURE:    int   = Field(..., ge=1,    le=3,    description="1=No furniture  2=Partial  3=Full")
    CONDITION:    int   = Field(..., ge=1,    le=5,    description="1=Rough/Open plan  2=Needs renovation  3=Neat/Average  4=Good  5=Fresh renovation")
    MATERIAL:     int   = Field(..., ge=1,    le=4,    description="1=Other  2=Panel  3=Monolith  4=Brick")
    YEAR:         int   = Field(..., ge=1900, le=2030, description="Year built")


# -- Routes --------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"yandex_key": YANDEX_KEY},
    )


@app.post("/predict")
async def predict(data: PredictionInput):
    if _pipeline is None or _nn is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet -- retry in a moment.")

    try:
        user_input = data.model_dump()
        lat = data.LATITUDE
        lon = data.LONGITUDE

        # Assemble model-ready feature matrix
        features_df = _pipeline.assemble(user_input)

        # Extract derived codes for display
        region_grid_code = int(features_df["REGION"].iloc[0]) \
            if "REGION" in features_df.columns else -1
        segment_code_val = int(features_df["segment_code"].iloc[0]) \
            if "segment_code" in features_df.columns else -1

        # NN prediction -> price per sqm in 2025Q4 KZT
        price_per_sqm = float(_nn.predict_kzt(features_df)[0])
        price_kzt     = price_per_sqm * data.TOTAL_AREA

        # Display-only info (region name, distances, stat summary for UI)
        display = _pipeline.get_display_info(lat, lon)

        return {
            "success":       True,
            "price_kzt":     round(price_kzt, 0),
            "price_per_sqm": round(price_per_sqm, 0),
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


# -- Batch predict -------------------------------------------------------------
REQUIRED_COLS = ["ROOMS", "LATITUDE", "LONGITUDE", "TOTAL_AREA", "FLOOR",
                 "TOTAL_FLOORS", "FURNITURE", "CONDITION", "MATERIAL", "YEAR"]


@app.post("/batch")
async def batch_predict(file: UploadFile = File(...)):
    if _pipeline is None or _nn is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    try:
        contents = await file.read()
        if file.filename and file.filename.lower().endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            df = pd.read_csv(io.StringIO(contents.decode("utf-8", errors="replace")),
                             comment='#')
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot parse file: {exc}") from exc

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    df = df.head(200)  # safety cap
    results: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        try:
            user_input = {c: row[c] for c in REQUIRED_COLS}
            features_df = _pipeline.assemble(user_input)
            price_per_sqm = float(_nn.predict_kzt(features_df)[0])
            if not math.isfinite(price_per_sqm):
                continue
            price_kzt     = price_per_sqm * float(row["TOTAL_AREA"])
            rec: dict[str, Any] = {c: (None if pd.isna(row[c]) else
                                        int(row[c]) if isinstance(row[c], (int,)) else
                                        float(row[c]) if hasattr(row[c], '__float__') else row[c])
                                   for c in REQUIRED_COLS}
            rec["pred_price_per_sqm"] = round(price_per_sqm, 0)
            rec["pred_price_kzt"] = round(price_kzt, 0)
            results.append(rec)
        except Exception:
            pass
    return jsonable_encoder(results)


@app.post("/batch/download/xlsx")
async def batch_download_xlsx(rows: list[dict[str, Any]]):
    df  = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=predictions.xlsx"},
    )


@app.get("/template/csv")
async def template_csv():
    # Header + one annotated sample row so users know the expected encoding
    lines = [
        ",".join(REQUIRED_COLS),
        "2,43.2567,76.9286,65.0,5,12,3,5,3,2015",
        "# FURNITURE: 1=No furniture  2=Partial  3=Full",
        "# CONDITION: 1=Rough/Open plan  2=Needs renovation  3=Neat/Average  4=Good  5=Fresh renovation",
        "# MATERIAL:  1=Other  2=Panel  3=Monolith  4=Brick",
    ]
    return StreamingResponse(
        io.StringIO("\n".join(lines)),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=template.csv"},
    )


@app.get("/template/xlsx")
async def template_xlsx():
    sample = {
        "ROOMS": [2], "LATITUDE": [43.2567], "LONGITUDE": [76.9286],
        "TOTAL_AREA": [65.0], "FLOOR": [5], "TOTAL_FLOORS": [12],
        "FURNITURE": [3], "CONDITION": [5],
        "MATERIAL": [3], "YEAR": [2015],
    }
    notes = {
        "ROOMS": [""], "LATITUDE": [""], "LONGITUDE": [""],
        "TOTAL_AREA": [""], "FLOOR": [""], "TOTAL_FLOORS": [""],
        "FURNITURE": ["1=No furniture  2=Partial  3=Full"],
        "CONDITION": ["1=Rough  2=Needs reno  3=Neat  4=Good  5=Fresh reno"],
        "MATERIAL": ["1=Other  2=Panel  3=Monolith  4=Brick"],
        "YEAR": [""],
    }
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame(sample).to_excel(writer, sheet_name="Data", index=False)
        pd.DataFrame(notes).to_excel(writer, sheet_name="Notes", index=False)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=template.xlsx"},
    )


# -- Entry point ---------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
