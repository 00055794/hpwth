# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# ── System dependencies for geospatial libraries ──────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgdal-dev \
        gdal-bin \
        libspatialindex-dev \
        libgeos-dev \
        libproj-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY main.py             ./
COPY feature_pipeline.py ./
COPY nn_inference.py     ./
COPY osm_distances.py    ./
COPY region_grid.py      ./
COPY stat_loader.py      ./
COPY templates/          ./templates/
COPY static/             ./static/

# ── Model artefacts & data files (no raw .shp files — distances pre-computed) ─
COPY nn_model/ ./nn_model/
COPY data/     ./data/

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 8000

# Health-check so Docker knows when the app is accepting requests
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
