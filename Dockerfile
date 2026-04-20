# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Build-time proxy (for corporate networks) ─────────────────────────────────
# Pass via: docker-compose build --build-arg HTTP_PROXY=http://user:pass@host:port
# Or set HTTP_PROXY / HTTPS_PROXY in your .env file.
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY=localhost,127.0.0.1

ENV http_proxy=${HTTP_PROXY} \
    https_proxy=${HTTPS_PROXY} \
    HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    no_proxy=${NO_PROXY}

# Write apt proxy config only when a proxy is supplied
RUN if [ -n "$HTTP_PROXY" ]; then \
        echo "Acquire::http::Proxy \"${HTTP_PROXY}\";"  > /etc/apt/apt.conf.d/01proxy && \
        echo "Acquire::https::Proxy \"${HTTPS_PROXY}\";" >> /etc/apt/apt.conf.d/01proxy; \
    fi

WORKDIR /app

# ── System dependencies for geospatial libraries ──────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
        libgdal-dev \
        gdal-bin \
        libspatialindex-dev \
        libgeos-dev \
        libproj-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --trusted-host pypi.org \
        --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org \
        --trusted-host download.pytorch.org \
        -r requirements.txt

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
