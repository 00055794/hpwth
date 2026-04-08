# Kazakhstan House Price Prediction App

A machine learning web application for predicting house prices in Kazakhstan using neural networks.

## Features
- **Single Property Prediction**: Interactive map-based location picker with geocoder search
- **Batch Prediction**: Upload CSV files to predict multiple properties at once
- **High Accuracy**: MAPE 0.44% on log scale, R² 0.97

## Model Performance
- Training R²: 0.90
- Test R²: 0.97
- Log MAPE: 0.44%
- KZT MAPE: ~8%

## Tech Stack
- PyTorch neural network (64→16→1 architecture)
- FastAPI + Uvicorn backend
- Leaflet.js interactive maps (ESRI Satellite)
- GeoPandas for spatial features
- Docker for containerized deployment
- 588,281 training samples with 13 features

## Running with Docker
```bash
docker-compose up --build
```
App will be available at `http://localhost:8000`

## Data Sources
- Region Grid: 4473 spatial cells for location encoding
- Segments: 2031 fine-grained polygons for neighborhood classification
- Training data from Kazakhstan real estate market (2024-2025)
