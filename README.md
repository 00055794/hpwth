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
- Streamlit for web interface
- GeoPandas for spatial features
- 588,281 training samples with 13 features

## Deployment
Deployed on Streamlit Cloud: [Your App URL]

## Local Development
```bash
pip install -r app/requirements.txt
cd app
streamlit run app_new.py
```

## Data Sources
- Region Grid: 4473 spatial cells for location encoding
- Segments: 2031 fine-grained polygons for neighborhood classification
- Training data from Kazakhstan real estate market (2024-2025)
