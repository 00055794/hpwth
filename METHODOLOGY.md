# Kazakhstan Apartment Price Prediction — Full Methodology

---

## 0. Model Results (Production)

**Accuracy (test set, 46,380 samples, listing date >= 2025-07-01):**

| Metric | Result |
|--------|--------|
| MAPE | 7.10% |
| MdAPE | 3.79% |
| Within ±10% | 77.4% |
| Within ±20% | 91.6% |
| Within ±30% | 97.1% |

**Architecture:** LightGBM (86.4%) + Neural Network (14.1%) stacked with Ridge meta-learner.

---

## 1. Key Challenges Unique to Ad Data

| # | Challenge | Approach |
|---|-----------|----------|
| 1.1 | **Asking vs. Transaction Price** | Train on asking price directly as target |
| 1.2 | **Temporal Drift** — 14-year span | Temporal feature engineering + time-aware CV |
| 1.3 | **Duplicate Ads** | Deduplication pipeline: hash on features + address |
| 1.4 | **Selection Bias** | Awareness of OOS limits; Heckman correction optional |
| 1.5 | **Stale Listings** | Use `listing_date` + `days_on_market` features |
| 1.6 | **Geocoding Errors** | Spatial validation + outlier filtering |

---

## 2. Target Variable

```python
target = log(asking_price_real)    # natural log, NOT log1p
```

All historical prices deflated to base period 2025-Q4 using the BMN repeat-sales price index:
```python
df['asking_price_real'] = df['asking_price_per_sqm'] * (price_index['2025Q4'] / df['year_quarter'].map(price_index))
target = np.log(df['asking_price_real'])
```

**Back-transform at inference:**
```python
price_per_sqm_kzt = np.exp(ensemble_prediction)   # NOT expm1
total_price       = price_per_sqm_kzt * TOTAL_AREA
```

Hybrid Hedonic + Repeat-Sales:  
`ln(P_it) = α_t + Σ β_k X_kit + γ_i + ε_it`

| Term | Meaning |
|------|---------|
| α_t | Time fixed effect = price index |
| X_kit | Attribute k of property i at time t |
| γ_i | Building fixed effect — permanent quality premium |
| ε_it | Idiosyncratic error |

---

## 3. Complete Feature Set (50 features in production)

### 3.1 User Input Features (11 collected, 9 used by model)

| Feature | Description |
|---------|-------------|
| ROOMS | Number of rooms | No (dropped by SHAP) |
| LONGITUDE | Longitude coordinate | Yes |
| LATITUDE | Latitude coordinate | Yes |
| TOTAL AREA | Apartment area sqm (note: space not underscore) | Yes |
| FLOOR | Apartment floor | No (dropped by SHAP) |
| TOTAL_FLOORS | Total floors in building | Yes |
| FURNITURE | 1=none, 2=partial, 3=full | Yes |
| CONDITION | 1=rough, 2=needs reno, 3=decent, 4=good, 5=excellent | Yes |
| CEILING | Ceiling height, m | Yes |
| MATERIAL | 1=other, 2=panel, 3=monolith, 4=brick | Yes |
| YEAR | Year of construction | Yes |

### 3.2 Derived Geographic Features (4)

| Feature | Description | Source |
|---------|-------------|--------|
| `REGION` | Integer region code (0-35) from lat/lon grid | `region_grid.py` + `data/region_grid_encoder.json` |
| `segment_code` | Fine-grained market segment integer | spatial join with `data/segments_fine_heuristic_polygons.geojson` |
| `is_almaty` | 1 if property is in Almaty, else 0 | derived from region name |
| `is_astana` | 1 if property is in Astana/Nur-Sultan, else 0 | derived from region name |

**City encoder build recipe (one-time):**
```python
from sklearn.neighbors import KNeighborsClassifier
import json

city_centroids = df.groupby('CITY')[['LATITUDE','LONGITUDE']].mean().reset_index()
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(city_centroids[['LATITUDE','LONGITUDE']], city_centroids['CITY'])
city_codes = {city: i for i, city in enumerate(sorted(city_centroids['CITY'].unique()))}
with open('data/city_encoder.json', 'w') as f:
    json.dump(city_codes, f, ensure_ascii=False)

# At inference:
def get_city_code(lat, lon, knn, city_codes):
    city = knn.predict([[lat, lon]])[0]
    return city_codes.get(city, -1)
```

### 3.3 Statistical / Construction Features (25) — `data/Stat_withConstruction_KZ092025.xlsx`

Nearest region found via `scipy.spatial.cKDTree` on lat/lon centroids. 25 features covering labour, demographics, CPI, and construction supply.

| Feature | Description |
|---------|-------------|
| `srednmes_zarplata` | Average monthly salary (₸) |
| `index_real_zarplaty` | Real wage index |
| `chislennost_naseleniya_092025` | Population (Sep 2025) |
| `prirost_naselenya` | Population growth |
| `temp_prirosta_percent` | Population growth rate % |
| `index_potreb_cen_tovary_uslugi` | CPI — goods & services |
| `index_potreb_cen_prodovolstv_tovary` | CPI — food |
| `index_potreb_cen_neprodovolstv_tovary` | CPI — non-food |
| `index_potreb_cen_platnye_uslugi` | CPI — paid services |
| `obsh_ploshad_expluat_new_buildings_vsego` | Total new floor area commissioned (sqm) |
| `obsh_ploshad_expluat_new_buildings_zhilye` | New residential floor area (sqm) |
| `obsh_ploshad_expluat_new_buildings_shkoly` | New school floor area (sqm) |
| `fakt_stoimost_str_va_vsego` | Total actual construction cost (mln ₸) |
| `fakt_stoimost_str_va_zhilye` | Residential construction cost (mln ₸) |
| `fakt_stoimost_str_va_shkoly` | School construction cost (mln ₸) |
| `fakt_stoimost_str_va_chastnoi_sobstv_ti_zastroishikami` | Private developer construction cost (mln ₸) |
| `obsh_ploshad_vveden_expluat_zhilyh_zdaniy_sqmeters` | Residential buildings commissioned (sqm) |
| `obsh_ploshad_expluat_new_buildings_zhilye_individualnye` | Individual housing commissioned (sqm) |
| `obsh_ploshad_expluat_new_buildings_zhilye_mnogokvartirnye` | Multi-apartment housing commissioned (sqm) |
| `poleznaya_ploshad_zhilyh_zdaniy_vvedeno` | Useful area commissioned (sqm) |
| `poleznaya_ploshad_zhilyh_zdaniy_vvedeno_zastroishikami` | Useful area by developers |
| `k_vo_vvedennyh_kvartir_vsego` | New apartments commissioned (total) |
| `sred_fakt_zatraty_str_vo_na_sqmeters_zhilyhdomov_vsego` | Avg construction cost per sqm (₸) |

### 3.4 Price Index Features (2) — `nn_model/price_index.json`

Computed from the BMN repeat-sales quarterly index.

| Feature | Description |
|---------|-------------|
| `price_index_current` | BMN index value of the latest quarter |
| `index_momentum_3m` | Quarter-over-quarter index change (momentum) |

### 3.5 Building Panel Features (5) — from training history

| Feature | Inference default | Description |
|---------|------------------|-------------|
| `building_last_known_price` | 0 | Last known asking price/sqm for this building |
| `months_since_building_last_listed` | 0 | Months since any apartment in this building was listed |
| `building_listing_count_historical` | 1 | Historical listing count for this building |
| `building_appreciation_rate_annualized` | 0 | Annualized price appreciation for this building |
| `building_price_volatility` | 0 | Standard deviation of past prices |

> Unknown buildings at inference time receive cold-start defaults. Known buildings are in `nn_model/building_fe_lookup.json`.

### 3.6 Macro / Time-Series Features — `data/macro2014_2024.xlsx` (not yet integrated)

44 quarters (2014–2024) × 477 columns. Join to training data by `year_quarter`.  
53 base indicators, each also available as lags 1–4, YoY ratio, absolute diff.  
**Use SHAP to select the most informative subset (see Section 4).**

| Group | Feature (Russian) | Description |
|-------|-------------------|-------------|
| **GDP** | Реальный ВВП | Real GDP |
| **Sectoral GVA** | ...Строительство | Construction GVA |
| | ...Операции с недвижимым имуществом | Real estate operations GVA |
| | ...Финансовая и страховая деятельность | Financial sector GVA |
| | ...Обрабатывающая промышленность | Manufacturing GVA |
| | ...Транспорт и складирование | Transport & logistics GVA |
| | ...Информация и связь | ICT GVA |
| **Deflators** | Дефлятор ВВП | GDP deflator |
| **Inflation** | Инфляция | Overall CPI |
| | Индекс цен на жилую недвижимость на первичном рынке | Primary RE price index KZ |
| | Индекс цен на жилую недвижимость на вторичном рынке | Secondary RE price index KZ |
| **Income** | Реальная среднемесячная заработная плата | Real avg monthly wage |
| | Реальные среднемесячные денежные доходы населения | Real avg income per capita |
| | Реальные среднемесячные денежные расходы населения | Real avg spending per capita |
| **Trade** | Импорт товаров | Goods imports |
| | Экспорт товаров | Goods exports |
| **Interest rates** | Средневзвешенная ставка по ипотечному кредитованию в KZT более года | Avg mortgage rate >1yr (KZT) |
| | TONIA | Interbank overnight rate |
| | Доходность ГЦБ в KZT, срочность 3 месяца / 1 год / 7 лет / 10 лет | Gov bond yields 3m/1y/7y/10y |
| | Ставка по федеральным фондам США | US Federal Funds Rate |
| | Ставки по кредитам нефинансовым организациям KZT/СКВ до / более года | Corporate lending rates |
| | Ставки по необеспеченным кредитам физлицам KZT до / более года | Unsecured consumer lending rates |
| **FX** | Обменный курс тенге к доллару США | KZT/USD |
| | Обменный курс тенге к евро | KZT/EUR |
| | Обменный курс тенге к российскому рублю | KZT/RUB |
| **Commodities** | Цена на нефть Brent | Brent crude price |
| **Russia spillover** | Индекс цен на жилую недвижимость на первичном рынке в РФ | Russia primary RE price index |
| | Индекс цен на жилую недвижимость на вторичном рынке в РФ | Russia secondary RE price index |

### 3.7 OSM Proximity Features (4 used in model) — `osm_shp/`

Pre-computed via `scripts/precompute_distances.py` → `data/distance_grid.parquet` (grid step = 0.01°). O(1) lookup at inference.

| Feature | OSM source | OSM fclass | In model |
|---------|-----------|------------|----------|
| `dist_to_pharmacy_km` | `gis_osm_pois_free_1` | `pharmacy` | Yes |
| `dist_to_hospital_km` | `gis_osm_pois_free_1` | `hospital` | Yes |
| `dist_to_kindergarten_km` | `gis_osm_pois_free_1` | `kindergarten` | Yes |
| `dist_to_main_road_km` | `gis_osm_roads_free_1` | `motorway`, `trunk`, `primary` | Yes |
| `dist_to_school_km` | `gis_osm_pois_free_1` | `school` | No (dropped by SHAP) |
| `dist_to_healthcare_km` | `gis_osm_pois_free_1` | `hospital`, `clinic` | No (dropped by SHAP) |

Stats (588,257 training samples): school median 0.34 km · pharmacy 0.34 km · main road 0.08 km  
Available but unused (evaluate via SHAP before adding): `buildings`, `railways`, `transport_a`, `waterways`, `water_a`, `natural`, `landuse`, `pofw`, `traffic`

---

## 4. Feature Selection — SHAP

Run SHAP **separately** for LightGBM and NN. Drop any feature with mean |SHAP| < 0.001 in **both** models.

### 4.1 LightGBM — TreeExplainer
```python
import shap
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_val)
shap_importance = pd.Series(
    np.abs(shap_values).mean(axis=0), index=X_val.columns
).sort_values(ascending=False)
shap.summary_plot(shap_values, X_val, max_display=30)
shap.summary_plot(shap_values, X_val, plot_type='bar', max_display=30)
```

### 4.2 NN — KernelExplainer + Permutation Importance
```python
# Option A: Permutation importance (faster)
from sklearn.inspection import permutation_importance
perm = permutation_importance(nn_wrapper, X_val, y_val, n_repeats=10, random_state=42)
perm_importance = pd.Series(perm.importances_mean, index=X_val.columns).sort_values(ascending=False)

# Option B: SHAP KernelExplainer (exact)
background = shap.kmeans(X_train, 50)
explainer_nn = shap.KernelExplainer(nn_predict_fn, background)
shap_values_nn = explainer_nn.shap_values(X_val.iloc[:200])
shap_importance_nn = pd.Series(
    np.abs(shap_values_nn).mean(axis=0), index=X_val.columns
).sort_values(ascending=False)
shap.summary_plot(shap_values_nn, X_val.iloc[:200], max_display=30)
```

### 4.3 Final Selection
```python
comparison = pd.DataFrame({'LGB': shap_importance, 'NN': shap_importance_nn}).fillna(0)
comparison['avg'] = comparison.mean(axis=1)
comparison = comparison.sort_values('avg', ascending=False)
print(comparison.head(30))

# Keep features important in at least one model
final_features = comparison[
    (comparison['LGB'] > 0.001) | (comparison['NN'] > 0.001)
].index.tolist()
print(f"Final feature count: {len(final_features)}")
# Update nn_model/feature_list.json with final_features
```

---

## 5. Data Pipeline

**Source:** `DDS.SARG_APARTMENTS_FL` via EDW Oracle (`edw_conn`)

1. Deduplicate: `ROW_NUMBER OVER (PARTITION BY UUID ORDER BY PARSED_AT DESC)`, keep RN=1
2. Clean columns: `AREA_WITH_KITCHEN` str→float (strip "м²"), `FLOOR` "5 из 9"→int pair, `CEILING` "3 м"→float
3. Build repeat-listings price index 2012–2026 (BMN regression) → quarterly index values
4. Deflate all asking prices to base period 2025-Q4 → `asking_price_real`
5. Compute building fixed effects (panel regression on buildings with ≥3 observations)
6. Run `scripts/precompute_distances.py` once → `data/distance_grid.parquet`
7. Spatial join with `segments_fine_heuristic_polygons.geojson` → `segment_code`
8. `region_grid.py` → `REGION` integer; build `city_encoder.py` → `CITY` integer
9. Nearest-region stat lookup via `cKDTree` → 23 stat columns
10. Macro quarterly join by `year_quarter` → selected macro features
11. Assign temporal sample weights: half-life=365 days; 2022 shock (Feb–Sep) ×0.4; pre-2018 ×0.1
12. Run SHAP feature selection → update `nn_model/feature_list.json`
13. Temporal split (Section 6)
14. Train NN + LightGBM → Ridge ensemble

---

## 6. Temporal Split & Cross-Validation

```
2012──────────────────────────────────────────────────►2026

┌──────────────────┬──────────────────┬──────────┬──────────┐
│ HISTORICAL PANEL │ RECENT TRAIN     │   VAL    │   TEST   │
│ 2012–2021        │ 2022–2024-06     │ 6mo gap  │ 6 months │
│ → Building FE    │ → Price surface  │ Tune     │ Final    │
│ → Price index    │ → Macro regime   │ hyperpars│ once     │
│ (low weight)     │ (full weight)    │          │          │
└──────────────────┴──────────────────┴──────────┴──────────┘
```

- **Test:** `>= 2025-10-01` (untouched until final eval)
- **Val:** `2025-04-01 – 2025-10-01`
- **Gap:** `2024-10-01 – 2025-04-01` (excluded)
- **Recent train:** `2022-01-01 – 2024-10-01`
- **Historical:** `< 2022-01-01` (low weight)

**Sample weights:** `w(t) = exp(-ln2/365 * (T_now - t))`  
**CV:** `TimeSeriesSplit(n_splits=5, gap=3*30)` — 3-month gap per fold

---

## 7. Model Architecture

### 7.1 Neural Network (14.1% ensemble weight)

```
Input (50) → Linear(50→64) → ReLU → Linear(64→16) → ReLU → Linear(16→1)
```

- Loss: `HuberLoss(delta=0.1)` · Optimizer: `AdamW(lr=1e-3, weight_decay=1e-4)`
- Scheduler: `CosineAnnealingWarmRestarts(T_0=10, T_mult=2)`
- Epochs: 200, patience=15, batch=512
- Input scaler: `QuantileTransformer(n_quantiles=1000, output_distribution='normal')`
- Output scaler: `MinMaxScaler` on log-price
- Inference: `exp(scaler_y.inverse_transform(model(scaler_X.transform(X))))`

### 7.2 LightGBM (86.4% ensemble weight)

```python
lgb_params = {
    'objective': 'huber', 'alpha': 0.9,
    'num_leaves': 127, 'max_depth': 8,
    'min_child_samples': 50,
    'learning_rate': 0.05, 'n_estimators': 3000,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 0.1, 'reg_lambda': 1.0,
    'monotone_constraints': [...],   # +1 for ROOMS/AREA/CONDITION, -1 for distances
    'monotone_constraints_method': 'advanced',
}
```

**Monotonic constraints (+1):** ROOMS, TOTAL AREA, CONDITION, CEILING, TOTAL_FLOORS, FLOOR  
**Monotonic constraints (-1):** dist_to_school_km, dist_to_hospital_km, dist_to_pharmacy_km, dist_to_main_road_km, dist_to_kindergarten_km

### 7.3 Ridge Meta-Learner (stacking)

```
LGB OOF predictions  ─┐
                       ├→ Ridge(alpha=1.0).fit → ensemble log-price
NN  OOF predictions  ─┘
```

**Fitted coefficients (production):**

| Model | Weight |
|-------|--------|
| LightGBM | 0.8638 |
| Neural Network | 0.1408 |
| Intercept | -0.0598 |

**Full inference:**
```python
X_sc      = scaler_X.transform(X[feature_list])          # QuantileTransformer
y_lgb     = lgb_model.predict(X_sc)                      # log-price
y_nn_sc   = nn_model(torch.tensor(X_sc))                 # scaled
y_nn      = scaler_y.inverse_transform(y_nn_sc)          # log-price
y_ens_log = ridge_meta.predict([[y_lgb, y_nn]])           # log-price
price_kzt = np.exp(y_ens_log) * TOTAL_AREA               # 2025Q4 real KZT
```

---

## 8. Artifact Inventory

**`nn_model/` (model artifacts):**

| File | Purpose |
|------|--------|
| `model.pt` | NN weights (Linear 50→64→16→1) |
| `lgb_model.txt` | LightGBM booster |
| `ridge_meta.joblib` | Ridge stacking meta-learner |
| `scaler_X.joblib` | QuantileTransformer — input features (fitted on X_train only) |
| `scaler_y.joblib` | MinMaxScaler — NN output (fitted on log-price train) |
| `feature_list.json` | 50 feature names in exact training column order |
| `cat_mappings.json` | REGION encoding by name |
| `price_index.json` | BMN quarterly index (2012Q4 to latest) |
| `building_fe_lookup.json` | Known building fixed effects |
| `metadata.json` | Full model params, ensemble coefs, test metrics |

**`data/` (lookup files):**

| File | Purpose |
|------|--------|
| `region_grid_lookup.json` | lat/lon grid (0.01°) → region name |
| `region_grid_encoder.json` | region name → integer code |
| `segment_code_map.json` | segment_id → integer code |
| `segments_fine_heuristic_polygons.geojson` | market segment polygons |
| `distance_grid.parquet` | Pre-computed OSM distances (~10MB) |
| `Stat_withConstruction_KZ092025.xlsx` | 25 stat features by region |

---

## 9. Evaluation

**Always apply `np.expm1()` before computing metrics — log-scale MAPE (~0.84%) is not meaningful.**

| Metric | Target |
|--------|--------|
| MAPE | < 8% |
| MdAPE (robust) | < 6% |
| Within 5% | > 40% |
| Within 10% | > 70% |
| Within 20% | > 90% |

Inspect top 5% largest residuals: luxury flats, 2022 shock listings, specific districts, new construction.

---

## 10. Retraining Schedule

| Trigger | Action |
|---------|--------|
| Weekly | Retrain Ridge meta-learner weights only |
| Monthly | Full LightGBM on sliding 24-month window |
| Quarterly | NN fine-tune on last 12 months |
| Central bank rate change > 1% | Immediate full retrain |
| KS drift test p-value < 0.05 | Alert + retrain |

---

## 11. Benchmarks

| Model | MAPE |
|-------|------|
| Naive (district median) | 18–25% |
| Linear hedonic | 12–18% |
| Random Forest | 8–12% |
| Single LightGBM | 6–9% |
| Single NN | 7–10% |
| **Ensemble + temporal weights** | **3–6%** |
