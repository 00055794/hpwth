"""
Statistical feature loader for KZ real estate.

Primary lookup (used at inference):
  data/region_stat_lookup.json -- pre-computed per-region averages that exactly
  replicate the training notebook's spatial join logic:
    stat centroids -> OSM region polygons (averaged) -> RegionGrid regions
  Call: model_features(lat, lon, region_name=<RegionGrid name>)

Fallback (if region_name not found):
  Nearest-centroid cKDTree on data/Stat_withConstruction_KZ092025.xlsx

Provides:
  - model_features(): dict of all numeric stat columns (for model input)
  - display_features(): dict with human-readable labels (for UI display)
"""
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# Columns shown in the UI (label -> display name)
_DISPLAY_LABELS = {
    "region":                                            "Region",
    "srednmes_zarplata":                                 "Avg monthly salary (₸)",
    "index_real_zarplaty":                               "Real wage index",
    "chislennost_naseleniya_092025":                     "Population (Sep 2025)",
    "prirost_naselenya":                                 "Population growth",
    "temp_prirosta_percent":                             "Growth rate %",
    "index_potreb_cen_tovary_uslugi":                    "CPI - goods & services",
    "index_potreb_cen_prodovolstv_tovary":               "CPI - food",
    "index_potreb_cen_neprodovolstv_tovary":             "CPI - non-food",
    "index_potreb_cen_platnye_uslugi":                   "CPI - paid services",
    "obsh_ploshad_expluat_new_buildings_zhilye":         "New residential floor area (sqm)",
    "fakt_stoimost_str_va_zhilye":                       "Residential construction cost (mln ₸)",
    "sred_fakt_zatraty_str_vo_na_sqmeters_zhilyhdomov_vsego": "Construction cost per sqm (₸)",
    "k_vo_vvedennyh_kvartir_vsego":                      "New apartments (total)",
    "k_vo_vvedennyh_novyh_zhilyh_zdaniy_vsego":          "New residential buildings",
}

# Columns to exclude when building the model feature dict
_EXCLUDE = {"region", "latitude", "longitude", "lat", "lon"}


class StatLoader:
    """
    Loads the Stat Excel once at startup.
    Primary: region_name -> pre-computed OSM-polygon averaged stat values.
    Fallback: O(log n) nearest-centroid lookup via cKDTree.
    """

    def __init__(
        self,
        excel_path: Path = DATA_DIR / "Stat_withConstruction_KZ092025.xlsx",
        lookup_path: Path = DATA_DIR / "region_stat_lookup.json",
    ):
        df = pd.read_excel(excel_path)

        # Normalise column names: lowercase + strip spaces
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Coerce all non-text columns to numeric
        for col in df.columns:
            if col not in ("region",):
                df[col] = pd.to_numeric(df[col], errors="coerce")

        self._df = df.reset_index(drop=True)

        # Identify lat/lon columns (try several spellings)
        lat_col = next((c for c in df.columns if c in ("latitude", "lat")), None)
        lon_col = next((c for c in df.columns if c in ("longitude", "lon")), None)
        if lat_col is None or lon_col is None:
            raise ValueError("Stat Excel must have 'latitude' and 'longitude' columns")

        coords = np.vstack([df[lon_col].values, df[lat_col].values]).T
        self._tree = cKDTree(coords)
        self._lat_col = lat_col
        self._lon_col = lon_col

        # All numeric feature column names (for model input)
        self._feature_cols = [
            c for c in df.columns
            if c.lower() not in _EXCLUDE and pd.api.types.is_numeric_dtype(df[c])
        ]

        # Load pre-computed region -> stat lookup (training-consistent values)
        self._region_lookup: dict = {}
        if lookup_path.exists():
            raw = json.loads(lookup_path.read_text(encoding="utf-8"))
            # Convert to {region_name: {feat: float}} -- filter to model feature cols
            feat_set = set(self._feature_cols)
            for rname, vals in raw.items():
                self._region_lookup[rname] = {
                    k: v for k, v in vals.items() if k in feat_set
                }
            print(f"[OK] StatLoader: {len(df)} centroids, {len(self._feature_cols)} feature cols, "
                  f"{len(self._region_lookup)} region-stat entries loaded")
        else:
            print(f"[WARN] StatLoader: region_stat_lookup.json not found -- using centroid fallback only")
            print(f"[OK] StatLoader: {len(df)} regions, {len(self._feature_cols)} model feature cols")

    def _nearest_row(self, lat: float, lon: float) -> pd.Series:
        _, idx = self._tree.query([[lon, lat]])
        return self._df.iloc[int(idx[0])]

    def model_features(self, lat: float, lon: float, region_name: str | None = None) -> dict:
        """
        Returns all numeric stat features keyed by their exact column names.
        If region_name is given and found in region_stat_lookup, uses the
        pre-computed OSM-polygon averaged values (training-consistent).
        Otherwise falls back to nearest centroid.
        """
        if region_name and region_name in self._region_lookup:
            vals = self._region_lookup[region_name]
            # Return all feature cols; missing ones become None
            return {col: vals.get(col) for col in self._feature_cols}

        # Fallback: nearest centroid
        row = self._nearest_row(lat, lon)
        return {col: (None if pd.isna(row[col]) else float(row[col]))
                for col in self._feature_cols}

    def display_features(self, lat: float, lon: float, region_name: str | None = None) -> dict:
        """
        Returns a human-readable dict for UI display only.
        """
        if region_name and region_name in self._region_lookup:
            vals = self._region_lookup[region_name]
            result = {}
            for col, label in _DISPLAY_LABELS.items():
                if col == "region":
                    result[label] = region_name
                    continue
                v = vals.get(col)
                if v is None:
                    continue
                result[label] = round(float(v), 2)
            return result

        row = self._nearest_row(lat, lon)
        result = {}
        for col, label in _DISPLAY_LABELS.items():
            if col not in row.index:
                continue
            val = row[col]
            if pd.isna(val):
                continue
            if col == "region":
                result[label] = str(val)
            elif isinstance(val, (int, np.integer)):
                result[label] = int(val)
            else:
                result[label] = round(float(val), 2)
        return result

