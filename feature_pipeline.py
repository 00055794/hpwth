"""
Full Feature Pipeline
=====================
Assembles ALL features from 11 user inputs, then selects only the features
listed in nn_model/feature_list.json for the model.

Feature assembly order (up to 47 total):
  11  user inputs:   ROOMS, LONGITUDE, LATITUDE, TOTAL_AREA, FLOOR,
                     TOTAL_FLOORS, FURNITURE, CONDITION, CEILING, MATERIAL, YEAR
   1  derived:       REGION_GRID  ← region_grid.py
   1  derived:       segment_code ← spatial join with segments GeoJSON
  28+ stat features  ← stat_loader.py
   6  OSM distances  ← osm_distances.py

The model only receives features listed in feature_list.json in that exact order.
If the list has 13 features, only 13 are passed. If 47, all 47 are passed.
"""
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point

from region_grid import RegionGrid
from stat_loader import StatLoader
from osm_distances import OSMDistances

DATA_DIR  = Path(__file__).resolve().parent / "data"
MODEL_DIR = Path(__file__).resolve().parent / "nn_model"

# User inputs: the 11 features entered through the UI
USER_FEATURES = [
    "ROOMS", "LONGITUDE", "LATITUDE", "TOTAL_AREA",
    "FLOOR", "TOTAL_FLOORS", "FURNITURE", "CONDITION",
    "CEILING", "MATERIAL", "YEAR",
]


class FeaturePipeline:
    """
    Loads all lookup resources once at startup.
    Call assemble(input_dict) → pd.DataFrame with model-ready features.
    """

    def __init__(self):
        self.region_grid = RegionGrid()
        self.stat_loader = StatLoader()
        self.osm_distances = OSMDistances()

        # Load segments GeoJSON for spatial join
        seg_path = DATA_DIR / "segments_fine_heuristic_polygons.geojson"
        self.segments_gdf = gpd.read_file(seg_path)

        # Build segment encoder: sorted unique segment_ids → integer code
        # Load from saved map if available (preferred — matches notebook exactly)
        seg_map_path = DATA_DIR / "segment_code_map.json"
        if seg_map_path.exists():
            with open(seg_map_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._segment_encoder = {k: int(v) for k, v in raw.items()}
            print(f"✔ FeaturePipeline: segment_code_map loaded ({len(self._segment_encoder)} segments)")
        else:
            seg_ids = sorted(self.segments_gdf["segment_id"].dropna().unique())
            self._segment_encoder = {seg_id: code for code, seg_id in enumerate(seg_ids)}
            print(f"✔ FeaturePipeline: segment_encoder built on-the-fly ({len(self._segment_encoder)} segments)")

        # Load the feature order required by the model
        with open(MODEL_DIR / "feature_list.json", "r", encoding="utf-8") as f:
            self.feature_list: list = json.load(f)
        print(f"✔ FeaturePipeline: model uses {len(self.feature_list)} features")

    # ── Segment code lookup ───────────────────────────────────────────────────
    def _get_segment_code(self, lat: float, lon: float) -> int:
        point = Point(lon, lat)
        pt_gdf = gpd.GeoDataFrame([{"geometry": point}], crs="EPSG:4326")
        joined = gpd.sjoin(pt_gdf, self.segments_gdf[["segment_id", "geometry"]],
                           how="left", predicate="within")

        if len(joined) == 0 or pd.isna(joined["segment_id"].iloc[0]):
            # Fallback: nearest polygon centroid
            dists = self.segments_gdf.geometry.distance(point)
            seg_id = self.segments_gdf.loc[dists.idxmin(), "segment_id"]
        else:
            seg_id = joined["segment_id"].iloc[0]

        return self._segment_encoder.get(seg_id, -1)

    # ── Main assembly ─────────────────────────────────────────────────────────
    def assemble(self, user_input: dict) -> pd.DataFrame:
        """
        Build a pd.DataFrame with all computed features.
        The returned DataFrame's column order matches feature_list.json.

        Args:
            user_input: dict with the 11 UI keys (ROOMS, LONGITUDE, …)

        Returns:
            pd.DataFrame with one row and columns = feature_list
        """
        lat = float(user_input["LATITUDE"])
        lon = float(user_input["LONGITUDE"])

        # Start with 11 user features
        row: dict = {k: float(user_input[k]) for k in USER_FEATURES}

        # ── Derived: REGION_GRID & segment_code ──────────────────────────────
        row["REGION_GRID"]  = self.region_grid.get_code(lat, lon)
        row["segment_code"] = self._get_segment_code(lat, lon)

        # ── Statistical features (28+ columns from Stat Excel) ───────────────
        stat = self.stat_loader.model_features(lat, lon)
        row.update(stat)

        # ── OSM distances (6 features) ────────────────────────────────────────
        distances = self.osm_distances.get_distances(lat, lon)
        row.update(distances)

        # ── Build DataFrame with exactly the features the model expects ───────
        df = pd.DataFrame([row])

        # Add any missing model features as 0
        for feat in self.feature_list:
            if feat not in df.columns:
                df[feat] = 0

        # Return only model features in correct order
        return df[self.feature_list]

    # ── Extra info for UI display ─────────────────────────────────────────────
    def get_display_info(self, lat: float, lon: float) -> dict:
        """Returns display-only info (not fed to model): region name, stat summary, distances."""
        return {
            "region_name": self.region_grid.get_region_name(lat, lon),
            "distances":   self.osm_distances.get_distances(lat, lon),
            "stat":        self.stat_loader.display_features(lat, lon),
        }
