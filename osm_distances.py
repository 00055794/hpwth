"""
OSM distance lookup from pre-computed distance_grid.parquet.

Replaces the slow on-the-fly KDTree over raw .shp files.
At startup: loads data/distance_grid.parquet and builds a fast dict lookup.
At query time: snaps (lat, lon) to nearest 0.01° grid cell → O(1) lookup.

Run scripts/precompute_distances.py ONCE locally to generate the parquet file.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR  = Path(__file__).resolve().parent / "data"
GRID_STEP = 0.01

DISTANCE_COLS = [
    "dist_to_school_km",
    "dist_to_kindergarten_km",
    "dist_to_hospital_km",
    "dist_to_healthcare_km",
    "dist_to_pharmacy_km",
    "dist_to_main_road_km",
]


class OSMDistances:
    """
    Fast O(1) distance lookup using a pre-computed grid parquet file.
    Falls back to NaN for coordinates outside the grid.
    """

    def __init__(self, parquet_path: Path = DATA_DIR / "distance_grid.parquet"):
        if not Path(parquet_path).exists():
            print(f"⚠️  {parquet_path} not found — distances will return NaN")
            self._grid: dict = {}
            self._available = False
            return

        df = pd.read_parquet(parquet_path)
        # Build dict: (lat_key, lon_key) → [d1, d2, d3, d4, d5, d6]
        # lat_key / lon_key are integers = round(coord / GRID_STEP)
        arr = df[DISTANCE_COLS].values.astype(np.float32)
        lat_keys = np.round(df["lat_grid"].values / GRID_STEP).astype(np.int32)
        lon_keys = np.round(df["lon_grid"].values / GRID_STEP).astype(np.int32)
        self._grid = {(int(la), int(lo)): arr[i] for i, (la, lo) in enumerate(zip(lat_keys, lon_keys))}
        self._available = True
        print(f"✔ OSMDistances: {len(self._grid):,} grid cells loaded from {parquet_path.name}")

    def get_distances(self, lat: float, lon: float) -> dict:
        """
        Returns dict of 6 distance features in km.
        Falls back gracefully if parquet is unavailable or coords out of range.
        """
        if not self._available:
            return {col: None for col in DISTANCE_COLS}

        lat_key = int(round(lat / GRID_STEP))
        lon_key = int(round(lon / GRID_STEP))
        vals = self._grid.get((lat_key, lon_key))

        if vals is None:
            # Try the 8 surrounding cells and take the nearest non-None
            for dlat in (-1, 0, 1):
                for dlon in (-1, 0, 1):
                    if dlat == 0 and dlon == 0:
                        continue
                    vals = self._grid.get((lat_key + dlat, lon_key + dlon))
                    if vals is not None:
                        break
                if vals is not None:
                    break

        if vals is None:
            return {col: None for col in DISTANCE_COLS}

        return {col: round(float(vals[i]), 3) for i, col in enumerate(DISTANCE_COLS)}
