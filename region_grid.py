"""
Region grid lookup: maps (lat, lon) → REGION_GRID integer code.
Reads from data/region_grid_lookup.json and data/region_grid_encoder.json.
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


class RegionGrid:
    """Snaps a coordinate to a 0.01° grid cell and returns an integer region code."""

    GRID_STEP = 0.01

    def __init__(self,
                 lookup_path: Path = DATA_DIR / "region_grid_lookup.json",
                 encoder_path: Path = DATA_DIR / "region_grid_encoder.json"):
        with open(lookup_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # The JSON may have the grid nested under a "grid" key
        self._lookup: dict = raw.get("grid", raw)

        with open(encoder_path, "r", encoding="utf-8") as f:
            self._encoder: dict = json.load(f)

        print(f"✔ RegionGrid: {len(self._lookup)} cells, {len(self._encoder)} regions")

    def get_code(self, lat: float, lon: float) -> int:
        """
        Returns integer region code (0-35) for a coordinate.
        Returns -1 if the cell is not in the lookup.
        """
        step = self.GRID_STEP
        lat_grid = int(round(lat / step))
        lon_grid = int(round(lon / step))
        key = f"{lat_grid},{lon_grid}"
        region_name = self._lookup.get(key)
        if region_name is None or region_name == "Unknown":
            return -1
        return self._encoder.get(region_name, -1)

    def get_region_name(self, lat: float, lon: float) -> str:
        """Returns the human-readable region name for display."""
        step = self.GRID_STEP
        key = f"{int(round(lat/step))},{int(round(lon/step))}"
        return self._lookup.get(key, "Unknown")
