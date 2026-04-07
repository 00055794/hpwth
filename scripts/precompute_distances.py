"""
PHASE 0 — LOCAL ONE-TIME SETUP
================================
Pre-compute OSM distance grid from raw .shp files.

Run ONCE locally (not in Docker) before pushing to GitHub:
    python scripts/precompute_distances.py

Input  : osm_shp/  directory with Kazakhstan OSM shapefiles
Output : data/distance_grid.parquet  (~10MB, grid_step=0.01 degrees)

The parquet contains columns:
    lat_grid, lon_grid,
    dist_to_school_km, dist_to_kindergarten_km, dist_to_hospital_km,
    dist_to_healthcare_km, dist_to_pharmacy_km, dist_to_main_road_km
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
OSM_DIR   = BASE_DIR / "osm_shp"
OUT_FILE  = BASE_DIR / "data" / "distance_grid.parquet"
GRID_STEP = 0.01                          # degrees (~1.1 km per cell)
# 1 degree ≈ 111 km; at 48 °N longitude is cos(48°) shorter
DEG_TO_KM = 111.0 * np.cos(np.radians(48.0))   # ≈ 74.3 km/deg

MAIN_ROAD_FCLASSES = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "motorway_link", "trunk_link", "primary_link",
    "secondary_link", "tertiary_link",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _to_point_array(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Extract (lon, lat) from any geometry (centroid for non-points)."""
    pts = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        c = geom.centroid if geom.geom_type != "Point" else geom
        pts.append((c.x, c.y))
    return np.asarray(pts, dtype=np.float64) if pts else np.empty((0, 2), dtype=np.float64)


def _road_vertices(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Flatten all road coordinates into (lon, lat) array for nearest-edge query."""
    pts = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            pts.extend(geom.coords)
        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                pts.extend(part.coords)
    return np.asarray(pts, dtype=np.float64) if pts else np.empty((0, 2), dtype=np.float64)


def _safe_tree(arr: np.ndarray) -> cKDTree:
    if arr.shape[0] == 0:
        arr = np.array([[76.9286, 43.2567]])   # Almaty fallback
    return cKDTree(arr)


# ── Load OSM data ─────────────────────────────────────────────────────────────
def load_osm(osm_dir: Path):
    print("Loading OSM shapefiles …")
    pois   = gpd.read_file(osm_dir / "gis_osm_pois_free_1.shp").to_crs("EPSG:4326")
    pois_a = gpd.read_file(osm_dir / "gis_osm_pois_a_free_1.shp").to_crs("EPSG:4326")
    roads  = gpd.read_file(osm_dir / "gis_osm_roads_free_1.shp").to_crs("EPSG:4326")
    print(f"  POIs: {len(pois)}, POI areas: {len(pois_a)}, Roads: {len(roads)}")

    def _merge(*fclasses):
        fc = set(fclasses)
        return pd.concat(
            [pois[pois["fclass"].isin(fc)], pois_a[pois_a["fclass"].isin(fc)]],
            ignore_index=True
        )

    categories = {
        "dist_to_school_km":       _safe_tree(_to_point_array(_merge("school"))),
        "dist_to_kindergarten_km": _safe_tree(_to_point_array(_merge("kindergarten"))),
        "dist_to_hospital_km":     _safe_tree(_to_point_array(_merge("hospital"))),
        "dist_to_healthcare_km":   _safe_tree(_to_point_array(_merge("hospital", "doctors", "clinic"))),
        "dist_to_pharmacy_km":     _safe_tree(_to_point_array(pois[pois["fclass"] == "pharmacy"])),
        "dist_to_main_road_km":    _safe_tree(_road_vertices(roads[roads["fclass"].isin(MAIN_ROAD_FCLASSES)])),
    }
    for name, tree in categories.items():
        print(f"  {name}: {tree.n} nodes in KDTree")
    return categories, pois


# ── Build grid ────────────────────────────────────────────────────────────────
def build_grid(pois_gdf: gpd.GeoDataFrame, step: float = GRID_STEP):
    """
    Create a grid covering the bounding box of all POI data + buffer.
    Using the POI bbox avoids generating millions of empty cells over desert.
    """
    # Get bounding box of all POI data
    bounds = pois_gdf.total_bounds  # [minx, miny, maxx, maxy] = [lon_min, lat_min, lon_max, lat_max]
    lon_min, lat_min, lon_max, lat_max = bounds
    # Add 0.5° buffer around the POI extent
    buf = 0.5
    lon_min -= buf; lat_min -= buf
    lon_max += buf; lat_max += buf

    # Snap to grid
    lon_min = round(lon_min / step) * step
    lat_min = round(lat_min / step) * step

    lat_vals = np.arange(lat_min, lat_max + step, step)
    lon_vals = np.arange(lon_min, lon_max + step, step)

    lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()
    print(f"Grid: {len(lat_vals)} lat × {len(lon_vals)} lon = {len(lat_flat):,} cells")
    return lat_flat, lon_flat


# ── Compute distances for all grid cells ──────────────────────────────────────
def compute_grid_distances(lat_flat, lon_flat, trees):
    n = len(lat_flat)
    query_pts = np.column_stack([lon_flat, lat_flat])   # KDTree indexed as (lon, lat)
    print(f"Computing distances for {n:,} grid cells …")

    result = {"lat_grid": lat_flat.astype(np.float32),
              "lon_grid": lon_flat.astype(np.float32)}

    for name, tree in trees.items():
        dists_deg, _ = tree.query(query_pts)
        result[name] = (dists_deg * DEG_TO_KM).astype(np.float32)
        print(f"  {name}: min={result[name].min():.2f} km, max={result[name].max():.2f} km")

    return pd.DataFrame(result)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not OSM_DIR.exists():
        print(f"ERROR: OSM directory not found: {OSM_DIR}")
        print("  Make sure the osm_shp/ folder is present locally.")
        return

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    trees, pois_gdf = load_osm(OSM_DIR)
    lat_flat, lon_flat = build_grid(pois_gdf)
    df = compute_grid_distances(lat_flat, lon_flat, trees)

    # Round grid coordinates to GRID_STEP precision for exact lookup
    df["lat_grid"] = (df["lat_grid"] / GRID_STEP).round().astype(np.int32) * GRID_STEP
    df["lon_grid"] = (df["lon_grid"] / GRID_STEP).round().astype(np.int32) * GRID_STEP

    df.to_parquet(OUT_FILE, index=False, compression="snappy")
    size_mb = OUT_FILE.stat().st_size / 1_048_576
    print(f"\n✅ Saved: {OUT_FILE}  ({size_mb:.1f} MB, {len(df):,} rows)")
    print("   Now commit data/distance_grid.parquet to git and push.")


if __name__ == "__main__":
    main()
