import os
import sys
import xarray as xr
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.weather import MarineEnvironmentLoader, resolve_marine_dataset_paths
from src.grid.cost_map import build_cost_map
from src.grid.no_go_zone import BUSAN_SHANGHAI_BOUNDS

def _remap_cmems_time_to_era5(cmems_path: str, era5_path: str):
    era5_ds = xr.open_dataset(era5_path)
    cmems_ds = xr.open_dataset(cmems_path)

    era5_time = "valid_time" if "valid_time" in era5_ds.coords else "time"
    cmems_time = "time" if "time" in cmems_ds.coords else "valid_time"

    era5_t0 = era5_ds[era5_time].values[0]
    cmems_t0 = cmems_ds[cmems_time].values[0]
    shift = era5_t0 - cmems_t0

    new_time = cmems_ds[cmems_time].values + shift
    cmems_ds = cmems_ds.assign_coords({cmems_time: new_time})

    era5_ds.close()
    return cmems_ds

def main():
    era5_path = os.path.join(PROJECT_ROOT, "data", "era5", "era5_marine_20250325T00_20250326T00.nc")
    _, cmems_path = resolve_marine_dataset_paths(PROJECT_ROOT)

    era5_ds = xr.open_dataset(era5_path)
    cmems_ds = _remap_cmems_time_to_era5(cmems_path, era5_path)

    loader = MarineEnvironmentLoader.from_datasets(era5_ds, cmems_ds)
    env_fn = loader.get_environment_fn()
    when_utc = datetime(2025, 3, 25, 6, 0, tzinfo=timezone.utc)

    # Resolution 0.05 is sufficient for a good look, or 0.01 for high quality
    # We will use 0.05 to make it reasonably fast while maintaining look
    # Wait, in previous step we used 0.01 because 0.05 land mask cache didn't exist
    # Let's use 0.01 to safely hit the cache and avoid geopandas
    cost_map = build_cost_map(resolution=0.01)

    fig, ax = plt.subplots(figsize=(16, 8.5))
    bounds = cost_map.bounds
    ax.set_aspect("equal")
    ax.set_xlim(bounds["lon_min"], bounds["lon_max"])
    ax.set_ylim(bounds["lat_min"], bounds["lat_max"])
    
    # Sea facecolor
    ax.set_facecolor("#dfeaf7")
    
    lon_grid, lat_grid = np.meshgrid(cost_map.lons, cost_map.lats)
    
    # Compute on a slightly coarser grid to save time if needed, but 0.01 is fine (601x1001)
    # To mimic wind_route_map, we use wind_speed in m/s, not knots for the background color!
    # "wind_speed" from env_loader is in m/s.
    
    # Downsample the calculation of wind speed for pcolormesh to save time 
    # since 0.01 is very dense, a 0.1 grid is usually enough for the pcolormesh background
    res_factor = 10 # 0.01 * 10 = 0.1 degree
    coarse_lons = cost_map.lons[::res_factor]
    coarse_lats = cost_map.lats[::res_factor]
    c_lon_grid, c_lat_grid = np.meshgrid(coarse_lons, coarse_lats)
    
    c_wind_speed = np.zeros_like(c_lon_grid, dtype=float)
    c_u_grid = np.zeros_like(c_lon_grid, dtype=float)
    c_v_grid = np.zeros_like(c_lon_grid, dtype=float)
    
    print("Computing wind speeds for environment map...")
    for i in range(len(coarse_lats)):
        for j in range(len(coarse_lons)):
            lat, lon = coarse_lats[i], coarse_lons[j]
            value = env_fn(lat, lon, when_utc)
            ws_ms = float(value.wind_speed_ms)
            wd_deg = float(value.wind_dir_deg or 0.0)
            
            theta = np.radians(wd_deg)
            c_u_grid[i, j] = -ws_ms * np.sin(theta)
            c_v_grid[i, j] = -ws_ms * np.cos(theta)
            c_wind_speed[i, j] = ws_ms
            
    # 1. Pcolormesh for wind speed (YlGnBu, alpha=0.45)
    mesh = ax.pcolormesh(
        c_lon_grid,
        c_lat_grid,
        c_wind_speed,
        cmap="YlGnBu",
        shading="auto",
        alpha=0.45,
        zorder=1,
    )
    
    # 2. Overlay land
    # cost_grid is 0 (sea) and 100 (land)
    extent = [bounds["lon_min"], bounds["lon_max"], bounds["lat_min"], bounds["lat_max"]]
    land_overlay = np.zeros((cost_map.cost_grid.shape[0], cost_map.cost_grid.shape[1], 4))
    land_mask = cost_map.cost_grid >= 50
    # #f5efe2 for land
    land_overlay[land_mask] = [245/255, 239/255, 226/255, 1.0] 
    land_overlay[~land_mask] = [0, 0, 0, 0.0]
    
    ax.imshow(land_overlay, extent=extent, origin="lower", aspect="equal", zorder=2)
    
    # 3. Coastlines (#8d8574, linewidth=0.45)
    X, Y = np.meshgrid(cost_map.lons, cost_map.lats)
    ax.contour(
        X,
        Y,
        cost_map.cost_grid,
        levels=[50],
        colors="#8d8574",
        linewidths=0.45,
        alpha=1.0,
        zorder=3
    )

    # 4. Wind barbs (#3f3f3f, length=5.2, linewidth=0.5)
    # The original script computes barb_stride = max(1, int(np.ceil(max(n_lat, n_lon) / 30)))
    # For a 0.1 degree grid, n_lat=60, n_lon=100. stride = ceil(100/30) = 4
    barb_stride = max(1, int(np.ceil(max(len(coarse_lats), len(coarse_lons)) / 30)))
    ax.barbs(
        c_lon_grid[::barb_stride, ::barb_stride],
        c_lat_grid[::barb_stride, ::barb_stride],
        c_u_grid[::barb_stride, ::barb_stride],
        c_v_grid[::barb_stride, ::barb_stride],
        length=5.2,
        linewidth=0.5,
        color="#3f3f3f",
        zorder=4,
    )

    # Gridlines
    ax.grid(True, linewidth=0.35, color="#8b93a0", alpha=0.35, linestyle="--", zorder=0)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("ERA5 2025-03-25T06:00 Wind Environment Map", fontsize=14, fontweight="bold")

    save_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "era5_0325T06_env_map.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {path}")

if __name__ == "__main__":
    main()
