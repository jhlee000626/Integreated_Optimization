"""
환경 맵 3-Panel 시각화
========================
동일 위경도 격자(125~130°E, 32~36°N) 위에:
  Panel 1: ERA5 바람 (풍속 컬러 + 풍향 화살표)
  Panel 2: CMEMS 해류 (유속 컬러 + 유향 화살표)
  Panel 3: GSHHG 고해상도 해안선 + 수심/육지 마스크

실행:
    python scripts/visualize_environment.py
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 프로젝트 루트 경로 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ─── 경로 설정 ───
ERA5_PATH = os.path.join(PROJECT_ROOT, "data", "era5", "era5_wind_2024_01.nc")
CMEMS_PATH = os.path.join(PROJECT_ROOT, "data", "cmems", "cmems_current.nc")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# ─── 공통 공간 범위 ───
LON_MIN, LON_MAX = 125.0, 130.0
LAT_MIN, LAT_MAX = 32.0, 36.0

# ─── GSHHG 해안선 로드 ───
from src.grid.no_go_zone import load_coastline, BUSAN_PORT, JEJU_PORT

BOUNDS = {
    "lat_min": LAT_MIN, "lat_max": LAT_MAX,
    "lon_min": LON_MIN, "lon_max": LON_MAX,
}


def load_era5():
    """ERA5 바람 데이터 로드 → (lats, lons, u10, v10, wind_speed)"""
    import xarray as xr
    ds = xr.open_dataset(ERA5_PATH)

    ds_sel = ds
    if 'expver' in ds_sel.dims:
        ds_sel = ds_sel.isel(expver=0)
    for dim in ['time', 'valid_time']:
        if dim in ds_sel.dims:
            ds_sel = ds_sel.isel({dim: 0})
            break

    u10 = ds_sel['u10'].values.squeeze()
    v10 = ds_sel['v10'].values.squeeze()
    lats = ds_sel['latitude'].values if 'latitude' in ds_sel.coords else ds_sel['lat'].values
    lons = ds_sel['longitude'].values if 'longitude' in ds_sel.coords else ds_sel['lon'].values

    # 바운딩 박스 클리핑
    lat_mask = (lats >= LAT_MIN) & (lats <= LAT_MAX)
    lon_mask = (lons >= LON_MIN) & (lons <= LON_MAX)
    lats = lats[lat_mask]
    lons = lons[lon_mask]
    u10 = u10[np.ix_(lat_mask, lon_mask)]
    v10 = v10[np.ix_(lat_mask, lon_mask)]

    wind_speed = np.sqrt(u10**2 + v10**2)
    ds.close()
    return lats, lons, u10, v10, wind_speed


def load_cmems():
    """CMEMS 해류 데이터 로드 → (lats, lons, uo, vo, current_speed)"""
    import xarray as xr
    ds = xr.open_dataset(CMEMS_PATH)

    ds_sel = ds
    # 시간/깊이 차원 축소
    for dim in ['time', 'valid_time']:
        if dim in ds_sel.dims:
            ds_sel = ds_sel.isel({dim: 0})
    for dim in ['depth']:
        if dim in ds_sel.dims:
            ds_sel = ds_sel.isel({dim: 0})

    uo = ds_sel['uo'].values.squeeze()
    vo = ds_sel['vo'].values.squeeze()

    lats = ds_sel['latitude'].values if 'latitude' in ds_sel.coords else ds_sel['lat'].values
    lons = ds_sel['longitude'].values if 'longitude' in ds_sel.coords else ds_sel['lon'].values

    # NaN 처리 (육지 영역)
    uo = np.nan_to_num(uo, nan=0.0)
    vo = np.nan_to_num(vo, nan=0.0)

    current_speed = np.sqrt(uo**2 + vo**2)
    ds.close()
    return lats, lons, uo, vo, current_speed


def rasterize_coastline(land_poly, lats, lons):
    """육지 폴리곤 → 래스터 마스크 (공통 격자)"""
    from shapely.geometry import Point
    from shapely.prepared import prep

    prepped = prep(land_poly)
    mask = np.zeros((len(lats), len(lons)), dtype=bool)
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            mask[i, j] = prepped.contains(Point(lon, lat))
    return mask


def plot_environment():
    """3-Panel 환경 맵 시각화"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ─── 데이터 로드 ───
    print("[1/4] Loading ERA5 wind data...")
    era5_lats, era5_lons, u10, v10, wind_speed = load_era5()
    print(f"       ERA5 grid: {len(era5_lats)}×{len(era5_lons)}, "
          f"lat {era5_lats.min():.2f}~{era5_lats.max():.2f}, "
          f"lon {era5_lons.min():.2f}~{era5_lons.max():.2f}")

    has_cmems = os.path.exists(CMEMS_PATH)
    if has_cmems:
        print("[2/4] Loading CMEMS ocean current data...")
        cmems_lats, cmems_lons, uo, vo, current_speed = load_cmems()
        print(f"       CMEMS grid: {len(cmems_lats)}×{len(cmems_lons)}, "
              f"lat {cmems_lats.min():.2f}~{cmems_lats.max():.2f}, "
              f"lon {cmems_lons.min():.2f}~{cmems_lons.max():.2f}")
    else:
        print("[2/4] CMEMS 데이터 미발견. 해류 패널은 비워둡니다.")
        print(f"       다운로드: python scripts/download_cmems.py")

    print("[3/4] Loading GSHHG coastline...")
    land_poly = load_coastline(source="natural_earth", bounds=BOUNDS)

    # GSHHG 래스터화용 공통 격자 (0.02° ≈ 2km)
    gshhg_res = 0.002
    gshhg_lats = np.arange(LAT_MIN, LAT_MAX, gshhg_res)
    gshhg_lons = np.arange(LON_MIN, LON_MAX, gshhg_res)

    print(f"       natural_earth raster: {len(gshhg_lats)}×{len(gshhg_lons)} @ {gshhg_res}°")
    print("[3.5/4] Rasterizing natural_earth coastline...")
    land_mask = rasterize_coastline(land_poly, gshhg_lats, gshhg_lons)
    print(f"       Land pixels: {land_mask.sum()} / {land_mask.size} "
          f"({land_mask.sum()/land_mask.size*100:.1f}%)")

    # ─── 시각화 ───
    print("[4/4] Rendering 3-panel figure...")

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), constrained_layout=True)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Panel 1: ERA5 Wind
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("[Plot 1/3] Saving ERA5 Wind Map...")
    fig1, ax1 = plt.subplots(figsize=(8, 7))
    lon_g, lat_g = np.meshgrid(era5_lons, era5_lats)
    im1 = ax1.pcolormesh(lon_g, lat_g, wind_speed, cmap='YlOrRd', vmin=0, vmax=15, shading='auto')
    skip_e = max(1, len(era5_lats) // 15)
    ax1.quiver(lon_g[::skip_e, ::skip_e], lat_g[::skip_e, ::skip_e],
               u10[::skip_e, ::skip_e], v10[::skip_e, ::skip_e], color='black', alpha=0.7)
    ax1.contour(gshhg_lons, gshhg_lats, land_mask.astype(float), levels=[0.5], colors='black')
    # plt.colorbar(im1, ax=ax1, label="Wind Speed (m/s)")
    # ax1.set_title("ERA5 Wind (10m)")
    ax1.set_aspect('equal')
    fig1.savefig(os.path.join(OUTPUT_DIR, "env_1_wind.png"), dpi=200, bbox_inches='tight')
    plt.axis('off')
    plt.close(fig1)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. CMEMS Current Map (개별 저장)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if has_cmems:
        print("[Plot 2/3] Saving CMEMS Current Map...")
        fig2, ax2 = plt.subplots(figsize=(8, 7))
        lon_c, lat_c = np.meshgrid(cmems_lons, cmems_lats)
        im2 = ax2.pcolormesh(lon_c, lat_c, current_speed, cmap='cool', vmin=0, vmax=1.5, shading='auto')
        skip_c = max(1, len(cmems_lats) // 20)
        ax2.quiver(lon_c[::skip_c, ::skip_c], lat_c[::skip_c, ::skip_c],
                   uo[::skip_c, ::skip_c], vo[::skip_c, ::skip_c], color='navy')
        ax2.contour(gshhg_lons, gshhg_lats, land_mask.astype(float), levels=[0.5], colors='black')
        # plt.colorbar(im2, ax=ax2, label="Current Speed (m/s)")
        # ax2.set_title("CMEMS Ocean Current")
        ax2.set_aspect('equal')
        fig2.savefig(os.path.join(OUTPUT_DIR, "env_2_current.png"), dpi=200, bbox_inches='tight')
        plt.axis('off')
        plt.close(fig2)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. GSHHG Coastline Map (개별 저장)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("[Plot 3/3] Saving GSHHG Coastline Map...")
    fig3, ax3 = plt.subplots(figsize=(8, 7))
    sea_land_cmap = mcolors.ListedColormap(['#1a5276', '#8B6914'])
    ax3.pcolormesh(gshhg_lons, gshhg_lats, land_mask.astype(float), cmap=sea_land_cmap, vmin=0, vmax=1)
    # ax3.set_title("GSHHG High-Res Coastline")
    ax3.set_aspect('equal')
    fig3.savefig(os.path.join(OUTPUT_DIR, "env_3_mask.png"), dpi=200, bbox_inches='tight')
    plt.close(fig3)

    print(f"\n ✅ All plots saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    plot_environment()
