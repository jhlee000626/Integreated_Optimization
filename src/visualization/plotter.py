"""
결과 시각화 모듈
================
GA 최적화 결과를 시각화:
    1. 최적 경로 지도
    2. GA 수렴 곡선
    3. 구간별 부하 + DG 스케줄
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from shapely.geometry import MultiPolygon


def _build_visual_route(route: dict) -> dict:
    """Pad the route for display so speed starts/ends at zero."""
    waypoints = list(route["waypoints"])
    speeds = list(route["speeds"])
    headings = list(route["headings"])

    if not waypoints or not speeds or not headings:
        return {"waypoints": waypoints, "speeds": speeds, "headings": headings}

    visual_waypoints = [waypoints[0], *waypoints, waypoints[-1]]
    visual_speeds = [0.0, *speeds, 0.0]
    visual_headings = [headings[0], *headings, headings[-1]]

    return {
        "waypoints": visual_waypoints,
        "speeds": visual_speeds,
        "headings": visual_headings,
    }


def plot_optimal_route(
    route: dict,
    cost_map,
    save_dir: str = "output",
):
    """
    최적 경로를 Cost Map 위에 시각화.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_aspect("equal")

    from src.grid.no_go_zone import BUSAN_JEJU_BOUNDS, BUSAN_PORT, JEJU_PORT
    bounds = BUSAN_JEJU_BOUNDS
    ax.set_xlim(bounds["lon_min"], bounds["lon_max"])
    ax.set_ylim(bounds["lat_min"], bounds["lat_max"])

    # ── Cost Map 배경 ──
    extent = [
        bounds["lon_min"], bounds["lon_max"],
        bounds["lat_min"], bounds["lat_max"],
    ]
    im = ax.imshow(
        cost_map.cost_grid,
        extent=extent,
        origin='lower',
        cmap='RdYlBu_r',
        vmin=0, vmax=100,
        aspect='equal',
        alpha=0.6,
    )
    
    # 이진 맵 등고선
    X, Y = np.meshgrid(cost_map.lons, cost_map.lats)
    cs = ax.contour(X, Y, cost_map.cost_grid, levels=[50],
                    colors='black', linewidths=0.3, alpha=0.3)
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.0f')

    # ── 최적 경로 ──
    visual_route = _build_visual_route(route)
    wps = visual_route["waypoints"]
    lats = [wp[0] for wp in wps]
    lons = [wp[1] for wp in wps]
    speeds = visual_route["speeds"]

    # 속도에 따른 색상 (느림=파랑, 빠름=빨강)
    v_min, v_max = min(speeds), max(speeds)
    norm = plt.Normalize(v_min - 0.5, v_max + 0.5)
    cmap = plt.cm.RdYlBu_r

    for i in range(len(speeds)):
        color = cmap(norm(speeds[i]))
        ax.plot(
            [lons[i], lons[i + 1]], [lats[i], lats[i + 1]],
            '-', color=color, linewidth=2.5, solid_capstyle='round',
        )
        # 구간 번호
        mid_lon = (lons[i] + lons[i + 1]) / 2
        mid_lat = (lats[i] + lats[i + 1]) / 2
        ax.text(mid_lon, mid_lat, f"{i}", fontsize=7, ha='center',
                color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))

    # 웨이포인트 마커
    ax.plot(lons, lats, 'o', color='white', markersize=5, markeredgecolor='#333', markeredgewidth=0.8)

    # 항구
    ax.plot(BUSAN_PORT[1], BUSAN_PORT[0], '*', color='#4CAF50',
            markersize=18, markeredgecolor='white', markeredgewidth=1.5, label='Busan', zorder=10)
    ax.plot(JEJU_PORT[1], JEJU_PORT[0], '*', color='#F44336',
            markersize=18, markeredgecolor='white', markeredgewidth=1.5, label='Jeju', zorder=10)

    # 컬러바
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Speed (knots)", fontsize=11)

    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)
    ax.set_title("Optimal Route — GA + MILP Integrated Optimization", fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, linewidth=0.3, alpha=0.5)

    path = os.path.join(save_dir, "optimal_route.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_convergence(logbook, save_dir: str = "output"):
    """GA 수렴 곡선."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    gen = logbook.select("gen")
    fit_min = logbook.select("min")
    fit_avg = logbook.select("avg")

    # BIG_PENALTY 필터링
    threshold = 1e8
    fit_avg_clean = [v if v < threshold else None for v in fit_avg]

    ax.plot(gen, fit_min, 'o-', color='#E53935', markersize=3, linewidth=1.5, label='Best (min)')
    valid_avg = [(g, v) for g, v in zip(gen, fit_avg_clean) if v is not None]
    if valid_avg:
        ax.plot([g for g, v in valid_avg], [v for g, v in valid_avg],
                's-', color='#1E88E5', markersize=2, linewidth=1, alpha=0.7, label='Average')

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("GA Objective", fontsize=12)
    ax.set_title("GA Convergence", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, "convergence.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_power_schedule(
    milp_result: dict,
    save_dir: str = "output",
):
    """구간별 전력 스케줄 (DG + ESS)."""
    os.makedirs(save_dir, exist_ok=True)

    schedule = milp_result["schedule"]
    T = len(schedule)
    t_arr = np.arange(T)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # ── (a) P_req + DG 출력 ──
    ax = axes[0]
    P_req = [s["P_req_MW"] for s in schedule]
    DG1 = [s["DG1_P_MW"] for s in schedule]
    DG2 = [s["DG2_P_MW"] for s in schedule]
    DG3 = [s["DG3_P_MW"] for s in schedule]

    ax.bar(t_arr, DG1, color='#E53935', alpha=0.8, label='DG1')
    ax.bar(t_arr, DG2, bottom=DG1, color='#1E88E5', alpha=0.8, label='DG2')
    dg12 = [a + b for a, b in zip(DG1, DG2)]
    ax.bar(t_arr, DG3, bottom=dg12, color='#43A047', alpha=0.8, label='DG3')
    ax.plot(t_arr, P_req, 'ko-', markersize=4, linewidth=1.5, label='P_req')
    ax.set_ylabel("Power (MW)", fontsize=11)
    ax.set_title("DG Output vs Required Load", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, ncol=4)
    ax.grid(True, alpha=0.3)

    # ── (b) ESS 충방전 ──
    ax = axes[1]
    P_dc = [s["P_dc_MW"] for s in schedule]
    P_c = [-s["P_c_MW"] for s in schedule]  # 충전은 음수 표시
    colors = ['#FF7043' if v > 0 else '#42A5F5' for v in [d + c for d, c in zip(P_dc, P_c)]]
    ax.bar(t_arr, P_dc, color='#FF7043', alpha=0.8, label='Discharge')
    ax.bar(t_arr, P_c, color='#42A5F5', alpha=0.8, label='Charge')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel("ESS Power (MW)", fontsize=11)
    ax.set_title("ESS Charge/Discharge", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── (c) SOC ──
    ax = axes[2]
    SOC = [s["SOC"] * 100 for s in schedule]
    ax.plot(t_arr, SOC, 'o-', color='#7E57C2', linewidth=2, markersize=5)
    ax.axhline(95, color='red', linestyle='--', alpha=0.5, label='SOC max (95%)')
    ax.axhline(10, color='red', linestyle='--', alpha=0.5, label='SOC min (10%)')
    ax.fill_between(t_arr, 10, SOC, alpha=0.15, color='#7E57C2')
    ax.set_ylabel("SOC (%)", fontsize=11)
    ax.set_xlabel("Time Step", fontsize=11)
    ax.set_title("Battery SOC", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "power_schedule.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_weather_map(
    env_fn,
    cost_map,
    route_wps=None,
    save_dir: str = "output",
    when_utc=None,
):
    """
    풍속/풍향을 바탕으로 Beaufort Number(BN) 맵 오버레이 시각화.
    """
    from datetime import datetime, timezone
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as patheffects
    from src.resistance.modified_dpm import wind_speed_to_beaufort
    from src.resistance.models import EnvironmentData
    from src.grid.no_go_zone import BUSAN_PORT, JEJU_PORT

    if when_utc is None:
        when_utc = datetime(2000, 1, 1, tzinfo=timezone.utc)

    def resolve_wind(lat, lon):
        try:
            value = env_fn(lat, lon, when_utc)
        except TypeError:
            value = env_fn(lat, lon)

        if isinstance(value, EnvironmentData):
            return float(value.wind_speed_ms), float(value.wind_dir_deg or 0.0)
        if isinstance(value, tuple) and len(value) == 2:
            return float(value[0]), float(value[1])
        raise TypeError("plot_weather_map expects env_fn to return EnvironmentData or (wind_speed_ms, wind_dir_deg).")

    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    
    bounds = cost_map.bounds
    ax.set_aspect("equal")
    ax.set_xlim(bounds["lon_min"], bounds["lon_max"])
    ax.set_ylim(bounds["lat_min"], bounds["lat_max"])
    
    lon_grid, lat_grid = np.meshgrid(cost_map.lons, cost_map.lats)
    
    bn_grid = np.zeros_like(lon_grid, dtype=float)
    u_grid = np.zeros_like(lon_grid, dtype=float)
    v_grid = np.zeros_like(lon_grid, dtype=float)
    
    for i in range(len(cost_map.lats)):
        for j in range(len(cost_map.lons)):
            lat, lon = cost_map.lats[i], cost_map.lons[j]
            ws, wd = resolve_wind(lat, lon)
            bn_grid[i, j] = wind_speed_to_beaufort(ws)
            
            # 기상학적 풍향: 바람이 불어오는 방향 (0 = N, 90 = E)
            # 벡터를 그리기 위해 바람이 "불어가는" 방향으로 성분 분해 (−값)
            theta = np.radians(wd)
            u_grid[i, j] = -ws * np.sin(theta)
            v_grid[i, j] = -ws * np.cos(theta)
            
    extent = [
        bounds["lon_min"], bounds["lon_max"],
        bounds["lat_min"], bounds["lat_max"],
    ]
    
    im = ax.imshow(
        bn_grid,
        extent=extent,
        origin='lower',
        cmap='YlOrRd',
        vmin=0, vmax=10,
        aspect='equal',
        alpha=0.6,
    )
    
    # Cost Map 육지 윤곽선 (레벨 50)
    ax.contour(lon_grid, lat_grid, cost_map.cost_grid, levels=[50],
               colors='black', linewidths=1.5, alpha=0.8)

    # 바람 벡터 (Quiver) - 너무 촘촘하지 않게 샘플링
    skip = max(1, int(len(cost_map.lats) / 30))
    M = np.hypot(u_grid, v_grid)
    ax.quiver(lon_grid[::skip, ::skip], lat_grid[::skip, ::skip],
              u_grid[::skip, ::skip], v_grid[::skip, ::skip],
              M[::skip, ::skip], cmap='winter', pivot='mid', alpha=0.6)

    # 최적 경로 (있으면)
    if route_wps:
        lats = [wp[0] for wp in route_wps]
        lons = [wp[1] for wp in route_wps]
        ax.plot(lons, lats, '-', color='#00FF00', linewidth=3,
                # path_effects=[patheffects.withStroke(linewidth=5, foreground='black')],
                label='Optimal Route')
        ax.plot(lons, lats, 'o', color='white', markersize=4)

    # 항구
    ax.plot(BUSAN_PORT[1], BUSAN_PORT[0], '*', color='#4CAF50',
            markersize=18, markeredgecolor='white', label='Busan')
    ax.plot(JEJU_PORT[1], JEJU_PORT[0], '*', color='#F44336',
            markersize=18, markeredgecolor='white', label='Jeju')

    # Colorbars
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Beaufort Number (BN)", fontsize=11)
    
    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)
    ax.set_title("Weather Map: Beaufort Number & Wind Vectors", fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, linewidth=0.3, alpha=0.5)

    path = os.path.join(save_dir, "weather_bn_map.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {path}")
