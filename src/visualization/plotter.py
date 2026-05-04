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
from matplotlib.colors import ListedColormap
from shapely.geometry import MultiPolygon


def _extract_schedule_dg_names(schedule: list[dict]) -> list[str]:
    """Infer DG names from MILP schedule keys like DG1_P_MW."""
    if not schedule:
        return []

    dg_names = []
    for key in schedule[0].keys():
        if key.startswith("DG") and key.endswith("_P_MW"):
            dg_names.append(key[:-5])

    def sort_key(name: str):
        suffix = name[2:]
        if suffix.isdigit():
            return (0, int(suffix))
        return (1, suffix)

    return sorted(dg_names, key=sort_key)


def _build_visual_route(route: dict) -> dict:
    """Return the original route geometry without synthetic display padding."""
    waypoints = list(route["waypoints"])
    speeds = list(route["speeds"])
    headings = list(route["headings"])

    if not waypoints or not speeds or not headings:
        return {"waypoints": waypoints, "speeds": speeds, "headings": headings}

    return {
        "waypoints": waypoints,
        "speeds": speeds,
        "headings": headings,
    }


def _infer_route_endpoints(
    waypoints,
    fallback_start: tuple[float, float],
    fallback_end: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    if waypoints and len(waypoints) >= 2:
        return tuple(waypoints[0]), tuple(waypoints[-1])
    return fallback_start, fallback_end


def plot_cost_map_route(
    route: dict,
    cost_map,
    save_dir: str = "output",
    filename: str = "cost_map_route.png",
):
    """Save a simple route view over the cost map without speed coloring."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    cost_map.plot(save_path=save_path, route_wps=route.get("waypoints"))


def plot_route_comparison_map(
    routes_by_case: dict[str, dict],
    cost_map,
    save_dir: str = "output",
    filename: str = "comparison_routes.png",
):
    """Save a single map with multiple case routes overlaid for comparison."""
    os.makedirs(save_dir, exist_ok=True)

    from src.grid.no_go_zone import BUSAN_SHANGHAI_BOUNDS, BUSAN_PORT, SHANGHAI_PORT

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_aspect("equal")

    bounds = BUSAN_SHANGHAI_BOUNDS
    ax.set_xlim(bounds["lon_min"], bounds["lon_max"])
    ax.set_ylim(bounds["lat_min"], bounds["lat_max"])

    extent = [
        bounds["lon_min"], bounds["lon_max"],
        bounds["lat_min"], bounds["lat_max"],
    ]
    ax.imshow(
        cost_map.cost_grid,
        extent=extent,
        origin="lower",
        cmap=ListedColormap(["white", "#bdbdbd"]),
        vmin=0,
        vmax=100,
        aspect="equal",
        alpha=0.55,
        interpolation="bilinear",
    )

    X, Y = np.meshgrid(cost_map.lons, cost_map.lats)
    ax.contour(
        X,
        Y,
        cost_map.cost_grid,
        levels=[50],
        colors="black",
        linewidths=0.5,
        alpha=0.35,
    )

    palette = ["#E53935", "#1E88E5", "#43A047", "#FB8C00", "#6D4C41", "#8E24AA"]
    first_route_with_waypoints = next(
        (route for route in routes_by_case.values() if route and route.get("waypoints")),
        None,
    )
    marker_start, marker_end = _infer_route_endpoints(
        None if first_route_with_waypoints is None else first_route_with_waypoints.get("waypoints"),
        BUSAN_PORT,
        SHANGHAI_PORT,
    )
    for index, (case_name, route) in enumerate(routes_by_case.items()):
        if not route or not route.get("waypoints"):
            continue

        color = palette[index % len(palette)]
        waypoints = route["waypoints"]
        lats = [wp[0] for wp in waypoints]
        lons = [wp[1] for wp in waypoints]
        ax.plot(
            lons,
            lats,
            "-",
            color=color,
            linewidth=2.4,
            solid_capstyle="round",
            label=case_name,
        )
        ax.plot(
            lons,
            lats,
            "o",
            color="white",
            markersize=2.8,
            markeredgecolor=color,
            markeredgewidth=0.8,
        )

    ax.plot(
        marker_start[1],
        marker_start[0],
        "*",
        color="#4CAF50",
        markersize=14,
        markeredgecolor="Black",
        markeredgewidth=1.2,
        label="Start",
        zorder=10,
    )
    ax.plot(
        marker_end[1],
        marker_end[0],
        "*",
        color="#F44336",
        markersize=14,
        markeredgecolor="Black",
        markeredgewidth=1.2,
        label="Destination",
        zorder=10,
    )

    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.set_title("Case Route Comparison", fontsize=12, fontweight="bold")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(fontsize=8.5, loc="upper right")

    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_optimal_route(
    route: dict,
    cost_map,
    save_dir: str = "output",
    raw_path_wps: list[tuple[float, float]] | None = None,
):
    """
    최적 경로를 Cost Map 위에 시각화.
    """
    os.makedirs(save_dir, exist_ok=True)
    plot_cost_map_route(route, cost_map, save_dir=save_dir, filename="route.png")
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_aspect("equal")

    from src.grid.no_go_zone import BUSAN_SHANGHAI_BOUNDS, BUSAN_PORT, SHANGHAI_PORT
    bounds = BUSAN_SHANGHAI_BOUNDS
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
    marker_start, marker_end = _infer_route_endpoints(wps, BUSAN_PORT, SHANGHAI_PORT)

    if raw_path_wps:
        raw_lats = [wp[0] for wp in raw_path_wps]
        raw_lons = [wp[1] for wp in raw_path_wps]
        ax.plot(
            raw_lons,
            raw_lats,
            "--",
            color="#212121",
            linewidth=1.0,
            alpha=0.75,
            dashes=(3, 2),
            label="Raw A* Path",
            zorder=3,
        )

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
    ax.plot(marker_start[1], marker_start[0], '*', color='#4CAF50',
            markersize=18, markeredgecolor='white', markeredgewidth=1.5, label='Start', zorder=10)
    ax.plot(marker_end[1], marker_end[0], '*', color='#F44336',
            markersize=18, markeredgecolor='white', markeredgewidth=1.5, label='Destination', zorder=10)

    # 컬러바
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    # cbar.set_label("Speed (knots)", fontsize=11)

    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)
    ax.set_title("Optimal Route — GA + MILP Integrated Optimization", fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, linewidth=0.3, alpha=0.5)

    path = os.path.join(save_dir, "optimal_route.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_raw_astar_only(
    raw_path_wps: list[tuple[float, float]] | None,
    cost_map,
    save_dir: str = "output",
    filename: str = "raw_astar_only.png",
):
    """Save the raw A* grid path on the cost map for comparison."""
    if not raw_path_wps:
        return

    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_aspect("equal")

    from src.grid.no_go_zone import BUSAN_SHANGHAI_BOUNDS, BUSAN_PORT, SHANGHAI_PORT

    bounds = BUSAN_SHANGHAI_BOUNDS
    ax.set_xlim(bounds["lon_min"], bounds["lon_max"])
    ax.set_ylim(bounds["lat_min"], bounds["lat_max"])

    extent = [
        bounds["lon_min"], bounds["lon_max"],
        bounds["lat_min"], bounds["lat_max"],
    ]
    ax.imshow(
        cost_map.cost_grid,
        extent=extent,
        origin="lower",
        cmap="RdYlBu_r",
        vmin=0,
        vmax=100,
        aspect="equal",
        alpha=0.6,
    )

    X, Y = np.meshgrid(cost_map.lons, cost_map.lats)
    cs = ax.contour(
        X,
        Y,
        cost_map.cost_grid,
        levels=[50],
        colors="black",
        linewidths=0.3,
        alpha=0.3,
    )
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")

    raw_lats = [wp[0] for wp in raw_path_wps]
    raw_lons = [wp[1] for wp in raw_path_wps]
    marker_start, marker_end = _infer_route_endpoints(raw_path_wps, BUSAN_PORT, SHANGHAI_PORT)
    ax.plot(
        raw_lons,
        raw_lats,
        "-",
        color="#111111",
        linewidth=1.4,
        alpha=0.95,
        label="Raw A* Path",
        zorder=4,
    )

    ax.plot(
        marker_start[1],
        marker_start[0],
        "*",
        color="#4CAF50",
        markersize=18,
        markeredgecolor="white",
        markeredgewidth=1.5,
        label="Start",
        zorder=10,
    )
    ax.plot(
        marker_end[1],
        marker_end[0],
        "*",
        color="#F44336",
        markersize=18,
        markeredgecolor="white",
        markeredgewidth=1.5,
        label="Destination",
        zorder=10,
    )

    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)
    ax.set_title("Raw A* Grid Path", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_convergence(
    logbook,
    save_dir: str = "output",
    start_gen: int = 0,
    y_max: float | None = None,
    y_label : str = 'Total Fuel (kg)'
):
    """GA 수렴 곡선."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    gen = logbook.select("gen")
    fit_min = logbook.select("min")
    fit_avg = logbook.select("avg")

    # BIG_PENALTY 필터링
    threshold = 1e+12
    fit_avg_clean = [v if v < threshold else None for v in fit_avg]

    mask = [g>= start_gen for g in gen]
    gen = [g for g, keep in zip(gen, mask) if keep]
    fit_min = [v for v, keep in zip(fit_min, mask) if keep]
    fit_avg_clean = [v for v, keep in zip(fit_avg_clean, mask) if keep]

    ax.plot(gen, fit_min, 'o-', color='#E53935', markersize=3, linewidth=1.5, label='Best (min)')
    # valid_avg = [(g, v) for g, v in zip(gen, fit_avg_clean) if v is not None]
    # if valid_avg:
    #     ax.plot([g for g, v in valid_avg], [v for g, v in valid_avg],
    #             's-', color='#1E88E5', markersize=2, linewidth=1, alpha=0.7, label='Average')

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel(f"{y_label}", fontsize=12)
    ax.set_title("GA Convergence", fontsize=14, fontweight='bold')
    if y_max is not None:
        ax.set_ylim(0, y_max)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, "convergence.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {path}")


def _plot_power_scheduling_panel(
    ax,
    schedule: list[dict],
    sog_profile: list[float] | None = None,
    title: str = "DG + ESS Output Scheduling Profile",
    title_size: float = 12,
    label_size: float = 11,
    tick_size: float = 10,
    legend_size: float = 9,
    bar_width: float = 0.8,
    show_xlabel: bool = False,
):
    """Draw the top scheduling panel used by the full power schedule plot."""
    T = len(schedule)
    t_arr = np.arange(T)
    P_req = [s["P_req_MW"] for s in schedule]
    dg_names = _extract_schedule_dg_names(schedule)
    dg_palette = ["#E53935", "#1E88E5", "#43A047", "#8E24AA", "#FB8C00", "#6D4C41"]
    dg_total = np.zeros(T, dtype=float)
    P_dc = [s["P_dc_MW"] for s in schedule]
    P_c = [s["P_c_MW"] for s in schedule]

    for index, dg_name in enumerate(dg_names):
        dg_power = np.array([s.get(f"{dg_name}_P_MW", 0.0) for s in schedule], dtype=float)
        ax.bar(
            t_arr,
            dg_power,
            bottom=dg_total,
            width=bar_width,
            color=dg_palette[index % len(dg_palette)],
            alpha=0.8,
            label=dg_name,
        )
        dg_total += dg_power

    ax.bar(t_arr, P_dc, bottom=dg_total, width=bar_width, color='#FB8C00', alpha=0.75, label='ESS discharge')
    ax.bar(t_arr, [-value for value in P_c], width=bar_width, color='#29B6F6', alpha=0.75, label='ESS charge')
    ax.plot(t_arr, P_req, 'ko-', markersize=4.5, linewidth=1.8, label='P_req')
    ax.axhline(0, color='black', linewidth=0.55)
    ax.set_ylabel("Power (MW)", fontsize=label_size)
    if show_xlabel:
        ax.set_xlabel("Time Step", fontsize=label_size)
    ax.set_title(title, fontsize=title_size, fontweight='bold', pad=24)
    ax.tick_params(axis='both', labelsize=tick_size)
    ax.grid(True, alpha=0.3)

    speed_ax = ax.twinx()
    if sog_profile:
        speed_ax.plot(
            t_arr[:len(sog_profile)],
            sog_profile,
            color='#8E24AA',
            linestyle='--',
            marker='o',
            markersize=3.8,
            linewidth=1.7,
            label='SOG',
        )
        speed_ax.set_ylabel("Speed (kts)", fontsize=label_size, color='#8E24AA')
        speed_ax.tick_params(axis='y', colors='#8E24AA', labelsize=tick_size)

    handles, labels = ax.get_legend_handles_labels()
    speed_handles, speed_labels = speed_ax.get_legend_handles_labels()
    ax.legend(
        handles + speed_handles,
        labels + speed_labels,
        fontsize=legend_size,
        ncol=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=True,
    )
    return ax


def plot_power_schedule(
    milp_result: dict,
    power_profile: list[dict] | None = None,
    save_dir: str = "output",
):
    """구간별 전력 스케줄 (DG + ESS)."""
    os.makedirs(save_dir, exist_ok=True)

    schedule = milp_result["schedule"]
    T = len(schedule)
    if T == 0:
        return

    t_arr = np.arange(T)
    sog_profile = []
    if power_profile is not None:
        sog_profile = [float(step["speed_sog_kts"]) for step in power_profile[:T]]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # ── (a) P_req + DG 출력 ──
    ax = axes[0]
    P_req = [s["P_req_MW"] for s in schedule]
    dg_names = _extract_schedule_dg_names(schedule)
    dg_palette = ["#E53935", "#1E88E5", "#43A047", "#8E24AA", "#FB8C00", "#6D4C41"]
    dg_total = np.zeros(T, dtype=float)
    P_dc = [s["P_dc_MW"] for s in schedule]
    P_c_signed = [-s["P_c_MW"] for s in schedule]
    P_c = [s["P_c_MW"] for s in schedule]

    for index, dg_name in enumerate(dg_names):
        dg_power = np.array([s.get(f"{dg_name}_P_MW", 0.0) for s in schedule], dtype=float)
        ax.bar(
            t_arr,
            dg_power,
            bottom=dg_total,
            color=dg_palette[index % len(dg_palette)],
            alpha=0.8,
            label=dg_name,
        )
        dg_total += dg_power

    ax.bar(t_arr, P_dc, bottom=dg_total, color='#FB8C00', alpha=0.75, label='ESS discharge')
    ax.bar(t_arr, [-value for value in P_c], color='#29B6F6', alpha=0.75, label='ESS charge')
    ax.plot(t_arr, P_req, 'ko-', markersize=4, linewidth=1.5, label='P_req')
    ax.set_ylabel("Power (MW)", fontsize=11)
    ax.set_title("DG + ESS Output Scheduling Profile", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    speed_ax = ax.twinx()
    if sog_profile:
        speed_ax.plot(
            t_arr[:len(sog_profile)],
            sog_profile,
            color='#8E24AA',
            linestyle='--',
            marker='o',
            markersize=3,
            linewidth=1.4,
            label='SOG',
        )
        speed_ax.set_ylabel("Speed (kts)", fontsize=11, color='#8E24AA')
        speed_ax.tick_params(axis='y', colors='#8E24AA')

    handles, labels = ax.get_legend_handles_labels()
    speed_handles, speed_labels = speed_ax.get_legend_handles_labels()
    ax.legend(handles + speed_handles, labels + speed_labels, fontsize=9, ncol=4)

    # ── (b) ESS 충방전 ──
    ax = axes[1]
    P_dc = [s["P_dc_MW"] for s in schedule]
    P_c = [-s["P_c_MW"] for s in schedule]  # 충전은 음수 표시
    colors = ['#FF7043' if v > 0 else '#42A5F5' for v in [d + c for d, c in zip(P_dc, P_c)]]
    ax.bar(t_arr, P_dc, color='#FF7043', alpha=0.8, label='Discharge')
    ax.bar(t_arr, P_c_signed, color='#42A5F5', alpha=0.8, label='Charge')
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


def plot_power_scheduling_detail(
    milp_result: dict,
    power_profile: list[dict] | None = None,
    save_dir: str = "output",
    filename: str = "power_scheduling_detail.png",
):
    """Save only the top DG/ESS scheduling panel as a large standalone figure."""
    os.makedirs(save_dir, exist_ok=True)

    schedule = milp_result["schedule"]
    T = len(schedule)
    if T == 0:
        return

    sog_profile = []
    if power_profile is not None:
        sog_profile = [float(step["speed_sog_kts"]) for step in power_profile[:T]]

    fig, ax = plt.subplots(figsize=(16, 6.2))
    _plot_power_scheduling_panel(
        ax,
        schedule,
        sog_profile=sog_profile,
        title="DG + ESS Output Scheduling Profile",
        title_size=16,
        label_size=13,
        tick_size=11,
        legend_size=11,
        bar_width=0.82,
        show_xlabel=True,
    )
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_weather_map(
    weather_fn,
    cost_map,
    route_wps=None,
    save_dir: str = "output",
    when_utc=None,
):
    """
    풍속/풍향을 바탕으로 Beaufort Number(BN) 맵 오버레이 시각화.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as patheffects
    from datetime import datetime, timezone
    from src.resistance.modified_dpm import wind_speed_to_beaufort
    from src.resistance.models import EnvironmentData
    from src.grid.no_go_zone import BUSAN_PORT, SHANGHAI_PORT

    if when_utc is None:
        when_utc = datetime(2000, 1, 1, tzinfo=timezone.utc)

    def resolve_wind(lat, lon):
        try:
            value = weather_fn(lat, lon, when_utc)
        except TypeError:
            value = weather_fn(lat, lon)

        if isinstance(value, EnvironmentData):
            return float(value.wind_speed_ms), float(value.wind_dir_deg or 0.0)
        if isinstance(value, tuple) and len(value) == 2:
            return float(value[0]), float(value[1])
        raise TypeError("plot_weather_map expects env_fn(lat, lon, when_utc) or weather_fn(lat, lon).")

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

    marker_start, marker_end = _infer_route_endpoints(route_wps, BUSAN_PORT, SHANGHAI_PORT)
    ax.plot(marker_start[1], marker_start[0], '*', color='#4CAF50',
            markersize=18, markeredgecolor='white', label='Start')
    ax.plot(marker_end[1], marker_end[0], '*', color='#F44336',
            markersize=18, markeredgecolor='white', label='Destination')

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
