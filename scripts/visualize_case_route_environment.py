"""
Visualize a saved case artifact as an environment raster + vectors + optimal route map.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

try:
    import cartopy.crs as ccrs

    CARTOPY_AVAILABLE = True
except ModuleNotFoundError:
    ccrs = None
    CARTOPY_AVAILABLE = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Validation.common import load_marine_environment
from src.grid.no_go_zone import load_coastline
from src.resistance.models import EnvironmentData

ROUTE_COLOR = "#d7191c"
ROUTE_NODE_FACE_COLOR = "#ffffff"
ROUTE_NODE_EDGE_COLOR = "#303030"
PORT_NODE_EDGE_COLOR = ROUTE_COLOR


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-json", required=True, help="Path to a saved case_artifact.json file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to <field>_route_map.png next to the case JSON.",
    )
    parser.add_argument(
        "--field",
        choices=("wind", "current"),
        default="wind",
        help="Environmental field to plot with the route.",
    )
    parser.add_argument(
        "--current-vector-scale",
        type=float,
        default=22.0,
        help="Quiver scale for current vectors. Larger values make arrows shorter.",
    )
    parser.add_argument("--era5-path", default=None, help="Explicit path to the ERA5 NetCDF dataset.")
    parser.add_argument("--cmems-path", default=None, help="Explicit path to the CMEMS NetCDF dataset.")
    parser.add_argument(
        "--hide-route",
        action="store_true",
        help="Do not draw the optimized route polyline or waypoint markers.",
    )
    parser.add_argument(
        "--hide-ports",
        action="store_true",
        help="Do not draw the start/end port markers.",
    )
    parser.add_argument(
        "--weather-only",
        action="store_true",
        help="Shortcut for --hide-route --hide-ports.",
    )
    return parser


def _load_case_artifact(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_output_path(case_json_path: str, output_path: str | None, field: str) -> str:
    if output_path:
        return output_path
    return os.path.join(os.path.dirname(os.path.abspath(case_json_path)), f"{field}_route_map.png")


def _parse_departure_time(case_artifact: dict) -> datetime:
    departure_time = case_artifact.get("departure_time_utc")
    if not departure_time:
        raise ValueError("case_artifact.json is missing departure_time_utc")
    return datetime.fromisoformat(departure_time)


def _find_dataset_for_departure(dataset_kind: str, when_utc: datetime) -> str | None:
    date_stamp = when_utc.strftime("%Y%m%d")
    if dataset_kind == "era5":
        directory = os.path.join(PROJECT_ROOT, "data", "era5")
        patterns = (
            f"era5_marine_{date_stamp}T*.nc",
            f"era5_*_{date_stamp}T*.nc",
            f"*{date_stamp}*.nc",
        )
    elif dataset_kind == "cmems":
        directory = os.path.join(PROJECT_ROOT, "data", "cmems")
        patterns = (
            f"cmems_current_{date_stamp}T*.nc",
            f"cmems_*_{date_stamp}T*.nc",
            f"*{date_stamp}*.nc",
        )
    else:
        raise ValueError(f"Unsupported dataset kind: {dataset_kind}")

    for pattern in patterns:
        matches = sorted(glob.glob(os.path.join(directory, pattern)))
        if matches:
            return matches[-1]
    return None


def _bounds_from_route(case_artifact: dict) -> dict[str, float]:
    route = case_artifact.get("route", {})
    waypoints = route.get("waypoints", [])
    if not waypoints:
        raise ValueError("case_artifact.json is missing route.waypoints")

    lats = [float(point[0]) for point in waypoints]
    lons = [float(point[1]) for point in waypoints]
    margin_lat = max(0.5, 0.08 * (max(lats) - min(lats) + 1e-6))
    margin_lon = max(0.5, 0.08 * (max(lons) - min(lons) + 1e-6))
    return {
        "lat_min": min(lats) - margin_lat,
        "lat_max": max(lats) + margin_lat,
        "lon_min": min(lons) - margin_lon,
        "lon_max": max(lons) + margin_lon,
    }


def _extract_map_config(case_artifact: dict) -> tuple[dict[str, float], float]:
    bounds = case_artifact.get("map_bounds") or _bounds_from_route(case_artifact)
    resolution = case_artifact.get("cost_map_resolution") or 0.1
    return {
        "lat_min": float(bounds["lat_min"]),
        "lat_max": float(bounds["lat_max"]),
        "lon_min": float(bounds["lon_min"]),
        "lon_max": float(bounds["lon_max"]),
    }, float(resolution)


def _build_target_grid(bounds: dict[str, float], resolution: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if resolution <= 0.0:
        raise ValueError("resolution must be positive")

    lats = np.arange(bounds["lat_min"], bounds["lat_max"] + resolution * 0.5, resolution, dtype=float)
    lons = np.arange(bounds["lon_min"], bounds["lon_max"] + resolution * 0.5, resolution, dtype=float)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return lats, lons, lat_grid, lon_grid


def _wind_barb_stride(n_lat: int, n_lon: int, max_density: int = 30) -> int:
    return max(1, int(np.ceil(max(n_lat, n_lon) / max_density)))


def _sample_wind_grid_from_loader(env_loader, bounds: dict[str, float], resolution: float, when_utc: datetime):
    lats, lons, lat_grid, lon_grid = _build_target_grid(bounds, resolution)

    lower, upper, time_weight = env_loader._time_bounds(when_utc)
    u_field = env_loader._u10[lower]
    v_field = env_loader._v10[lower]
    if lower != upper:
        u_field = (1.0 - time_weight) * env_loader._u10[lower] + time_weight * env_loader._u10[upper]
        v_field = (1.0 - time_weight) * env_loader._v10[lower] + time_weight * env_loader._v10[upper]

    interp_u = RegularGridInterpolator(
        (env_loader._era5_lats, env_loader._era5_lons),
        u_field,
        bounds_error=False,
        fill_value=np.nan,
    )
    interp_v = RegularGridInterpolator(
        (env_loader._era5_lats, env_loader._era5_lons),
        v_field,
        bounds_error=False,
        fill_value=np.nan,
    )

    points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    u_grid = interp_u(points).reshape(lat_grid.shape)
    v_grid = interp_v(points).reshape(lat_grid.shape)
    u_grid = np.nan_to_num(u_grid, nan=0.0)
    v_grid = np.nan_to_num(v_grid, nan=0.0)
    wind_speed = np.hypot(u_grid, v_grid)
    return lats, lons, lat_grid, lon_grid, u_grid, v_grid, wind_speed


def _sample_current_grid_from_loader(env_loader, bounds: dict[str, float], resolution: float, when_utc: datetime):
    lats, lons, lat_grid, lon_grid = _build_target_grid(bounds, resolution)

    lower, upper, time_weight = env_loader._time_bounds(when_utc)
    u_field = env_loader._uo[lower]
    v_field = env_loader._vo[lower]
    if lower != upper:
        u_field = (1.0 - time_weight) * env_loader._uo[lower] + time_weight * env_loader._uo[upper]
        v_field = (1.0 - time_weight) * env_loader._vo[lower] + time_weight * env_loader._vo[upper]

    interp_u = RegularGridInterpolator(
        (env_loader._cmems_lats, env_loader._cmems_lons),
        u_field,
        bounds_error=False,
        fill_value=np.nan,
    )
    interp_v = RegularGridInterpolator(
        (env_loader._cmems_lats, env_loader._cmems_lons),
        v_field,
        bounds_error=False,
        fill_value=np.nan,
    )

    points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    u_grid = interp_u(points).reshape(lat_grid.shape)
    v_grid = interp_v(points).reshape(lat_grid.shape)
    u_grid = np.nan_to_num(u_grid, nan=0.0)
    v_grid = np.nan_to_num(v_grid, nan=0.0)
    current_speed = np.hypot(u_grid, v_grid)
    return lats, lons, lat_grid, lon_grid, u_grid, v_grid, current_speed


def _resolve_wind_from_env(env) -> tuple[float, float]:
    if isinstance(env, EnvironmentData):
        wind_speed = float(env.wind_speed_ms)
        wind_dir = float(env.wind_dir_deg or 0.0)
        return wind_speed, wind_dir
    if isinstance(env, tuple) and len(env) == 2:
        return float(env[0]), float(env[1])
    raise TypeError("env_fn must return EnvironmentData or (wind_speed_ms, wind_dir_deg)")


def _sample_wind_grid_from_env_fn(env_fn, bounds: dict[str, float], resolution: float, when_utc: datetime):
    lats, lons, lat_grid, lon_grid = _build_target_grid(bounds, resolution)
    u_grid = np.zeros_like(lat_grid, dtype=float)
    v_grid = np.zeros_like(lat_grid, dtype=float)

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            wind_speed, wind_dir_deg = _resolve_wind_from_env(env_fn(float(lat), float(lon), when_utc))
            theta = np.radians(wind_dir_deg)
            u_grid[i, j] = -wind_speed * np.sin(theta)
            v_grid[i, j] = -wind_speed * np.cos(theta)

    wind_speed = np.hypot(u_grid, v_grid)
    return lats, lons, lat_grid, lon_grid, u_grid, v_grid, wind_speed


def _resolve_current_from_env(env) -> tuple[float, float]:
    if isinstance(env, EnvironmentData):
        current_speed = float(env.current_speed_ms)
        current_dir = float(env.current_dir_deg or 0.0)
        return current_speed, current_dir
    raise TypeError("env_fn must return EnvironmentData when plotting current")


def _sample_current_grid_from_env_fn(env_fn, bounds: dict[str, float], resolution: float, when_utc: datetime):
    lats, lons, lat_grid, lon_grid = _build_target_grid(bounds, resolution)
    u_grid = np.zeros_like(lat_grid, dtype=float)
    v_grid = np.zeros_like(lat_grid, dtype=float)

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            current_speed, current_dir_deg = _resolve_current_from_env(env_fn(float(lat), float(lon), when_utc))
            theta = np.radians(current_dir_deg)
            u_grid[i, j] = current_speed * np.sin(theta)
            v_grid[i, j] = current_speed * np.cos(theta)

    current_speed = np.hypot(u_grid, v_grid)
    return lats, lons, lat_grid, lon_grid, u_grid, v_grid, current_speed


def _route_points(case_artifact: dict) -> tuple[list[float], list[float]]:
    route = case_artifact["route"]
    waypoints = route["waypoints"]
    route_lats = [float(point[0]) for point in waypoints]
    route_lons = [float(point[1]) for point in waypoints]
    return route_lats, route_lons


def render_wind_route_map(
    case_artifact: dict,
    output_path: str,
    env_loader=None,
    env_fn=None,
    land_geometry=None,
    show_route: bool = True,
    show_ports: bool = True,
) -> str:
    bounds, resolution = _extract_map_config(case_artifact)
    when_utc = _parse_departure_time(case_artifact)

    if env_loader is not None and hasattr(env_loader, "_u10") and hasattr(env_loader, "_v10"):
        lats, lons, lat_grid, lon_grid, u_grid, v_grid, wind_speed = _sample_wind_grid_from_loader(
            env_loader, bounds, resolution, when_utc
        )
    elif env_fn is not None:
        lats, lons, lat_grid, lon_grid, u_grid, v_grid, wind_speed = _sample_wind_grid_from_env_fn(
            env_fn, bounds, resolution, when_utc
        )
    else:
        raise ValueError("Either env_loader or env_fn must be provided")

    if land_geometry is None:
        try:
            land_geometry = load_coastline(source="natural_earth", bounds=bounds)
        except Exception as exc:
            print(f"[WARN] Failed to load coastline geometry: {exc}")
            land_geometry = None

    route = case_artifact["route"]
    waypoints = route["waypoints"]
    route_lats = [float(point[0]) for point in waypoints]
    route_lons = [float(point[1]) for point in waypoints]

    fig = plt.figure(figsize=(16, 8.5))
    if CARTOPY_AVAILABLE:
        projection = ccrs.PlateCarree()
        ax = plt.axes(projection=projection)
        ax.set_extent(
            [bounds["lon_min"], bounds["lon_max"], bounds["lat_min"], bounds["lat_max"]],
            crs=projection,
        )
        ax.set_facecolor("#dfeaf7")
        mesh = ax.pcolormesh(
            lon_grid,
            lat_grid,
            wind_speed,
            cmap="YlGnBu",
            shading="auto",
            alpha=0.45,
            transform=projection,
            zorder=1,
        )

        if land_geometry is not None and hasattr(land_geometry, "geoms") and len(land_geometry.geoms) > 0:
            ax.add_geometries(
                land_geometry.geoms,
                crs=projection,
                facecolor="#f5efe2",
                edgecolor="#8d8574",
                linewidth=0.45,
                zorder=2,
            )

        barb_stride = _wind_barb_stride(len(lats), len(lons))
        ax.barbs(
            lon_grid[::barb_stride, ::barb_stride],
            lat_grid[::barb_stride, ::barb_stride],
            u_grid[::barb_stride, ::barb_stride],
            v_grid[::barb_stride, ::barb_stride],
            length=5.2,
            linewidth=0.5,
            color="#3f3f3f",
            transform=projection,
            zorder=4,
        )
        if show_route:
            ax.plot(
                route_lons,
                route_lats,
                color=ROUTE_COLOR,
                linewidth=2.7,
                transform=projection,
                zorder=6,
            )
            ax.scatter(
                route_lons,
                route_lats,
                s=30,
                facecolor=ROUTE_NODE_FACE_COLOR,
                edgecolor=ROUTE_NODE_EDGE_COLOR,
                linewidth=0.45,
                transform=projection,
                zorder=7,
            )
        if show_ports:
            ax.scatter(
                [route_lons[0], route_lons[-1]],
                [route_lats[0], route_lats[-1]],
                s=135,
                facecolor=ROUTE_NODE_FACE_COLOR,
                edgecolor=PORT_NODE_EDGE_COLOR,
                linewidth=1.5,
                transform=projection,
                zorder=8,
            )
        ax.gridlines(
            crs=projection,
            draw_labels=True,
            linewidth=0.35,
            color="#8b93a0",
            alpha=0.35,
            linestyle="--",
        )
    else:
        ax = fig.add_subplot(111)
        ax.set_facecolor("#dfeaf7")
        ax.set_xlim(bounds["lon_min"], bounds["lon_max"])
        ax.set_ylim(bounds["lat_min"], bounds["lat_max"])
        mesh = ax.pcolormesh(
            lon_grid,
            lat_grid,
            wind_speed,
            cmap="YlGnBu",
            shading="auto",
            alpha=0.45,
            zorder=1,
        )

        if land_geometry is not None and hasattr(land_geometry, "geoms"):
            for geom in land_geometry.geoms:
                x, y = geom.exterior.xy
                ax.fill(x, y, facecolor="#f5efe2", edgecolor="#8d8574", linewidth=0.45, zorder=2)

        barb_stride = _wind_barb_stride(len(lats), len(lons))
        ax.barbs(
            lon_grid[::barb_stride, ::barb_stride],
            lat_grid[::barb_stride, ::barb_stride],
            u_grid[::barb_stride, ::barb_stride],
            v_grid[::barb_stride, ::barb_stride],
            length=5.2,
            linewidth=0.5,
            color="#3f3f3f",
            zorder=4,
        )
        if show_route:
            ax.plot(route_lons, route_lats, color=ROUTE_COLOR, linewidth=2.7, zorder=6)
            ax.scatter(
                route_lons,
                route_lats,
                s=30,
                facecolor=ROUTE_NODE_FACE_COLOR,
                edgecolor=ROUTE_NODE_EDGE_COLOR,
                linewidth=0.45,
                zorder=7,
            )
        if show_ports:
            ax.scatter(
                [route_lons[0], route_lons[-1]],
                [route_lats[0], route_lats[-1]],
                s=135,
                facecolor=ROUTE_NODE_FACE_COLOR,
                edgecolor=PORT_NODE_EDGE_COLOR,
                linewidth=1.5,
                zorder=8,
            )
        ax.grid(True, linewidth=0.35, color="#8b93a0", alpha=0.35, linestyle="--")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    title = "Wind Raster and Wind Barbs"
    if show_route:
        title += " with Optimal Route"
    ax.set_title(title, fontsize=14, fontweight="bold")

    cbar = fig.colorbar(mesh, ax=ax, shrink=0.76, pad=0.03)
    cbar.set_label("Wind Speed (m/s)")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_current_route_map(
    case_artifact: dict,
    output_path: str,
    env_loader=None,
    env_fn=None,
    land_geometry=None,
    vector_scale: float = 22.0,
    show_route: bool = True,
    show_ports: bool = True,
) -> str:
    bounds, resolution = _extract_map_config(case_artifact)
    when_utc = _parse_departure_time(case_artifact)

    if env_loader is not None and hasattr(env_loader, "_uo") and hasattr(env_loader, "_vo"):
        lats, lons, lat_grid, lon_grid, u_grid, v_grid, current_speed = _sample_current_grid_from_loader(
            env_loader, bounds, resolution, when_utc
        )
    elif env_fn is not None:
        lats, lons, lat_grid, lon_grid, u_grid, v_grid, current_speed = _sample_current_grid_from_env_fn(
            env_fn, bounds, resolution, when_utc
        )
    else:
        raise ValueError("Either env_loader or env_fn must be provided")

    if land_geometry is None:
        try:
            land_geometry = load_coastline(source="natural_earth", bounds=bounds)
        except Exception as exc:
            print(f"[WARN] Failed to load coastline geometry: {exc}")
            land_geometry = None

    route_lats, route_lons = _route_points(case_artifact)

    fig = plt.figure(figsize=(16, 8.5))
    if CARTOPY_AVAILABLE:
        projection = ccrs.PlateCarree()
        ax = plt.axes(projection=projection)
        ax.set_extent(
            [bounds["lon_min"], bounds["lon_max"], bounds["lat_min"], bounds["lat_max"]],
            crs=projection,
        )
        ax.set_facecolor("#dbeef3")
        mesh = ax.pcolormesh(
            lon_grid,
            lat_grid,
            current_speed,
            cmap="PuBuGn",
            shading="auto",
            alpha=0.68,
            transform=projection,
            zorder=1,
        )

        if land_geometry is not None and hasattr(land_geometry, "geoms") and len(land_geometry.geoms) > 0:
            ax.add_geometries(
                land_geometry.geoms,
                crs=projection,
                facecolor="#f4ebd8",
                edgecolor="#746d5e",
                linewidth=0.45,
                zorder=2,
            )

        vector_stride = _wind_barb_stride(len(lats), len(lons), max_density=26)
        ax.quiver(
            lon_grid[::vector_stride, ::vector_stride],
            lat_grid[::vector_stride, ::vector_stride],
            u_grid[::vector_stride, ::vector_stride],
            v_grid[::vector_stride, ::vector_stride],
            color="#08306b",
            alpha=0.72,
            width=0.0017,
            scale=vector_scale,
            transform=projection,
            zorder=4,
        )
        if show_route:
            ax.plot(
                route_lons,
                route_lats,
                color=ROUTE_COLOR,
                linewidth=2.7,
                transform=projection,
                zorder=6,
            )
            ax.scatter(
                route_lons,
                route_lats,
                s=30,
                facecolor=ROUTE_NODE_FACE_COLOR,
                edgecolor=ROUTE_NODE_EDGE_COLOR,
                linewidth=0.45,
                transform=projection,
                zorder=7,
            )
        if show_ports:
            ax.scatter(
                [route_lons[0], route_lons[-1]],
                [route_lats[0], route_lats[-1]],
                s=135,
                facecolor=ROUTE_NODE_FACE_COLOR,
                edgecolor=PORT_NODE_EDGE_COLOR,
                linewidth=1.5,
                transform=projection,
                zorder=8,
            )
        ax.gridlines(
            crs=projection,
            draw_labels=True,
            linewidth=0.35,
            color="#52606d",
            alpha=0.35,
            linestyle="--",
        )
    else:
        ax = fig.add_subplot(111)
        ax.set_facecolor("#dbeef3")
        ax.set_xlim(bounds["lon_min"], bounds["lon_max"])
        ax.set_ylim(bounds["lat_min"], bounds["lat_max"])
        mesh = ax.pcolormesh(
            lon_grid,
            lat_grid,
            current_speed,
            cmap="PuBuGn",
            shading="auto",
            alpha=0.68,
            zorder=1,
        )

        if land_geometry is not None and hasattr(land_geometry, "geoms"):
            for geom in land_geometry.geoms:
                x, y = geom.exterior.xy
                ax.fill(x, y, facecolor="#f4ebd8", edgecolor="#746d5e", linewidth=0.45, zorder=2)

        vector_stride = _wind_barb_stride(len(lats), len(lons), max_density=26)
        ax.quiver(
            lon_grid[::vector_stride, ::vector_stride],
            lat_grid[::vector_stride, ::vector_stride],
            u_grid[::vector_stride, ::vector_stride],
            v_grid[::vector_stride, ::vector_stride],
            color="#08306b",
            alpha=0.72,
            width=0.0017,
            scale=vector_scale,
            zorder=4,
        )
        if show_route:
            ax.plot(route_lons, route_lats, color=ROUTE_COLOR, linewidth=2.7, zorder=6)
            ax.scatter(
                route_lons,
                route_lats,
                s=30,
                facecolor=ROUTE_NODE_FACE_COLOR,
                edgecolor=ROUTE_NODE_EDGE_COLOR,
                linewidth=0.45,
                zorder=7,
            )
        if show_ports:
            ax.scatter(
                [route_lons[0], route_lons[-1]],
                [route_lats[0], route_lats[-1]],
                s=135,
                facecolor=ROUTE_NODE_FACE_COLOR,
                edgecolor=PORT_NODE_EDGE_COLOR,
                linewidth=1.5,
                zorder=8,
            )
        ax.grid(True, linewidth=0.35, color="#52606d", alpha=0.35, linestyle="--")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    title = "Ocean Current Speed and Current Vectors"
    if show_route:
        title += " with Optimal Route"
    ax.set_title(title, fontsize=14, fontweight="bold")

    cbar = fig.colorbar(mesh, ax=ax, shrink=0.76, pad=0.03)
    cbar.set_label("Current Speed (m/s)")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = _build_parser().parse_args()
    case_artifact = _load_case_artifact(args.case_json)
    output_path = _resolve_output_path(args.case_json, args.output, args.field)
    departure_time = _parse_departure_time(case_artifact)
    era5_path = args.era5_path or _find_dataset_for_departure("era5", departure_time)
    cmems_path = args.cmems_path or _find_dataset_for_departure("cmems", departure_time)
    show_route = not (args.hide_route or args.weather_only)
    show_ports = not (args.hide_ports or args.weather_only)

    env_loader, env_fn, _ = load_marine_environment(era5_path=era5_path, cmems_path=cmems_path)
    try:
        if args.field == "current":
            saved_path = render_current_route_map(
                case_artifact=case_artifact,
                output_path=output_path,
                env_loader=env_loader,
                env_fn=env_fn,
                vector_scale=args.current_vector_scale,
                show_route=show_route,
                show_ports=show_ports,
            )
        else:
            saved_path = render_wind_route_map(
                case_artifact=case_artifact,
                output_path=output_path,
                env_loader=env_loader,
                env_fn=env_fn,
                show_route=show_route,
                show_ports=show_ports,
            )
    finally:
        del env_loader

    print(f"[SAVED] {saved_path}")


if __name__ == "__main__":
    main()
