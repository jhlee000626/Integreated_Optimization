"""
Visualize multiple saved case_artifact.json routes on a single Cartopy map.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon, Polygon

try:
    import cartopy.crs as ccrs

    CARTOPY_AVAILABLE = True
except ModuleNotFoundError:
    ccrs = None
    CARTOPY_AVAILABLE = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.grid.no_go_zone import load_coastline

DEFAULT_CASES_ROOT = os.path.join(PROJECT_ROOT, "output", "verification")
ROUTE_COLORS = [
    "#d32f2f",
    "#1976d2",
    "#2e7d32",
    "#f57c00",
    "#7b1fa2",
    "#00838f",
    "#5d4037",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-json",
        action="append",
        default=[],
        help="Path to a case_artifact.json file. Repeat to overlay multiple cases.",
    )
    parser.add_argument(
        "--cases-root",
        default=DEFAULT_CASES_ROOT,
        help="Directory that contains saved case subdirectories.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for case_artifact.json under --cases-root.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to saved_case_routes_cartopy.png under --cases-root.",
    )
    parser.add_argument(
        "--title",
        default="Saved Optimal Route Comparison",
        help="Figure title.",
    )
    return parser


def _discover_case_json_paths(cases_root: str, recursive: bool) -> list[str]:
    if recursive:
        pattern = os.path.join(os.path.abspath(cases_root), "**", "case_artifact.json")
        return sorted(glob.glob(pattern, recursive=True))

    pattern = os.path.join(os.path.abspath(cases_root), "*", "case_artifact.json")
    return sorted(glob.glob(pattern))


def _load_case_artifact(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    artifact.setdefault("case_name", os.path.basename(os.path.dirname(os.path.abspath(path))))
    return artifact


def _extract_waypoints(case_artifact: dict) -> list[tuple[float, float]]:
    route = case_artifact.get("route", {})
    waypoints = route.get("waypoints", [])
    if not waypoints:
        raise ValueError(f"{case_artifact.get('case_name', 'unknown')} is missing route.waypoints")
    return [(float(point[0]), float(point[1])) for point in waypoints]


def _bounds_from_waypoints(waypoints: list[tuple[float, float]]) -> dict[str, float]:
    lats = [lat for lat, _ in waypoints]
    lons = [lon for _, lon in waypoints]
    lat_span = max(lats) - min(lats)
    lon_span = max(lons) - min(lons)
    margin_lat = max(0.4, 0.08 * (lat_span + 1e-6))
    margin_lon = max(0.4, 0.08 * (lon_span + 1e-6))
    return {
        "lat_min": min(lats) - margin_lat,
        "lat_max": max(lats) + margin_lat,
        "lon_min": min(lons) - margin_lon,
        "lon_max": max(lons) + margin_lon,
    }


def _bounds_from_artifact(case_artifact: dict) -> dict[str, float]:
    bounds = case_artifact.get("map_bounds")
    if bounds:
        return {
            "lat_min": float(bounds["lat_min"]),
            "lat_max": float(bounds["lat_max"]),
            "lon_min": float(bounds["lon_min"]),
            "lon_max": float(bounds["lon_max"]),
        }
    return _bounds_from_waypoints(_extract_waypoints(case_artifact))


def _combine_bounds(case_artifacts: list[dict]) -> dict[str, float]:
    bounds_list = [_bounds_from_artifact(case_artifact) for case_artifact in case_artifacts]
    lat_min = min(bounds["lat_min"] for bounds in bounds_list)
    lat_max = max(bounds["lat_max"] for bounds in bounds_list)
    lon_min = min(bounds["lon_min"] for bounds in bounds_list)
    lon_max = max(bounds["lon_max"] for bounds in bounds_list)

    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min
    margin_lat = max(0.3, 0.04 * (lat_span + 1e-6))
    margin_lon = max(0.3, 0.04 * (lon_span + 1e-6))
    return {
        "lat_min": lat_min - margin_lat,
        "lat_max": lat_max + margin_lat,
        "lon_min": lon_min - margin_lon,
        "lon_max": lon_max + margin_lon,
    }


def _resolve_output_path(cases_root: str, output_path: str | None) -> str:
    if output_path:
        return os.path.abspath(output_path)
    return os.path.join(os.path.abspath(cases_root), "saved_case_routes_cartopy.png")


def _iter_polygon_geometries(land_geometry):
    if land_geometry is None:
        return []
    if isinstance(land_geometry, Polygon):
        return [land_geometry]
    if isinstance(land_geometry, MultiPolygon):
        return list(land_geometry.geoms)
    if hasattr(land_geometry, "geoms"):
        return [geom for geom in land_geometry.geoms if isinstance(geom, Polygon)]
    return []


def _draw_land_fallback(ax, land_geometry) -> None:
    for geom in _iter_polygon_geometries(land_geometry):
        x_coords, y_coords = geom.exterior.xy
        ax.fill(
            x_coords,
            y_coords,
            facecolor="#efe7da",
            edgecolor="#8d8574",
            linewidth=0.45,
            zorder=1,
        )


def render_saved_case_routes_map(
    case_artifacts: list[dict],
    output_path: str,
    title: str = "Saved Optimal Route Comparison",
    land_geometry=None,
    use_cartopy: bool | None = None,
) -> str:
    if not case_artifacts:
        raise ValueError("At least one case artifact is required")

    if use_cartopy is None:
        use_cartopy = CARTOPY_AVAILABLE
    if use_cartopy and not CARTOPY_AVAILABLE:
        raise RuntimeError("Cartopy is not installed but use_cartopy=True was requested")

    bounds = _combine_bounds(case_artifacts)
    if land_geometry is None:
        try:
            land_geometry = load_coastline(source="natural_earth", bounds=bounds)
        except Exception as exc:
            print(f"[WARN] Failed to load coastline geometry: {exc}")
            land_geometry = None

    fig = plt.figure(figsize=(15.5, 9))
    projection = ccrs.PlateCarree() if use_cartopy else None

    if use_cartopy:
        ax = plt.axes(projection=projection)
        ax.set_extent(
            [bounds["lon_min"], bounds["lon_max"], bounds["lat_min"], bounds["lat_max"]],
            crs=projection,
        )
        ax.set_facecolor("#dfeaf7")
        if land_geometry is not None and hasattr(land_geometry, "geoms") and len(land_geometry.geoms) > 0:
            ax.add_geometries(
                land_geometry.geoms,
                crs=projection,
                facecolor="#efe7da",
                edgecolor="#8d8574",
                linewidth=0.45,
                zorder=1,
            )
        gridliner = ax.gridlines(
            crs=projection,
            draw_labels=True,
            linewidth=0.35,
            color="#8b93a0",
            alpha=0.45,
            linestyle="--",
            x_inline=False,
            y_inline=False,
        )
        gridliner.top_labels = False
        gridliner.right_labels = False
    else:
        ax = fig.add_subplot(111)
        ax.set_facecolor("#dfeaf7")
        ax.set_xlim(bounds["lon_min"], bounds["lon_max"])
        ax.set_ylim(bounds["lat_min"], bounds["lat_max"])
        _draw_land_fallback(ax, land_geometry)
        ax.grid(True, linewidth=0.35, color="#8b93a0", alpha=0.45, linestyle="--")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    first_waypoints = _extract_waypoints(case_artifacts[0])
    marker_start = first_waypoints[0]
    marker_end = first_waypoints[-1]

    for index, case_artifact in enumerate(case_artifacts):
        case_name = str(case_artifact.get("case_name", f"case_{index + 1}"))
        waypoints = _extract_waypoints(case_artifact)
        route_lats = [lat for lat, _ in waypoints]
        route_lons = [lon for _, lon in waypoints]
        color = ROUTE_COLORS[index % len(ROUTE_COLORS)]
        transform = {"transform": projection} if use_cartopy else {}

        ax.plot(
            route_lons,
            route_lats,
            color=color,
            linewidth=2.5,
            solid_capstyle="round",
            label=case_name,
            zorder=4,
            **transform,
        )
        ax.scatter(
            route_lons,
            route_lats,
            s=28,
            color=color,
            edgecolor=color,
            linewidth=0.8,
            zorder=5,
            **transform,
        )
        # ax.text(
        #     route_lons[-1],
        #     route_lats[-1],
        #     f" {case_name}",
        #     fontsize=9,
        #     color=color,
        #     fontweight="bold",
        #     bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "alpha": 0.78, "edgecolor": color},
        #     zorder=6,
        #     **transform,
        

    marker_transform = {"transform": projection} if use_cartopy else {}
    ax.scatter(
        [marker_start[1]],
        [marker_start[0]],
        s=170,
        color="#43a047",
        edgecolor="white",
        linewidth=1.2,
        marker="*",
        label="Start",
        zorder=7,
        **marker_transform,
    )
    ax.scatter(
        [marker_end[1]],
        [marker_end[0]],
        s=170,
        color="#e53935",
        edgecolor="white",
        linewidth=1.2,
        marker="*",
        label="Destination",
        zorder=7,
        **marker_transform,
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9, frameon=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = _build_parser().parse_args()

    case_json_paths = [os.path.abspath(path) for path in args.case_json]
    if not case_json_paths:
        case_json_paths = _discover_case_json_paths(args.cases_root, recursive=args.recursive)
    if not case_json_paths:
        raise FileNotFoundError(
            f"No case_artifact.json files found under {os.path.abspath(args.cases_root)}"
        )

    case_artifacts = [_load_case_artifact(path) for path in case_json_paths]
    output_path = _resolve_output_path(args.cases_root, args.output)

    if not CARTOPY_AVAILABLE:
        print("[WARN] Cartopy is not installed. Falling back to a plain Matplotlib map.")

    saved_path = render_saved_case_routes_map(
        case_artifacts=case_artifacts,
        output_path=output_path,
        title=args.title,
    )
    print(f"[SAVED] {saved_path}")


if __name__ == "__main__":
    main()
