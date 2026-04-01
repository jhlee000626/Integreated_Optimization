"""
Verification Case 1

Fixed-speed A* baseline:
- build an A* route on the land-mask grid
- interpolate to the voyage segment count
- run MILP scheduling on the fixed route
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Verification.common import (
    VERIFICATION_COST_MAP_RESOLUTION,
    ensure_output_dir,
    load_marine_environment,
    make_milp_solver,
    validate_route,
)
from src.grid.cost_map import build_cost_map
from src.grid.pathfinding import build_astar_route_points
from src.grid.no_go_zone import BUSAN_PORT, JEJU_PORT
from src.optimizer.ga_engine import N_SEGMENTS, RTA_HOURS, build_required_power_profile, compute_heading, haversine_nm
from src.visualization.plotter import plot_optimal_route, plot_power_schedule, plot_weather_map


DT_HOURS = RTA_HOURS / N_SEGMENTS


def build_fixed_route(waypoints, base_speed_knots, n_segments: int = N_SEGMENTS, rta_h: float = RTA_HOURS):
    dt_hours = rta_h / n_segments
    route = {
        "waypoints": waypoints,
        "speeds": [],
        "speeds_sog": [],
        "speeds_stw": [],
        "headings": [],
        "delta_headings": [],
        "dt": [dt_hours] * n_segments,
        "distances_nm": [],
        "valid": False,
        "valid_departure_heading": False,
        "valid_turning": False,
        "valid_speed": False,
        "valid_heading": False,
        "land_violation": 0.0,
        "last_speed": float("nan"),
        "last_speed_sog": 0.0,
        "final_heading_delta": 0.0,
    }

    prev_heading = compute_heading(BUSAN_PORT, JEJU_PORT)
    for idx in range(n_segments):
        wp_from = waypoints[idx]
        wp_to = waypoints[idx + 1]
        heading = compute_heading(wp_from, wp_to)
        if idx == 0 or idx == n_segments - 1:
            speed = base_speed_knots * 0.7
        else:
            speed = base_speed_knots
        delta = ((heading - prev_heading + 180.0) % 360.0) - 180.0

        route["speeds"].append(speed)
        route["speeds_sog"].append(speed)
        route["headings"].append(heading)
        route["delta_headings"].append(delta)
        route["distances_nm"].append(haversine_nm(wp_from, wp_to))
        prev_heading = heading

    route["last_speed_sog"] = route["speeds"][-1]
    route["final_heading_delta"] = route["delta_headings"][-1]
    route["nodes"] = [
        {
            "lat": wp[0],
            "lon": wp[1],
            "time_h": idx * dt_hours,
            "time_utc": None,
            "speed_out_kts": route["speeds"][idx] if idx < n_segments else None,
            "speed_over_ground_kts": route["speeds"][idx] if idx < n_segments else None,
            "speed_through_water_kts": None,
            "heading_out_deg": route["headings"][idx] if idx < n_segments else None,
        }
        for idx, wp in enumerate(route["waypoints"])
    ]
    return route


def main():
    print("=" * 60)
    print(" [Verification Case 1] A* Fixed Route + MILP Scheduling ")
    print("=" * 60)

    out_dir = ensure_output_dir("case1")
    env_loader, env_fn, departure_time_utc = load_marine_environment()
    try:
        cost_map = build_cost_map(resolution=VERIFICATION_COST_MAP_RESOLUTION)
        raw_path, waypoints, total_dist_nm = build_astar_route_points(cost_map, BUSAN_PORT, JEJU_PORT, N_SEGMENTS)
        base_speed = total_dist_nm / (DT_HOURS * (N_SEGMENTS - 0.6))

        route = build_fixed_route(waypoints, base_speed)
        validation = validate_route(route, cost_map, env_fn, departure_time_utc)
        power_profile = build_required_power_profile(
            route,
            env_fn=env_fn,
            departure_time_utc=departure_time_utc,
        )
        p_req_list = [segment["P_req"] for segment in power_profile]

        milp = make_milp_solver()
        milp_result = milp.solve(P_req=p_req_list, dt=route["dt"], initial_SOC=0.7, msg=False)

        print(f"  Raw path points: {len(raw_path)}")
        print(f"  Resampled waypoints: {len(waypoints)}")
        print(f"  Resampled distance: {total_dist_nm:.2f} nm")
        print(f"  Base speed: {base_speed:.2f} kts")
        print(
            "  Route validity: "
            f"overall={route['valid']} "
            f"departure={route['valid_departure_heading']} "
            f"turning={route['valid_turning']} "
            f"last_speed={route['valid_speed']} "
            f"last_heading={route['valid_heading']}"
        )
        print(f"  Land violation: {validation.land_violation:.1f}")
        print(f"  Last STW: {route['last_speed']:.2f} kts | Last SOG: {route['last_speed_sog']:.2f} kts")

        if milp_result["feasible"]:
            print(f"  MILP fuel: {milp_result['total_fuel_kg']:.2f} kg")
        else:
            print("  MILP infeasible")

        plot_optimal_route(route, cost_map, save_dir=out_dir)
        plot_weather_map(
            env_fn,
            cost_map,
            route_wps=route["waypoints"],
            save_dir=out_dir,
            when_utc=departure_time_utc,
        )
        if milp_result["feasible"]:
            plot_power_schedule(milp_result, save_dir=out_dir)
    finally:
        del env_loader


if __name__ == "__main__":
    main()
