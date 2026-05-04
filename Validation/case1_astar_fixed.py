"""
Verification Case 1

Fixed-speed A* baseline:
- build an A* route on the land-mask grid
- interpolate to the voyage segment count
- run MILP scheduling on the fixed route
"""

from __future__ import annotations

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Validation.common import (
    VERIFICATION_COST_MAP_RESOLUTION,
    build_case_metrics,
    ensure_output_dir,
    load_marine_environment,
    make_milp_solver,
    save_case_artifacts,
    validate_route,
    get_or_build_astar_seed,
)
from src.grid.cost_map import build_cost_map
from src.grid.no_go_zone import BUSAN_PORT, SHANGHAI_PORT
from src.optimizer.ga_engine import N_SEGMENTS, RTA_HOURS, build_required_power_profile, compute_heading, haversine_nm
from src.visualization.plotter import (
    plot_optimal_route,
    plot_power_schedule,
    plot_raw_astar_only,
    plot_weather_map,
)


DT_HOURS = RTA_HOURS / N_SEGMENTS
CASE1_PATH_DISTANCE_MODE = "grid"
VOYAGE_START_PORT = SHANGHAI_PORT
VOYAGE_END_PORT = BUSAN_PORT


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    smoothing_group = parser.add_mutually_exclusive_group()
    smoothing_group.add_argument(
        "--astar-smoothing",
        dest="astar_smoothing",
        action="store_true",
        help="Apply waypoint smoothing to the cached/generated A* route.",
    )
    smoothing_group.add_argument(
        "--no-astar-smoothing",
        dest="astar_smoothing",
        action="store_false",
        help="Disable waypoint smoothing and use the compressed A* waypoints directly.",
    )
    parser.add_argument("--era5-path", type=str, default=None, help="Explicit path to ERA5 netCDF dataset")
    parser.add_argument("--cmems-path", type=str, default=None, help="Explicit path to CMEMS netCDF dataset")
    parser.set_defaults(astar_smoothing=False)
    return parser


def compute_route_distance_nm(waypoints) -> float:
    return sum(haversine_nm(waypoints[idx], waypoints[idx + 1]) for idx in range(len(waypoints) - 1))


def compute_case1_base_speed_knots(waypoints, rta_h: float = RTA_HOURS) -> float:
    if rta_h <= 0.0:
        return 0.0
    return compute_route_distance_nm(waypoints) / rta_h


def build_fixed_route(
    waypoints,
    base_speed_knots,
    n_segments: int = N_SEGMENTS,
    rta_h: float = RTA_HOURS,
    start_port=BUSAN_PORT,
    end_port=SHANGHAI_PORT,
):
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

    prev_heading = compute_heading(start_port, end_port)
    for idx in range(n_segments):
        wp_from = waypoints[idx]
        wp_to = waypoints[idx + 1]
        heading = compute_heading(wp_from, wp_to)
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


def apply_case1_baseline_validation(route, cost_map, env_fn, departure_time_utc):
    validation = validate_route(route, cost_map, env_fn, departure_time_utc)
    route["valid_departure_heading"] = True
    route["valid_turning"] = True
    route["valid_speed"] = True
    route["valid_heading"] = True
    route["valid"] = route["land_violation"] <= 0.0
    return validation


def extract_schedule_dg_names(schedule: list[dict]) -> list[str]:
    if not schedule:
        return []

    dg_names = []
    for key in schedule[0]:
        if key.startswith("DG") and key.endswith("_P_MW"):
            dg_names.append(key[:-5])

    def sort_key(name: str):
        suffix = name[2:]
        if suffix.isdigit():
            return (0, int(suffix))
        return (1, suffix)

    return sorted(dg_names, key=sort_key)


def print_schedule_log(milp_result: dict):
    schedule = milp_result.get("schedule", [])
    if not schedule:
        print("  MILP schedule: empty")
        return

    dg_names = extract_schedule_dg_names(schedule)
    summary = milp_result.get("summary", {})
    dg_hours = summary.get("dg_running_hours", {})
    dg_fuel = summary.get("dg_fuel_kg", {})
    dg_starts = summary.get("dg_start_count", {})

    print("  DG summary:")
    for dg_name in dg_names:
        print(
            f"    {dg_name}: "
            f"hours={dg_hours.get(dg_name, 0.0):.2f} h, "
            f"fuel={dg_fuel.get(dg_name, 0.0):.2f} kg, "
            f"starts={dg_starts.get(dg_name, 0)}"
        )

    print("  Schedule by step:")
    for step in schedule:
        dg_parts = []
        for dg_name in dg_names:
            dg_parts.append(
                f"{dg_name}={step.get(f'{dg_name}_P_MW', 0.0):5.2f} MW"
                f"(ON={step.get(f'{dg_name}_ON', 0)}, Start={step.get(f'{dg_name}_Start', 0)})"
            )

        print(
            f"    t={step['t']:02d} | "
            f"P_req={step['P_req_MW']:5.2f} MW | "
            f"P_dc={step['P_dc_MW']:5.2f} MW | "
            f"P_c={step['P_c_MW']:5.2f} MW | "
            f"SOC={step['SOC'] * 100:5.1f}% | "
            + " | ".join(dg_parts)
        )


def main():
    args = _build_parser().parse_args()
    start_time = time.time()
    print("=" * 60)
    print(" [Verification Case 1] A* Fixed Route + MILP Scheduling ")
    print("=" * 60)

    out_dir = ensure_output_dir("case1")
    env_loader, env_fn, departure_time_utc = load_marine_environment(
        era5_path=args.era5_path,
        cmems_path=args.cmems_path,
    )
    try:
        cost_map = build_cost_map(resolution=VERIFICATION_COST_MAP_RESOLUTION)
        raw_path, waypoints, _ = get_or_build_astar_seed(
            cost_map,
            VOYAGE_START_PORT,
            VOYAGE_END_PORT,
            N_SEGMENTS,
            distance_mode=CASE1_PATH_DISTANCE_MODE,
            apply_smoothing=args.astar_smoothing,
        )
        total_dist_nm = compute_route_distance_nm(waypoints)
        base_speed = compute_case1_base_speed_knots(waypoints, rta_h=RTA_HOURS)
        base_speed = round(base_speed, 1)  # 사용자 요청: 고정 속도를 소수점 첫째 자리까지만 사용

        route = build_fixed_route(
            waypoints,
            base_speed,
            start_port=VOYAGE_START_PORT,
            end_port=VOYAGE_END_PORT,
        )
        validation = apply_case1_baseline_validation(route, cost_map, env_fn, departure_time_utc)
        power_profile = build_required_power_profile(
            route,
            env_fn=env_fn,
            departure_time_utc=departure_time_utc,
        )
        p_req_list = [segment["P_req"] for segment in power_profile]

        milp = make_milp_solver()
        milp_result = milp.solve(P_req=p_req_list, dt=route["dt"], initial_SOC=0.7, msg=False)
        runtime = time.time() - start_time

        print(f"  Raw path points: {len(raw_path)}")
        print(f"  Resampled waypoints: {len(waypoints)}")
        print(f"  Physical route distance: {total_dist_nm:.2f} nm")
        print(f"  Path metric: {CASE1_PATH_DISTANCE_MODE}")
        print(f"  A* smoothing: {'on' if args.astar_smoothing else 'off'}")
        print(f"  Base speed: {base_speed:.2f} kts")
        print(
            "  Route validity: "
            f"overall={route['valid']} "
            f"departure={route['valid_departure_heading']} "
            f"last_speed={route['valid_speed']} "
            f"land={(route['land_violation'] <= 0.0)}"
        )
        print(f"  Land violation: {validation.land_violation:.1f}")
        print(f"  Turning (diagnostic): {route['valid_turning']}")
        print(f"  Final heading ok (diagnostic): {route['valid_heading']}")
        print(f"  Last STW: {route['last_speed']:.2f} kts | Last SOG: {route['last_speed_sog']:.2f} kts")

        if milp_result["feasible"]:
            print(f"  MILP fuel: {milp_result['total_fuel_kg']:.2f} kg")
            print_schedule_log(milp_result)
        else:
            print("  MILP infeasible")

        metrics = build_case_metrics(
            case_name="case1_astar_fixed",
            route=route,
            power_profile=power_profile,
            objective=milp_result["total_fuel_kg"] if milp_result["feasible"] else None,
            objective_kind="milp_fuel_kg",
            milp_result=milp_result,
            route_violation=validation.land_violation,
            runtime_sec=runtime,
            note=(
                f"astar_res={VERIFICATION_COST_MAP_RESOLUTION}, "
                f"smoothing={'on' if args.astar_smoothing else 'off'}"
            ),
        )
        artifact_paths = save_case_artifacts(
            case_name="case1_astar_fixed",
            output_dir=out_dir,
            route=route,
            power_profile=power_profile,
            milp_result=milp_result,
            departure_time_utc=departure_time_utc,
            cost_map=cost_map,
            metrics=metrics,
        )

        plot_optimal_route(route, cost_map, save_dir=out_dir, raw_path_wps=raw_path)
        plot_raw_astar_only(raw_path, cost_map, save_dir=out_dir)
        # plot_weather_map(
        #     env_fn,
        #     cost_map,
        #     route_wps=route["waypoints"],
        #     save_dir=out_dir,
        #     when_utc=departure_time_utc,
        # )
        if milp_result["feasible"]:
            plot_power_schedule(milp_result, power_profile=power_profile, save_dir=out_dir)
        print(f"  Case artifact: {artifact_paths['case_artifact']}")
    finally:
        del env_loader


if __name__ == "__main__":
    main()
