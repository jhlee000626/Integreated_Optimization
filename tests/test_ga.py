"""
Manual GA integration verification script.

This file is a scenario runner, not a pytest-style unit test.
Run it directly with `python tests/test_ga.py`.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.grid.cost_map import build_cost_map
from src.grid.no_go_zone import BUSAN_PORT, SHANGHAI_PORT
from src.optimizer.ga_engine import (
    N_SEGMENTS,
    RTA_HOURS,
    build_required_power_profile,
    compute_violation,
    haversine_nm,
    setup_ga,
)
from src.optimizer.milp_solver import MILPSolver
from src.visualization.plotter import (
    plot_convergence,
    plot_cost_map_route,
    plot_optimal_route,
    plot_power_schedule,
    plot_weather_map,
)
from src.weather import MarineEnvironmentLoader, resolve_marine_dataset_paths


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SFOC_PATH = os.path.join(
    PROJECT_ROOT,
    "config",
    "sfoc.json",
)
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "output",
)


def main():
    t0 = time.time()

    print("=" * 60)
    print(" [Phase 1] Cost Map build")
    print("=" * 60)
    cmap = build_cost_map(resolution=0.01)

    print("\n" + "=" * 60)
    print(" [Phase 2] Marine environment setup")
    print("=" * 60)
    era5_nc_path, cmems_nc_path = resolve_marine_dataset_paths(PROJECT_ROOT)
    env_loader = MarineEnvironmentLoader(era5_nc_path, cmems_nc_path)
    env_fn = env_loader.get_environment_fn()
    departure_time_utc = env_loader.available_times_utc[0]
    print(f"  ERA5 file: {os.path.basename(era5_nc_path)}")
    print(f"  CMEMS file: {os.path.basename(cmems_nc_path)}")

    busan_env = env_fn(*BUSAN_PORT, departure_time_utc)
    print(
        f"  Busan env: wind {busan_env.wind_speed_ms:.1f} m/s @ {busan_env.wind_dir_deg:.0f} deg | "
        f"current {busan_env.current_speed_ms:.2f} m/s @ {busan_env.current_dir_deg:.0f} deg | "
        f"wave Hs {busan_env.wave_height_m:.2f} m"
    )
    shanghai_env = env_fn(*SHANGHAI_PORT, departure_time_utc)
    print(
        f"  Shanghai env: wind {shanghai_env.wind_speed_ms:.1f} m/s @ {shanghai_env.wind_dir_deg:.0f} deg | "
        f"current {shanghai_env.current_speed_ms:.2f} m/s @ {shanghai_env.current_dir_deg:.0f} deg | "
        f"wave Hs {shanghai_env.wave_height_m:.2f} m"
    )
    print(f"  Departure time (UTC): {departure_time_utc.isoformat()}")

    print("\n" + "=" * 60)
    print(" [Phase 3] MILP setup")
    print("=" * 60)
    milp = MILPSolver(sfoc_json_path=SFOC_PATH, n_pwl_segments=5, solver_name="cplex")
    print("  MILP ready")

    print("\n" + "=" * 60)
    print(" [Phase 4] GA run")
    print("=" * 60)
    direct_dist = haversine_nm(BUSAN_PORT, SHANGHAI_PORT)
    print(f"  Direct distance: {direct_dist:.1f} nm")

    ga_result = setup_ga(
        cost_map=cmap,
        milp_solver=milp,
        env_fn=env_fn,
        departure_time_utc=departure_time_utc,
        n_segments=N_SEGMENTS,
        pop_size=100,
        n_gen=100,
        seed=42,
        n_workers=0,
        sfoc_path=SFOC_PATH,
        rta_h=RTA_HOURS,
        smoothing_weight=0.0,
    )

    elapsed = time.time() - t0
    route = ga_result["best_route"]

    print("\n" + "=" * 60)
    print(" GA result")
    print("=" * 60)
    print(f"  GA objective: {ga_result['best_fitness']:.1f}")
    print("  GA seed: random")
    print(f"  Elapsed: {elapsed:.1f} sec")
    print(
        "  Route validity: "
        f"overall={route['valid']} "
        f"departure={route['valid_departure_heading']} "
        f"last_speed={route['valid_speed']} "
        f"land={(route['land_violation'] <= 0.0)}"
    )
    print(f"  Land violation: {route['land_violation']:.1f}")
    print(f"  Turning (diagnostic): {route['valid_turning']}")
    print(f"  Final heading ok (diagnostic): {route['valid_heading']}")
    print(f"  Last STW: {route['last_speed']:.2f} kts | Last SOG: {route['last_speed_sog']:.2f} kts")
    print(f"  Final heading delta: {route['final_heading_delta']:.2f} deg")

    print("\n  Segments:")
    print(f"  {'Seg':>4s} {'SOG':>8s} {'hdg(deg)':>9s} {'dist(nm)':>9s} {'lat':>8s} {'lon':>9s}")
    print(f"  {'-' * 57}")
    for i in range(len(route["speeds"])):
        wp = route["waypoints"][i]
        print(
            f"  {i:>4d} {route['speeds'][i]:>8.2f} {route['headings'][i]:>9.1f} "
            f"{route['distances_nm'][i]:>9.2f} {wp[0]:>8.4f} {wp[1]:>9.4f}"
        )
    wp = route["waypoints"][-1]
    print(f"  {'END':>4s} {'':>8s} {'':>9s} {'':>9s} {wp[0]:>8.4f} {wp[1]:>9.4f}")

    print("\n" + "=" * 60)
    print(" [Phase 6] Post-check")
    print("=" * 60)
    power_profile = build_required_power_profile(
        route,
        env_fn=env_fn,
        departure_time_utc=departure_time_utc,
    )
    p_req_best = [segment["P_req"] for segment in power_profile]
    milp_check = MILPSolver(sfoc_json_path=SFOC_PATH, n_pwl_segments=5, solver_name="cplex")
    milp_detail = milp_check.solve(P_req=p_req_best, dt=route["dt"], initial_SOC=0.7, msg=False, enable_sos2=False)
    if milp_detail["feasible"]:
        print(f"  MILP pure fuel: {milp_detail['total_fuel_kg']:.1f} kg")

    plot_cost_map_route(route, cmap, save_dir=OUTPUT_DIR)
    plot_weather_map(env_fn, cmap, route_wps=route["waypoints"], save_dir=OUTPUT_DIR, when_utc=departure_time_utc)
    plot_optimal_route(route, cmap, save_dir=OUTPUT_DIR)
    plot_convergence(ga_result["logbook"], save_dir=OUTPUT_DIR)
    if milp_detail["feasible"]:
        plot_power_schedule(milp_detail, power_profile=power_profile, save_dir=OUTPUT_DIR)

    print(f"\n  Total elapsed: {elapsed:.1f} sec")
    print(f"  Output: {OUTPUT_DIR}/")
    del env_loader


if __name__ == "__main__":
    main()
