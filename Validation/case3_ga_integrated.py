"""
Verification Case 3

Integrated GA + MILP.
This is the same workflow as tests/test_ga.py, but kept as a named scenario.
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
    solve_route_schedule,
)
from src.grid.cost_map import build_cost_map
from src.optimizer.ga_engine import N_SEGMENTS, RTA_HOURS, setup_ga
from src.visualization.plotter import plot_convergence, plot_optimal_route, plot_power_schedule, plot_weather_map


def main():
    print("=" * 60)
    print(" [Verification Case 3] Integrated GA + MILP ")
    print("=" * 60)

    out_dir = ensure_output_dir("case3")
    env_loader, env_fn, departure_time_utc = load_marine_environment()
    try:
        cost_map = build_cost_map(resolution=VERIFICATION_COST_MAP_RESOLUTION)
        milp = make_milp_solver()
        ga_result = setup_ga(
            cost_map=cost_map,
            milp_solver=milp,
            env_fn=env_fn,
            departure_time_utc=departure_time_utc,
            n_segments=N_SEGMENTS,
            rta_h=RTA_HOURS,
            pop_size=100,
            n_gen=500,
            seed=42,
            n_workers=0,
            smoothing_weight=0.0,
        )

        route = ga_result["best_route"]
        _, power_profile, milp_result = solve_route_schedule(route, env_fn, departure_time_utc, initial_soc=0.7)

        print(f"  Best fitness: {ga_result['best_fitness']:.2f} kg")
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
        print(f"  Final last STW: {route['last_speed']:.2f} kts | Last SOG: {route['last_speed_sog']:.2f} kts")
        if milp_result["feasible"]:
            print(f"  Recomputed MILP fuel: {milp_result['total_fuel_kg']:.2f} kg")
        else:
            print("  Final MILP infeasible")

        plot_optimal_route(route, cost_map, save_dir=out_dir)
        plot_weather_map(env_fn, cost_map, route_wps=route["waypoints"], save_dir=out_dir, when_utc=departure_time_utc)
        plot_convergence(ga_result["logbook"], save_dir=out_dir)
        if milp_result["feasible"]:
            plot_power_schedule(milp_result, power_profile=power_profile, save_dir=out_dir)
    finally:
        del env_loader


if __name__ == "__main__":
    main()
