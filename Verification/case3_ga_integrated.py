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

from Verification.common import ensure_output_dir, load_era5_weather, make_milp_solver, solve_route_schedule
from src.grid.cost_map import build_cost_map
from src.optimizer.ga_engine import N_SEGMENTS, setup_ga
from src.visualization.plotter import plot_convergence, plot_optimal_route, plot_power_schedule, plot_weather_map


def main():
    print("=" * 60)
    print(" [Verification Case 3] Integrated GA + MILP ")
    print("=" * 60)

    out_dir = ensure_output_dir("case3")
    weather_loader, weather_fn = load_era5_weather(time_index=0)
    try:
        cost_map = build_cost_map(resolution=0.005)
        milp = make_milp_solver()
        ga_result = setup_ga(
            cost_map=cost_map,
            milp_solver=milp,
            weather_fn=weather_fn,
            n_segments=N_SEGMENTS,
            pop_size=100,
            n_gen=100,
            seed=42,
            n_workers=0,
        )

        route = ga_result["best_route"]
        _, _, milp_result = solve_route_schedule(route, weather_fn, initial_soc=0.7)

        print(f"  Best fitness: {ga_result['best_fitness']:.2f} kg")
        print(f"  Final last speed: {route['last_speed']:.2f} kts")
        if milp_result["feasible"]:
            print(f"  Recomputed MILP fuel: {milp_result['total_fuel_kg']:.2f} kg")
        else:
            print("  Final MILP infeasible")

        plot_optimal_route(route, cost_map, save_dir=out_dir)
        plot_weather_map(weather_fn, cost_map, route_wps=route["waypoints"], save_dir=out_dir)
        plot_convergence(ga_result["logbook"], save_dir=out_dir)
        if milp_result["feasible"]:
            plot_power_schedule(milp_result, save_dir=out_dir)
    finally:
        weather_loader.close()


if __name__ == "__main__":
    main()
