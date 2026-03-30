"""
Verification Case 2

Two-stage optimization:
1. GA searches a route by minimizing energy demand only.
2. MILP schedules the resulting load profile.
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Verification.common import ensure_output_dir, load_era5_weather, solve_route_schedule
from src.grid.cost_map import build_cost_map
from src.optimizer.ga_engine import N_SEGMENTS, setup_ga
from src.visualization.plotter import plot_convergence, plot_optimal_route, plot_power_schedule, plot_weather_map


class EnergyObjectiveSolver:
    """MILP-shaped adapter that turns setup_ga into an energy-only optimizer."""

    @staticmethod
    def solve(P_req, dt, initial_SOC=0.7, msg=False):
        total_energy = sum(power * duration for power, duration in zip(P_req, dt))
        return {
            "feasible": True,
            "total_fuel_kg": total_energy,
            "schedule": [],
            "summary": {},
        }


def main():
    print("=" * 60)
    print(" [Verification Case 2] Energy-Only GA -> MILP ")
    print("=" * 60)

    out_dir = ensure_output_dir("case2")
    weather_loader, weather_fn = load_era5_weather(time_index=0)
    try:
        cost_map = build_cost_map(resolution=0.01)
        ga_result = setup_ga(
            cost_map=cost_map,
            milp_solver=EnergyObjectiveSolver(),
            weather_fn=weather_fn,
            n_segments=N_SEGMENTS,
            pop_size=60,
            n_gen=60,
            seed=42,
            n_workers=1,
        )

        route = ga_result["best_route"]
        milp, power_profile, milp_result = solve_route_schedule(route, weather_fn, initial_soc=0.7)

        print(f"  Energy objective value: {ga_result['best_fitness']:.3f}")
        if milp_result["feasible"]:
            print(f"  MILP fuel: {milp_result['total_fuel_kg']:.2f} kg")
        else:
            print("  MILP infeasible")

        plot_optimal_route(route, cost_map, save_dir=out_dir)
        plot_weather_map(weather_fn, cost_map, route_wps=route["waypoints"], save_dir=out_dir)
        plot_convergence(ga_result["logbook"], save_dir=out_dir)
        if milp_result["feasible"]:
            plot_power_schedule(milp_result, save_dir=out_dir)
    finally:
        weather_loader.close()


if __name__ == "__main__":
    main()
