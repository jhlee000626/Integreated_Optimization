"""
Verification Case 5

SOC sensitivity study:
- optimize one route with the standard integrated workflow
- compare MILP scheduling with charged ESS vs empty ESS
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Verification.common import ensure_output_dir, load_era5_weather, make_milp_solver
from src.grid.cost_map import build_cost_map
from src.optimizer.ga_engine import N_SEGMENTS, build_required_power_profile, setup_ga
from src.visualization.plotter import plot_convergence, plot_optimal_route, plot_power_schedule, plot_weather_map


def main():
    print("=" * 60)
    print(" [Verification Case 5] SOC Sensitivity ")
    print("=" * 60)

    out_dir = ensure_output_dir("case5")
    weather_loader, weather_fn = load_era5_weather(time_index=0)
    try:
        cost_map = build_cost_map(resolution=0.005)
        ga_result = setup_ga(
            cost_map=cost_map,
            milp_solver=make_milp_solver(),
            weather_fn=weather_fn,
            n_segments=N_SEGMENTS,
            pop_size=100,
            n_gen=100,
            seed=42,
            n_workers=0,
        )

        route = ga_result["best_route"]
        power_profile = build_required_power_profile(route, weather_fn=weather_fn)
        p_req = [segment["P_req"] for segment in power_profile]
        dt = route["dt"]

        milp_with_soc = make_milp_solver()
        result_with_soc = milp_with_soc.solve(P_req=p_req, dt=dt, initial_SOC=0.7, msg=False)

        milp_empty_soc = make_milp_solver()
        result_empty_soc = milp_empty_soc.solve(P_req=p_req, dt=dt, initial_SOC=0.0, msg=False)

        print(f"  Best integrated fitness: {ga_result['best_fitness']:.2f} kg")
        print(f"  With initial SOC=0.7 feasible: {result_with_soc['feasible']}")
        print(f"  With initial SOC=0.0 feasible: {result_empty_soc['feasible']}")
        if result_with_soc["feasible"]:
            print(f"  Fuel with charged ESS: {result_with_soc['total_fuel_kg']:.2f} kg")
        if result_empty_soc["feasible"]:
            print(f"  Fuel with empty ESS: {result_empty_soc['total_fuel_kg']:.2f} kg")

        plot_optimal_route(route, cost_map, save_dir=out_dir)
        plot_weather_map(weather_fn, cost_map, route_wps=route["waypoints"], save_dir=out_dir)
        plot_convergence(ga_result["logbook"], save_dir=out_dir)

        if result_with_soc["feasible"]:
            plot_power_schedule(result_with_soc, save_dir=os.path.join(out_dir, "with_soc"))
        if result_empty_soc["feasible"]:
            plot_power_schedule(result_empty_soc, save_dir=os.path.join(out_dir, "empty_soc"))
    finally:
        weather_loader.close()


if __name__ == "__main__":
    main()
