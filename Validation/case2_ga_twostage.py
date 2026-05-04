"""
Verification Case 2

Two-stage optimization:
1. GA searches a route by minimizing energy demand only.
2. MILP schedules the resulting load profile.
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
    get_or_build_astar_seed,
    load_marine_environment,
    save_case2_seed_cache,
    save_case_artifacts,
    solve_route_schedule,
)
from Validation.case1_astar_fixed import compute_case1_base_speed_knots
from src.grid.cost_map import build_cost_map
from src.grid.no_go_zone import BUSAN_PORT, SHANGHAI_PORT
from src.optimizer.ga_engine import (
    GENOTYPE_DECOUPLED,
    N_SEGMENTS,
    RTA_HOURS,
    encode_fixed_speed_route_seed,
    setup_ga,
)
from src.visualization.plotter import plot_convergence, plot_optimal_route, plot_power_schedule, plot_weather_map


VOYAGE_START_PORT = SHANGHAI_PORT
VOYAGE_END_PORT = BUSAN_PORT
ENFORCE_DEPARTURE_HEADING = False


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    smoothing_group = parser.add_mutually_exclusive_group()
    smoothing_group.add_argument(
        "--astar-smoothing",
        dest="astar_smoothing",
        action="store_true",
        help="Apply waypoint smoothing to the cached/generated A* seed route.",
    )
    smoothing_group.add_argument(
        "--no-astar-smoothing",
        dest="astar_smoothing",
        action="store_false",
        help="Disable waypoint smoothing for the A* seed route.",
    )
    parser.add_argument("--era5-path", type=str, default=None, help="Explicit path to ERA5 netCDF dataset")
    parser.add_argument("--cmems-path", type=str, default=None, help="Explicit path to CMEMS netCDF dataset")
    parser.set_defaults(astar_smoothing=True)
    return parser


def main():
    args = _build_parser().parse_args()
    start_time = time.time()
    print("=" * 60)
    print(" [Verification Case 2] Energy-Only GA -> MILP ")
    print("=" * 60)

    out_dir = ensure_output_dir("case2")
    env_loader, env_fn, departure_time_utc = load_marine_environment(
        era5_path=args.era5_path,
        cmems_path=args.cmems_path,
    )
    try:
        cost_map = build_cost_map(resolution=VERIFICATION_COST_MAP_RESOLUTION)

        # A* 경로를 먼저 구해서 GA 초기 시드로 사용
        print("  [A* pre-solve] Loading A* route for GA seeding...")
        _, astar_waypoints, _ = get_or_build_astar_seed(
        cost_map,
        VOYAGE_START_PORT,
        VOYAGE_END_PORT,
        N_SEGMENTS,
        distance_mode="grid",
        apply_smoothing=args.astar_smoothing,
    )
        print(f"  [A* pre-solve] {len(astar_waypoints)} waypoints ready")
        print(f"  [A* pre-solve] Smoothing: {'on' if args.astar_smoothing else 'off'}")

        case1_base_speed = compute_case1_base_speed_knots(astar_waypoints, rta_h=RTA_HOURS)
        case1_seed_individual = encode_fixed_speed_route_seed(
            astar_waypoints,
            case1_base_speed,
            n_segments=N_SEGMENTS,
            genotype_layout=GENOTYPE_DECOUPLED,
        )
        print("  [Case2 seed] Injecting Case1 fixed-route seed")

        ga_result = setup_ga(
            cost_map=cost_map,
            milp_solver=EnergyObjectiveSolver(),
            env_fn=env_fn,
            departure_time_utc=departure_time_utc,
            n_segments=N_SEGMENTS,
            rta_h=RTA_HOURS,
            pop_size=100,
            n_gen=2000,
            seed=42,
            n_workers=0,
            smoothing_weight=0.0,
            astar_seed_waypoints=astar_waypoints,
            seed_individual=case1_seed_individual,
            start_port=VOYAGE_START_PORT,
            end_port=VOYAGE_END_PORT,
            enforce_departure_heading=ENFORCE_DEPARTURE_HEADING,
            genotype_layout=GENOTYPE_DECOUPLED,
        )

        route = ga_result["best_route"]
        milp, power_profile, milp_result = solve_route_schedule(
            route,
            env_fn,
            departure_time_utc,
            initial_soc=0.7,
        )
        runtime = time.time() - start_time

        print(f"  Energy objective value: {ga_result['best_fitness']:.3f}")
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
        if milp_result["feasible"]:
            print(f"  MILP fuel: {milp_result['total_fuel_kg']:.2f} kg")
        else:
            print("  MILP infeasible")

        metrics = build_case_metrics(
            case_name="case2_ga_twostage",
            route=route,
            power_profile=power_profile,
            objective=ga_result["best_fitness"],
            objective_kind="energy_objective",
            milp_result=milp_result,
            route_violation=route["land_violation"],
            runtime_sec=runtime,
            note=(
                f"ga_res={VERIFICATION_COST_MAP_RESOLUTION}, "
                f"smoothing={'on' if args.astar_smoothing else 'off'}"
            ),
        )
        artifact_paths = save_case_artifacts(
            case_name="case2_ga_twostage",
            output_dir=out_dir,
            route=route,
            power_profile=power_profile,
            milp_result=milp_result,
            departure_time_utc=departure_time_utc,
            cost_map=cost_map,
            metrics=metrics,
            ga_best_individual=list(ga_result["best_individual"]),
            ga_best_fitness=ga_result["best_fitness"],
            ga_logbook=ga_result["logbook"],
        )
        cache_path = save_case2_seed_cache(
            best_individual=list(ga_result["best_individual"]),
            best_fitness=ga_result["best_fitness"],
            route=route,
            start_port=VOYAGE_START_PORT,
            end_port=VOYAGE_END_PORT,
            n_segments=N_SEGMENTS,
            rta_h=RTA_HOURS,
            ga_cost_resolution=VERIFICATION_COST_MAP_RESOLUTION,
            departure_time_utc=departure_time_utc,
            astar_seed_smoothing=args.astar_smoothing,
            genotype_layout=GENOTYPE_DECOUPLED,
        )

        plot_optimal_route(route, cost_map, save_dir=out_dir)
        plot_weather_map(env_fn, cost_map, route_wps=route["waypoints"], save_dir=out_dir, when_utc=departure_time_utc)
        plot_convergence(ga_result["logbook"], save_dir=out_dir)
        if milp_result["feasible"]:
            plot_power_schedule(milp_result, power_profile=power_profile, save_dir=out_dir)
        print(f"  Case artifact: {artifact_paths['case_artifact']}")
        print(f"  GA seed cache: {cache_path}")
    finally:
        del env_loader


if __name__ == "__main__":
    main()
