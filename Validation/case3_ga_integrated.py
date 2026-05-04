"""
Verification Case 3

Integrated GA + MILP.
This is the same workflow as tests/test_ga.py, but kept as a named scenario.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Validation.case1_astar_fixed import compute_case1_base_speed_knots
from Validation.common import (
    VERIFICATION_COST_MAP_RESOLUTION,
    build_case_metrics,
    ensure_output_dir,
    get_or_build_astar_seed,
    load_marine_environment,
    make_milp_solver,
    resolve_case3_seed_individual,
    save_case_artifacts,
    solve_route_schedule,
)
from src.grid.cost_map import build_cost_map
from src.grid.no_go_zone import BUSAN_PORT, SHANGHAI_PORT
from src.optimizer.ga_engine import (
    GENOTYPE_DECOUPLED,
    N_SEGMENTS,
    RTA_HOURS,
    encode_fixed_speed_route_seed,
    setup_ga,
)
from src.visualization.plotter import (
    plot_convergence,
    plot_optimal_route,
    plot_power_schedule,
    plot_power_scheduling_detail,
)


VOYAGE_START_PORT = SHANGHAI_PORT
VOYAGE_END_PORT = BUSAN_PORT
ENFORCE_DEPARTURE_HEADING = False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ga-pop-size", type=int, default=100, help="GA population size.")
    parser.add_argument("--ga-n-gen", type=int, default=5000, help="GA generation count.")
    parser.add_argument("--ga-workers", type=int, default=0, help="GA worker count.")
    parser.add_argument(
        "--ga-milp-time-limit-sec",
        type=int,
        default=20,
        help="MILP time limit for each GA fitness evaluation.",
    )
    parser.add_argument(
        "--final-milp-time-limit-sec",
        type=int,
        default=300,
        help="MILP time limit for final schedule recomputation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for GA initialization.")
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
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--use-case2-seed-for-case3",
        dest="use_case2_seed_for_case3",
        action="store_true",
        help="Use a compatible Case2 optimal solution as the preferred initial seed.",
    )
    seed_group.add_argument(
        "--no-case2-seed-for-case3",
        dest="use_case2_seed_for_case3",
        action="store_false",
        help="Disable Case2 seed reuse and keep the existing A* seeding only.",
    )
    parser.add_argument("--era5-path", type=str, default=None, help="Explicit path to ERA5 netCDF dataset")
    parser.add_argument("--cmems-path", type=str, default=None, help="Explicit path to CMEMS netCDF dataset")
    parser.set_defaults(astar_smoothing=True)
    parser.set_defaults(use_case2_seed_for_case3=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    start_time = time.time()

    print("=" * 60)
    print(" [Verification Case 3] Integrated GA + MILP ")
    print("=" * 60)

    out_dir = ensure_output_dir("case3")
    env_loader, env_fn, departure_time_utc = load_marine_environment(
        era5_path=args.era5_path,
        cmems_path=args.cmems_path,
    )
    try:
        cost_map = build_cost_map(resolution=VERIFICATION_COST_MAP_RESOLUTION)

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
        print("  [Case3 seed] Injecting Case1 fixed-route seed")

        case2_seed_individual, seed_source = resolve_case3_seed_individual(
            use_case2_seed=args.use_case2_seed_for_case3,
            start_port=VOYAGE_START_PORT,
            end_port=VOYAGE_END_PORT,
            n_segments=N_SEGMENTS,
            rta_h=RTA_HOURS,
            ga_cost_resolution=VERIFICATION_COST_MAP_RESOLUTION,
            departure_time_utc=departure_time_utc,
            astar_seed_smoothing=args.astar_smoothing,
            genotype_layout=GENOTYPE_DECOUPLED,
        )
        if seed_source == "cache":
            print("  [Case3 seed] Loaded Case2 seed from canonical cache")
        elif seed_source == "disabled":
            print("  [Case3 seed] Disabled, using existing A* seeding only")
        else:
            print("  [Case3 seed] No compatible Case2 seed found, using existing A* seeding only")

        milp = make_milp_solver()
        ga_result = setup_ga(
            cost_map=cost_map,
            milp_solver=milp,
            env_fn=env_fn,
            departure_time_utc=departure_time_utc,
            n_segments=N_SEGMENTS,
            rta_h=RTA_HOURS,
            pop_size=args.ga_pop_size,
            n_gen=args.ga_n_gen,
            seed=args.seed,
            n_workers=args.ga_workers,
            smoothing_weight=0.0,
            astar_seed_waypoints=astar_waypoints,
            seed_individual=case1_seed_individual,
            seed_individuals=[] if case2_seed_individual is None else [case2_seed_individual],
            start_port=VOYAGE_START_PORT,
            end_port=VOYAGE_END_PORT,
            enforce_departure_heading=ENFORCE_DEPARTURE_HEADING,
            genotype_layout=GENOTYPE_DECOUPLED,
            milp_time_limit_sec=args.ga_milp_time_limit_sec,
        )

        route = ga_result["best_route"]
        _, power_profile, milp_result = solve_route_schedule(
            route,
            env_fn,
            departure_time_utc,
            initial_soc=0.7,
            time_limit_sec=args.final_milp_time_limit_sec,
        )
        runtime = time.time() - start_time

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

        metrics = build_case_metrics(
            case_name="case3_ga_integrated",
            route=route,
            power_profile=power_profile,
            objective=ga_result["best_fitness"],
            objective_kind="ga_objective",
            milp_result=milp_result,
            route_violation=route["land_violation"],
            runtime_sec=runtime,
            note=(
                f"ga_res={VERIFICATION_COST_MAP_RESOLUTION}, "
                f"smoothing={'on' if args.astar_smoothing else 'off'}"
            ),
        )
        artifact_paths = save_case_artifacts(
            case_name="case3_ga_integrated",
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

        plot_optimal_route(route, cost_map, save_dir=out_dir)
        plot_convergence(ga_result["logbook"], save_dir=out_dir)
        if milp_result["feasible"]:
            plot_power_schedule(milp_result, power_profile=power_profile, save_dir=out_dir)
            plot_power_scheduling_detail(milp_result, power_profile=power_profile, save_dir=out_dir)
        print(f"  Case artifact: {artifact_paths['case_artifact']}")
    finally:
        del env_loader


if __name__ == "__main__":
    main()
