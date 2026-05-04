"""
Run Case 1 / 2 / 3 under diverse real-weather scenarios and compare.

Each scenario uses date-matched ERA5 wind/wave data and CMEMS current data.
Three optimization strategies are evaluated under each weather condition to
demonstrate when Weather Routing provides meaningful fuel savings for electric
propulsion ships.

Scenarios (ERA5 and CMEMS data must be pre-downloaded):
  - calm_summer     : 2024-07-15  (Pacific high, calm seas)
  - typhoon_bebinca : 2024-09-14  (Typhoon traversing the route)
  - winter_severe   : 2025-01-15  (Strong NW monsoon)

Usage:
    python Validation/run_weather_scenarios.py --ga-n-gen 200
    python Validation/run_weather_scenarios.py --scenario typhoon_bebinca
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Validation.case1_astar_fixed import (
    CASE1_PATH_DISTANCE_MODE,
    apply_case1_baseline_validation,
    build_fixed_route,
    compute_case1_base_speed_knots,
)
from Validation.case2_ga_twostage import EnergyObjectiveSolver
from Validation.common import (
    VERIFICATION_COST_MAP_RESOLUTION,
    WEATHER_SCENARIOS,
    build_case_metrics,
    build_node_rows,
    build_segment_rows,
    ensure_output_dir,
    get_or_build_astar_seed,
    load_scenario_environment,
    make_milp_solver,
    resolve_case3_seed_individual,
    save_case2_seed_cache,
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
    plot_cost_map_route,
    plot_power_schedule,
    plot_route_comparison_map,
    plot_weather_map,
)


VOYAGE_START_PORT = SHANGHAI_PORT
VOYAGE_END_PORT = BUSAN_PORT
ENFORCE_DEPARTURE_HEADING = False


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class CaseMetrics:
    case_name: str
    route_valid: bool
    route_violation: float
    objective: float | None
    objective_kind: str
    milp_fuel_kg: float | None
    milp_feasible: bool
    total_distance_nm: float
    p_req_mean_mw: float
    p_req_std_mw: float
    p_req_min_mw: float
    p_req_max_mw: float
    last_speed_kts: float
    final_heading_delta_deg: float
    runtime_sec: float
    note: str = ""


@dataclass
class CaseRunResult:
    metrics: CaseMetrics
    route: dict
    cost_map: object
    power_profile: list[dict]
    milp_result: dict | None
    ga_logbook: object | None = None
    ga_best_individual: list[float] | None = None


# ── CLI ───────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=list(WEATHER_SCENARIOS.keys()) + ["all"],
        default="all",
        help="Which weather scenario to run (default: all).",
    )
    parser.add_argument("--ga-pop-size", type=int, default=100)
    parser.add_argument("--ga-n-gen", type=int, default=500, help="GA generation count.")
    parser.add_argument("--ga-workers", type=int, default=0)
    parser.add_argument(
        "--ga-milp-time-limit-sec",
        type=int,
        default=20,
        help="MILP time limit for each integrated Case3 GA fitness evaluation.",
    )
    parser.add_argument(
        "--final-milp-time-limit-sec",
        type=int,
        default=300,
        help="MILP time limit for final case schedule recomputation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["1", "2", "3"],
        default=["1", "2", "3"],
        help="Which cases to run (default: 1 2 3).",
    )
    smoothing_group = parser.add_mutually_exclusive_group()
    smoothing_group.add_argument("--astar-smoothing", dest="astar_smoothing", action="store_true")
    smoothing_group.add_argument("--no-astar-smoothing", dest="astar_smoothing", action="store_false")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--use-case2-seed-for-case3",
        dest="use_case2_seed_for_case3",
        action="store_true",
        help="Use the compatible Case2 best individual as the preferred seed for Case3.",
    )
    seed_group.add_argument(
        "--no-case2-seed-for-case3",
        dest="use_case2_seed_for_case3",
        action="store_false",
        help="Disable Case2 -> Case3 seed reuse and fall back to the existing A* seeding only.",
    )
    parser.set_defaults(astar_smoothing=True, use_case2_seed_for_case3=True)
    return parser


# ── Summarizer ────────────────────────────────────────────────────────

def _summarize_case(
    case_name: str,
    route: dict,
    power_profile: list[dict],
    objective: float | None,
    objective_kind: str,
    milp_result: dict | None,
    route_violation: float,
    runtime_sec: float,
    note: str = "",
) -> CaseMetrics:
    return CaseMetrics(
        **build_case_metrics(
            case_name=case_name,
            route=route,
            power_profile=power_profile,
            objective=objective,
            objective_kind=objective_kind,
            milp_result=milp_result,
            route_violation=route_violation,
            runtime_sec=runtime_sec,
            note=note,
        )
    )


# ── Case runners (reuse existing logic, inject env_fn) ────────────────

def _run_case1(
    env_fn,
    departure_time_utc,
    n_segments,
    rta_h,
    astar_smoothing,
    final_milp_time_limit_sec,
) -> CaseRunResult:
    t0 = time.time()
    cost_map = build_cost_map(resolution=VERIFICATION_COST_MAP_RESOLUTION)
    _, waypoints, _ = get_or_build_astar_seed(
        cost_map, VOYAGE_START_PORT, VOYAGE_END_PORT, n_segments,
        distance_mode=CASE1_PATH_DISTANCE_MODE, apply_smoothing=astar_smoothing,
    )
    base_speed = compute_case1_base_speed_knots(waypoints, rta_h=rta_h)
    route = build_fixed_route(
        waypoints, base_speed, n_segments=n_segments, rta_h=rta_h,
        start_port=VOYAGE_START_PORT, end_port=VOYAGE_END_PORT,
    )
    validation = apply_case1_baseline_validation(route, cost_map, env_fn, departure_time_utc)
    _, power_profile, milp_result = solve_route_schedule(
        route,
        env_fn,
        departure_time_utc,
        time_limit_sec=final_milp_time_limit_sec,
    )
    return CaseRunResult(
        metrics=_summarize_case(
            "case1_astar_fixed", route, power_profile,
            objective=milp_result["total_fuel_kg"] if milp_result["feasible"] else None,
            objective_kind="milp_fuel_kg", milp_result=milp_result,
            route_violation=validation.land_violation, runtime_sec=time.time() - t0,
        ),
        route=route, cost_map=cost_map, power_profile=power_profile, milp_result=milp_result,
    )


def _run_case2(
    env_fn, departure_time_utc, n_segments, rta_h,
    ga_pop_size, ga_n_gen, ga_workers, seed, astar_smoothing,
    final_milp_time_limit_sec,
) -> CaseRunResult:
    t0 = time.time()
    cost_map = build_cost_map(resolution=VERIFICATION_COST_MAP_RESOLUTION)
    _, astar_waypoints, _ = get_or_build_astar_seed(
        cost_map, VOYAGE_START_PORT, VOYAGE_END_PORT, n_segments,
        distance_mode=CASE1_PATH_DISTANCE_MODE, apply_smoothing=astar_smoothing,
    )
    case1_base_speed = compute_case1_base_speed_knots(astar_waypoints, rta_h=rta_h)
    case1_seed_individual = encode_fixed_speed_route_seed(
        astar_waypoints,
        case1_base_speed,
        n_segments=n_segments,
        genotype_layout=GENOTYPE_DECOUPLED,
    )
    print("  [Case2 seed] Injecting Case1 fixed-route seed")

    ga_result = setup_ga(
        cost_map=cost_map, milp_solver=EnergyObjectiveSolver(), env_fn=env_fn,
        departure_time_utc=departure_time_utc, n_segments=n_segments, rta_h=rta_h,
        pop_size=ga_pop_size, n_gen=ga_n_gen, seed=seed, n_workers=ga_workers,
        astar_seed_waypoints=astar_waypoints,
        seed_individual=case1_seed_individual,
        start_port=VOYAGE_START_PORT, end_port=VOYAGE_END_PORT,
        enforce_departure_heading=ENFORCE_DEPARTURE_HEADING,
        genotype_layout=GENOTYPE_DECOUPLED,
    )
    route = ga_result["best_route"]
    _, power_profile, milp_result = solve_route_schedule(
        route,
        env_fn,
        departure_time_utc,
        time_limit_sec=final_milp_time_limit_sec,
    )
    return CaseRunResult(
        metrics=_summarize_case(
            "case2_ga_twostage", route, power_profile,
            objective=ga_result["best_fitness"], objective_kind="energy_objective",
            milp_result=milp_result, route_violation=route["land_violation"],
            runtime_sec=time.time() - t0,
        ),
        route=route, cost_map=cost_map, power_profile=power_profile,
        milp_result=milp_result, ga_logbook=ga_result["logbook"],
        ga_best_individual=list(ga_result["best_individual"]),
    )


def _run_case3(
    env_fn, departure_time_utc, n_segments, rta_h,
    ga_pop_size, ga_n_gen, ga_workers, seed, astar_smoothing,
    ga_milp_time_limit_sec,
    final_milp_time_limit_sec,
    use_case2_seed_for_case3,
    case2_best_individual=None,
) -> CaseRunResult:
    t0 = time.time()
    cost_map = build_cost_map(resolution=VERIFICATION_COST_MAP_RESOLUTION)
    _, astar_waypoints, _ = get_or_build_astar_seed(
        cost_map, VOYAGE_START_PORT, VOYAGE_END_PORT, n_segments,
        distance_mode=CASE1_PATH_DISTANCE_MODE, apply_smoothing=astar_smoothing,
    )
    case1_base_speed = compute_case1_base_speed_knots(astar_waypoints, rta_h=rta_h)
    case1_seed_individual = encode_fixed_speed_route_seed(
        astar_waypoints,
        case1_base_speed,
        n_segments=n_segments,
        genotype_layout=GENOTYPE_DECOUPLED,
    )
    print("  [Case3 seed] Injecting Case1 fixed-route seed")

    case2_seed_individual, seed_source = resolve_case3_seed_individual(
        use_case2_seed=use_case2_seed_for_case3,
        start_port=VOYAGE_START_PORT,
        end_port=VOYAGE_END_PORT,
        n_segments=n_segments,
        rta_h=rta_h,
        ga_cost_resolution=VERIFICATION_COST_MAP_RESOLUTION,
        departure_time_utc=departure_time_utc,
        astar_seed_smoothing=astar_smoothing,
        genotype_layout=GENOTYPE_DECOUPLED,
        same_run_seed_individual=case2_best_individual,
    )
    if seed_source == "same_run":
        print("  [Case3 seed] Using same-run Case2 best individual")
    elif seed_source == "cache":
        print("  [Case3 seed] Using cached Case2 best individual")
    elif seed_source == "disabled":
        print("  [Case3 seed] Disabled, using existing A* seeding only")
    else:
        print("  [Case3 seed] No compatible Case2 seed found, using existing A* seeding only")

    milp = make_milp_solver()
    ga_result = setup_ga(
        cost_map=cost_map, milp_solver=milp, env_fn=env_fn,
        departure_time_utc=departure_time_utc, n_segments=n_segments, rta_h=rta_h,
        pop_size=ga_pop_size, n_gen=ga_n_gen, seed=seed, n_workers=ga_workers,
        smoothing_weight=0.0, astar_seed_waypoints=astar_waypoints,
        seed_individual=case1_seed_individual,
        seed_individuals=[] if case2_seed_individual is None else [case2_seed_individual],
        start_port=VOYAGE_START_PORT, end_port=VOYAGE_END_PORT,
        enforce_departure_heading=ENFORCE_DEPARTURE_HEADING,
        genotype_layout=GENOTYPE_DECOUPLED,
        milp_time_limit_sec=ga_milp_time_limit_sec,
    )
    route = ga_result["best_route"]
    _, power_profile, milp_result = solve_route_schedule(
        route,
        env_fn,
        departure_time_utc,
        time_limit_sec=final_milp_time_limit_sec,
    )
    return CaseRunResult(
        metrics=_summarize_case(
            "case3_ga_integrated", route, power_profile,
            objective=ga_result["best_fitness"], objective_kind="ga_objective",
            milp_result=milp_result, route_violation=route["land_violation"],
            runtime_sec=time.time() - t0,
        ),
        route=route, cost_map=cost_map, power_profile=power_profile,
        milp_result=milp_result, ga_logbook=ga_result["logbook"],
        ga_best_individual=list(ga_result["best_individual"]),
    )


# ── Output helpers ────────────────────────────────────────────────────

def _print_scenario_summary(scenario_name: str, rows: list[CaseMetrics]) -> None:
    print(f"\n{'=' * 100}")
    print(f"  {WEATHER_SCENARIOS[scenario_name]['label']} — Case Comparison")
    print(f"{'=' * 100}")
    header = (
        f"{'case':<20} {'valid':<6} {'milp_fuel_kg':>14} {'dist_nm':>10} "
        f"{'P_mean':>10} {'P_std':>10} {'P_min':>10} {'P_max':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        fuel_text = "N/A" if row.milp_fuel_kg is None else f"{row.milp_fuel_kg:.1f}"
        print(
            f"{row.case_name:<20} {str(row.route_valid):<6} {fuel_text:>14} "
            f"{row.total_distance_nm:>10.1f} {row.p_req_mean_mw:>10.2f} "
            f"{row.p_req_std_mw:>10.2f} {row.p_req_min_mw:>10.2f} {row.p_req_max_mw:>10.2f}"
        )


def _save_scenario_outputs(
    scenario_name: str,
    case_runs: list[CaseRunResult],
    departure_time_utc,
    output_dir: str,
) -> None:
    """Save per-scenario route maps, power schedules, and artifacts."""
    scenario_dir = os.path.join(output_dir, scenario_name)
    os.makedirs(scenario_dir, exist_ok=True)

    for case_run in case_runs:
        case_dir = os.path.join(scenario_dir, case_run.metrics.case_name)
        save_case_artifacts(
            case_name=case_run.metrics.case_name,
            output_dir=case_dir,
            route=case_run.route,
            power_profile=case_run.power_profile,
            milp_result=case_run.milp_result,
            departure_time_utc=departure_time_utc,
            cost_map=case_run.cost_map,
            metrics=asdict(case_run.metrics),
            ga_best_individual=case_run.ga_best_individual,
            ga_best_fitness=case_run.metrics.objective,
            ga_logbook=case_run.ga_logbook,
        )
        plot_cost_map_route(case_run.route, case_run.cost_map, save_dir=case_dir, filename="route.png")
        if case_run.milp_result and case_run.milp_result.get("feasible"):
            plot_power_schedule(case_run.milp_result, power_profile=case_run.power_profile, save_dir=case_dir)
        if case_run.ga_logbook is not None:
            plot_convergence(case_run.ga_logbook, save_dir=case_dir)

    # Route comparison map
    comparison_cost_map = build_cost_map(resolution=VERIFICATION_COST_MAP_RESOLUTION)
    plot_route_comparison_map(
        {run.metrics.case_name: run.route for run in case_runs},
        comparison_cost_map,
        save_dir=scenario_dir,
    )


def _save_cross_scenario_summary(
    all_results: dict[str, list[CaseRunResult]],
    output_dir: str,
) -> None:
    """Save a combined comparison across all scenarios."""
    rows = []
    for scenario_name, case_runs in all_results.items():
        for run in case_runs:
            row = asdict(run.metrics)
            row["scenario"] = scenario_name
            row["scenario_label"] = WEATHER_SCENARIOS[scenario_name]["label"]
            rows.append(row)

    if not rows:
        return

    # JSON
    json_path = os.path.join(output_dir, "cross_scenario_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    # CSV
    csv_path = os.path.join(output_dir, "cross_scenario_comparison.csv")
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    # XLSX with summary + pivot
    xlsx_path = os.path.join(output_dir, "cross_scenario_comparison.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="details", index=False)
        pivot = df.pivot_table(
            index="scenario_label",
            columns="case_name",
            values=["milp_fuel_kg", "total_distance_nm", "p_req_mean_mw", "runtime_sec"],
            aggfunc="first",
        )
        pivot.to_excel(writer, sheet_name="pivot")

    print(f"\n  Cross-scenario outputs:")
    print(f"    JSON: {json_path}")
    print(f"    CSV : {csv_path}")
    print(f"    XLSX: {xlsx_path}")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    args = _build_parser().parse_args()
    output_dir = ensure_output_dir("weather_scenarios")

    scenarios = list(WEATHER_SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]
    requested_cases = set(args.cases)

    print("=" * 80)
    print("  Weather Scenario Comparison")
    print("=" * 80)
    print(f"  Scenarios   : {', '.join(scenarios)}")
    print(f"  Cases       : {', '.join(sorted(requested_cases))}")
    print(f"  GA config   : pop={args.ga_pop_size}, gen={args.ga_n_gen}, workers={args.ga_workers}")
    print(
        f"  MILP limits : ga_eval={args.ga_milp_time_limit_sec}s, "
        f"final={args.final_milp_time_limit_sec}s"
    )
    print(f"  A* smoothing: {'on' if args.astar_smoothing else 'off'}")

    all_results: dict[str, list[CaseRunResult]] = {}

    for scenario_name in scenarios:
        print(f"\n{'#' * 80}")
        print(f"  SCENARIO: {WEATHER_SCENARIOS[scenario_name]['label']}")
        print(f"{'#' * 80}")

        env_loader, env_fn, departure_time_utc = load_scenario_environment(scenario_name)
        try:
            case_runs: list[CaseRunResult] = []

            if "1" in requested_cases:
                print(f"\n  ── Case 1: A* Fixed Route ──")
                run = _run_case1(
                    env_fn,
                    departure_time_utc,
                    N_SEGMENTS,
                    RTA_HOURS,
                    args.astar_smoothing,
                    args.final_milp_time_limit_sec,
                )
                case_runs.append(run)

            if "2" in requested_cases:
                print(f"\n  ── Case 2: Energy-Only GA → MILP ──")
                run = _run_case2(
                    env_fn, departure_time_utc, N_SEGMENTS, RTA_HOURS,
                    args.ga_pop_size, args.ga_n_gen, args.ga_workers, args.seed, args.astar_smoothing,
                    args.final_milp_time_limit_sec,
                )
                case_runs.append(run)
                if run.ga_best_individual is not None:
                    cache_path = save_case2_seed_cache(
                        best_individual=run.ga_best_individual,
                        best_fitness=run.metrics.objective,
                        route=run.route,
                        start_port=VOYAGE_START_PORT,
                        end_port=VOYAGE_END_PORT,
                        n_segments=N_SEGMENTS,
                        rta_h=RTA_HOURS,
                        ga_cost_resolution=VERIFICATION_COST_MAP_RESOLUTION,
                        departure_time_utc=departure_time_utc,
                        astar_seed_smoothing=args.astar_smoothing,
                        genotype_layout=GENOTYPE_DECOUPLED,
                    )
                    print(f"  [Case2 seed] Saved canonical cache: {cache_path}")

            if "3" in requested_cases:
                print(f"\n  ── Case 3: Integrated GA + MILP ──")
                case2_best = None
                for r in case_runs:
                    if r.metrics.case_name == "case2_ga_twostage":
                        case2_best = r.ga_best_individual
                run = _run_case3(
                    env_fn, departure_time_utc, N_SEGMENTS, RTA_HOURS,
                    args.ga_pop_size, args.ga_n_gen, args.ga_workers, args.seed,
                    args.astar_smoothing,
                    args.ga_milp_time_limit_sec,
                    args.final_milp_time_limit_sec,
                    args.use_case2_seed_for_case3,
                    case2_best_individual=case2_best,
                )
                case_runs.append(run)

            # Print summary
            _print_scenario_summary(scenario_name, [r.metrics for r in case_runs])

            # Save outputs
            _save_scenario_outputs(scenario_name, case_runs, departure_time_utc, output_dir)

            # Weather map (using first case's cost_map)
            if case_runs:
                scenario_dir = os.path.join(output_dir, scenario_name)
                try:
                    plot_weather_map(
                        env_fn, case_runs[0].cost_map,
                        route_wps=case_runs[0].route.get("waypoints"),
                        save_dir=scenario_dir, when_utc=departure_time_utc,
                    )
                except Exception as e:
                    print(f"  [Warning] Weather map failed: {e}")

            all_results[scenario_name] = case_runs
        finally:
            del env_loader

    # Cross-scenario summary
    _save_cross_scenario_summary(all_results, output_dir)

    # Final summary
    print(f"\n{'=' * 80}")
    print("  ALL SCENARIOS COMPLETE")
    print(f"{'=' * 80}")
    for scenario_name, runs in all_results.items():
        label = WEATHER_SCENARIOS[scenario_name]["label"]
        for run in runs:
            fuel = f"{run.metrics.milp_fuel_kg:.0f}" if run.metrics.milp_fuel_kg else "N/A"
            print(
                f"  {label:<35} {run.metrics.case_name:<22} "
                f"fuel={fuel:>8} kg  dist={run.metrics.total_distance_nm:.1f} nm"
            )


if __name__ == "__main__":
    main()
