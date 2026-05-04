"""
Compare Verification cases under a shared environment/time/configuration.

This runner keeps the comparison fair by using the same:
- departure_time_utc
- env_fn
- segment count
- RTA

Default settings are intentionally lightweight so we can iterate quickly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

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
    build_case_metrics,
    build_node_rows,
    build_segment_rows,
    ensure_output_dir,
    load_marine_environment,
    make_milp_solver,
    resolve_case3_seed_individual,
    save_case2_seed_cache,
    save_case_artifacts,
    solve_route_schedule,
    get_or_build_astar_seed,
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
)


VOYAGE_START_PORT = SHANGHAI_PORT
VOYAGE_END_PORT = BUSAN_PORT
ENFORCE_DEPARTURE_HEADING = False


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
    ga_best_individual : list[float] | None = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-segments", type=int, default=N_SEGMENTS, help="Common segment count.")
    parser.add_argument("--rta-h", type=float, default=RTA_HOURS, help="Common voyage RTA in hours.")
    parser.add_argument("--ga-pop-size", type=int, default=100, help="GA population size for cases 2 and 3.")
    parser.add_argument("--ga-n-gen", type=int, default=1000, help="GA generation count for cases 2 and 3.")
    parser.add_argument("--ga-workers", type=int, default=0, help="GA worker count.")
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for GA cases. Use None for random initialization.")
    parser.add_argument(
        "--ga-cost-resolution",
        type=float,
        default=VERIFICATION_COST_MAP_RESOLUTION,
        help="Cost-map resolution for GA cases.",
    )
    parser.add_argument(
        "--astar-cost-resolution",
        type=float,
        default=VERIFICATION_COST_MAP_RESOLUTION,
        help="Cost-map resolution for the A* baseline.",
    )
    smoothing_group = parser.add_mutually_exclusive_group()
    smoothing_group.add_argument(
        "--astar-smoothing",
        dest="astar_smoothing",
        action="store_true",
        help="Apply waypoint smoothing to cached/generated A* routes.",
    )
    smoothing_group.add_argument(
        "--no-astar-smoothing",
        dest="astar_smoothing",
        action="store_false",
        help="Disable waypoint smoothing and use compressed A* waypoints directly.",
    )
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


def _run_case1(
    env_fn,
    departure_time_utc,
    n_segments: int,
    rta_h: float,
    astar_cost_resolution: float,
    astar_smoothing: bool,
    final_milp_time_limit_sec: int,
) -> CaseRunResult:
    start_time = time.time()
    cost_map = build_cost_map(resolution=astar_cost_resolution)
    _, waypoints, _ = get_or_build_astar_seed(
        cost_map,
        VOYAGE_START_PORT,
        VOYAGE_END_PORT,
        n_segments,
        distance_mode=CASE1_PATH_DISTANCE_MODE,
        apply_smoothing=astar_smoothing,
    )
    base_speed = compute_case1_base_speed_knots(waypoints, rta_h=rta_h)

    route = build_fixed_route(
        waypoints,
        base_speed,
        n_segments=n_segments,
        rta_h=rta_h,
        start_port=VOYAGE_START_PORT,
        end_port=VOYAGE_END_PORT,
    )
    validation = apply_case1_baseline_validation(route, cost_map, env_fn, departure_time_utc)
    _, power_profile, milp_result = solve_route_schedule(
        route,
        env_fn,
        departure_time_utc,
        initial_soc=0.7,
        time_limit_sec=final_milp_time_limit_sec,
    )

    runtime = time.time() - start_time
    return CaseRunResult(
        metrics=_summarize_case(
            "case1_astar_fixed",
            route,
            power_profile,
            objective=milp_result["total_fuel_kg"] if milp_result["feasible"] else None,
            objective_kind="milp_fuel_kg",
            milp_result=milp_result,
            route_violation=validation.land_violation,
            runtime_sec=runtime,
            note=f"astar_res={astar_cost_resolution}, smoothing={'on' if astar_smoothing else 'off'}",
        ),
        route=route,
        cost_map=cost_map,
        power_profile=power_profile,
        milp_result=milp_result,
        ga_logbook=None,
    )


def _run_case2(
    env_fn,
    departure_time_utc,
    n_segments: int,
    rta_h: float,
    ga_pop_size: int,
    ga_n_gen: int,
    ga_workers: int,
    ga_cost_resolution: float,
    seed: int,
    astar_smoothing: bool,
    final_milp_time_limit_sec: int,
) -> CaseRunResult:
    start_time = time.time()
    cost_map = build_cost_map(resolution=ga_cost_resolution)

    _, astar_waypoints, _ = get_or_build_astar_seed(
        cost_map,
        VOYAGE_START_PORT,
        VOYAGE_END_PORT,
        n_segments,
        distance_mode=CASE1_PATH_DISTANCE_MODE,
        apply_smoothing=astar_smoothing,
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
        cost_map=cost_map,
        milp_solver=EnergyObjectiveSolver(),
        env_fn=env_fn,
        departure_time_utc=departure_time_utc,
        n_segments=n_segments,
        rta_h=rta_h,
        pop_size=ga_pop_size,
        n_gen=ga_n_gen,
        seed=seed,
        n_workers=ga_workers,
        astar_seed_waypoints=astar_waypoints,
        seed_individual=case1_seed_individual,
        start_port=VOYAGE_START_PORT,
        end_port=VOYAGE_END_PORT,
        enforce_departure_heading=ENFORCE_DEPARTURE_HEADING,
        genotype_layout=GENOTYPE_DECOUPLED,
    )
    route = ga_result["best_route"]
    _, power_profile, milp_result = solve_route_schedule(
        route,
        env_fn,
        departure_time_utc,
        initial_soc=0.7,
        time_limit_sec=final_milp_time_limit_sec,
    )

    runtime = time.time() - start_time
    return CaseRunResult(
        metrics=_summarize_case(
            "case2_ga_twostage",
            route,
            power_profile,
            objective=ga_result["best_fitness"],
            objective_kind="energy_objective",
            milp_result=milp_result,
            route_violation=route["land_violation"],
            runtime_sec=runtime,
            note=f"ga_res={ga_cost_resolution}, smoothing={'on' if astar_smoothing else 'off'}",
        ),
        route=route,
        cost_map=cost_map,
        power_profile=power_profile,
        milp_result=milp_result,
        ga_logbook=ga_result["logbook"],
        ga_best_individual=list(ga_result["best_individual"])
    )


def _run_case3(
    env_fn,
    departure_time_utc,
    n_segments: int,
    rta_h: float,
    ga_pop_size: int,
    ga_n_gen: int,
    ga_workers: int,
    ga_cost_resolution: float,
    seed: int,
    astar_smoothing: bool,
    ga_milp_time_limit_sec: int,
    final_milp_time_limit_sec: int,
    use_case2_seed_for_case3: bool,
    same_run_case2_best_individual: list[float] | None = None,
) -> CaseRunResult:
    start_time = time.time()
    cost_map = build_cost_map(resolution=ga_cost_resolution)

    _, astar_waypoints, _ = get_or_build_astar_seed(
        cost_map,
        VOYAGE_START_PORT,
        VOYAGE_END_PORT,
        n_segments,
        distance_mode=CASE1_PATH_DISTANCE_MODE,
        apply_smoothing=astar_smoothing,
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
        ga_cost_resolution=ga_cost_resolution,
        departure_time_utc=departure_time_utc,
        astar_seed_smoothing=astar_smoothing,
        genotype_layout=GENOTYPE_DECOUPLED,
        same_run_seed_individual=same_run_case2_best_individual,
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
        cost_map=cost_map,
        milp_solver=milp,
        env_fn=env_fn,
        departure_time_utc=departure_time_utc,
        n_segments=n_segments,
        rta_h=rta_h,
        pop_size=ga_pop_size,
        n_gen=ga_n_gen,
        seed=seed,
        n_workers=ga_workers,
        smoothing_weight=0.0,
        astar_seed_waypoints=astar_waypoints,
        seed_individual=case1_seed_individual,
        seed_individuals=[] if case2_seed_individual is None else [case2_seed_individual],
        start_port=VOYAGE_START_PORT,
        end_port=VOYAGE_END_PORT,
        enforce_departure_heading=ENFORCE_DEPARTURE_HEADING,
        genotype_layout=GENOTYPE_DECOUPLED,
        milp_time_limit_sec=ga_milp_time_limit_sec,
    )
    route = ga_result["best_route"]
    _, power_profile, milp_result = solve_route_schedule(
        route,
        env_fn,
        departure_time_utc,
        initial_soc=0.7,
        time_limit_sec=final_milp_time_limit_sec,
    )

    runtime = time.time() - start_time
    return CaseRunResult(
        metrics=_summarize_case(
            "case3_ga_integrated",
            route,
            power_profile,
            objective=ga_result["best_fitness"],
            objective_kind="ga_objective",
            milp_result=milp_result,
            route_violation=route["land_violation"],
            runtime_sec=runtime,
            note=f"ga_res={ga_cost_resolution}, smoothing={'on' if astar_smoothing else 'off'}",
        ),
        route=route,
        cost_map=cost_map,
        power_profile=power_profile,
        milp_result=milp_result,
        ga_logbook=ga_result["logbook"],
        ga_best_individual=list(ga_result["best_individual"]),
    )


def _print_summary(rows: list[CaseMetrics]) -> None:
    print("\n" + "=" * 120)
    print(" Case Comparison Summary")
    print("=" * 120)
    header = (
        f"{'case':<20} {'valid':<6} {'viol':>8} {'objective':>14} {'milp_fuel':>12} "
        f"{'dist_nm':>10} {'P_mean':>10} {'P_std':>10} {'last_v':>10} {'last_dh':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        objective_text = "None" if row.objective is None else f"{row.objective:.2f}"
        fuel_text = "None" if row.milp_fuel_kg is None else f"{row.milp_fuel_kg:.2f}"
        print(
            f"{row.case_name:<20} {str(row.route_valid):<6} {row.route_violation:>8.1f} "
            f"{objective_text:>14} {fuel_text:>12} "
            f"{row.total_distance_nm:>10.2f} {row.p_req_mean_mw:>10.3f} {row.p_req_std_mw:>10.3f} "
            f"{row.last_speed_kts:>10.2f} {row.final_heading_delta_deg:>10.2f}"
        )
        if row.note:
            print(f"  note: {row.note}")
        print(f"  objective_kind: {row.objective_kind} | milp_feasible: {row.milp_feasible}")


def _save_detail_workbook(case_runs: list[CaseRunResult], output_dir: str) -> str:
    summary_df = pd.DataFrame([asdict(case_run.metrics) for case_run in case_runs])
    segments_df = pd.DataFrame(
        [
            row
            for case_run in case_runs
            for row in build_segment_rows(
                case_run.metrics.case_name,
                case_run.route,
                case_run.power_profile,
                case_run.milp_result,
            )
        ]
    )
    nodes_df = pd.DataFrame(
        [
            row
            for case_run in case_runs
            for row in build_node_rows(
                case_run.metrics.case_name,
                case_run.route,
                case_run.power_profile,
                case_run.milp_result,
            )
        ]
    )

    workbook_path = os.path.join(output_dir, "case_comparison_details.xlsx")
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        segments_df.to_excel(writer, sheet_name="segments", index=False)
        nodes_df.to_excel(writer, sheet_name="nodes", index=False)

    return workbook_path


def _save_summary(rows: list[CaseMetrics], output_dir: str) -> tuple[str, str]:
    json_path = os.path.join(output_dir, "case_comparison_summary.json")
    csv_path = os.path.join(output_dir, "case_comparison_summary.csv")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump([asdict(row) for row in rows], handle, indent=2, ensure_ascii=False)

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    return json_path, csv_path


def _save_case_power_schedules(case_runs: list[CaseRunResult], output_dir: str) -> list[str]:
    saved_dirs: list[str] = []

    for case_run in case_runs:
        if not case_run.milp_result:
            continue

        case_dir = os.path.join(output_dir, case_run.metrics.case_name)
        plot_power_schedule(
            case_run.milp_result,
            power_profile=case_run.power_profile,
            save_dir=case_dir,
        )

        if os.path.exists(os.path.join(case_dir, "power_schedule.png")):
            saved_dirs.append(case_dir)

    return saved_dirs


def _save_case_artifact_payloads(
    case_runs: list[CaseRunResult],
    output_dir: str,
    departure_time_utc,
) -> list[str]:
    saved_dirs: list[str] = []

    for case_run in case_runs:
        case_dir = os.path.join(output_dir, case_run.metrics.case_name)
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
        saved_dirs.append(case_dir)

    return saved_dirs


def _save_case_routes(case_runs: list[CaseRunResult], output_dir: str) -> list[str]:
    saved_dirs: list[str] = []

    for case_run in case_runs:
        if not case_run.route or not case_run.route.get("waypoints"):
            continue

        case_dir = os.path.join(output_dir, case_run.metrics.case_name)
        plot_cost_map_route(
            case_run.route,
            case_run.cost_map,
            save_dir=case_dir,
            filename="route.png",
        )

        if os.path.exists(os.path.join(case_dir, "route.png")):
            saved_dirs.append(case_dir)

    return saved_dirs


def _save_case_convergences(case_runs: list[CaseRunResult], output_dir: str) -> list[str]:
    saved_dirs: list[str] = []
    y_max_by_case = {
    }

    for case_run in case_runs:
        if case_run.ga_logbook is None:
            continue

        case_dir = os.path.join(output_dir, case_run.metrics.case_name)
        plot_convergence(
            case_run.ga_logbook,
            save_dir=case_dir,
            y_max=y_max_by_case.get(case_run.metrics.case_name),
        )

        if os.path.exists(os.path.join(case_dir, "convergence.png")):
            saved_dirs.append(case_dir)

    return saved_dirs


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = ensure_output_dir("comparison")

    print("=" * 80)
    print(" Verification Case Comparison")
    print("=" * 80)
    print(
        f"  Config: segments={args.n_segments}, rta_h={args.rta_h}, "
        f"ga_pop={args.ga_pop_size}, ga_gen={args.ga_n_gen}, ga_workers={args.ga_workers}, seed={args.seed}, "
        f"astar_smoothing={'on' if args.astar_smoothing else 'off'}, "
        f"milp_limits=ga_eval:{args.ga_milp_time_limit_sec}s/final:{args.final_milp_time_limit_sec}s"
    )

    env_loader, env_fn, departure_time_utc = load_marine_environment()
    try:
        case1_run = _run_case1(
            env_fn=env_fn,
            departure_time_utc=departure_time_utc,
            n_segments=args.n_segments,
            rta_h=args.rta_h,
            astar_cost_resolution=args.astar_cost_resolution,
            astar_smoothing=args.astar_smoothing,
            final_milp_time_limit_sec=args.final_milp_time_limit_sec,
        )
        case2_run = _run_case2(
            env_fn=env_fn,
            departure_time_utc=departure_time_utc,
            n_segments=args.n_segments,
            rta_h=args.rta_h,
            ga_pop_size=args.ga_pop_size,
            ga_n_gen=args.ga_n_gen,
            ga_workers=args.ga_workers,
            ga_cost_resolution=args.ga_cost_resolution,
            seed=args.seed,
            astar_smoothing=args.astar_smoothing,
            final_milp_time_limit_sec=args.final_milp_time_limit_sec,
        )
        if case2_run.ga_best_individual is not None:
            cache_path = save_case2_seed_cache(
                best_individual=case2_run.ga_best_individual,
                best_fitness=case2_run.metrics.objective,
                route=case2_run.route,
                start_port=VOYAGE_START_PORT,
                end_port=VOYAGE_END_PORT,
                n_segments=args.n_segments,
                rta_h=args.rta_h,
                ga_cost_resolution=args.ga_cost_resolution,
                departure_time_utc=departure_time_utc,
                astar_seed_smoothing=args.astar_smoothing,
                genotype_layout=GENOTYPE_DECOUPLED,
            )
            print(f"  [Case2 seed] Saved canonical cache: {cache_path}")
        case3_run = _run_case3(
            env_fn=env_fn,
            departure_time_utc=departure_time_utc,
            n_segments=args.n_segments,
            rta_h=args.rta_h,
            ga_pop_size=args.ga_pop_size,
            ga_n_gen=args.ga_n_gen,
            ga_workers=args.ga_workers,
            ga_cost_resolution=args.ga_cost_resolution,
            seed=args.seed,
            astar_smoothing=args.astar_smoothing,
            ga_milp_time_limit_sec=args.ga_milp_time_limit_sec,
            final_milp_time_limit_sec=args.final_milp_time_limit_sec,
            use_case2_seed_for_case3=args.use_case2_seed_for_case3,
            same_run_case2_best_individual=case2_run.ga_best_individual,
        )
        case_runs = [case1_run, case2_run, case3_run]
    finally:
        del env_loader

    rows = [case_run.metrics for case_run in case_runs]
    _print_summary(rows)
    json_path, csv_path = _save_summary(rows, out_dir)
    workbook_path = _save_detail_workbook(case_runs, out_dir)
    artifact_dirs = _save_case_artifact_payloads(case_runs, out_dir, departure_time_utc)
    route_dirs = _save_case_routes(case_runs, out_dir)
    power_schedule_dirs = _save_case_power_schedules(case_runs, out_dir)
    convergence_dirs = _save_case_convergences(case_runs, out_dir)

    comparison_cost_map = build_cost_map(
        resolution=min(args.astar_cost_resolution, args.ga_cost_resolution)
    )
    comparison_route_path = os.path.join(out_dir, "comparison_routes.png")
    plot_route_comparison_map(
        {case_run.metrics.case_name: case_run.route for case_run in case_runs},
        comparison_cost_map,
        save_dir=out_dir,
    )

    print(f"\nSaved summary:")
    print(f"  JSON: {json_path}")
    print(f"  CSV : {csv_path}")
    print(f"  XLSX: {workbook_path}")
    if os.path.exists(comparison_route_path):
        print(f"  Route comparison: {comparison_route_path}")
    if artifact_dirs:
        print("  Artifacts:")
        for case_dir in artifact_dirs:
            print(f"    {os.path.join(case_dir, 'case_artifact.json')}")
    if route_dirs:
        print("  Routes:")
        for case_dir in route_dirs:
            print(f"    {os.path.join(case_dir, 'route.png')}")
    if power_schedule_dirs:
        print("  Power schedules:")
        for case_dir in power_schedule_dirs:
            print(f"    {os.path.join(case_dir, 'power_schedule.png')}")
    if convergence_dirs:
        print("  Convergence plots:")
        for case_dir in convergence_dirs:
            print(f"    {os.path.join(case_dir, 'convergence.png')}")


if __name__ == "__main__":
    main()
