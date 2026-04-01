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
from dataclasses import asdict, dataclass

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Verification.case1_astar_fixed import build_fixed_route
from Verification.case2_ga_twostage import EnergyObjectiveSolver
from Verification.common import (
    VERIFICATION_COST_MAP_RESOLUTION,
    ensure_output_dir,
    load_marine_environment,
    make_milp_solver,
    solve_route_schedule,
    validate_route,
)
from src.grid.cost_map import build_cost_map
from src.grid.pathfinding import build_astar_route_points
from src.grid.no_go_zone import BUSAN_PORT, JEJU_PORT
from src.optimizer.ga_engine import (
    N_SEGMENTS,
    RTA_HOURS,
    setup_ga,
)


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
    note: str = ""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-segments", type=int, default=N_SEGMENTS, help="Common segment count.")
    parser.add_argument("--rta-h", type=float, default=RTA_HOURS, help="Common voyage RTA in hours.")
    parser.add_argument("--ga-pop-size", type=int, default=20, help="GA population size for cases 2 and 3.")
    parser.add_argument("--ga-n-gen", type=int, default=10, help="GA generation count for cases 2 and 3.")
    parser.add_argument("--ga-workers", type=int, default=1, help="GA worker count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for GA cases.")
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
    return parser


def _summarize_case(
    case_name: str,
    route: dict,
    power_profile: list[dict],
    objective: float | None,
    objective_kind: str,
    milp_result: dict | None,
    route_violation: float,
    note: str = "",
) -> CaseMetrics:
    p_req = np.asarray([segment["P_req"] for segment in power_profile], dtype=float)
    total_distance = float(sum(route["distances_nm"]))
    milp_feasible = bool(milp_result and milp_result.get("feasible"))
    milp_fuel = float(milp_result["total_fuel_kg"]) if milp_feasible else None

    return CaseMetrics(
        case_name=case_name,
        route_valid=bool(route.get("valid", False)) and float(route_violation) <= 0.0,
        route_violation=float(route_violation),
        objective=None if objective is None else float(objective),
        objective_kind=objective_kind,
        milp_fuel_kg=milp_fuel,
        milp_feasible=milp_feasible,
        total_distance_nm=total_distance,
        p_req_mean_mw=float(p_req.mean()),
        p_req_std_mw=float(p_req.std()),
        p_req_min_mw=float(p_req.min()),
        p_req_max_mw=float(p_req.max()),
        last_speed_kts=float(route["last_speed"]),
        final_heading_delta_deg=float(route["final_heading_delta"]),
        note=note,
    )


def _run_case1(env_fn, departure_time_utc, n_segments: int, rta_h: float, astar_cost_resolution: float) -> CaseMetrics:
    cost_map = build_cost_map(resolution=astar_cost_resolution)
    _, waypoints, total_dist_nm = build_astar_route_points(cost_map, BUSAN_PORT, JEJU_PORT, n_segments)
    dt_h = rta_h / n_segments
    base_speed = total_dist_nm / (dt_h * (n_segments - 0.6))

    route = build_fixed_route(waypoints, base_speed, n_segments=n_segments, rta_h=rta_h)
    validation = validate_route(route, cost_map, env_fn, departure_time_utc)
    _, power_profile, milp_result = solve_route_schedule(route, env_fn, departure_time_utc, initial_soc=0.7)
    return _summarize_case(
        "case1_astar_fixed",
        route,
        power_profile,
        objective=milp_result["total_fuel_kg"] if milp_result["feasible"] else None,
        objective_kind="milp_fuel_kg",
        milp_result=milp_result,
        route_violation=validation.land_violation,
        note=f"astar_res={astar_cost_resolution}",
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
) -> CaseMetrics:
    cost_map = build_cost_map(resolution=ga_cost_resolution)
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
    )
    route = ga_result["best_route"]
    _, power_profile, milp_result = solve_route_schedule(route, env_fn, departure_time_utc, initial_soc=0.7)
    return _summarize_case(
        "case2_ga_twostage",
        route,
        power_profile,
        objective=ga_result["best_fitness"],
        objective_kind="energy_objective",
        milp_result=milp_result,
        route_violation=route["land_violation"],
        note=f"ga_res={ga_cost_resolution}",
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
) -> CaseMetrics:
    cost_map = build_cost_map(resolution=ga_cost_resolution)
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
    )
    route = ga_result["best_route"]
    _, power_profile, milp_result = solve_route_schedule(route, env_fn, departure_time_utc, initial_soc=0.7)
    return _summarize_case(
        "case3_ga_integrated",
        route,
        power_profile,
        objective=ga_result["best_fitness"],
        objective_kind="ga_objective",
        milp_result=milp_result,
        route_violation=route["land_violation"],
        note=f"ga_res={ga_cost_resolution}",
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


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = ensure_output_dir("comparison")

    print("=" * 80)
    print(" Verification Case Comparison")
    print("=" * 80)
    print(
        f"  Config: segments={args.n_segments}, rta_h={args.rta_h}, "
        f"ga_pop={args.ga_pop_size}, ga_gen={args.ga_n_gen}, ga_workers={args.ga_workers}, seed={args.seed}"
    )

    env_loader, env_fn, departure_time_utc = load_marine_environment()
    try:
        rows = [
            _run_case1(
                env_fn=env_fn,
                departure_time_utc=departure_time_utc,
                n_segments=args.n_segments,
                rta_h=args.rta_h,
                astar_cost_resolution=args.astar_cost_resolution,
            ),
            _run_case2(
                env_fn=env_fn,
                departure_time_utc=departure_time_utc,
                n_segments=args.n_segments,
                rta_h=args.rta_h,
                ga_pop_size=args.ga_pop_size,
                ga_n_gen=args.ga_n_gen,
                ga_workers=args.ga_workers,
                ga_cost_resolution=args.ga_cost_resolution,
                seed=args.seed,
            ),
            _run_case3(
                env_fn=env_fn,
                departure_time_utc=departure_time_utc,
                n_segments=args.n_segments,
                rta_h=args.rta_h,
                ga_pop_size=args.ga_pop_size,
                ga_n_gen=args.ga_n_gen,
                ga_workers=args.ga_workers,
                ga_cost_resolution=args.ga_cost_resolution,
                seed=args.seed,
            ),
        ]
    finally:
        del env_loader

    _print_summary(rows)
    json_path, csv_path = _save_summary(rows, out_dir)
    print(f"\nSaved summary:")
    print(f"  JSON: {json_path}")
    print(f"  CSV : {csv_path}")


if __name__ == "__main__":
    main()
