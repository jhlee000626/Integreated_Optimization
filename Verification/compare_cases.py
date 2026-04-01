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
import math
import os
import sys
from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Verification.case1_astar_fixed import build_astar_graph, get_closest_node, interpolate_path
from Verification.case2_ga_twostage import EnergyObjectiveSolver
from Verification.common import ensure_output_dir, load_marine_environment, make_milp_solver, solve_route_schedule
from src.grid.cost_map import build_cost_map
from src.grid.no_go_zone import BUSAN_PORT, JEJU_PORT
from src.optimizer.ga_engine import (
    N_SEGMENTS,
    RTA_HOURS,
    build_required_power_profile,
    compute_heading,
    compute_violation,
    haversine_nm,
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
        default=0.01,
        help="Cost-map resolution for GA cases.",
    )
    parser.add_argument(
        "--astar-cost-resolution",
        type=float,
        default=0.1,
        help="Cost-map resolution for the A* baseline.",
    )
    return parser


def _build_fixed_route(waypoints: list[tuple[float, float]], base_speed_knots: float, n_segments: int, rta_h: float) -> dict:
    dt_h = rta_h / n_segments
    route = {
        "waypoints": waypoints,
        "speeds": [],
        "headings": [],
        "delta_headings": [],
        "dt": [dt_h] * n_segments,
        "distances_nm": [],
        "valid": True,
        "valid_departure_heading": True,
        "valid_speed": True,
        "valid_heading": True,
        "last_speed": 0.0,
        "final_heading_delta": 0.0,
    }

    prev_heading = compute_heading(BUSAN_PORT, JEJU_PORT)
    for idx in range(n_segments):
        wp_from = waypoints[idx]
        wp_to = waypoints[idx + 1]
        heading = compute_heading(wp_from, wp_to)
        speed = base_speed_knots * 0.7 if idx == 0 or idx == n_segments - 1 else base_speed_knots
        delta = ((heading - prev_heading + 180.0) % 360.0) - 180.0

        route["speeds"].append(speed)
        route["headings"].append(heading)
        route["delta_headings"].append(delta)
        route["distances_nm"].append(haversine_nm(wp_from, wp_to))
        prev_heading = heading

    route["last_speed"] = route["speeds"][-1]
    route["final_heading_delta"] = route["delta_headings"][-1]
    route["nodes"] = [
        {
            "lat": wp[0],
            "lon": wp[1],
            "time_h": idx * dt_h,
            "time_utc": None,
            "speed_out_kts": route["speeds"][idx] if idx < n_segments else None,
            "heading_out_deg": route["headings"][idx] if idx < n_segments else None,
        }
        for idx, wp in enumerate(route["waypoints"])
    ]
    return route


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
    graph = build_astar_graph(cost_map)
    start_node = get_closest_node(cost_map, BUSAN_PORT, graph)
    goal_node = get_closest_node(cost_map, JEJU_PORT, graph)

    def heuristic(node_a, node_b):
        return math.hypot(node_b[0] - node_a[0], node_b[1] - node_a[1]) * cost_map.resolution

    path_idx = nx.astar_path(graph, start_node, goal_node, heuristic=heuristic, weight="weight")
    raw_path = [(cost_map.lats[i], cost_map.lons[j]) for i, j in path_idx]
    waypoints, total_dist_nm = interpolate_path(raw_path, n_segments)
    dt_h = rta_h / n_segments
    base_speed = total_dist_nm / (dt_h * (n_segments - 0.6))

    route = _build_fixed_route(waypoints, base_speed, n_segments=n_segments, rta_h=rta_h)
    power_profile = build_required_power_profile(route, env_fn=env_fn, departure_time_utc=departure_time_utc)
    milp = make_milp_solver()
    milp_result = milp.solve(
        P_req=[segment["P_req"] for segment in power_profile],
        dt=route["dt"],
        initial_SOC=0.7,
        msg=False,
    )
    return _summarize_case(
        "case1_astar_fixed",
        route,
        power_profile,
        objective=milp_result["total_fuel_kg"] if milp_result["feasible"] else None,
        objective_kind="milp_fuel_kg",
        milp_result=milp_result,
        route_violation=cost_map.route_violation(route["waypoints"]),
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
        route_violation=compute_violation(route, cost_map),
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
        route_violation=compute_violation(route, cost_map),
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
