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
import pandas as pd

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
from src.visualization.plotter import plot_route_comparison_map


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


@dataclass
class CaseRunResult:
    metrics: CaseMetrics
    route: dict
    power_profile: list[dict]
    milp_result: dict | None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-segments", type=int, default=N_SEGMENTS, help="Common segment count.")
    parser.add_argument("--rta-h", type=float, default=RTA_HOURS, help="Common voyage RTA in hours.")
    parser.add_argument("--ga-pop-size", type=int, default=200, help="GA population size for cases 2 and 3.")
    parser.add_argument("--ga-n-gen", type=int, default=200, help="GA generation count for cases 2 and 3.")
    parser.add_argument("--ga-workers", type=int, default=0, help="GA worker count.")
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


def _run_case1(env_fn, departure_time_utc, n_segments: int, rta_h: float, astar_cost_resolution: float) -> CaseRunResult:
    cost_map = build_cost_map(resolution=astar_cost_resolution)
    _, waypoints, total_dist_nm = build_astar_route_points(cost_map, BUSAN_PORT, JEJU_PORT, n_segments)
    dt_h = rta_h / n_segments
    base_speed = total_dist_nm / (dt_h * (n_segments - 0.6))

    route = build_fixed_route(waypoints, base_speed, n_segments=n_segments, rta_h=rta_h)
    validation = validate_route(route, cost_map, env_fn, departure_time_utc)
    _, power_profile, milp_result = solve_route_schedule(route, env_fn, departure_time_utc, initial_soc=0.7)
    return CaseRunResult(
        metrics=_summarize_case(
            "case1_astar_fixed",
            route,
            power_profile,
            objective=milp_result["total_fuel_kg"] if milp_result["feasible"] else None,
            objective_kind="milp_fuel_kg",
            milp_result=milp_result,
            route_violation=validation.land_violation,
            note=f"astar_res={astar_cost_resolution}",
        ),
        route=route,
        power_profile=power_profile,
        milp_result=milp_result,
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
) -> CaseRunResult:
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
    return CaseRunResult(
        metrics=_summarize_case(
            "case2_ga_twostage",
            route,
            power_profile,
            objective=ga_result["best_fitness"],
            objective_kind="energy_objective",
            milp_result=milp_result,
            route_violation=route["land_violation"],
            note=f"ga_res={ga_cost_resolution}",
        ),
        route=route,
        power_profile=power_profile,
        milp_result=milp_result,
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
) -> CaseRunResult:
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
        smoothing_weight=0.0,
    )
    route = ga_result["best_route"]
    _, power_profile, milp_result = solve_route_schedule(route, env_fn, departure_time_utc, initial_soc=0.7)
    return CaseRunResult(
        metrics=_summarize_case(
            "case3_ga_integrated",
            route,
            power_profile,
            objective=ga_result["best_fitness"],
            objective_kind="ga_objective",
            milp_result=milp_result,
            route_violation=route["land_violation"],
            note=f"ga_res={ga_cost_resolution}",
        ),
        route=route,
        power_profile=power_profile,
        milp_result=milp_result,
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


def _serialize_datetime(value) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _segment_rows(case_run: CaseRunResult) -> list[dict]:
    route = case_run.route
    power_profile = case_run.power_profile
    milp_result = case_run.milp_result or {}
    schedule_by_t = {int(step["t"]): step for step in milp_result.get("schedule", [])}
    rows: list[dict] = []

    for segment in power_profile:
        step_index = int(segment["segment"])
        schedule_step = schedule_by_t.get(step_index, {})
        row = {
            "case_name": case_run.metrics.case_name,
            "segment": step_index,
            "phase": segment["phase"],
            "when_utc": _serialize_datetime(segment["when_utc"]),
            "dt_h": float(route["dt"][step_index]),
            "distance_nm": float(route["distances_nm"][step_index]),
            "node_from_lat": float(segment["waypoint_from"][0]),
            "node_from_lon": float(segment["waypoint_from"][1]),
            "node_to_lat": float(segment["waypoint_to"][0]),
            "node_to_lon": float(segment["waypoint_to"][1]),
            "heading_deg": float(segment["heading_deg"]),
            "encounter_angle_deg": float(segment["encounter_angle_deg"]),
            "speed_sog_kts": float(segment["speed_sog_kts"]),
            "speed_stw_kts": float(segment["speed_stw_kts"]),
            "speed_stw_power_kts": float(segment["speed_stw_power_kts"]),
            "current_component_kts": float(segment["current_component_kts"]),
            "wind_speed_ms": float(segment["wind_speed_ms"]),
            "wind_dir_deg": float(segment["wind_dir_deg"]) if segment["wind_dir_deg"] is not None else None,
            "current_speed_ms": float(segment["current_speed_ms"]),
            "current_dir_deg": float(segment["current_dir_deg"]),
            "wave_height_m": float(segment["wave_height_m"]),
            "wave_period_s": float(segment["wave_period_s"]),
            "wave_dir_deg": float(segment["wave_dir_deg"]),
            "P_prop_MW": float(segment["P_prop"]),
            "P_service_MW": float(segment["P_service"]),
            "P_req_MW": float(segment["P_req"]),
            "PD_kW": float(segment["PD_kW"]),
            "BHP_kW": float(segment["BHP_kW"]),
            "total_resistance_N": float(segment["total_resistance_N"]),
            "R_calm_N": float(segment["R_calm_N"]),
            "R_wind_N": float(segment["R_wind_N"]),
            "R_wave_N": float(segment["R_wave_N"]),
            "relative_wind_speed_ms": float(segment["relative_wind_speed_ms"]),
            "relative_wind_dir_deg": float(segment["relative_wind_dir_deg"]),
        }

        if schedule_step:
            row["fuel_step_kg"] = float(
                sum(
                    float(value) * float(schedule_step["dt_h"])
                    for key, value in schedule_step.items()
                    if key.endswith("_FC_kgh")
                )
            )
            for key, value in schedule_step.items():
                if key in {"t", "dt_h", "P_req_MW"}:
                    continue
                row[f"milp_{key}"] = value
        else:
            row["fuel_step_kg"] = None

        rows.append(row)

    return rows


def _node_rows(case_run: CaseRunResult) -> list[dict]:
    route = case_run.route
    power_by_segment = {int(segment["segment"]): segment for segment in case_run.power_profile}
    schedule_by_t = {int(step["t"]): step for step in (case_run.milp_result or {}).get("schedule", [])}
    rows: list[dict] = []

    for node_index, node in enumerate(route.get("nodes", [])):
        row = {
            "case_name": case_run.metrics.case_name,
            "node_index": node_index,
            "time_h": float(node["time_h"]),
            "time_utc": _serialize_datetime(node.get("time_utc")),
            "lat": float(node["lat"]),
            "lon": float(node["lon"]),
            "speed_out_kts": float(node["speed_out_kts"]) if node.get("speed_out_kts") is not None else None,
            "speed_over_ground_kts": (
                float(node["speed_over_ground_kts"])
                if node.get("speed_over_ground_kts") is not None
                else None
            ),
            "speed_through_water_kts": (
                float(node["speed_through_water_kts"])
                if node.get("speed_through_water_kts") is not None
                else None
            ),
            "heading_out_deg": float(node["heading_out_deg"]) if node.get("heading_out_deg") is not None else None,
        }

        segment = power_by_segment.get(node_index)
        schedule_step = schedule_by_t.get(node_index, {})
        if segment is not None:
            row.update(
                {
                    "outgoing_phase": segment["phase"],
                    "outgoing_speed_sog_kts": float(segment["speed_sog_kts"]),
                    "outgoing_speed_stw_kts": float(segment["speed_stw_kts"]),
                    "wind_speed_ms": float(segment["wind_speed_ms"]),
                    "wind_dir_deg": float(segment["wind_dir_deg"]) if segment["wind_dir_deg"] is not None else None,
                    "current_speed_ms": float(segment["current_speed_ms"]),
                    "current_dir_deg": float(segment["current_dir_deg"]),
                    "wave_height_m": float(segment["wave_height_m"]),
                    "wave_period_s": float(segment["wave_period_s"]),
                    "wave_dir_deg": float(segment["wave_dir_deg"]),
                    "P_prop_MW": float(segment["P_prop"]),
                    "P_service_MW": float(segment["P_service"]),
                    "P_req_MW": float(segment["P_req"]),
                }
            )

        if schedule_step:
            row["fuel_step_kg"] = float(
                sum(
                    float(value) * float(schedule_step["dt_h"])
                    for key, value in schedule_step.items()
                    if key.endswith("_FC_kgh")
                )
            )
            row["SOC"] = schedule_step.get("SOC")

        rows.append(row)

    return rows


def _save_detail_workbook(case_runs: list[CaseRunResult], output_dir: str) -> str:
    summary_df = pd.DataFrame([asdict(case_run.metrics) for case_run in case_runs])
    segments_df = pd.DataFrame(
        [row for case_run in case_runs for row in _segment_rows(case_run)]
    )
    nodes_df = pd.DataFrame(
        [row for case_run in case_runs for row in _node_rows(case_run)]
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
        case_runs = [
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

    rows = [case_run.metrics for case_run in case_runs]
    _print_summary(rows)
    json_path, csv_path = _save_summary(rows, out_dir)
    workbook_path = _save_detail_workbook(case_runs, out_dir)

    comparison_cost_map = build_cost_map(
        resolution=min(args.astar_cost_resolution, args.ga_cost_resolution)
    )
    plot_route_comparison_map(
        {case_run.metrics.case_name: case_run.route for case_run in case_runs},
        comparison_cost_map,
        save_dir=out_dir,
    )

    print(f"\nSaved summary:")
    print(f"  JSON: {json_path}")
    print(f"  CSV : {csv_path}")
    print(f"  XLSX: {workbook_path}")


if __name__ == "__main__":
    main()
