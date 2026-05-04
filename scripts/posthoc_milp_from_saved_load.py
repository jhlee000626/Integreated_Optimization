"""
Rerun MILP scheduling from a saved load profile without rerunning routing/GA.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.optimizer.milp_solver import MILPSolver
from src.visualization.plotter import plot_power_schedule


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--case-json",
        help="Saved case_artifact.json path. Uses route.dt and power_profile[*].P_req.",
    )
    source_group.add_argument(
        "--route-segments-csv",
        help="Saved route_segments.csv path. Uses dt_h and P_req_MW columns.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to a sibling posthoc_milp folder next to the input.",
    )
    parser.add_argument(
        "--initial-soc",
        type=float,
        default=0.5,
        help="Initial ESS SOC for the post-hoc MILP.",
    )
    parser.add_argument(
        "--initial-u",
        default=None,
        help="Optional DG initial status, e.g. DG1=1,DG2=0,DG3=0",
    )
    parser.add_argument(
        "--solver-name",
        default="cplex_cmd",
        help="MILP solver backend passed to MILPSolver.",
    )
    parser.add_argument(
        "--sfoc-json",
        default=os.path.join(PROJECT_ROOT, "config", "sfoc.json"),
        help="SFOC config JSON path.",
    )
    parser.add_argument(
        "--time-limit-sec",
        type=int,
        default=120,
        help="MILP time limit in seconds.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip power_schedule.png generation.",
    )
    return parser


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _normalize_for_json(sub_value) for key, sub_value in value.items()}
    if isinstance(value, list):
        return [_normalize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_for_json(item) for item in value]
    return value


def _write_csv_rows(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline=""):
            return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_normalize_for_json(row))


def _parse_initial_u(raw_value: str | None) -> dict[str, int] | None:
    if raw_value is None:
        return None

    parsed: dict[str, int] = {}
    for chunk in raw_value.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid --initial-u token: {token}")
        key, value = token.split("=", 1)
        parsed[key.strip()] = int(value.strip())
    return parsed or None


def _default_output_dir(input_path: str) -> str:
    parent_dir = os.path.dirname(os.path.abspath(input_path))
    return os.path.join(parent_dir, "posthoc_milp")


def _load_case_artifact(path: str) -> tuple[str, list[float], list[float], list[dict[str, Any]], dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        artifact = json.load(handle)

    route = artifact.get("route", {})
    power_profile = artifact.get("power_profile", [])
    if not power_profile:
        raise ValueError("case_artifact.json is missing power_profile")

    dt = [float(value) for value in route.get("dt", [])]
    p_req = [float(step["P_req"]) for step in power_profile]
    if not dt:
        raise ValueError("case_artifact.json is missing route.dt")
    if len(dt) != len(p_req):
        raise ValueError(f"Length mismatch: len(dt)={len(dt)} != len(P_req)={len(p_req)}")

    case_name = str(artifact.get("case_name") or os.path.basename(os.path.dirname(os.path.abspath(path))))
    source_info = {
        "source_kind": "case_artifact",
        "source_path": os.path.abspath(path),
        "case_name": case_name,
    }
    return case_name, p_req, dt, power_profile, source_info


def _load_route_segments_csv(path: str) -> tuple[str, list[float], list[float], list[dict[str, Any]], dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError("route_segments.csv is empty")

    p_req_key = "P_req_MW" if "P_req_MW" in rows[0] else "P_req"
    if p_req_key not in rows[0]:
        raise ValueError("route_segments.csv must contain P_req_MW or P_req")
    if "dt_h" not in rows[0]:
        raise ValueError("route_segments.csv must contain dt_h")

    p_req: list[float] = []
    dt: list[float] = []
    power_profile: list[dict[str, Any]] = []
    for row in rows:
        p_req.append(float(row[p_req_key]))
        dt.append(float(row["dt_h"]))
        power_row: dict[str, Any] = {"P_req": float(row[p_req_key])}
        if row.get("speed_sog_kts") not in {None, ""}:
            power_row["speed_sog_kts"] = float(row["speed_sog_kts"])
        power_profile.append(power_row)

    case_name = os.path.basename(os.path.dirname(os.path.abspath(path))) or os.path.splitext(os.path.basename(path))[0]
    source_info = {
        "source_kind": "route_segments_csv",
        "source_path": os.path.abspath(path),
        "case_name": case_name,
    }
    return case_name, p_req, dt, power_profile, source_info


def _load_saved_load(args) -> tuple[str, list[float], list[float], list[dict[str, Any]], dict[str, Any], str]:
    if args.case_json:
        case_name, p_req, dt, power_profile, source_info = _load_case_artifact(args.case_json)
        input_path = args.case_json
    else:
        case_name, p_req, dt, power_profile, source_info = _load_route_segments_csv(args.route_segments_csv)
        input_path = args.route_segments_csv
    return case_name, p_req, dt, power_profile, source_info, input_path


def _build_load_rows(p_req: list[float], dt: list[float], power_profile: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (power, duration) in enumerate(zip(p_req, dt)):
        row = {
            "t": index,
            "dt_h": float(duration),
            "P_req_MW": float(power),
        }
        if index < len(power_profile) and "speed_sog_kts" in power_profile[index]:
            row["speed_sog_kts"] = float(power_profile[index]["speed_sog_kts"])
        rows.append(row)
    return rows


def _plot_if_possible(milp_result: dict, power_profile: list[dict[str, Any]], output_dir: str) -> bool:
    if not milp_result.get("feasible"):
        return False
    if not power_profile:
        plot_power_schedule(milp_result, power_profile=None, save_dir=output_dir)
        return True

    can_use_profile = all("speed_sog_kts" in step for step in power_profile[: len(milp_result.get("schedule", []))])
    plot_power_schedule(
        milp_result,
        power_profile=power_profile if can_use_profile else None,
        save_dir=output_dir,
    )
    return True


def main() -> None:
    args = _build_parser().parse_args()
    initial_u = _parse_initial_u(args.initial_u)

    case_name, p_req, dt, power_profile, source_info, input_path = _load_saved_load(args)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else _default_output_dir(input_path)
    os.makedirs(output_dir, exist_ok=True)

    solver = MILPSolver(
        sfoc_json_path=args.sfoc_json,
        n_pwl_segments=5,
        solver_name=args.solver_name,
    )
    milp_result = solver.solve(
        P_req=p_req,
        dt=dt,
        initial_SOC=args.initial_soc,
        initial_u=initial_u,
        time_limit_sec=args.time_limit_sec,
        msg=False,
    )

    payload = {
        "case_name": case_name,
        "source": source_info,
        "settings": {
            "initial_soc": float(args.initial_soc),
            "initial_u": initial_u,
            "solver_name": args.solver_name,
            "time_limit_sec": int(args.time_limit_sec),
            "sfoc_json": os.path.abspath(args.sfoc_json),
        },
        "input_load": _build_load_rows(p_req, dt, power_profile),
        "milp_result": _normalize_for_json(milp_result),
    }

    json_path = os.path.join(output_dir, "posthoc_milp_result.json")
    schedule_csv_path = os.path.join(output_dir, "posthoc_schedule.csv")
    load_csv_path = os.path.join(output_dir, "posthoc_input_load.csv")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    _write_csv_rows(schedule_csv_path, milp_result.get("schedule", []))
    _write_csv_rows(load_csv_path, payload["input_load"])

    plotted = False
    if not args.no_plot:
        plotted = _plot_if_possible(milp_result, power_profile, output_dir)

    print("=" * 72)
    print(" Post-hoc MILP Scheduling")
    print("=" * 72)
    print(f"  case_name     : {case_name}")
    print(f"  source        : {source_info['source_path']}")
    print(f"  steps         : {len(p_req)}")
    print(f"  initial_soc   : {args.initial_soc:.4f}")
    print(f"  terminal_soc  : {args.initial_soc:.4f} (fixed to initial_soc)")
    print(f"  feasible      : {milp_result['feasible']}")
    print(f"  status        : {milp_result['status']}")
    if milp_result.get("feasible"):
        print(f"  total_fuel_kg : {milp_result['total_fuel_kg']:.6f}")
        summary = milp_result.get("summary", {})
        print(f"  final_soc_out : {summary.get('ess_final_SOC')}")
    print(f"  saved_json    : {json_path}")
    print(f"  saved_schedule: {schedule_csv_path}")
    print(f"  saved_load    : {load_csv_path}")
    if plotted:
        print(f"  saved_plot    : {os.path.join(output_dir, 'power_schedule.png')}")


if __name__ == "__main__":
    main()
