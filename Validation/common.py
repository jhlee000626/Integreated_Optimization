"""
Shared helpers for maintained verification scenarios.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Callable, Mapping, Tuple

import numpy as np

from src.optimizer.ga_engine import (
    apply_route_validation,
    build_required_power_profile,
    evaluate_route_validity,
    ga_gene_count,
)
from src.optimizer.milp_solver import MILPSolver
from src.weather import MarineEnvironmentLoader, resolve_marine_dataset_paths


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SFOC_PATH = os.path.join(PROJECT_ROOT, "config", "sfoc.json")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output", "verification")
VERIFICATION_COST_MAP_RESOLUTION = 0.01


def ensure_output_dir(case_name: str) -> str:
    path = os.path.join(OUTPUT_ROOT, case_name)
    os.makedirs(path, exist_ok=True)
    return path


def _data_dir(project_root: str = PROJECT_ROOT) -> str:
    path = os.path.join(project_root, "data")
    os.makedirs(path, exist_ok=True)
    return path


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return _normalize_for_json(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _normalize_for_json(sub_value) for key, sub_value in value.items()}
    if isinstance(value, tuple):
        return [_normalize_for_json(item) for item in value]
    if isinstance(value, list):
        return [_normalize_for_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_normalize_for_json(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _normalize_for_json(value.item())
    return value


def _serialize_datetime(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _ordered_fieldnames(rows: list[dict]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    return fieldnames


def _write_csv_rows(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline=""):
            return

    fieldnames = _ordered_fieldnames(rows)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_normalize_for_json(row))


def summarize_ga_logbook(logbook) -> dict[str, list[float] | list[int]] | None:
    if logbook is None or not hasattr(logbook, "select"):
        return None

    summary: dict[str, list[float] | list[int]] = {}
    for field in ("gen", "min", "avg", "max"):
        try:
            values = logbook.select(field)
        except Exception:
            continue
        if values is None:
            continue
        summary[field] = [float(value) if field != "gen" else int(value) for value in values]
    return summary or None


def build_case_metrics(
    case_name: str,
    route: dict,
    power_profile: list[dict],
    objective: float | None,
    objective_kind: str,
    milp_result: dict | None,
    route_violation: float,
    runtime_sec: float,
    note: str = "",
) -> dict[str, Any]:
    p_req = np.asarray([segment["P_req"] for segment in power_profile], dtype=float)
    total_distance = float(sum(route["distances_nm"]))
    milp_feasible = bool(milp_result and milp_result.get("feasible"))
    milp_fuel = float(milp_result["total_fuel_kg"]) if milp_feasible else None

    if p_req.size == 0:
        p_req_mean = float("nan")
        p_req_std = float("nan")
        p_req_min = float("nan")
        p_req_max = float("nan")
    else:
        p_req_mean = float(p_req.mean())
        p_req_std = float(p_req.std())
        p_req_min = float(p_req.min())
        p_req_max = float(p_req.max())

    return {
        "case_name": case_name,
        "route_valid": bool(route.get("valid", False)) and float(route_violation) <= 0.0,
        "route_violation": float(route_violation),
        "objective": None if objective is None else float(objective),
        "objective_kind": objective_kind,
        "milp_fuel_kg": milp_fuel,
        "milp_feasible": milp_feasible,
        "total_distance_nm": total_distance,
        "p_req_mean_mw": p_req_mean,
        "p_req_std_mw": p_req_std,
        "p_req_min_mw": p_req_min,
        "p_req_max_mw": p_req_max,
        "last_speed_kts": float(route["last_speed"]),
        "final_heading_delta_deg": float(route["final_heading_delta"]),
        "runtime_sec": float(runtime_sec),
        "note": note,
    }


def build_segment_rows(
    case_name: str,
    route: dict,
    power_profile: list[dict],
    milp_result: dict | None,
) -> list[dict]:
    milp_result = milp_result or {}
    schedule_by_t = {int(step["t"]): step for step in milp_result.get("schedule", [])}
    rows: list[dict] = []

    for segment in power_profile:
        step_index = int(segment["segment"])
        schedule_step = schedule_by_t.get(step_index, {})
        row = {
            "case_name": case_name,
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
            row["milp_total_dg_MW"] = float(
                sum(
                    float(value)
                    for key, value in schedule_step.items()
                    if key.endswith("_P_MW") and key.startswith("DG")
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


def build_node_rows(
    case_name: str,
    route: dict,
    power_profile: list[dict],
    milp_result: dict | None,
) -> list[dict]:
    power_by_segment = {int(segment["segment"]): segment for segment in power_profile}
    schedule_by_t = {int(step["t"]): step for step in (milp_result or {}).get("schedule", [])}
    rows: list[dict] = []

    for node_index, node in enumerate(route.get("nodes", [])):
        row = {
            "case_name": case_name,
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
            row["milp_total_dg_MW"] = float(
                sum(
                    float(value)
                    for key, value in schedule_step.items()
                    if key.endswith("_P_MW") and key.startswith("DG")
                )
            )
            for key, value in schedule_step.items():
                if key in {"t", "dt_h", "P_req_MW"}:
                    continue
                row[f"milp_{key}"] = value

        rows.append(row)

    return rows


def build_schedule_rows(case_name: str, milp_result: dict | None) -> list[dict]:
    schedule = [] if milp_result is None else milp_result.get("schedule", [])
    return [{"case_name": case_name, **dict(step)} for step in schedule]


def _build_ga_seed_payload(
    case_name: str,
    best_individual: list[float],
    best_fitness: float | None,
    route: dict,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "case_name": case_name,
        "best_individual": [float(value) for value in best_individual],
        "best_fitness": None if best_fitness is None else float(best_fitness),
        "route_summary": {
            "waypoints": route.get("waypoints", []),
            "speeds": route.get("speeds", []),
            "headings": route.get("headings", []),
            "valid": route.get("valid"),
            "land_violation": route.get("land_violation"),
        },
        "metadata": dict(metadata or {}),
    }
    return _normalize_for_json(payload)


def save_case_artifacts(
    case_name: str,
    output_dir: str,
    route: dict,
    power_profile: list[dict],
    milp_result: dict | None,
    departure_time_utc: datetime,
    cost_map=None,
    metrics: Mapping[str, Any] | None = None,
    ga_best_individual: list[float] | None = None,
    ga_best_fitness: float | None = None,
    ga_logbook=None,
) -> dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    node_rows = build_node_rows(case_name, route, power_profile, milp_result)
    segment_rows = build_segment_rows(case_name, route, power_profile, milp_result)
    schedule_rows = build_schedule_rows(case_name, milp_result)

    ga_payload = None
    if ga_best_individual is not None:
        ga_payload = _build_ga_seed_payload(
            case_name=case_name,
            best_individual=ga_best_individual,
            best_fitness=ga_best_fitness,
            route=route,
            metadata={
                "departure_time_utc": departure_time_utc,
                "cost_map_resolution": None if cost_map is None else getattr(cost_map, "resolution", None),
            },
        )

    artifact_payload = {
        "case_name": case_name,
        "metrics": None if metrics is None else _normalize_for_json(dict(metrics)),
        "route": _normalize_for_json(route),
        "power_profile": _normalize_for_json(power_profile),
        "milp_result": None
        if milp_result is None
        else {
            "feasible": bool(milp_result.get("feasible")),
            "total_fuel_kg": milp_result.get("total_fuel_kg"),
            "summary": _normalize_for_json(milp_result.get("summary", {})),
            "schedule": _normalize_for_json(milp_result.get("schedule", [])),
        },
        "ga": None
        if ga_payload is None
        else {
            "best_individual": ga_payload["best_individual"],
            "best_fitness": ga_payload["best_fitness"],
            "logbook_summary": summarize_ga_logbook(ga_logbook),
        },
        "departure_time_utc": departure_time_utc,
        "map_bounds": None if cost_map is None else getattr(cost_map, "bounds", None),
        "cost_map_resolution": None if cost_map is None else getattr(cost_map, "resolution", None),
    }

    paths = {
        "case_artifact": os.path.join(output_dir, "case_artifact.json"),
        "route_nodes": os.path.join(output_dir, "route_nodes.csv"),
        "route_segments": os.path.join(output_dir, "route_segments.csv"),
        "schedule": os.path.join(output_dir, "schedule.csv"),
    }
    if ga_payload is not None:
        paths["ga_seed"] = os.path.join(output_dir, "ga_seed.json")

    with open(paths["case_artifact"], "w", encoding="utf-8") as handle:
        json.dump(_normalize_for_json(artifact_payload), handle, indent=2, ensure_ascii=False)

    _write_csv_rows(paths["route_nodes"], node_rows)
    _write_csv_rows(paths["route_segments"], segment_rows)
    _write_csv_rows(paths["schedule"], schedule_rows)

    if ga_payload is not None:
        with open(paths["ga_seed"], "w", encoding="utf-8") as handle:
            json.dump(ga_payload, handle, indent=2, ensure_ascii=False)

    return paths


def _build_case2_seed_metadata(
    start_port: tuple[float, float],
    end_port: tuple[float, float],
    n_segments: int,
    rta_h: float,
    ga_cost_resolution: float,
    departure_time_utc: datetime,
    astar_seed_smoothing: bool | None = None,
    genotype_layout: str | None = None,
) -> dict[str, Any]:
    metadata = {
        "start_port": start_port,
        "end_port": end_port,
        "n_segments": int(n_segments),
        "rta_h": float(rta_h),
        "ga_cost_resolution": float(ga_cost_resolution),
        "departure_time_utc": departure_time_utc,
    }
    if astar_seed_smoothing is not None:
        metadata["astar_seed_smoothing"] = bool(astar_seed_smoothing)
    if genotype_layout is not None:
        metadata["genotype_layout"] = str(genotype_layout)
    return metadata


def _case2_seed_cache_path(
    start_port: tuple[float, float],
    end_port: tuple[float, float],
    n_segments: int,
    rta_h: float,
    ga_cost_resolution: float,
    departure_time_utc: datetime,
    astar_seed_smoothing: bool | None = None,
    genotype_layout: str | None = None,
    data_dir: str | None = None,
) -> tuple[str, dict[str, Any]]:
    metadata = _build_case2_seed_metadata(
        start_port=start_port,
        end_port=end_port,
        n_segments=n_segments,
        rta_h=rta_h,
        ga_cost_resolution=ga_cost_resolution,
        departure_time_utc=departure_time_utc,
        astar_seed_smoothing=astar_seed_smoothing,
        genotype_layout=genotype_layout,
    )
    metadata_blob = json.dumps(_normalize_for_json(metadata), sort_keys=True, separators=(",", ":"))
    cache_hash = hashlib.md5(metadata_blob.encode("utf-8")).hexdigest()
    resolved_data_dir = data_dir or _data_dir()
    return os.path.join(resolved_data_dir, f"case2_seed_{cache_hash}.json"), metadata


def save_case2_seed_cache(
    best_individual: list[float],
    best_fitness: float | None,
    route: dict,
    start_port: tuple[float, float],
    end_port: tuple[float, float],
    n_segments: int,
    rta_h: float,
    ga_cost_resolution: float,
    departure_time_utc: datetime,
    astar_seed_smoothing: bool | None = None,
    genotype_layout: str | None = None,
    data_dir: str | None = None,
) -> str:
    cache_path, metadata = _case2_seed_cache_path(
        start_port=start_port,
        end_port=end_port,
        n_segments=n_segments,
        rta_h=rta_h,
        ga_cost_resolution=ga_cost_resolution,
        departure_time_utc=departure_time_utc,
        astar_seed_smoothing=astar_seed_smoothing,
        genotype_layout=genotype_layout,
        data_dir=data_dir,
    )
    payload = _build_ga_seed_payload(
        case_name="case2_ga_twostage",
        best_individual=best_individual,
        best_fitness=best_fitness,
        route=route,
        metadata=metadata,
    )
    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return cache_path


def load_case2_seed_cache(
    start_port: tuple[float, float],
    end_port: tuple[float, float],
    n_segments: int,
    rta_h: float,
    ga_cost_resolution: float,
    departure_time_utc: datetime,
    astar_seed_smoothing: bool | None = None,
    genotype_layout: str | None = None,
    data_dir: str | None = None,
) -> dict[str, Any] | None:
    cache_path, metadata = _case2_seed_cache_path(
        start_port=start_port,
        end_port=end_port,
        n_segments=n_segments,
        rta_h=rta_h,
        ga_cost_resolution=ga_cost_resolution,
        departure_time_utc=departure_time_utc,
        astar_seed_smoothing=astar_seed_smoothing,
        genotype_layout=genotype_layout,
        data_dir=data_dir,
    )
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        print(f"  [Case2 seed] Failed to load cache {cache_path}: {exc}")
        return None

    expected_metadata = _normalize_for_json(metadata)
    actual_metadata = payload.get("metadata")
    if actual_metadata != expected_metadata:
        print(f"  [Case2 seed] Ignoring cache with mismatched metadata: {cache_path}")
        return None

    return payload


def resolve_case3_seed_individual(
    use_case2_seed: bool,
    start_port: tuple[float, float],
    end_port: tuple[float, float],
    n_segments: int,
    rta_h: float,
    ga_cost_resolution: float,
    departure_time_utc: datetime,
    astar_seed_smoothing: bool | None = None,
    genotype_layout: str | None = None,
    same_run_seed_individual: list[float] | None = None,
    data_dir: str | None = None,
) -> tuple[list[float] | None, str]:
    if not use_case2_seed:
        return None, "disabled"

    expected_gene_count = ga_gene_count(n_segments, genotype_layout)
    if same_run_seed_individual is not None:
        if len(same_run_seed_individual) == expected_gene_count:
            return list(same_run_seed_individual), "same_run"
        print(
            "  [Case2 seed] Ignoring same-run seed with unexpected length: "
            f"{len(same_run_seed_individual)} != {expected_gene_count}"
        )

    payload = load_case2_seed_cache(
        start_port=start_port,
        end_port=end_port,
        n_segments=n_segments,
        rta_h=rta_h,
        ga_cost_resolution=ga_cost_resolution,
        departure_time_utc=departure_time_utc,
        astar_seed_smoothing=astar_seed_smoothing,
        genotype_layout=genotype_layout,
        data_dir=data_dir,
    )
    if payload is None:
        return None, "fallback"

    best_individual = payload.get("best_individual")
    if not isinstance(best_individual, list) or len(best_individual) != expected_gene_count:
        print("  [Case2 seed] Cache payload is missing a compatible best_individual")
        return None, "fallback"

    return [float(value) for value in best_individual], "cache"


def get_or_build_astar_seed(
    cost_map,
    start_port: tuple[float, float],
    end_port: tuple[float, float],
    n_segments: int,
    distance_mode: str = "grid",
    apply_smoothing: bool = True,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], float]:
    """
    Fetch the compressed A* waypoints from cache, or generate and cache them.
    This saves significant computation time across multiple case validations.
    """
    import json
    from src.grid.pathfinding import ASTAR_ALGORITHM_VERSION, DEFAULT_ASTAR_CONNECTIVITY, build_astar_route_points
    from src.grid.pathfinding import haversine_nm

    # Hash the key parameters for the cache filename
    import hashlib
    data_str = (
        f"{start_port}_{end_port}_{n_segments}_{VERIFICATION_COST_MAP_RESOLUTION}_"
        f"{distance_mode}_smooth_{int(bool(apply_smoothing))}_"
        f"conn_{DEFAULT_ASTAR_CONNECTIVITY}_algo_{ASTAR_ALGORITHM_VERSION}"
    )
    cache_hash = hashlib.md5(data_str.encode()).hexdigest()

    # Use the same data directory as other artifacts
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    cache_file = os.path.join(data_dir, f"astar_seed_{cache_hash}.json")

    if os.path.exists(cache_file):
        try:
            smoothing_label = "smoothed" if apply_smoothing else "unsmoothed"
            print(f"  [Cache] Loading {smoothing_label} A* seed from {cache_file}")
            with open(cache_file, "r") as f:
                data = json.load(f)
            # Reconstruct tuples from lists
            raw_path = [tuple(p) for p in data["raw_path"]]
            waypoints = [tuple(p) for p in data["waypoints"]]
            total_dist_nm = data["total_dist_nm"]
            return raw_path, waypoints, total_dist_nm
        except Exception as e:
            print(f"  [Cache] Failed to load cache: {e}. Rebuilding...")

    smoothing_label = "smoothed" if apply_smoothing else "unsmoothed"
    print(f"  [A* build] Generating new {smoothing_label} A* path...")
    raw_path, waypoints, total_dist_nm = build_astar_route_points(
        cost_map,
        start_port,
        end_port,
        n_segments,
        distance_mode=distance_mode,
        apply_smoothing=apply_smoothing,
    )

    # Save to cache
    try:
        with open(cache_file, "w") as f:
            json.dump({
                "raw_path": raw_path,
                "waypoints": waypoints,
                "total_dist_nm": total_dist_nm,
                "apply_smoothing": bool(apply_smoothing),
                "astar_connectivity": DEFAULT_ASTAR_CONNECTIVITY,
            }, f)
        print(f"  [Cache] Saved {smoothing_label} A* seed to {cache_file}")
    except Exception as e:
        print(f"  [Cache] Failed to save cache: {e}")

    return raw_path, waypoints, total_dist_nm


def load_marine_environment(
    era5_path: str | None = None,
    cmems_path: str | None = None,
) -> Tuple[MarineEnvironmentLoader, Callable, datetime]:
    resolved_era5, resolved_cmems = resolve_marine_dataset_paths(
        PROJECT_ROOT, era5_path=era5_path, cmems_path=cmems_path
    )
    loader = MarineEnvironmentLoader(resolved_era5, resolved_cmems)
    from datetime import datetime, timezone

    departure_time_utc = datetime(2025, 3, 25, 6, 0, tzinfo=timezone.utc)

    return loader, loader.get_environment_fn(), departure_time_utc


def make_milp_solver(solver_name: str = "cplex_cmd") -> MILPSolver:
    return MILPSolver(sfoc_json_path=SFOC_PATH, n_pwl_segments=5, solver_name=solver_name)


# ── Weather scenario definitions ──────────────────────────────────────

WEATHER_SCENARIOS: dict[str, dict[str, Any]] = {
    "calm_summer": {
        "era5": "era5_marine_20240715T00_20240716T06.nc",
        "cmems": "cmems_current_20240715T00_20240716T06.nc",
        "departure": "2024-07-15T06:00:00+00:00",
        "label": "Calm Summer (2024-07-15)",
        "description": "North Pacific high → calm seas, Hs<1m, Wind<7m/s",
    },
    "typhoon_bebinca": {
        "era5": "era5_marine_20240914T00_20240915T06.nc",
        "cmems": "cmems_current_20240914T00_20240915T06.nc",
        "departure": "2024-09-14T06:00:00+00:00",
        "label": "Typhoon Bebinca (2024-09-14)",
        "description": "Typhoon Bebinca crossing East China Sea toward Shanghai",
    },
    "winter_severe": {
        "era5": "era5_marine_20250115T00_20250116T06.nc",
        "cmems": "cmems_current_20250115T00_20250116T06.nc",
        "departure": "2025-01-15T06:00:00+00:00",
        "label": "Winter Monsoon (2025-01-15)",
        "description": "Siberian high → strong NW monsoon, Hs 2-3m, Wind 15-20m/s",
    },
}


def _remap_cmems_time_to_era5(cmems_path: str, era5_path: str):
    """Load CMEMS and shift its time axis to overlap with the ERA5 dataset.

    The existing CMEMS current files cover different dates than the scenario
    ERA5 files.  Since ocean currents vary slowly compared to wind/waves, we
    simply remap (shift) the CMEMS time coordinates so they align with the
    ERA5 time window.  This lets ``MarineEnvironmentLoader.from_datasets``
    build a valid shared time grid.
    """
    import xarray as xr

    era5_ds = xr.open_dataset(era5_path)
    cmems_ds = xr.open_dataset(cmems_path)

    # Detect time coordinate names
    era5_time = "valid_time" if "valid_time" in era5_ds.coords else "time"
    cmems_time = "time" if "time" in cmems_ds.coords else "valid_time"

    era5_t0 = era5_ds[era5_time].values[0]
    cmems_t0 = cmems_ds[cmems_time].values[0]
    shift = era5_t0 - cmems_t0

    # Shift CMEMS time to match ERA5 window
    new_time = cmems_ds[cmems_time].values + shift
    cmems_ds = cmems_ds.assign_coords({cmems_time: new_time})

    era5_ds.close()
    return cmems_ds


def load_scenario_environment(
    scenario_name: str,
) -> Tuple[MarineEnvironmentLoader, Callable, datetime]:
    """Load a weather scenario using date-matched ERA5 and CMEMS data.

    Parameters
    ----------
    scenario_name : str
        One of the keys in ``WEATHER_SCENARIOS``.

    Returns
    -------
    loader : MarineEnvironmentLoader
    env_fn : Callable
    departure_time_utc : datetime
    """
    from datetime import datetime as dt_cls

    if scenario_name not in WEATHER_SCENARIOS:
        available = ", ".join(sorted(WEATHER_SCENARIOS))
        raise ValueError(
            f"Unknown scenario '{scenario_name}'. Available: {available}"
        )

    scenario = WEATHER_SCENARIOS[scenario_name]
    era5_dir = os.path.join(PROJECT_ROOT, "data", "era5")
    cmems_dir = os.path.join(PROJECT_ROOT, "data", "cmems")
    era5_path = os.path.join(era5_dir, scenario["era5"])
    cmems_path = os.path.join(cmems_dir, scenario["cmems"])

    if not os.path.exists(era5_path):
        raise FileNotFoundError(
            f"ERA5 file for scenario '{scenario_name}' not found: {era5_path}\n"
            f"Run: python scripts/download_scenario_data.py --scenario {scenario_name}"
        )
    if not os.path.exists(cmems_path):
        raise FileNotFoundError(
            f"CMEMS file for scenario '{scenario_name}' not found: {cmems_path}\n"
            f"Run: python scripts/download_scenario_data.py --scenario {scenario_name}"
        )

    print(f"  [Scenario] {scenario['label']}")
    print(f"  [Scenario] {scenario['description']}")
    print(f"  [ERA5]     {era5_path}")
    print(f"  [CMEMS]    {cmems_path}")

    loader = MarineEnvironmentLoader(era5_path, cmems_path)

    departure_time_utc = dt_cls.fromisoformat(scenario["departure"])
    return loader, loader.get_environment_fn(), departure_time_utc


def solve_route_schedule(
    route: dict,
    env_fn: Callable,
    departure_time_utc: datetime,
    initial_soc: float = 0.7,
    enable_sos2: bool = False,
    time_limit_sec: int = 120,
):
    milp = make_milp_solver(solver_name="cplex_cmd")
    power_profile = build_required_power_profile(
        route,
        env_fn=env_fn,
        departure_time_utc=departure_time_utc,
    )
    result = milp.solve(
        P_req=[segment["P_req"] for segment in power_profile],
        dt=route["dt"],
        initial_SOC=initial_soc,
        time_limit_sec=int(time_limit_sec),
        msg=False,
        enable_sos2=enable_sos2,
    )
    return milp, power_profile, result


def validate_route(
    route: dict,
    cost_map,
    env_fn: Callable,
    departure_time_utc: datetime,
):
    validation = evaluate_route_validity(
        route,
        cost_map=cost_map,
        env_fn=env_fn,
        departure_time_utc=departure_time_utc,
    )
    apply_route_validation(route, validation)
    return validation
