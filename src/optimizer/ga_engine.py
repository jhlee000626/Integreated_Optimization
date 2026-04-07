"""
DEAP GA engine for integrated route and speed optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import copy
import math
import multiprocessing
import random
from typing import Callable, Sequence, Tuple

import numpy as np
from deap import algorithms, base, creator, tools

from src.grid.no_go_zone import BUSAN_PORT, JEJU_PORT
from src.resistance.models import EnvironmentData
from src.resistance.modified_dpm import compute_P_req
from src.ship.kcs_specs import POWER_MODEL, SERVICE_LOAD


BIG_PENALTY = 1e12
RTA_HOURS = 11.0
N_SEGMENTS = 22
MAX_HEADING_DELTA = 30.0
FIRST_HEADING_MIN = 160.0
FIRST_HEADING_MAX = 220.0
V_MIN, V_MAX = 5.0, 24.0
GAMMA = 0.7
V_MIN_PORT = V_MIN
V_MAX_PORT = V_MAX * GAMMA
SMOOTHING_WEIGHT = 0.0
DEPARTURE_CORRIDOR_HEADING_STEP_DEG = 2.0
DEPARTURE_CORRIDOR_SPEED_STEP_KTS = 0.5
DEPARTURE_CORRIDOR_MAX_CANDIDATES = 120
MILP_INFEASIBLE_BASE = 100000.0
MILP_POWER_EXCESS_WEIGHT = 40000.0
MILP_ENERGY_DEFICIT_WEIGHT = 12000.0
MILP_RAMP_EXCESS_WEIGHT = 6000.0
LAST_SPEED_PENALTY_LINEAR = 2500.0
LAST_SPEED_PENALTY_QUADRATIC = 1000.0


@dataclass
class RouteValidationResult:
    valid: bool
    land_violation: float
    valid_departure_heading: bool
    valid_turning: bool
    valid_speed: bool
    valid_heading: bool
    last_speed_sog_kts: float
    last_speed_stw_kts: float
    final_heading_delta_deg: float


_worker_cost_map = None
_worker_milp = None
_worker_env_fn = None
_worker_n_segments = N_SEGMENTS
_worker_rta_h = RTA_HOURS
_worker_departure_time_utc = None
_worker_smoothing_weight = SMOOTHING_WEIGHT


def _default_departure_time_utc() -> datetime:
    return datetime(2000, 1, 1, tzinfo=timezone.utc)


def _coerce_departure_time_utc(value: datetime | None) -> datetime:
    if value is None:
        return _default_departure_time_utc()
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _wrap_wind_weather_fn(weather_fn: Callable | None) -> Callable | None:
    if weather_fn is None:
        return None

    def env_fn(lat: float, lon: float, when_utc: datetime) -> EnvironmentData:
        del when_utc
        wind_speed, wind_dir = weather_fn(lat, lon)
        return EnvironmentData(wind_speed_ms=float(wind_speed), wind_dir_deg=float(wind_dir))

    return env_fn


def _normalize_env_fn(env_fn: Callable | None = None, weather_fn: Callable | None = None) -> Callable | None:
    if env_fn is not None:
        return env_fn
    return _wrap_wind_weather_fn(weather_fn)


def _resolve_environment(
    env_fn: Callable | None,
    lat: float,
    lon: float,
    when_utc: datetime,
) -> EnvironmentData:
    if env_fn is None:
        return EnvironmentData()

    try:
        value = env_fn(lat, lon, when_utc)
    except TypeError:
        value = env_fn(lat, lon)

    if isinstance(value, EnvironmentData):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        wind_speed, wind_dir = value
        return EnvironmentData(wind_speed_ms=float(wind_speed), wind_dir_deg=float(wind_dir))
    raise TypeError("env_fn must return EnvironmentData or (wind_speed_ms, wind_dir_deg).")


def _angle_delta_deg(angle_a: float, angle_b: float) -> float:
    return ((angle_a - angle_b + 180.0) % 360.0) - 180.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _current_component_knots(env: EnvironmentData, heading_deg: float) -> float:
    if not math.isfinite(env.current_speed_ms) or not math.isfinite(env.current_dir_deg):
        return 0.0
    if abs(env.current_speed_ms) <= 1e-12:
        return 0.0
    heading_delta = _angle_delta_deg(env.current_dir_deg, heading_deg)
    return (env.current_speed_ms * math.cos(math.radians(heading_delta))) / 0.514444


def compute_speed_through_water_knots(
    v_sog_knots: float,
    heading_deg: float,
    env: EnvironmentData,
) -> float:
    if not math.isfinite(v_sog_knots):
        return float("nan")
    return v_sog_knots - _current_component_knots(env, heading_deg)


def _power_stw_knots(v_stw_knots: float) -> float:
    if not math.isfinite(v_stw_knots):
        return 0.1
    return max(0.1, v_stw_knots)


def _worker_init(
    cost_map_data,
    milp_solver_data,
    env_fn_data,
    n_segments: int,
    rta_h: float,
    departure_time_utc: datetime,
    smoothing_weight: float,
):
    global _worker_cost_map, _worker_milp, _worker_env_fn
    global _worker_n_segments, _worker_rta_h, _worker_departure_time_utc, _worker_smoothing_weight

    _worker_milp = milp_solver_data
    _worker_cost_map = cost_map_data
    _worker_env_fn = env_fn_data
    _worker_n_segments = n_segments
    _worker_rta_h = rta_h
    _worker_departure_time_utc = departure_time_utc
    _worker_smoothing_weight = smoothing_weight


def _evaluate_parallel(individual):
    return evaluate(
        individual,
        cost_map=_worker_cost_map,
        milp_solver=_worker_milp,
        env_fn=_worker_env_fn,
        n_segments=_worker_n_segments,
        rta_h=_worker_rta_h,
        departure_time_utc=_worker_departure_time_utc,
        smoothing_weight=_worker_smoothing_weight,
    )


def move_position(
    lat: float,
    lon: float,
    bearing_deg: float,
    distance_nm: float,
) -> Tuple[float, float]:
    radius_nm = 3440.065
    d = distance_nm / radius_nm
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    brg_r = math.radians(bearing_deg)

    new_lat_r = math.asin(
        math.sin(lat_r) * math.cos(d)
        + math.cos(lat_r) * math.sin(d) * math.cos(brg_r)
    )
    new_lon_r = lon_r + math.atan2(
        math.sin(brg_r) * math.sin(d) * math.cos(lat_r),
        math.cos(d) - math.sin(lat_r) * math.sin(new_lat_r),
    )
    return math.degrees(new_lat_r), math.degrees(new_lon_r)


def compute_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlon = lon2_r - lon1_r
    x = math.sin(dlon) * math.cos(lat2_r)
    y = (
        math.cos(lat1_r) * math.sin(lat2_r)
        - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
    )
    return math.degrees(math.atan2(x, y)) % 360.0


def compute_heading(wp_from: tuple[float, float], wp_to: tuple[float, float]) -> float:
    return compute_bearing(wp_from[0], wp_from[1], wp_to[0], wp_to[1])


def haversine_nm(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    radius_nm = 3440.065
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return radius_nm * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def compute_encounter_angle(
    heading_deg: float,
    wind_dir_deg: float,
    ship_speed_knots: float = 0.0,
    wind_speed_ms: float = 0.0,
) -> float:
    encounter_rad = math.radians(heading_deg - wind_dir_deg)
    encounter_x = ship_speed_knots * 0.514444 + wind_speed_ms * math.cos(encounter_rad)
    encounter_y = wind_speed_ms * math.sin(encounter_rad)
    return math.degrees(math.atan2(encounter_y, encounter_x)) % 360.0


def _make_node(lat: float, lon: float, time_h: float) -> dict:
    return {
        "lat": lat,
        "lon": lon,
        "time_h": time_h,
        "time_utc": None,
        "speed_out_kts": None,
        "speed_over_ground_kts": None,
        "speed_through_water_kts": None,
        "heading_out_deg": None,
    }


def _sog_bounds_for_segment(segment_index: int) -> tuple[float, float]:
    if segment_index == 0:
        return V_MIN_PORT, V_MAX_PORT
    return V_MIN, V_MAX


def decode_route(
    individual: list,
    n_segments: int = N_SEGMENTS,
    rta_h: float = RTA_HOURS,
    env_fn: Callable | None = None,
    departure_time_utc: datetime | None = None,
    cost_map=None,
) -> dict:
    del env_fn, departure_time_utc, cost_map

    n_free = n_segments - 1
    dt_per_seg = rta_h / n_segments

    nodes = [_make_node(BUSAN_PORT[0], BUSAN_PORT[1], 0.0)]
    speeds_sog: list[float] = []
    headings: list[float] = []
    delta_headings: list[float] = []
    distances: list[float] = []
    dt_list: list[float] = []

    current_lat, current_lon = BUSAN_PORT
    base_heading = compute_bearing(*BUSAN_PORT, *JEJU_PORT)
    prev_heading = base_heading

    for segment_index in range(n_free):
        sog_kts = float(individual[2 * segment_index])
        heading_gene = float(individual[2 * segment_index + 1])
        if segment_index == 0:
            heading_deg = heading_gene % 360.0
            delta_deg = _angle_delta_deg(heading_deg, base_heading)
        else:
            delta_deg = heading_gene
            heading_deg = (prev_heading + delta_deg) % 360.0

        dist_nm = sog_kts * dt_per_seg
        next_lat, next_lon = move_position(current_lat, current_lon, heading_deg, dist_nm)

        nodes[-1]["speed_out_kts"] = sog_kts
        nodes[-1]["speed_over_ground_kts"] = sog_kts
        nodes[-1]["heading_out_deg"] = heading_deg

        speeds_sog.append(sog_kts)
        headings.append(heading_deg)
        delta_headings.append(delta_deg)
        distances.append(dist_nm)
        dt_list.append(dt_per_seg)
        nodes.append(_make_node(next_lat, next_lon, (segment_index + 1) * dt_per_seg))

        current_lat, current_lon = next_lat, next_lon
        prev_heading = heading_deg

    final_heading = compute_bearing(current_lat, current_lon, JEJU_PORT[0], JEJU_PORT[1])
    final_dist_nm = haversine_nm((current_lat, current_lon), JEJU_PORT)
    final_sog_kts = final_dist_nm / dt_per_seg if dt_per_seg > 0.0 else 0.0
    final_heading_delta = _angle_delta_deg(final_heading, prev_heading)

    nodes[-1]["speed_out_kts"] = final_sog_kts
    nodes[-1]["speed_over_ground_kts"] = final_sog_kts
    nodes[-1]["heading_out_deg"] = final_heading

    speeds_sog.append(final_sog_kts)
    headings.append(final_heading)
    delta_headings.append(final_heading_delta)
    distances.append(final_dist_nm)
    dt_list.append(dt_per_seg)
    nodes.append(_make_node(JEJU_PORT[0], JEJU_PORT[1], n_segments * dt_per_seg))

    waypoints = [(node["lat"], node["lon"]) for node in nodes]
    return {
        "nodes": nodes,
        "waypoints": waypoints,
        "speeds": list(speeds_sog),
        "speeds_sog": list(speeds_sog),
        "speeds_stw": [None] * len(speeds_sog),
        "headings": headings,
        "delta_headings": delta_headings,
        "dt": dt_list,
        "distances_nm": distances,
        "valid": False,
        "valid_departure_heading": False,
        "valid_turning": True,
        "valid_speed": False, 
        "valid_heading": True,
        "land_violation": 0.0,
        "last_speed": float("nan"),
        "last_speed_sog": final_sog_kts,
        "final_heading_delta": final_heading_delta,
    }


def _build_node_transitions(
    route: dict,
    env_fn: Callable | None,
    departure_time_utc: datetime | None,
) -> list[dict]:
    resolved_env_fn = _normalize_env_fn(env_fn=env_fn)
    departure_time_utc = _coerce_departure_time_utc(departure_time_utc)
    transitions: list[dict] = []
    speeds_stw: list[float] = []

    for segment_index in range(len(route["nodes"]) - 1):
        node = route["nodes"][segment_index]
        next_node = route["nodes"][segment_index + 1]
        when_utc = departure_time_utc + timedelta(hours=float(node["time_h"]))
        env = _resolve_environment(resolved_env_fn, node["lat"], node["lon"], when_utc)
        sog_kts = float(node["speed_out_kts"])
        heading_deg = float(node["heading_out_deg"])
        stw_kts = compute_speed_through_water_knots(sog_kts, heading_deg, env)
        current_component_kts = _current_component_knots(env, heading_deg)

        node["time_utc"] = when_utc
        node["speed_over_ground_kts"] = sog_kts
        node["speed_through_water_kts"] = stw_kts

        transitions.append(
            {
                "segment": segment_index,
                "node": node,
                "next_node": next_node,
                "when_utc": when_utc,
                "environment": env,
                "speed_sog_kts": sog_kts,
                "speed_stw_kts": stw_kts,
                "speed_stw_power_kts": _power_stw_knots(stw_kts),
                "heading_deg": heading_deg,
                "current_component_kts": current_component_kts,
            }
        )
        speeds_stw.append(stw_kts)

    if route["nodes"]:
        last_node = route["nodes"][-1]
        last_node["time_utc"] = departure_time_utc + timedelta(hours=float(last_node["time_h"]))

    route["speeds_stw"] = speeds_stw
    return transitions


def evaluate_route_validity(
    route: dict,
    cost_map,
    env_fn: Callable | None = None,
    departure_time_utc: datetime | None = None,
) -> RouteValidationResult:
    transitions = _build_node_transitions(route, env_fn=env_fn, departure_time_utc=departure_time_utc)

    headings = route["headings"]
    first_heading = headings[0] if headings else compute_bearing(*BUSAN_PORT, *JEJU_PORT)
    valid_departure_heading = FIRST_HEADING_MIN <= first_heading <= FIRST_HEADING_MAX

    internal_deltas = route["delta_headings"][1:-1] if len(route["delta_headings"]) > 2 else []
    valid_turning = all(abs(delta_deg) <= MAX_HEADING_DELTA for delta_deg in internal_deltas)

    final_heading_delta = route["final_heading_delta"]
    valid_heading = abs(final_heading_delta) <= MAX_HEADING_DELTA

    last_speed_sog = route["last_speed_sog"]
    valid_speed = V_MIN_PORT <= last_speed_sog <= V_MAX_PORT

    last_speed_stw = transitions[-1]["speed_stw_kts"] if transitions else float("nan")
    land_violation = compute_violation(route, cost_map) if cost_map is not None else 0.0
    valid = valid_departure_heading and land_violation <= 0.0

    return RouteValidationResult(
        valid=valid,
        land_violation=float(land_violation),
        valid_departure_heading=valid_departure_heading,
        valid_turning=valid_turning,
        valid_speed=valid_speed,
        valid_heading=valid_heading,
        last_speed_sog_kts=float(last_speed_sog),
        last_speed_stw_kts=float(last_speed_stw),
        final_heading_delta_deg=float(final_heading_delta),
    )


def apply_route_validation(route: dict, validation: RouteValidationResult) -> dict:
    route["valid"] = validation.valid
    route["valid_departure_heading"] = validation.valid_departure_heading
    route["valid_turning"] = validation.valid_turning
    route["valid_speed"] = validation.valid_speed
    route["valid_heading"] = validation.valid_heading
    route["land_violation"] = validation.land_violation
    route["last_speed"] = validation.last_speed_stw_kts
    route["last_speed_sog"] = validation.last_speed_sog_kts
    route["final_heading_delta"] = validation.final_heading_delta_deg
    return route


def repair_individual(individual: list) -> list:
    n_free = len(individual) // 2
    for segment_index in range(n_free):
        speed_index = 2 * segment_index
        heading_index = speed_index + 1
        low, high = _sog_bounds_for_segment(segment_index)
        individual[speed_index] = _clamp(float(individual[speed_index]), low, high)
        if segment_index == 0:
            heading = float(individual[heading_index]) % 360.0
            individual[heading_index] = _clamp(heading, FIRST_HEADING_MIN, FIRST_HEADING_MAX)
        else:
            delta_heading = float(individual[heading_index])
            individual[heading_index] = _clamp(delta_heading, -MAX_HEADING_DELTA, MAX_HEADING_DELTA)
    return individual


def _build_departure_corridor(
    cost_map,
    dt_per_seg: float,
    base_bearing: float,
) -> list[tuple[float, float]]:
    if cost_map is None:
        return []

    candidates: list[tuple[float, float, float, float]] = []
    heading_deg = FIRST_HEADING_MIN
    while heading_deg <= FIRST_HEADING_MAX + 1e-9:
        speed_kts = V_MIN_PORT
        while speed_kts <= V_MAX_PORT + 1e-9:
            dist_nm = speed_kts * dt_per_seg
            next_lat, next_lon = move_position(
                BUSAN_PORT[0],
                BUSAN_PORT[1],
                heading_deg,
                dist_nm,
            )
            segment_cost = float(
                cost_map.segment_cost(
                    BUSAN_PORT[0],
                    BUSAN_PORT[1],
                    next_lat,
                    next_lon,
                )
            )
            if segment_cost <= 0.0:
                candidates.append(
                    (
                        round(speed_kts, 3),
                        round(heading_deg, 3),
                        abs(_angle_delta_deg(heading_deg, base_bearing)),
                        abs(speed_kts - 0.5 * (V_MIN_PORT + V_MAX_PORT)),
                    )
                )
            speed_kts += DEPARTURE_CORRIDOR_SPEED_STEP_KTS
        heading_deg += DEPARTURE_CORRIDOR_HEADING_STEP_DEG

    candidates.sort(key=lambda item: (item[2], item[3], item[1]))
    trimmed = candidates[:DEPARTURE_CORRIDOR_MAX_CANDIDATES]
    return [(speed_kts, heading_deg) for speed_kts, heading_deg, _, _ in trimmed]


def _apply_departure_corridor(individual: list, departure_corridor: Sequence[tuple[float, float]]) -> list:
    if not departure_corridor:
        return individual

    target_speed = float(individual[0])
    target_heading = float(individual[1]) % 360.0
    best_speed, best_heading = min(
        departure_corridor,
        key=lambda candidate: (
            abs(_angle_delta_deg(candidate[1], target_heading)),
            abs(candidate[0] - target_speed),
        ),
    )
    individual[0] = best_speed
    individual[1] = best_heading
    return individual


def build_required_power_profile(
    route: dict,
    env_fn: Callable | None = None,
    departure_time_utc: datetime | None = None,
    weather_fn: Callable | None = None,
) -> list[dict]:
    resolved_env_fn = _normalize_env_fn(env_fn=env_fn, weather_fn=weather_fn)
    transitions = _build_node_transitions(route, env_fn=resolved_env_fn, departure_time_utc=departure_time_utc)
    power_profile: list[dict] = []

    for transition in transitions:
        segment_index = transition["segment"]
        heading = transition["heading_deg"]
        env = transition["environment"]
        sog_kts = transition["speed_sog_kts"]
        stw_kts = transition["speed_stw_kts"]
        encounter_deg = compute_encounter_angle(
            heading_deg=heading,
            wind_dir_deg=float(env.wind_dir_deg or 0.0),
            ship_speed_knots=sog_kts,
            wind_speed_ms=float(env.wind_speed_ms),
        )

        if segment_index == 0:
            phase = "departure"
        elif segment_index == len(transitions) - 1:
            phase = "approach"
        else:
            phase = "cruising"

        p_service = SERVICE_LOAD[phase]
        result = compute_P_req(
            v_ship_knots=transition["speed_stw_power_kts"],
            v_wind_ms=float(env.wind_speed_ms),
            encounter_angle_deg=encounter_deg,
            a1=POWER_MODEL["a1"],
            P_service=p_service,
            heading_deg=heading,
            env=env,
        )
        power_profile.append(
            {
                "segment": segment_index,
                "phase": phase,
                "when_utc": transition["when_utc"],
                "node": transition["node"],
                "waypoint_from": (transition["node"]["lat"], transition["node"]["lon"]),
                "waypoint_to": (transition["next_node"]["lat"], transition["next_node"]["lon"]),
                "speed_kts": sog_kts,
                "speed_sog_kts": sog_kts,
                "speed_stw_kts": stw_kts,
                "speed_stw_power_kts": transition["speed_stw_power_kts"],
                "heading_deg": heading,
                "encounter_angle_deg": encounter_deg,
                "environment": env,
                "wind_speed_ms": env.wind_speed_ms,
                "wind_dir_deg": env.wind_dir_deg,
                "current_speed_ms": env.current_speed_ms,
                "current_dir_deg": env.current_dir_deg,
                "wave_height_m": env.wave_height_m,
                "wave_period_s": env.wave_period_s,
                "wave_dir_deg": env.wave_dir_deg,
                "current_component_kts": transition["current_component_kts"],
                "P_service": p_service,
                **result,
            }
        )

    return power_profile


def compute_violation(route: dict, cost_map) -> float:
    return cost_map.route_violation(route["waypoints"])


def compute_smoothing_penalty(power_values: list[float], weight: float = SMOOTHING_WEIGHT) -> float:
    if weight <= 0.0:
        return 0.0
    values = np.asarray(power_values, dtype=float)
    if values.size == 0:
        return 0.0
    mean_load = max(float(values.mean()), 1e-9)
    normalized_std = float(values.std()) / mean_load
    if values.size > 1:
        normalized_mean_abs_ramp = float(np.abs(np.diff(values)).mean()) / mean_load
    else:
        normalized_mean_abs_ramp = 0.0
    return weight * (normalized_std + 0.5 * normalized_mean_abs_ramp)


def compute_last_speed_penalty(last_speed_sog_kts: float) -> float:
    if not math.isfinite(last_speed_sog_kts):
        return BIG_PENALTY
    if V_MIN_PORT <= last_speed_sog_kts <= V_MAX_PORT:
        return 0.0

    if last_speed_sog_kts < V_MIN_PORT:
        deviation = V_MIN_PORT - last_speed_sog_kts
    else:
        deviation = last_speed_sog_kts - V_MAX_PORT

    return (
        LAST_SPEED_PENALTY_LINEAR * deviation
        + LAST_SPEED_PENALTY_QUADRATIC * deviation * deviation
    )


def compute_milp_infeasible_surrogate(
    p_req_list: Sequence[float],
    dt: Sequence[float],
    milp_solver,
    initial_soc: float = 0.7,
) -> float:
    dg_specs = milp_solver.dg_specs
    ess = milp_solver.ess

    total_dg_power_cap = sum(float(spec["P_max"]) for spec in dg_specs.values())
    total_supply_cap = total_dg_power_cap + float(ess["P_dc_max"])
    power_excess_mwh = sum(
        max(0.0, float(power) - total_supply_cap) * float(duration)
        for power, duration in zip(p_req_list, dt)
    )

    total_required_energy = sum(float(power) * float(duration) for power, duration in zip(p_req_list, dt))
    total_dg_energy_cap = total_dg_power_cap * sum(float(duration) for duration in dt)
    ess_deliverable_energy = (
        float(ess["capacity"])
        * max(0.0, float(initial_soc) - float(ess["SOC_min"]))
        * float(ess["eta_dc"])
    )
    energy_deficit_mwh = max(0.0, total_required_energy - total_dg_energy_cap - ess_deliverable_energy)

    ramp_excess_mw = 0.0
    ess_net_ramp_cap = float(ess["P_dc_max"]) + float(ess["P_c_max"])
    for step_index in range(1, len(p_req_list)):
        dg_ramp_cap = sum(
            float(spec["ramp_rate"]) * float(spec["P_max"]) * float(dt[step_index])
            for spec in dg_specs.values()
        )
        total_ramp_cap = dg_ramp_cap + ess_net_ramp_cap
        load_delta = abs(float(p_req_list[step_index]) - float(p_req_list[step_index - 1]))
        ramp_excess_mw += max(0.0, load_delta - total_ramp_cap)

    return (
        MILP_INFEASIBLE_BASE
        + MILP_POWER_EXCESS_WEIGHT * power_excess_mwh
        + MILP_ENERGY_DEFICIT_WEIGHT * energy_deficit_mwh
        + MILP_RAMP_EXCESS_WEIGHT * ramp_excess_mw
    )


def evaluate(
    individual: list,
    cost_map,
    milp_solver,
    env_fn: Callable | None = None,
    weather_fn: Callable | None = None,
    n_segments: int = N_SEGMENTS,
    rta_h: float = RTA_HOURS,
    departure_time_utc: datetime | None = None,
    smoothing_weight: float = SMOOTHING_WEIGHT,
) -> Tuple[float]:
    resolved_env_fn = _normalize_env_fn(env_fn=env_fn, weather_fn=weather_fn)
    route = decode_route(individual, n_segments=n_segments, rta_h=rta_h)
    validation = evaluate_route_validity(
        route,
        cost_map=cost_map,
        env_fn=resolved_env_fn,
        departure_time_utc=departure_time_utc,
    )
    apply_route_validation(route, validation)
    if not validation.valid:
        return (BIG_PENALTY,)
    last_speed_penalty = compute_last_speed_penalty(validation.last_speed_sog_kts)

    power_profile = build_required_power_profile(
        route,
        env_fn=resolved_env_fn,
        departure_time_utc=departure_time_utc,
    )
    p_req_list = [segment["P_req"] for segment in power_profile]
    milp_result = milp_solver.solve(
        P_req=p_req_list,
        dt=route["dt"],
        initial_SOC=0.7,
        msg=False,
    )
    if not milp_result["feasible"]:
        return (
            compute_milp_infeasible_surrogate(
                p_req_list=p_req_list,
                dt=route["dt"],
                milp_solver=milp_solver,
                initial_soc=0.7,
            )
            + last_speed_penalty,
        )

    fuel = float(milp_result["total_fuel_kg"])
    smoothing_penalty = compute_smoothing_penalty(p_req_list, weight=smoothing_weight)
    return (fuel + smoothing_penalty + last_speed_penalty,)


def _evaluate_invalid_individuals(population: Sequence, toolbox: base.Toolbox) -> int:
    invalid_individuals = [individual for individual in population if not individual.fitness.valid]
    if not invalid_individuals:
        return 0

    fitnesses = list(toolbox.map(toolbox.evaluate, invalid_individuals))
    for individual, fitness in zip(invalid_individuals, fitnesses):
        individual.fitness.values = fitness
    return len(invalid_individuals)


def _run_simple_ga(
    population: list,
    toolbox: base.Toolbox,
    n_gen: int,
    cx_prob: float,
    mut_prob: float,
    stats,
    halloffame,
    departure_corridor: Sequence[tuple[float, float]],
    elite_count: int,
):
    logbook = tools.Logbook()
    header = ["gen", "nevals"]
    if stats is not None:
        header.extend(stats.fields)
    logbook.header = header

    nevals = _evaluate_invalid_individuals(population, toolbox)
    if halloffame is not None:
        halloffame.update(population)
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=nevals, **record)
    if record:
        print(logbook.stream)

    for generation in range(1, n_gen + 1):
        elites = []
        if elite_count > 0:
            elites = [toolbox.clone(individual) for individual in tools.selBest(population, min(elite_count, len(population)))]

        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=cx_prob, mutpb=mut_prob)
        for individual in offspring:
            repair_individual(individual)
            _apply_departure_corridor(individual, departure_corridor)

        nevals = _evaluate_invalid_individuals(offspring, toolbox)

        if elites:
            worst_indices = sorted(
                range(len(offspring)),
                key=lambda index: offspring[index].fitness.values[0],
                reverse=True,
            )[:len(elites)]
            for index, elite in zip(worst_indices, elites):
                offspring[index] = elite

        if halloffame is not None:
            halloffame.update(offspring)
        population[:] = offspring
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=generation, nevals=nevals, **record)
        if record:
            print(logbook.stream)

    return population, logbook


def setup_ga(
    cost_map,
    milp_solver,
    env_fn: Callable | None = None,
    weather_fn: Callable | None = None,
    departure_time_utc: datetime | None = None,
    n_segments: int = N_SEGMENTS,
    rta_h: float = RTA_HOURS,
    pop_size: int = 100,
    n_gen: int = 50,
    cx_prob: float = 0.7,
    mut_prob: float = 0.3,
    tournament_size: int = 3,
    seed: int | None = None,
    n_workers: int = 1,
    sfoc_path: str = "config/sfoc.json",
    smoothing_weight: float = SMOOTHING_WEIGHT,
    elite_count: int = 5,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    resolved_env_fn = _normalize_env_fn(env_fn=env_fn, weather_fn=weather_fn)
    departure_time_utc = _coerce_departure_time_utc(departure_time_utc)

    n_free = n_segments - 1
    n_genes = n_free * 2

    if "FitnessMin" not in dir(creator):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in dir(creator):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("clone", copy.deepcopy)
    base_bearing = compute_bearing(*BUSAN_PORT, *JEJU_PORT)
    departure_corridor = _build_departure_corridor(
        cost_map=cost_map,
        dt_per_seg=rta_h / n_segments,
        base_bearing=base_bearing,
    )

    def init_individual():
        genes: list[float] = []
        for segment_index in range(n_free):
            if segment_index == 0:
                if departure_corridor:
                    sog_kts, heading_deg = random.choice(departure_corridor)
                else:
                    low, high = _sog_bounds_for_segment(segment_index)
                    sog_kts = random.uniform(low, high)
                    heading_deg = random.uniform(FIRST_HEADING_MIN, FIRST_HEADING_MAX)
                genes.extend([sog_kts, heading_deg])
            else:
                low, high = _sog_bounds_for_segment(segment_index)
                sog_kts = random.uniform(low, high)
                delta_heading_deg = random.uniform(-MAX_HEADING_DELTA, MAX_HEADING_DELTA)
                genes.extend([sog_kts, delta_heading_deg])
        individual = repair_individual(creator.Individual(genes))
        return _apply_departure_corridor(individual, departure_corridor)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    if n_workers == 0:
        n_workers = multiprocessing.cpu_count()

    pool = None
    if n_workers > 1:
        print(f"  Multiprocessing: {n_workers} workers")
        pool = multiprocessing.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(
                cost_map,
                milp_solver,
                resolved_env_fn,
                n_segments,
                rta_h,
                departure_time_utc,
                smoothing_weight,
            ),
        )
        toolbox.register("map", pool.map)
        toolbox.register("evaluate", _evaluate_parallel)
    else:
        print("  Sequential mode (1 worker)")
        import functools

        toolbox.register("map", map)
        toolbox.register(
            "evaluate",
            functools.partial(
                evaluate,
                cost_map=cost_map,
                milp_solver=milp_solver,
                env_fn=resolved_env_fn,
                n_segments=n_segments,
                rta_h=rta_h,
                departure_time_utc=departure_time_utc,
                smoothing_weight=smoothing_weight,
            ),
        )

    low_bounds = [V_MIN_PORT, FIRST_HEADING_MIN] + [V_MIN, -MAX_HEADING_DELTA] * (n_free - 1)
    up_bounds = [V_MAX_PORT, FIRST_HEADING_MAX] + [V_MAX, MAX_HEADING_DELTA] * (n_free - 1)

    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low_bounds, up=up_bounds, eta=20.0)
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        low=low_bounds,
        up=up_bounds,
        eta=20.0,
        indpb=1.0 / n_genes,
    )
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    hof = tools.HallOfFame(5)
    stats = tools.Statistics(lambda individual: individual.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    print(f"\n{'=' * 60}")
    print(" DEAP GA run")
    print(f" Population: {pop_size} | Generations: {n_gen} | Segments: {n_segments}")
    print(f" RTA: {rta_h}h | SOG bounds: {V_MIN}~{V_MAX} kts")
    print(
        f" Base bearing: {base_bearing:.1f} deg | "
        f"Smoothing weight: {smoothing_weight:.1f} | "
        f"Seed: {'random' if seed is None else seed}"
    )
    if n_workers > 1:
        print(f" Parallel workers: {n_workers}")
    print(f"{'=' * 60}")

    population = toolbox.population(n=pop_size)

    try:
        result_pop, logbook = _run_simple_ga(
            population=population,
            toolbox=toolbox,
            n_gen=n_gen,
            cx_prob=cx_prob,
            mut_prob=mut_prob,
            stats=stats,
            halloffame=hof,
            departure_corridor=departure_corridor,
            elite_count=elite_count,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    best_individual = hof[0]
    best_route = decode_route(list(best_individual), n_segments=n_segments, rta_h=rta_h)
    validation = evaluate_route_validity(
        best_route,
        cost_map=cost_map,
        env_fn=resolved_env_fn,
        departure_time_utc=departure_time_utc,
    )
    apply_route_validation(best_route, validation)

    return {
        "best_individual": list(best_individual),
        "best_fitness": float(best_individual.fitness.values[0]),
        "best_route": best_route,
        "logbook": logbook,
        "population": result_pop,
        "hall_of_fame": hof,
    }
