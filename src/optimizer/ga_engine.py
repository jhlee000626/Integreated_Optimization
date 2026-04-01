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
from src.grid.pathfinding import build_astar_route_points
from src.resistance.modified_dpm import compute_P_req
from src.resistance.models import EnvironmentData
from src.ship.kcs_specs import POWER_MODEL, SERVICE_LOAD


BIG_PENALTY = 1e12
SOFT_PENALTY_BASE = 5_000.0
RTA_HOURS = 11.0
N_SEGMENTS = 22
MAX_HEADING_DELTA = 30.0
FIRST_HEADING_MIN = 40.0
FIRST_HEADING_MAX = 220.0
V_MIN, V_MAX = 5.0, 24.0
GAMMA = 0.7
V_MIN_PORT = V_MIN
V_MAX_PORT = V_MAX * GAMMA
SMOOTHING_WEIGHT = 50.0
GUIDED_INIT_FRACTION = 0.7
GUIDED_HEADING_NOISE_DEG = 8.0
GUIDED_SPEED_NOISE_KTS = 1.0


@dataclass
class RouteValidationResult:
    valid: bool
    land_violation: float
    valid_departure_heading: bool
    valid_turning: bool
    valid_speed: bool
    valid_heading: bool
    first_heading_deg: float
    max_internal_heading_delta_deg: float
    last_speed_sog_kts: float
    last_speed_stw_kts: float
    final_heading_delta_deg: float
    departure_excess_deg: float
    max_internal_turn_excess_deg: float
    speed_excess_kts: float
    heading_excess_deg: float


_worker_cost_map = None
_worker_milp = None
_worker_env_fn = None
_worker_n_segments = N_SEGMENTS
_worker_rta_h = RTA_HOURS
_worker_departure_time_utc = None
_worker_smoothing_weight = SMOOTHING_WEIGHT
_worker_generation_progress = None


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


def _clamp_heading_relative(candidate_deg: float, reference_deg: float, max_delta_deg: float) -> float:
    delta_deg = _angle_delta_deg(candidate_deg, reference_deg)
    return (reference_deg + _clamp(delta_deg, -max_delta_deg, max_delta_deg)) % 360.0


def _current_component_knots(env: EnvironmentData, heading_deg: float) -> float:
    if not math.isfinite(env.current_speed_ms) or not math.isfinite(env.current_dir_deg):
        return 0.0
    if abs(env.current_speed_ms) <= 1e-12:
        return 0.0
    heading_delta = _angle_delta_deg(env.current_dir_deg, heading_deg)
    return (env.current_speed_ms * math.cos(math.radians(heading_delta))) / 0.514444


def compute_speed_over_ground_knots(
    v_stw_knots: float,
    heading_deg: float,
    env: EnvironmentData,
) -> float:
    v_sog = v_stw_knots + _current_component_knots(env, heading_deg)
    if not math.isfinite(v_sog):
        return max(0.1, v_stw_knots)
    return max(0.1, v_sog)


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
    sfoc_path: str,
    n_pwl: int,
    env_fn_data,
    n_segments: int,
    rta_h: float,
    departure_time_utc: datetime,
    smoothing_weight: float,
    generation_progress,
):
    global _worker_cost_map, _worker_milp, _worker_env_fn
    global _worker_n_segments, _worker_rta_h, _worker_departure_time_utc
    global _worker_smoothing_weight, _worker_generation_progress

    from src.optimizer.milp_solver import MILPSolver

    _worker_milp = MILPSolver(sfoc_json_path=sfoc_path, n_pwl_segments=n_pwl)
    _worker_cost_map = cost_map_data
    _worker_env_fn = env_fn_data
    _worker_n_segments = n_segments
    _worker_rta_h = rta_h
    _worker_departure_time_utc = departure_time_utc
    _worker_smoothing_weight = smoothing_weight
    _worker_generation_progress = generation_progress


def _evaluate_parallel(individual):
    generation_progress = 1.0
    if _worker_generation_progress is not None:
        generation_progress = float(_worker_generation_progress.value)
    return evaluate(
        individual,
        cost_map=_worker_cost_map,
        milp_solver=_worker_milp,
        env_fn=_worker_env_fn,
        n_segments=_worker_n_segments,
        rta_h=_worker_rta_h,
        departure_time_utc=_worker_departure_time_utc,
        smoothing_weight=_worker_smoothing_weight,
        generation_progress=generation_progress,
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
    encounter = math.degrees(math.atan2(encounter_y, encounter_x)) % 360.0
    return encounter


def _build_nodes(
    waypoints: Sequence[tuple[float, float]],
    speeds_sog: Sequence[float],
    headings: Sequence[float],
    dt_per_seg: float,
    speeds_stw: Sequence[float | None] | None = None,
) -> list[dict]:
    nodes: list[dict] = []
    for index, (lat, lon) in enumerate(waypoints):
        node = {
            "lat": lat,
            "lon": lon,
            "time_h": index * dt_per_seg,
            "time_utc": None,
            "speed_out_kts": speeds_sog[index] if index < len(speeds_sog) else None,
            "speed_over_ground_kts": speeds_sog[index] if index < len(speeds_sog) else None,
            "speed_through_water_kts": speeds_stw[index] if speeds_stw is not None and index < len(speeds_stw) else None,
            "heading_out_deg": headings[index] if index < len(headings) else None,
        }
        nodes.append(node)
    return nodes


def _sog_bounds_for_segment(segment_index: int, n_free: int) -> tuple[float, float]:
    if segment_index == 0:
        return V_MIN_PORT, V_MAX_PORT
    return V_MIN, V_MAX


def _intermediate_delta_headings(headings: Sequence[float]) -> list[float]:
    if len(headings) <= 2:
        return []
    deltas = []
    for index in range(1, len(headings) - 1):
        deltas.append(_angle_delta_deg(headings[index], headings[index - 1]))
    return deltas


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

    speeds_sog = [float(individual[i * 2]) for i in range(n_free)]
    heading_genes = [float(individual[i * 2 + 1]) % 360.0 for i in range(n_free)]

    waypoints = [BUSAN_PORT]
    speeds: list[float] = []
    headings: list[float] = []
    delta_headings: list[float] = []
    distances: list[float] = []
    dt_list: list[float] = []

    current_lat, current_lon = BUSAN_PORT
    base_heading = compute_bearing(*BUSAN_PORT, *JEJU_PORT)
    prev_heading = base_heading

    for index in range(n_free):
        v_sog_kts = speeds_sog[index]
        theta_deg = heading_genes[index]
        delta_deg = _angle_delta_deg(theta_deg, prev_heading)
        dist_nm = v_sog_kts * dt_per_seg
        new_lat, new_lon = move_position(current_lat, current_lon, theta_deg, dist_nm)

        waypoints.append((new_lat, new_lon))
        speeds.append(v_sog_kts)
        headings.append(theta_deg)
        delta_headings.append(delta_deg)
        distances.append(dist_nm)
        dt_list.append(dt_per_seg)

        current_lat, current_lon = new_lat, new_lon
        prev_heading = theta_deg

    dest_lat, dest_lon = JEJU_PORT
    final_heading = compute_bearing(current_lat, current_lon, dest_lat, dest_lon)
    final_dist_nm = haversine_nm((current_lat, current_lon), (dest_lat, dest_lon))
    final_sog_kts = final_dist_nm / dt_per_seg if dt_per_seg > 0.0 else 0.0
    final_heading_delta = _angle_delta_deg(final_heading, prev_heading)

    waypoints.append(JEJU_PORT)
    speeds.append(final_sog_kts)
    headings.append(final_heading)
    delta_headings.append(final_heading_delta)
    distances.append(final_dist_nm)
    dt_list.append(dt_per_seg)

    nodes = _build_nodes(waypoints, speeds, headings, dt_per_seg)

    return {
        "waypoints": waypoints,
        "nodes": nodes,
        "speeds": speeds,
        "speeds_sog": list(speeds),
        "speeds_stw": [None] * len(speeds),
        "headings": headings,
        "delta_headings": delta_headings,
        "dt": dt_list,
        "distances_nm": distances,
        "valid": False,
        "valid_departure_heading": False,
        "valid_turning": False,
        "valid_speed": False,
        "valid_heading": False,
        "land_violation": 0.0,
        "last_speed": float("nan"),
        "last_speed_sog": final_sog_kts,
        "final_heading_delta": final_heading_delta,
       }


def _build_segment_states(
    route: dict,
    env_fn: Callable | None,
    departure_time_utc: datetime | None,
) -> list[dict]:
    resolved_env_fn = _normalize_env_fn(env_fn=env_fn)
    departure_time_utc = _coerce_departure_time_utc(departure_time_utc)
    segment_states: list[dict] = []

    for segment_index in range(len(route["speeds"])):
        node = route["nodes"][segment_index]
        when_utc = departure_time_utc + timedelta(hours=float(node["time_h"]))
        env = _resolve_environment(resolved_env_fn, node["lat"], node["lon"], when_utc)
        sog_kts = float(route["speeds"][segment_index])
        heading_deg = float(route["headings"][segment_index])
        stw_kts = compute_speed_through_water_knots(sog_kts, heading_deg, env)
        current_component_kts = _current_component_knots(env, heading_deg)

        node["time_utc"] = when_utc
        node["speed_out_kts"] = sog_kts
        node["speed_over_ground_kts"] = sog_kts
        node["speed_through_water_kts"] = stw_kts

        segment_states.append(
            {
                "segment": segment_index,
                "node": node,
                "when_utc": when_utc,
                "environment": env,
                "speed_sog_kts": sog_kts,
                "speed_stw_kts": stw_kts,
                "speed_stw_power_kts": _power_stw_knots(stw_kts),
                "heading_deg": heading_deg,
                "current_component_kts": current_component_kts,
                "waypoint_from": route["waypoints"][segment_index],
                "waypoint_to": route["waypoints"][segment_index + 1],
            }
        )

    if route["nodes"]:
        route["nodes"][-1]["time_utc"] = departure_time_utc + timedelta(hours=float(route["nodes"][-1]["time_h"]))

    route["speeds_stw"] = [state["speed_stw_kts"] for state in segment_states]
    return segment_states


def evaluate_route_validity(
    route: dict,
    cost_map,
    env_fn: Callable | None = None,
    departure_time_utc: datetime | None = None,
) -> RouteValidationResult:
    segment_states = _build_segment_states(route, env_fn=env_fn, departure_time_utc=departure_time_utc)
    headings = route["headings"]
    first_heading = headings[0] if headings else compute_bearing(*BUSAN_PORT, *JEJU_PORT)
    valid_departure_heading = FIRST_HEADING_MIN <= first_heading <= FIRST_HEADING_MAX
    departure_excess = 0.0
    if first_heading < FIRST_HEADING_MIN:
        departure_excess = FIRST_HEADING_MIN - first_heading
    elif first_heading > FIRST_HEADING_MAX:
        departure_excess = first_heading - FIRST_HEADING_MAX

    internal_deltas = _intermediate_delta_headings(headings)
    if internal_deltas:
        max_internal_delta = max(abs(delta_deg) for delta_deg in internal_deltas)
    else:
        max_internal_delta = 0.0
    valid_turning = all(abs(delta_deg) <= MAX_HEADING_DELTA for delta_deg in internal_deltas)
    max_internal_turn_excess = max(0.0, max_internal_delta - MAX_HEADING_DELTA)

    final_heading_delta = route["delta_headings"][-1] if route["delta_headings"] else 0.0
    valid_heading = abs(final_heading_delta) <= MAX_HEADING_DELTA
    heading_excess = max(0.0, abs(final_heading_delta) - MAX_HEADING_DELTA)

    last_speed_sog = route["speeds"][-1] if route["speeds"] else 0.0
    last_speed_stw = segment_states[-1]["speed_stw_kts"] if segment_states else float("nan")
    valid_speed = math.isfinite(last_speed_stw) and V_MIN_PORT <= last_speed_stw <= V_MAX_PORT
    speed_excess = 0.0
    if not valid_speed and math.isfinite(last_speed_stw):
        if last_speed_stw < V_MIN_PORT:
            speed_excess = V_MIN_PORT - last_speed_stw
        else:
            speed_excess = last_speed_stw - V_MAX_PORT
    elif not math.isfinite(last_speed_stw):
        speed_excess = V_MAX_PORT

    land_violation = compute_violation(route, cost_map) if cost_map is not None else 0.0
    valid_land = land_violation <= 0.0
    valid = valid_departure_heading and valid_turning and valid_speed and valid_heading and valid_land

    return RouteValidationResult(
        valid=valid,
        land_violation=float(land_violation),
        valid_departure_heading=valid_departure_heading,
        valid_turning=valid_turning,
        valid_speed=valid_speed,
        valid_heading=valid_heading,
        first_heading_deg=float(first_heading),
        max_internal_heading_delta_deg=float(max_internal_delta),
        last_speed_sog_kts=float(last_speed_sog),
        last_speed_stw_kts=float(last_speed_stw),
        final_heading_delta_deg=float(final_heading_delta),
        departure_excess_deg=float(departure_excess),
        max_internal_turn_excess_deg=float(max_internal_turn_excess),
        speed_excess_kts=float(speed_excess),
        heading_excess_deg=float(heading_excess),
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
    route["max_internal_heading_delta"] = validation.max_internal_heading_delta_deg
    return route


def _speed_excess_for_remaining_sog(speed_kts: float, segment_index: int, n_segments: int) -> float:
    if segment_index == n_segments - 1:
        return max(0.0, V_MIN_PORT - speed_kts, speed_kts - V_MAX_PORT)
    return max(0.0, V_MIN - speed_kts, speed_kts - V_MAX)


def _candidate_headings(
    desired_heading: float,
    previous_heading: float,
    segment_index: int,
) -> list[float]:
    if segment_index == 0:
        base = _clamp(desired_heading, FIRST_HEADING_MIN, FIRST_HEADING_MAX)
        candidates = [base]
        for offset in (5.0, 10.0, 15.0, 20.0, 30.0):
            candidates.append(_clamp(base + offset, FIRST_HEADING_MIN, FIRST_HEADING_MAX))
            candidates.append(_clamp(base - offset, FIRST_HEADING_MIN, FIRST_HEADING_MAX))
        candidates.extend([
            FIRST_HEADING_MIN,
            FIRST_HEADING_MAX,
            random.uniform(FIRST_HEADING_MIN, FIRST_HEADING_MAX),
            random.uniform(FIRST_HEADING_MIN, FIRST_HEADING_MAX),
        ])
    else:
        base = _clamp_heading_relative(desired_heading, previous_heading, MAX_HEADING_DELTA)
        candidates = [base]
        for offset in (5.0, 10.0, 15.0, 20.0, 25.0, 30.0):
            candidates.append(_clamp_heading_relative(base + offset, previous_heading, MAX_HEADING_DELTA))
            candidates.append(_clamp_heading_relative(base - offset, previous_heading, MAX_HEADING_DELTA))
        candidates.extend([
            _clamp_heading_relative(random.uniform(0.0, 360.0), previous_heading, MAX_HEADING_DELTA),
            _clamp_heading_relative(random.uniform(0.0, 360.0), previous_heading, MAX_HEADING_DELTA),
        ])
    unique = []
    for heading in candidates:
        value = heading % 360.0
        if not any(abs(_angle_delta_deg(value, existing)) < 1e-6 for existing in unique):
            unique.append(value)
    return unique


def _candidate_speeds(target_sog_kts: float, segment_index: int, n_free: int) -> list[float]:
    low, high = _sog_bounds_for_segment(segment_index, n_free)
    base = _clamp(target_sog_kts, low, high)
    candidates = [base]
    for delta in (0.5, 1.0, 1.5, 2.0):
        candidates.append(_clamp(base + delta, low, high))
        candidates.append(_clamp(base - delta, low, high))
    candidates.extend([
        random.uniform(low, high),
        random.uniform(low, high),
    ])
    unique = []
    for speed in candidates:
        if not any(abs(speed - existing) < 1e-6 for existing in unique):
            unique.append(speed)
    return unique


def _select_rollout_candidate(
    current_position: tuple[float, float],
    previous_heading: float,
    segment_index: int,
    n_segments: int,
    dt_per_seg: float,
    cost_map,
    desired_heading: float,
    target_sog_kts: float,
    corridor_target: tuple[float, float] | None = None,
) -> tuple[float, float, tuple[float, float]]:
    n_free = n_segments - 1
    best: tuple[float, float, tuple[float, float]] | None = None
    best_score = float("inf")
    heading_candidates = _candidate_headings(desired_heading, previous_heading, segment_index)
    speed_candidates = _candidate_speeds(target_sog_kts, segment_index, n_free)
    remaining_after = n_segments - (segment_index + 1)

    for heading in heading_candidates:
        for sog_kts in speed_candidates:
            next_position = move_position(current_position[0], current_position[1], heading, sog_kts * dt_per_seg)
            segment_cost = cost_map.segment_cost(
                current_position[0],
                current_position[1],
                next_position[0],
                next_position[1],
            )
            remaining_dist = haversine_nm(next_position, JEJU_PORT)
            score = segment_cost * 1e9
            if remaining_after > 0:
                required_sog = remaining_dist / max(remaining_after * dt_per_seg, 1e-9)
                score += 100.0 * _speed_excess_for_remaining_sog(required_sog, segment_index + 1, n_segments)
            next_heading_to_dest = compute_heading(next_position, JEJU_PORT)
            score += 10.0 * max(0.0, abs(_angle_delta_deg(next_heading_to_dest, heading)) - MAX_HEADING_DELTA)
            if corridor_target is not None:
                score += 0.25 * haversine_nm(next_position, corridor_target)
            score += 0.05 * abs(_angle_delta_deg(heading, desired_heading))

            if segment_cost <= 0.0 and score < best_score:
                best_score = score
                best = (sog_kts, heading, next_position)
            elif best is None and score < best_score:
                best_score = score
                best = (sog_kts, heading, next_position)

    if best is None:
        fallback_heading = heading_candidates[0]
        fallback_sog = speed_candidates[0]
        fallback_position = move_position(current_position[0], current_position[1], fallback_heading, fallback_sog * dt_per_seg)
        best = (fallback_sog, fallback_heading, fallback_position)
    return best


def repair_individual(individual: list) -> list:
    n_free = len(individual) // 2
    previous_heading = None
    for segment_index in range(n_free):
        speed_index = 2 * segment_index
        heading_index = speed_index + 1
        low, high = _sog_bounds_for_segment(segment_index, n_free)
        individual[speed_index] = _clamp(float(individual[speed_index]), low, high)
        heading = float(individual[heading_index]) % 360.0
        if segment_index == 0:
            heading = _clamp(heading, FIRST_HEADING_MIN, FIRST_HEADING_MAX)
        else:
            heading = _clamp_heading_relative(heading, previous_heading, MAX_HEADING_DELTA)
        individual[heading_index] = heading
        previous_heading = heading
    return individual


def _build_guided_individual(
    n_segments: int,
    rta_h: float,
    cost_map,
    corridor_waypoints: Sequence[tuple[float, float]] | None = None,
    heading_noise_deg: float = GUIDED_HEADING_NOISE_DEG,
    speed_noise_kts: float = GUIDED_SPEED_NOISE_KTS,
):
    n_free = n_segments - 1
    dt_per_seg = rta_h / n_segments
    genes: list[float] = []
    current_position = BUSAN_PORT
    previous_heading = compute_bearing(*BUSAN_PORT, *JEJU_PORT)

    for segment_index in range(n_free):
        corridor_target = None
        if corridor_waypoints is not None and segment_index + 1 < len(corridor_waypoints):
            corridor_target = corridor_waypoints[segment_index + 1]
        target_point = corridor_target or JEJU_PORT
        desired_heading = compute_heading(current_position, target_point)
        desired_heading += random.uniform(-heading_noise_deg, heading_noise_deg)
        remaining_time_h = max((n_segments - segment_index) * dt_per_seg, 1e-9)
        if corridor_target is not None:
            target_sog = haversine_nm(current_position, corridor_target) / dt_per_seg
        else:
            target_sog = haversine_nm(current_position, JEJU_PORT) / remaining_time_h
        target_sog += random.uniform(-speed_noise_kts, speed_noise_kts)

        sog_kts, heading_deg, next_position = _select_rollout_candidate(
            current_position=current_position,
            previous_heading=previous_heading,
            segment_index=segment_index,
            n_segments=n_segments,
            dt_per_seg=dt_per_seg,
            cost_map=cost_map,
            desired_heading=desired_heading,
            target_sog_kts=target_sog,
            corridor_target=corridor_target,
        )
        genes.extend([sog_kts, heading_deg])
        current_position = next_position
        previous_heading = heading_deg

    individual = creator.Individual(genes)
    return repair_individual(individual)


def _build_rollout_individual(
    n_segments: int,
    rta_h: float,
    cost_map,
):
    n_free = n_segments - 1
    dt_per_seg = rta_h / n_segments
    genes: list[float] = []
    current_position = BUSAN_PORT
    previous_heading = compute_bearing(*BUSAN_PORT, *JEJU_PORT)

    for segment_index in range(n_free):
        desired_heading = compute_heading(current_position, JEJU_PORT)
        remaining_time_h = max((n_segments - segment_index) * dt_per_seg, 1e-9)
        target_sog = haversine_nm(current_position, JEJU_PORT) / remaining_time_h
        sog_kts, heading_deg, next_position = _select_rollout_candidate(
            current_position=current_position,
            previous_heading=previous_heading,
            segment_index=segment_index,
            n_segments=n_segments,
            dt_per_seg=dt_per_seg,
            cost_map=cost_map,
            desired_heading=desired_heading,
            target_sog_kts=target_sog,
            corridor_target=None,
        )
        genes.extend([sog_kts, heading_deg])
        current_position = next_position
        previous_heading = heading_deg

    individual = creator.Individual(genes)
    return repair_individual(individual)


def build_required_power_profile(
    route: dict,
    env_fn: Callable | None = None,
    departure_time_utc: datetime | None = None,
    weather_fn: Callable | None = None,
) -> list[dict]:
    resolved_env_fn = _normalize_env_fn(env_fn=env_fn, weather_fn=weather_fn)
    segment_states = _build_segment_states(route, env_fn=resolved_env_fn, departure_time_utc=departure_time_utc)
    power_profile: list[dict] = []

    for segment_index, state in enumerate(segment_states):
        heading = state["heading_deg"]
        env = state["environment"]
        sog_kts = state["speed_sog_kts"]
        stw_kts = state["speed_stw_kts"]
        encounter_deg = compute_encounter_angle(
            heading_deg=heading,
            wind_dir_deg=float(env.wind_dir_deg or 0.0),
            ship_speed_knots=sog_kts,
            wind_speed_ms=float(env.wind_speed_ms),
        )

        if segment_index == 0:
            phase = "departure"
        elif segment_index == len(segment_states) - 1:
            phase = "approach"
        else:
            phase = "cruising"

        p_service = SERVICE_LOAD[phase]
        result = compute_P_req(
            v_ship_knots=state["speed_stw_power_kts"],
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
                "when_utc": state["when_utc"],
                "node": state["node"],
                "waypoint_from": state["waypoint_from"],
                "waypoint_to": state["waypoint_to"],
                "speed_kts": sog_kts,
                "speed_sog_kts": sog_kts,
                "speed_stw_kts": stw_kts,
                "speed_stw_power_kts": state["speed_stw_power_kts"],
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
                "current_component_kts": state["current_component_kts"],
                "P_service": p_service,
                **result,
            }
        )

    return power_profile


def compute_violation(route: dict, cost_map) -> float:
    return cost_map.route_violation(route["waypoints"])


def compute_smoothing_penalty(power_values: list[float], weight: float = SMOOTHING_WEIGHT) -> float:
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


def _violation_severity(validation: RouteValidationResult) -> float:
    severity = validation.land_violation / 100.0
    severity += validation.departure_excess_deg / 180.0
    severity += validation.max_internal_turn_excess_deg / MAX_HEADING_DELTA
    severity += validation.speed_excess_kts / max(V_MAX_PORT, 1e-9)
    severity += validation.heading_excess_deg / MAX_HEADING_DELTA
    return max(0.0, severity)


def compute_violation_penalty(severity: float, generation_progress: float) -> float:
    progress = _clamp(float(generation_progress), 0.0, 1.0)
    hard_weight = progress * progress
    soft_penalty = SOFT_PENALTY_BASE * (1.0 + severity)
    hard_penalty = BIG_PENALTY * (1.0 + severity)
    return (1.0 - hard_weight) * soft_penalty + hard_weight * hard_penalty


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
    generation_progress: float = 1.0,
) -> Tuple[float]:
    resolved_env_fn = _normalize_env_fn(env_fn=env_fn, weather_fn=weather_fn)
    route = decode_route(
        individual,
        n_segments=n_segments,
        rta_h=rta_h,
    )
    validation = evaluate_route_validity(
        route,
        cost_map=cost_map,
        env_fn=resolved_env_fn,
        departure_time_utc=departure_time_utc,
    )
    apply_route_validation(route, validation)

    if not validation.valid:
        severity = _violation_severity(validation)
        return (compute_violation_penalty(severity, generation_progress),)

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
        return (compute_violation_penalty(1.0, generation_progress),)

    fuel = float(milp_result["total_fuel_kg"])
    smoothing_penalty = compute_smoothing_penalty(p_req_list, weight=smoothing_weight)
    return (fuel + smoothing_penalty,)


def _evaluate_invalid_individuals(
    population: Sequence,
    toolbox: base.Toolbox,
    generation_progress: float,
    n_workers: int,
    cost_map,
    milp_solver,
    env_fn: Callable | None,
    n_segments: int,
    rta_h: float,
    departure_time_utc: datetime,
    smoothing_weight: float,
    generation_progress_shared=None,
) -> int:
    invalid_individuals = [individual for individual in population if not individual.fitness.valid]
    if not invalid_individuals:
        return 0

    if n_workers > 1:
        generation_progress_shared.value = generation_progress
        fitnesses = list(toolbox.map(_evaluate_parallel, invalid_individuals))
    else:
        fitnesses = [
            evaluate(
                individual,
                cost_map=cost_map,
                milp_solver=milp_solver,
                env_fn=env_fn,
                n_segments=n_segments,
                rta_h=rta_h,
                departure_time_utc=departure_time_utc,
                smoothing_weight=smoothing_weight,
                generation_progress=generation_progress,
            )
            for individual in invalid_individuals
        ]

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
    n_workers: int,
    cost_map,
    milp_solver,
    env_fn: Callable | None,
    n_segments: int,
    rta_h: float,
    departure_time_utc: datetime,
    smoothing_weight: float,
    generation_progress_shared=None,
):
    logbook = tools.Logbook()
    header = ["gen", "nevals"]
    if stats is not None:
        header.extend(stats.fields)
    logbook.header = header

    nevals = _evaluate_invalid_individuals(
        population,
        toolbox,
        generation_progress=0.0,
        n_workers=n_workers,
        cost_map=cost_map,
        milp_solver=milp_solver,
        env_fn=env_fn,
        n_segments=n_segments,
        rta_h=rta_h,
        departure_time_utc=departure_time_utc,
        smoothing_weight=smoothing_weight,
        generation_progress_shared=generation_progress_shared,
    )
    if halloffame is not None:
        halloffame.update(population)
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=nevals, **record)
    if record:
        print(logbook.stream)

    for generation in range(1, n_gen + 1):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=cx_prob, mutpb=mut_prob)
        for individual in offspring:
            repair_individual(individual)

        nevals = _evaluate_invalid_individuals(
            offspring,
            toolbox,
            generation_progress=float(generation) / max(float(n_gen), 1.0),
            n_workers=n_workers,
            cost_map=cost_map,
            milp_solver=milp_solver,
            env_fn=env_fn,
            n_segments=n_segments,
            rta_h=rta_h,
            departure_time_utc=departure_time_utc,
            smoothing_weight=smoothing_weight,
            generation_progress_shared=generation_progress_shared,
        )
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
    seed: int = 42,
    n_workers: int = 1,
    sfoc_path: str = "config/sfoc.json",
    smoothing_weight: float = SMOOTHING_WEIGHT,
):
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

    corridor_waypoints = None
    try:
        _, corridor_waypoints, _ = build_astar_route_points(cost_map, BUSAN_PORT, JEJU_PORT, n_segments)
    except Exception:
        corridor_waypoints = None

    def init_individual():
        if random.random() < GUIDED_INIT_FRACTION:
            return _build_guided_individual(
                n_segments=n_segments,
                rta_h=rta_h,
                cost_map=cost_map,
                corridor_waypoints=corridor_waypoints,
            )
        return _build_rollout_individual(
            n_segments=n_segments,
            rta_h=rta_h,
            cost_map=cost_map,
        )

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    if n_workers == 0:
        n_workers = multiprocessing.cpu_count()

    pool = None
    generation_progress_shared = None
    if n_workers > 1:
        print(f"  Multiprocessing: {n_workers} workers")
        generation_progress_shared = multiprocessing.Value("d", 0.0)
        pool = multiprocessing.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(
                cost_map,
                sfoc_path,
                5,
                resolved_env_fn,
                n_segments,
                rta_h,
                departure_time_utc,
                smoothing_weight,
                generation_progress_shared,
            ),
        )
        toolbox.register("map", pool.map)
    else:
        print("  Sequential mode (1 worker)")

    low_bounds = [V_MIN_PORT, FIRST_HEADING_MIN] + [V_MIN, 0.0] * (n_free - 1)
    up_bounds = [V_MAX_PORT, FIRST_HEADING_MAX] + [V_MAX, 360.0] * (n_free - 1)

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
    print(f" Base bearing: {base_bearing:.1f} deg | Smoothing weight: {smoothing_weight:.1f}")
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
            n_workers=n_workers,
            cost_map=cost_map,
            milp_solver=milp_solver,
            env_fn=resolved_env_fn,
            n_segments=n_segments,
            rta_h=rta_h,
            departure_time_utc=departure_time_utc,
            smoothing_weight=smoothing_weight,
            generation_progress_shared=generation_progress_shared,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    best_individual = hof[0]
    best_route = decode_route(
        list(best_individual),
        n_segments=n_segments,
        rta_h=rta_h,
    )
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
