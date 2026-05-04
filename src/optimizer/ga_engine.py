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

from src.grid.no_go_zone import BUSAN_PORT, SHANGHAI_PORT
from src.resistance.models import EnvironmentData
from src.resistance.modified_dpm import compute_P_req
from src.ship.kcs_specs import SERVICE_LOAD


BIG_PENALTY = 1e12
RTA_HOURS = 24.0
N_SEGMENTS = 48
MAX_HEADING_DELTA = 30.0
FIRST_HEADING_MIN = 135.0
FIRST_HEADING_MAX = 240.0
V_MIN, V_MAX = 5.0, 22.0
V_MIN_PORT = V_MIN
V_MAX_PORT = V_MAX
SMOOTHING_WEIGHT = 0.0
GENOTYPE_COUPLED = "coupled"
GENOTYPE_DECOUPLED = "decoupled"
DEPARTURE_CORRIDOR_HEADING_STEP_DEG = 2.0
DEPARTURE_CORRIDOR_SPEED_STEP_KTS = 0.5
DEPARTURE_CORRIDOR_MAX_CANDIDATES = 120
MILP_INFEASIBLE_BASE = 100000.0
MILP_POWER_EXCESS_WEIGHT = 40000.0
MILP_ENERGY_DEFICIT_WEIGHT = 12000.0
MILP_RAMP_EXCESS_WEIGHT = 6000.0
LAST_SPEED_PENALTY_LINEAR = 15000.0
LAST_SPEED_PENALTY_QUADRATIC = 15000.0
SPEED_DISTANCE_MISMATCH_PENALTY_WEIGHT = 50000.0

# ── 점진적 패널티 상수 ──
LAND_PENALTY_BASE = 200000.0      # 육지 위반 시 기본 패널티 (정상 연료량 7.3만보다 무조건 높아야 함)
LAND_PENALTY_WEIGHT = 100000.0    # 위반 정도에 비례하는 가중치
HEADING_PENALTY_WEIGHT = 50000.0  # 출항 방위각 위반 패널티


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
_worker_start_port = BUSAN_PORT
_worker_end_port = SHANGHAI_PORT
_worker_enforce_departure_heading = True
_worker_genotype_layout = GENOTYPE_COUPLED
_worker_milp_time_limit_sec = 120


def _default_departure_time_utc() -> datetime:
    return datetime(2000, 1, 1, tzinfo=timezone.utc)


def _coerce_departure_time_utc(value: datetime | None) -> datetime:
    if value is None:
        return _default_departure_time_utc()
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _first_heading_bounds(enforce_departure_heading: bool) -> tuple[float, float]:
    if enforce_departure_heading:
        return FIRST_HEADING_MIN, FIRST_HEADING_MAX
    return 0.0, 360.0


def normalize_genotype_layout(genotype_layout: str | None) -> str:
    if genotype_layout in (None, "", GENOTYPE_COUPLED):
        return GENOTYPE_COUPLED
    if genotype_layout == GENOTYPE_DECOUPLED:
        return GENOTYPE_DECOUPLED
    raise ValueError(
        f"Unsupported genotype_layout={genotype_layout!r}. "
        f"Use {GENOTYPE_COUPLED!r} or {GENOTYPE_DECOUPLED!r}."
    )


def ga_gene_count(n_segments: int, genotype_layout: str | None = GENOTYPE_COUPLED) -> int:
    n_free = max(0, int(n_segments) - 1)
    layout = normalize_genotype_layout(genotype_layout)
    if layout == GENOTYPE_DECOUPLED:
        return n_free * 3 + 1
    return n_free * 2


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
    start_port: tuple[float, float],
    end_port: tuple[float, float],
    enforce_departure_heading: bool,
    genotype_layout: str,
    milp_time_limit_sec: int,
):
    global _worker_cost_map, _worker_milp, _worker_env_fn
    global _worker_n_segments, _worker_rta_h, _worker_departure_time_utc, _worker_smoothing_weight
    global _worker_start_port, _worker_end_port, _worker_enforce_departure_heading, _worker_genotype_layout
    global _worker_milp_time_limit_sec

    _worker_milp = milp_solver_data
    _worker_cost_map = cost_map_data
    _worker_env_fn = env_fn_data
    _worker_n_segments = n_segments
    _worker_rta_h = rta_h
    _worker_departure_time_utc = departure_time_utc
    _worker_smoothing_weight = smoothing_weight
    _worker_start_port = start_port
    _worker_end_port = end_port
    _worker_enforce_departure_heading = enforce_departure_heading
    _worker_genotype_layout = normalize_genotype_layout(genotype_layout)
    _worker_milp_time_limit_sec = int(milp_time_limit_sec)


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
        start_port=_worker_start_port,
        end_port=_worker_end_port,
        enforce_departure_heading=_worker_enforce_departure_heading,
        genotype_layout=_worker_genotype_layout,
        milp_time_limit_sec=_worker_milp_time_limit_sec,
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
    start_port: tuple[float, float] = BUSAN_PORT,
    end_port: tuple[float, float] = SHANGHAI_PORT,
    genotype_layout: str | None = GENOTYPE_COUPLED,
) -> dict:
    layout = normalize_genotype_layout(genotype_layout)
    if layout == GENOTYPE_DECOUPLED:
        return _decode_route_decoupled(
            individual,
            n_segments=n_segments,
            rta_h=rta_h,
            env_fn=env_fn,
            departure_time_utc=departure_time_utc,
            cost_map=cost_map,
            start_port=start_port,
            end_port=end_port,
        )

    del env_fn, departure_time_utc, cost_map

    n_free = n_segments - 1
    dt_per_seg = rta_h / n_segments

    nodes = [_make_node(start_port[0], start_port[1], 0.0)]
    speeds_sog: list[float] = []
    headings: list[float] = []
    delta_headings: list[float] = []
    distances: list[float] = []
    dt_list: list[float] = []

    current_lat, current_lon = start_port
    base_heading = compute_bearing(*start_port, *end_port)
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

    final_heading = compute_bearing(current_lat, current_lon, end_port[0], end_port[1])
    final_dist_nm = haversine_nm((current_lat, current_lon), end_port)
    final_sog_kts = final_dist_nm / dt_per_seg if dt_per_seg > 0.0 else 0.0
    final_heading_delta = _angle_delta_deg(final_heading, prev_heading)

    # ── 마지막 구간 속도 보정: 속도 편차를 전체에 분산 ──
    if speeds_sog and final_sog_kts > 0.0:
        total_dist = sum(distances) + final_dist_nm
        total_time = n_segments * dt_per_seg
        desired_avg = total_dist / total_time if total_time > 0.0 else final_sog_kts
        current_avg = (sum(speeds_sog) + final_sog_kts) / n_segments
        if current_avg > 1e-6:
            scale = desired_avg / current_avg
            # 위반 방지를 위해 보수적 스케일링 (±20%)
            scale = _clamp(scale, 0.8, 1.2)
            speeds_sog = [spd * scale for spd in speeds_sog]
            final_sog_kts *= scale
            distances = [spd * dt_per_seg for spd in speeds_sog]
            final_dist_nm = final_sog_kts * dt_per_seg
            # 보정된 좌표 재계산
            current_lat, current_lon = start_port
            for seg_i in range(n_free):
                nodes[seg_i]["speed_out_kts"] = speeds_sog[seg_i]
                nodes[seg_i]["speed_over_ground_kts"] = speeds_sog[seg_i]
                next_lat, next_lon = move_position(
                    current_lat, current_lon, headings[seg_i], distances[seg_i]
                )
                nodes[seg_i + 1]["lat"] = next_lat
                nodes[seg_i + 1]["lon"] = next_lon
                current_lat, current_lon = next_lat, next_lon
            # 마지막 구간 방위각 재계산
            final_heading = compute_bearing(current_lat, current_lon, end_port[0], end_port[1])
            final_dist_nm = haversine_nm((current_lat, current_lon), end_port)
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
    nodes.append(_make_node(end_port[0], end_port[1], n_segments * dt_per_seg))

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
        "start_port": tuple(start_port),
        "end_port": tuple(end_port),
        "genotype_layout": GENOTYPE_COUPLED,
    }


def _decode_route_decoupled(
    individual: list,
    n_segments: int = N_SEGMENTS,
    rta_h: float = RTA_HOURS,
    env_fn: Callable | None = None,
    departure_time_utc: datetime | None = None,
    cost_map=None,
    start_port: tuple[float, float] = BUSAN_PORT,
    end_port: tuple[float, float] = SHANGHAI_PORT,
) -> dict:
    del env_fn, departure_time_utc, cost_map

    n_free = n_segments - 1
    dt_per_seg = rta_h / n_segments

    nodes = [_make_node(start_port[0], start_port[1], 0.0)]
    speeds_sog: list[float] = []
    headings: list[float] = []
    delta_headings: list[float] = []
    distances: list[float] = []
    dt_list: list[float] = []

    current_lat, current_lon = start_port
    base_heading = compute_bearing(*start_port, *end_port)
    prev_heading = base_heading

    for segment_index in range(n_free):
        gene_index = 3 * segment_index
        sog_kts = float(individual[gene_index])
        dist_nm = max(0.0, float(individual[gene_index + 1]))
        heading_gene = float(individual[gene_index + 2])
        if segment_index == 0:
            heading_deg = heading_gene % 360.0
            delta_deg = _angle_delta_deg(heading_deg, base_heading)
        else:
            delta_deg = heading_gene
            heading_deg = (prev_heading + delta_deg) % 360.0

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

    final_heading = compute_bearing(current_lat, current_lon, end_port[0], end_port[1])
    final_dist_nm = haversine_nm((current_lat, current_lon), end_port)
    final_speed_index = 3 * n_free
    if final_speed_index < len(individual):
        final_sog_kts = float(individual[final_speed_index])
    else:
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
    nodes.append(_make_node(end_port[0], end_port[1], n_segments * dt_per_seg))

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
        "start_port": tuple(start_port),
        "end_port": tuple(end_port),
        "genotype_layout": GENOTYPE_DECOUPLED,
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
    start_port: tuple[float, float] = BUSAN_PORT,
    end_port: tuple[float, float] = SHANGHAI_PORT,
    enforce_departure_heading: bool = True,
) -> RouteValidationResult:
    transitions = _build_node_transitions(route, env_fn=env_fn, departure_time_utc=departure_time_utc)

    headings = route["headings"]
    first_heading = headings[0] if headings else compute_bearing(*start_port, *end_port)
    if enforce_departure_heading:
        valid_departure_heading = FIRST_HEADING_MIN <= first_heading <= FIRST_HEADING_MAX
    else:
        valid_departure_heading = True

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


def repair_individual(
    individual: list,
    enforce_departure_heading: bool = True,
    genotype_layout: str | None = GENOTYPE_COUPLED,
    dt_per_seg: float | None = None,
) -> list:
    layout = normalize_genotype_layout(genotype_layout)
    if layout == GENOTYPE_DECOUPLED:
        n_free = max(0, (len(individual) - 1) // 3)
        distance_dt = 1.0 if dt_per_seg is None else float(dt_per_seg)
        for segment_index in range(n_free):
            gene_index = 3 * segment_index
            speed_index = gene_index
            distance_index = gene_index + 1
            heading_index = gene_index + 2
            low, high = _sog_bounds_for_segment(segment_index)
            individual[speed_index] = _clamp(float(individual[speed_index]), low, high)
            individual[distance_index] = _clamp(
                float(individual[distance_index]),
                low * distance_dt,
                high * distance_dt,
            )
            if segment_index == 0:
                heading = float(individual[heading_index]) % 360.0
                if enforce_departure_heading:
                    individual[heading_index] = _clamp(heading, FIRST_HEADING_MIN, FIRST_HEADING_MAX)
                else:
                    individual[heading_index] = heading
            else:
                delta_heading = float(individual[heading_index])
                individual[heading_index] = _clamp(delta_heading, -MAX_HEADING_DELTA, MAX_HEADING_DELTA)

        final_speed_index = 3 * n_free
        if final_speed_index < len(individual):
            individual[final_speed_index] = _clamp(
                float(individual[final_speed_index]),
                V_MIN_PORT,
                V_MAX_PORT,
            )
        return individual

    n_free = len(individual) // 2
    for segment_index in range(n_free):
        speed_index = 2 * segment_index
        heading_index = speed_index + 1
        low, high = _sog_bounds_for_segment(segment_index)
        individual[speed_index] = _clamp(float(individual[speed_index]), low, high)
        if segment_index == 0:
            heading = float(individual[heading_index]) % 360.0
            if enforce_departure_heading:
                individual[heading_index] = _clamp(heading, FIRST_HEADING_MIN, FIRST_HEADING_MAX)
            else:
                individual[heading_index] = heading
        else:
            delta_heading = float(individual[heading_index])
            individual[heading_index] = _clamp(delta_heading, -MAX_HEADING_DELTA, MAX_HEADING_DELTA)
    return individual


def _build_departure_corridor(
    cost_map,
    dt_per_seg: float,
    base_bearing: float,
    start_port: tuple[float, float] = BUSAN_PORT,
    enforce_departure_heading: bool = True,
) -> list[tuple[float, float]]:
    if cost_map is None or not enforce_departure_heading:
        return []

    candidates: list[tuple[float, float, float, float]] = []
    heading_deg = FIRST_HEADING_MIN
    while heading_deg <= FIRST_HEADING_MAX + 1e-9:
        speed_kts = V_MIN_PORT
        while speed_kts <= V_MAX_PORT + 1e-9:
            dist_nm = speed_kts * dt_per_seg
            next_lat, next_lon = move_position(
                start_port[0],
                start_port[1],
                heading_deg,
                dist_nm,
            )
            segment_cost = float(
                cost_map.segment_cost(
                    start_port[0],
                    start_port[1],
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
    # 항만 좌표 전진 배치로, 더 이상 첫 구간 궤적을 이산(Discrete)된 안전 지대에 강제로 맞추지 않습니다.
    # GA가 소수점 단위의 정밀한 연속 최적화를 수행하도록 원본 그대로 리턴합니다.
    return individual


def _adjust_astar_seed_waypoints_for_departure(
    waypoints: Sequence[tuple[float, float]],
    departure_corridor: Sequence[tuple[float, float]],
    cost_map,
    n_segments: int,
    rta_h: float,
    start_port: tuple[float, float] = BUSAN_PORT,
    enforce_departure_heading: bool = True,
) -> list[tuple[float, float]]:
    """Project the first A* leg into the valid departure corridor with minimal drift."""
    adjusted = list(waypoints)
    if not enforce_departure_heading or len(adjusted) < 2 or not departure_corridor:
        return adjusted

    original_heading = compute_bearing(*adjusted[0], *adjusted[1])
    if FIRST_HEADING_MIN <= original_heading <= FIRST_HEADING_MAX:
        return adjusted

    dt_per_seg = rta_h / n_segments
    if dt_per_seg <= 0.0:
        return adjusted

    original_wp1 = adjusted[1]
    original_wp2 = adjusted[2] if len(adjusted) > 2 else None
    original_speed = haversine_nm(adjusted[0], adjusted[1]) / dt_per_seg

    best_choice = None
    best_score = None

    for speed_kts, heading_deg in departure_corridor:
        candidate_wp1 = move_position(
            start_port[0],
            start_port[1],
            heading_deg,
            speed_kts * dt_per_seg,
        )

        second_leg_violation = 0.0
        turn_excess = 0.0
        if original_wp2 is not None:
            if cost_map is not None:
                second_leg_violation = float(
                    cost_map.segment_cost(
                        candidate_wp1[0],
                        candidate_wp1[1],
                        original_wp2[0],
                        original_wp2[1],
                    )
                )
            next_heading = compute_bearing(
                candidate_wp1[0],
                candidate_wp1[1],
                original_wp2[0],
                original_wp2[1],
            )
            turn_excess = max(
                0.0,
                abs(_angle_delta_deg(next_heading, heading_deg)) - MAX_HEADING_DELTA,
            )

        score = (
            second_leg_violation > 0.0,
            second_leg_violation,
            haversine_nm(candidate_wp1, original_wp1),
            turn_excess,
            abs(_angle_delta_deg(heading_deg, original_heading)),
            abs(speed_kts - original_speed),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_choice = (candidate_wp1, speed_kts, heading_deg)

    if best_choice is None:
        return adjusted

    adjusted[1] = best_choice[0]
    print(
        "  [A* seed] Adjusted departure: "
        f"heading {original_heading:.2f} -> {best_choice[2]:.2f} deg, "
        f"speed {original_speed:.2f} -> {best_choice[1]:.2f} kts"
    )
    return adjusted


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


def compute_speed_distance_mismatch_penalty(
    route: dict,
    weight: float = SPEED_DISTANCE_MISMATCH_PENALTY_WEIGHT,
) -> float:
    if weight <= 0.0:
        return 0.0
    distance_nm = sum(float(value) for value in route.get("distances_nm", []))
    speed_distance_nm = sum(
        float(speed) * float(duration)
        for speed, duration in zip(route.get("speeds_sog", route.get("speeds", [])), route.get("dt", []))
    )
    mismatch_nm = abs(distance_nm - speed_distance_nm)
    if mismatch_nm <= 1e-6:
        return 0.0
    return weight * mismatch_nm


def compute_milp_infeasible_surrogate(
    p_req_list: Sequence[float],
    dt: Sequence[float],
    milp_solver,
    initial_soc: float = 0.7,
) -> float:
    dg_specs = milp_solver.dg_specs
    ess = milp_solver.ess

    total_dg_power_cap = sum(float(spec["P_max"]) for spec in dg_specs.values())
    ess_power_cap = float(ess["power_limit"])
    total_supply_cap = total_dg_power_cap + ess_power_cap
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

    return (
        MILP_INFEASIBLE_BASE
        + MILP_POWER_EXCESS_WEIGHT * power_excess_mwh
        + MILP_ENERGY_DEFICIT_WEIGHT * energy_deficit_mwh
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
    start_port: tuple[float, float] = BUSAN_PORT,
    end_port: tuple[float, float] = SHANGHAI_PORT,
    enforce_departure_heading: bool = True,
    genotype_layout: str | None = GENOTYPE_COUPLED,
    milp_time_limit_sec: int = 120,
) -> Tuple[float]:
    resolved_env_fn = _normalize_env_fn(env_fn=env_fn, weather_fn=weather_fn)
    layout = normalize_genotype_layout(genotype_layout)
    route = decode_route(
        individual,
        n_segments=n_segments,
        rta_h=rta_h,
        start_port=start_port,
        end_port=end_port,
        genotype_layout=layout,
    )
    validation = evaluate_route_validity(
        route,
        cost_map=cost_map,
        env_fn=resolved_env_fn,
        departure_time_utc=departure_time_utc,
        start_port=start_port,
        end_port=end_port,
        enforce_departure_heading=enforce_departure_heading,
    )
    apply_route_validation(route, validation)

    # ── 점진적 패널티: BIG_PENALTY 대신 위반 정도에 비례 ──
    land_penalty = 0.0
    heading_penalty = 0.0
    if validation.land_violation > 0.0:
        land_penalty = LAND_PENALTY_BASE + LAND_PENALTY_WEIGHT * validation.land_violation
    if not validation.valid_departure_heading:
        first_heading = route["headings"][0] if route["headings"] else 0.0
        heading_dev = min(
            abs(_angle_delta_deg(first_heading, FIRST_HEADING_MIN)),
            abs(_angle_delta_deg(first_heading, FIRST_HEADING_MAX)),
        )
        heading_penalty = HEADING_PENALTY_WEIGHT * heading_dev

    constraint_penalty = land_penalty + heading_penalty
    if constraint_penalty > 0.0:
        last_speed_penalty = compute_last_speed_penalty(validation.last_speed_sog_kts)
        return (constraint_penalty + last_speed_penalty,)

    last_speed_penalty = compute_last_speed_penalty(validation.last_speed_sog_kts)
    speed_distance_penalty = (
        compute_speed_distance_mismatch_penalty(route)
        if layout == GENOTYPE_DECOUPLED
        else 0.0
    )

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
        time_limit_sec=int(milp_time_limit_sec),
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
            + last_speed_penalty
            + speed_distance_penalty,
        )

    fuel = float(milp_result["total_fuel_kg"])
    smoothing_penalty = compute_smoothing_penalty(p_req_list, weight=smoothing_weight)
    return (fuel + smoothing_penalty + last_speed_penalty + speed_distance_penalty,)


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
    enforce_departure_heading: bool,
    genotype_layout: str | None = GENOTYPE_COUPLED,
    dt_per_seg: float | None = None,
):
    layout = normalize_genotype_layout(genotype_layout)

    logbook = tools.Logbook()
    generation_records: list[dict] = []
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
            repair_individual(
                individual,
                enforce_departure_heading=enforce_departure_heading,
                genotype_layout=layout,
                dt_per_seg=dt_per_seg,
            )
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

    return population, logbook, generation_records


def encode_astar_seed(
    waypoints: list[tuple[float, float]],
    n_segments: int = N_SEGMENTS,
    rta_h: float = RTA_HOURS,
    genotype_layout: str | None = GENOTYPE_COUPLED,
) -> list[float]:
    """
    A* 경로의 waypoints를 GA 유전자(개체) 형식으로 인코딩.

    Parameters
    ----------
    waypoints : list of (lat, lon)
        A* 경로의 waypoint 리스트 (n_segments + 1 개)
    n_segments : int
        전체 구간 수
    rta_h : float
        목표 도착 시간 (h)

    Returns
    -------
    list[float]
        GA individual 유전자 리스트 [speed0, heading0, speed1, delta1, ...]
    """
    layout = normalize_genotype_layout(genotype_layout)
    n_free = n_segments - 1
    dt_per_seg = rta_h / n_segments

    genes: list[float] = []
    if len(waypoints) >= 2:
        base_heading = compute_bearing(waypoints[0][0], waypoints[0][1], waypoints[-1][0], waypoints[-1][1])
    else:
        base_heading = 0.0
    prev_heading = base_heading

    for seg_idx in range(n_free):
        wp_from = waypoints[seg_idx]
        wp_to = waypoints[seg_idx + 1]

        # ── 기하학적 궤적 오차(Drift) 방지 ──
        # A* 구간별 실제 거리가 미세하게 다르므로, 정확한 개별 구간 속도를 사용해야
        # GA의 decode_route에서 거리를 계산할 때 원래 A* 경로 좌표를 정확히 따라갑니다!
        # (균일 속도를 쓰면 거리가 안 맞아서 좌표가 누적 점프되며 땅에 부딪힙니다.)
        dist_nm = haversine_nm(wp_from, wp_to)
        sog_kts = dist_nm / dt_per_seg if dt_per_seg > 0.0 else 0.0

        # GA 제약조건(V_MIN, V_MAX 등)으로 강제 변환 시
        # 기하학적인 거리가 틀어져 육지에 들이박으므로 원본 속도(sog_kts) 그대로 인코딩합니다.

        # 방위각 계산
        heading_deg = compute_bearing(wp_from[0], wp_from[1], wp_to[0], wp_to[1])

        if layout == GENOTYPE_DECOUPLED:
            if seg_idx == 0:
                genes.extend([sog_kts, dist_nm, heading_deg])
            else:
                delta = _angle_delta_deg(heading_deg, prev_heading)
                genes.extend([sog_kts, dist_nm, delta])
        elif seg_idx == 0:
            # 첫 구간: 절대 방위각 (시드의 원본 궤적 유지를 위해 강제 꺾임 방지)
            genes.extend([sog_kts, heading_deg])
        else:
            # 이후 구간: 이전 대비 변화량 (delta)
            delta = _angle_delta_deg(heading_deg, prev_heading)
            genes.extend([sog_kts, delta])

        prev_heading = heading_deg

    if layout == GENOTYPE_DECOUPLED:
        final_sog_kts = 0.0
        if len(waypoints) >= n_segments + 1:
            final_dist_nm = haversine_nm(waypoints[n_free], waypoints[n_segments])
            final_sog_kts = final_dist_nm / dt_per_seg if dt_per_seg > 0.0 else 0.0
        genes.append(final_sog_kts)

    return genes


def encode_fixed_speed_route_seed(
    waypoints: list[tuple[float, float]],
    base_speed_knots: float,
    n_segments: int = N_SEGMENTS,
    genotype_layout: str | None = GENOTYPE_COUPLED,
) -> list[float]:
    """Encode a fixed-speed waypoint route into the GA gene layout."""
    layout = normalize_genotype_layout(genotype_layout)
    n_free = n_segments - 1
    genes: list[float] = []
    if len(waypoints) >= 2:
        base_heading = compute_bearing(waypoints[0][0], waypoints[0][1], waypoints[-1][0], waypoints[-1][1])
    else:
        base_heading = 0.0
    prev_heading = base_heading

    for seg_idx in range(min(n_free, len(waypoints) - 1)):
        wp_from = waypoints[seg_idx]
        wp_to = waypoints[seg_idx + 1]
        heading_deg = compute_bearing(wp_from[0], wp_from[1], wp_to[0], wp_to[1])
        dist_nm = haversine_nm(wp_from, wp_to)

        if layout == GENOTYPE_DECOUPLED:
            if seg_idx == 0:
                genes.extend([float(base_speed_knots), dist_nm, heading_deg])
            else:
                genes.extend([float(base_speed_knots), dist_nm, _angle_delta_deg(heading_deg, prev_heading)])
        elif seg_idx == 0:
            genes.extend([float(base_speed_knots), heading_deg])
        else:
            genes.extend([float(base_speed_knots), _angle_delta_deg(heading_deg, prev_heading)])

        prev_heading = heading_deg

    if layout == GENOTYPE_DECOUPLED:
        genes.append(float(base_speed_knots))

    return genes


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
    astar_seed_waypoints: list[tuple[float, float]] | None = None,
    seed_individual: list[float] | None = None,
    seed_individuals: Sequence[list[float]] | None = None,
    capture_generation_records: bool = True,
    start_port: tuple[float, float] = BUSAN_PORT,
    end_port: tuple[float, float] = SHANGHAI_PORT,
    enforce_departure_heading: bool = True,
    genotype_layout: str | None = GENOTYPE_COUPLED,
    milp_time_limit_sec: int = 120,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    resolved_env_fn = _normalize_env_fn(env_fn=env_fn, weather_fn=weather_fn)
    departure_time_utc = _coerce_departure_time_utc(departure_time_utc)
    layout = normalize_genotype_layout(genotype_layout)
    dt_per_seg = rta_h / n_segments

    n_free = n_segments - 1
    n_genes = ga_gene_count(n_segments, layout)

    if "FitnessMin" not in dir(creator):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in dir(creator):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("clone", copy.deepcopy)
    base_bearing = compute_bearing(*start_port, *end_port)
    departure_corridor = _build_departure_corridor(
        cost_map=cost_map,
        dt_per_seg=dt_per_seg,
        base_bearing=base_bearing,
        start_port=start_port,
        enforce_departure_heading=enforce_departure_heading,
    )
    first_heading_low, first_heading_high = _first_heading_bounds(enforce_departure_heading)

    def init_individual():
        genes: list[float] = []
        if layout == GENOTYPE_DECOUPLED:
            for segment_index in range(n_free):
                low, high = _sog_bounds_for_segment(segment_index)
                if segment_index == 0:
                    if departure_corridor:
                        corridor_speed, heading_deg = random.choice(departure_corridor)
                        sog_kts = corridor_speed
                        dist_nm = corridor_speed * dt_per_seg
                    else:
                        sog_kts = random.uniform(low, high)
                        dist_nm = random.uniform(low * dt_per_seg, high * dt_per_seg)
                        heading_deg = random.uniform(first_heading_low, first_heading_high)
                    genes.extend([sog_kts, dist_nm, heading_deg])
                else:
                    sog_kts = random.uniform(low, high)
                    dist_nm = random.uniform(low * dt_per_seg, high * dt_per_seg)
                    delta_heading_deg = random.uniform(-MAX_HEADING_DELTA, MAX_HEADING_DELTA)
                    genes.extend([sog_kts, dist_nm, delta_heading_deg])
            genes.append(random.uniform(V_MIN_PORT, V_MAX_PORT))
        else:
            for segment_index in range(n_free):
                if segment_index == 0:
                    if departure_corridor:
                        sog_kts, heading_deg = random.choice(departure_corridor)
                    else:
                        low, high = _sog_bounds_for_segment(segment_index)
                        sog_kts = random.uniform(low, high)
                        heading_deg = random.uniform(first_heading_low, first_heading_high)
                    genes.extend([sog_kts, heading_deg])
                else:
                    low, high = _sog_bounds_for_segment(segment_index)
                    sog_kts = random.uniform(low, high)
                    delta_heading_deg = random.uniform(-MAX_HEADING_DELTA, MAX_HEADING_DELTA)
                    genes.extend([sog_kts, delta_heading_deg])
        individual = repair_individual(
            creator.Individual(genes),
            enforce_departure_heading=enforce_departure_heading,
            genotype_layout=layout,
            dt_per_seg=dt_per_seg,
        )
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
                start_port,
                end_port,
                enforce_departure_heading,
                layout,
                int(milp_time_limit_sec),
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
                start_port=start_port,
                end_port=end_port,
                enforce_departure_heading=enforce_departure_heading,
                genotype_layout=layout,
                milp_time_limit_sec=int(milp_time_limit_sec),
            ),
        )

    if layout == GENOTYPE_DECOUPLED:
        low_bounds = [V_MIN_PORT, V_MIN_PORT * dt_per_seg, first_heading_low]
        up_bounds = [V_MAX_PORT, V_MAX_PORT * dt_per_seg, first_heading_high]
        for _ in range(n_free - 1):
            low_bounds.extend([V_MIN, V_MIN * dt_per_seg, -MAX_HEADING_DELTA])
            up_bounds.extend([V_MAX, V_MAX * dt_per_seg, MAX_HEADING_DELTA])
        low_bounds.append(V_MIN_PORT)
        up_bounds.append(V_MAX_PORT)
    else:
        low_bounds = [V_MIN_PORT, first_heading_low] + [V_MIN, -MAX_HEADING_DELTA] * (n_free - 1)
        up_bounds = [V_MAX_PORT, first_heading_high] + [V_MAX, MAX_HEADING_DELTA] * (n_free - 1)

    # ── A* 시드 주입 준비 및 Bounds 확장 ──
    # DEAP의 cxSimulatedBinaryBounded는 유전자가 bounds를 벗어나면 complex number(허수) 에러를 발생시킵니다.
    # A* 원본 시드가 기하학적 보존을 위해 V_MAX나 MAX_HEADING_DELTA를 미세하게 초과할 수 있으므로,
    # GA 엔진이 다운타임 없이 교차(Crossover) 연산을 수행할 수 있도록 bounds를 동적으로 확장합니다.
    astar_genes = None
    if astar_seed_waypoints is not None:
        astar_seed_waypoints = _adjust_astar_seed_waypoints_for_departure(
            astar_seed_waypoints,
            departure_corridor=departure_corridor,
            cost_map=cost_map,
            n_segments=n_segments,
            rta_h=rta_h,
            start_port=start_port,
            enforce_departure_heading=enforce_departure_heading,
        )
        astar_genes = encode_astar_seed(
            astar_seed_waypoints,
            n_segments=n_segments,
            rta_h=rta_h,
            genotype_layout=layout,
        )
        for i in range(len(astar_genes)):
            low_bounds[i] = min(low_bounds[i], astar_genes[i])
            up_bounds[i] = max(up_bounds[i], astar_genes[i])

    explicit_seed_individuals: list[list[float]] = []
    if seed_individual is not None:
        explicit_seed_individuals.append(list(seed_individual))
    if seed_individuals is not None:
        explicit_seed_individuals.extend(list(seed) for seed in seed_individuals if seed is not None)

    compatible_seed_individuals: list[list[float]] = []
    for index, explicit_seed in enumerate(explicit_seed_individuals):
        if len(explicit_seed) != n_genes:
            print(
                "  [GA seed] Ignoring explicit seed with unexpected length: "
                f"{len(explicit_seed)} != {n_genes} (index={index})"
            )
            continue
        compatible_seed_individuals.append([float(value) for value in explicit_seed])
        for i in range(n_genes):
            low_bounds[i] = min(low_bounds[i], explicit_seed[i])
            up_bounds[i] = max(up_bounds[i], explicit_seed[i])

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
        f"Seed: {'random' if seed is None else seed} | "
        f"Departure heading: {'on' if enforce_departure_heading else 'off'} | "
        f"Genotype: {layout} | "
        f"MILP limit: {int(milp_time_limit_sec)}s"
    )
    if n_workers > 1:
        print(f" Parallel workers: {n_workers}")
    print(f"{'=' * 60}")

    population = toolbox.population(n=pop_size)

    seed_insert_index = 0
    for explicit_seed_genes in compatible_seed_individuals:
        if seed_insert_index >= len(population):
            break
        explicit_seed = creator.Individual(list(explicit_seed_genes))
        del explicit_seed.fitness.values
        population[seed_insert_index] = explicit_seed
        seed_insert_index += 1
    if seed_insert_index > 0:
        print("  [GA seed] Injected explicit seed individual")

    # ── A* 시드 주입 ──
    if astar_seed_waypoints is not None and astar_genes is not None:
        # 원본 A* 시드는 수학적 기하학 정밀도가 생명입니다.
        # repair_individual이나 _apply_departure_corridor를 거치면 속도/각도가 반올림되거나
        # 클램핑되어 기하학적 궤적이 틀어지고 육지에 충돌하게 되므로 절대 건드리지 않습니다.

        # 시드 개체를 인구의 일부 (최대 20%)에 주입
        n_seed = max(1, pop_size // 5)
        print(f"  [A* seed] Injecting {n_seed} seeded individuals from A* route")
        for i in range(n_seed):
            population_index = seed_insert_index + i
            if population_index >= len(population):
                break
            seeded = creator.Individual(list(astar_genes))
            # 첫 번째 개체는 완벽한 원본 무결성 유지 (노이즈, repair, corridor 적용 배제)
            if i > 0:
                for g in range(len(seeded)):
                    noise_scale = 0.05 * abs(up_bounds[g] - low_bounds[g])
                    seeded[g] += random.gauss(0.0, noise_scale)
                repair_individual(
                    seeded,
                    enforce_departure_heading=enforce_departure_heading,
                    genotype_layout=layout,
                    dt_per_seg=dt_per_seg,
                )
                _apply_departure_corridor(seeded, departure_corridor)
            del seeded.fitness.values
            population[population_index] = seeded

    try:
        result_pop, logbook, generation_records = _run_simple_ga(
            population=population,
            toolbox=toolbox,
            n_gen=n_gen,
            cx_prob=cx_prob,
            mut_prob=mut_prob,
            stats=stats,
            halloffame=hof,
            departure_corridor=departure_corridor,
            elite_count=elite_count,
            enforce_departure_heading=enforce_departure_heading,
            genotype_layout=layout,
            dt_per_seg=dt_per_seg,
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
        start_port=start_port,
        end_port=end_port,
        genotype_layout=layout,
    )
    validation = evaluate_route_validity(
        best_route,
        cost_map=cost_map,
        env_fn=resolved_env_fn,
        departure_time_utc=departure_time_utc,
        start_port=start_port,
        end_port=end_port,
        enforce_departure_heading=enforce_departure_heading,
    )
    apply_route_validation(best_route, validation)

    return {
        "best_individual": list(best_individual),
        "best_fitness": float(best_individual.fitness.values[0]),
        "best_route": best_route,
        "logbook": logbook,
        "population": result_pop,
        "hall_of_fame": hof,
        "generation_records": generation_records if capture_generation_records else None,
        "genotype_layout": layout,
    }
