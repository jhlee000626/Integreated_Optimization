"""
DEAP GA engine for integrated route and speed optimization.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import multiprocessing
import random
from typing import Callable, Tuple

import numpy as np
from deap import algorithms, base, creator, tools

from src.grid.no_go_zone import BUSAN_PORT, JEJU_PORT
from src.resistance.modified_dpm import compute_P_req
from src.resistance.models import EnvironmentData
from src.ship.kcs_specs import POWER_MODEL, SERVICE_LOAD


BIG_PENALTY = 1e12
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


def _current_component_knots(env: EnvironmentData, heading_deg: float) -> float:
    if abs(env.current_speed_ms) <= 1e-12:
        return 0.0
    heading_delta = _angle_delta_deg(env.current_dir_deg, heading_deg)
    return (env.current_speed_ms * math.cos(math.radians(heading_delta))) / 0.514444


def compute_speed_over_ground_knots(
    v_stw_knots: float,
    heading_deg: float,
    env: EnvironmentData,
) -> float:
    return max(0.1, v_stw_knots + _current_component_knots(env, heading_deg))


def _required_stw_for_ground_speed(
    target_ground_speed_knots: float,
    heading_deg: float,
    env: EnvironmentData,
) -> float:
    return target_ground_speed_knots - _current_component_knots(env, heading_deg)


def _worker_init(
    cost_map_data,
    sfoc_path: str,
    n_pwl: int,
    env_fn_data,
    n_segments: int,
    rta_h: float,
    departure_time_utc: datetime,
    smoothing_weight: float,
):
    global _worker_cost_map, _worker_milp, _worker_env_fn
    global _worker_n_segments, _worker_rta_h, _worker_departure_time_utc, _worker_smoothing_weight

    from src.optimizer.milp_solver import MILPSolver

    _worker_milp = MILPSolver(sfoc_json_path=sfoc_path, n_pwl_segments=n_pwl)
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
    """Move a point along a great-circle arc."""
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
    """Return the true bearing from point 1 to point 2."""
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
    """Return the great-circle distance in nautical miles."""
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
    waypoints: list[tuple[float, float]],
    speeds: list[float],
    speeds_sog: list[float],
    headings: list[float],
    dt_per_seg: float,
) -> list[dict]:
    nodes: list[dict] = []
    for index, (lat, lon) in enumerate(waypoints):
        node = {
            "lat": lat,
            "lon": lon,
            "time_h": index * dt_per_seg,
            "time_utc": None,
            "speed_out_kts": speeds[index] if index < len(speeds) else None,
            "speed_over_ground_kts": speeds_sog[index] if index < len(speeds_sog) else None,
            "heading_out_deg": headings[index] if index < len(headings) else None,
        }
        nodes.append(node)
    return nodes


def decode_route(
    individual: list,
    n_segments: int = N_SEGMENTS,
    rta_h: float = RTA_HOURS,
    env_fn: Callable | None = None,
    departure_time_utc: datetime | None = None,
) -> dict:
    """Decode a chromosome into a route with node states."""
    n_free = n_segments - 1
    dt_per_seg = rta_h / n_segments
    resolved_env_fn = _normalize_env_fn(env_fn=env_fn)
    departure_time_utc = _coerce_departure_time_utc(departure_time_utc)

    speeds_free = [individual[i * 2] for i in range(n_free)]
    heading_genes = [individual[i * 2 + 1] for i in range(n_free)]

    waypoints = [BUSAN_PORT]
    speeds: list[float] = []
    speeds_sog: list[float] = []
    headings: list[float] = []
    delta_headings: list[float] = []
    distances: list[float] = []
    dt_list: list[float] = []

    current_lat, current_lon = BUSAN_PORT
    base_heading = compute_bearing(*BUSAN_PORT, *JEJU_PORT)
    prev_heading = base_heading

    for index in range(n_free):
        v_kts = speeds_free[index]
        if index == 0:
            theta_deg = heading_genes[index] % 360.0
            delta_deg = ((theta_deg - base_heading + 180.0) % 360.0) - 180.0
        else:
            delta_deg = heading_genes[index]
            theta_deg = (prev_heading + delta_deg) % 360.0
        when_utc = departure_time_utc + timedelta(hours=index * dt_per_seg)
        env = _resolve_environment(resolved_env_fn, current_lat, current_lon, when_utc)
        v_sog_kts = compute_speed_over_ground_knots(v_kts, theta_deg, env)
        dist_nm = v_sog_kts * dt_per_seg
        new_lat, new_lon = move_position(current_lat, current_lon, theta_deg, dist_nm)

        waypoints.append((new_lat, new_lon))
        speeds.append(v_kts)
        speeds_sog.append(v_sog_kts)
        headings.append(theta_deg)
        delta_headings.append(delta_deg)
        distances.append(dist_nm)
        dt_list.append(dt_per_seg)

        current_lat, current_lon = new_lat, new_lon
        prev_heading = theta_deg

    dest_lat, dest_lon = JEJU_PORT
    final_heading = compute_bearing(current_lat, current_lon, dest_lat, dest_lon)
    final_dist_nm = haversine_nm((current_lat, current_lon), (dest_lat, dest_lon))
    final_ground_speed_kts = final_dist_nm / dt_per_seg if dt_per_seg > 0.0 else 0.0
    final_when_utc = departure_time_utc + timedelta(hours=n_free * dt_per_seg)
    final_env = _resolve_environment(resolved_env_fn, current_lat, current_lon, final_when_utc)
    v_last = _required_stw_for_ground_speed(final_ground_speed_kts, final_heading, final_env)
    final_heading_delta = ((final_heading - prev_heading + 180.0) % 360.0) - 180.0

    waypoints.append(JEJU_PORT)
    speeds.append(v_last)
    speeds_sog.append(final_ground_speed_kts)
    headings.append(final_heading)
    delta_headings.append(final_heading_delta)
    distances.append(final_dist_nm)
    dt_list.append(dt_per_seg)

    first_heading = headings[0] if headings else base_heading
    valid_departure_heading = FIRST_HEADING_MIN <= first_heading <= FIRST_HEADING_MAX
    valid_speed = V_MIN_PORT <= v_last <= V_MAX_PORT
    valid_heading = abs(final_heading_delta) <= MAX_HEADING_DELTA
    valid = valid_departure_heading and valid_speed and valid_heading

    nodes = _build_nodes(waypoints, speeds, speeds_sog, headings, dt_per_seg)

    return {
        "waypoints": waypoints,
        "nodes": nodes,
        "speeds": speeds,
        "speeds_sog": speeds_sog,
        "headings": headings,
        "delta_headings": delta_headings,
        "dt": dt_list,
        "distances_nm": distances,
        "valid": valid,
        "valid_departure_heading": valid_departure_heading,
        "valid_speed": valid_speed,
        "valid_heading": valid_heading,
        "last_speed": v_last,
        "final_heading_delta": final_heading_delta,
    }


def _build_guided_individual(
    n_segments: int,
    rta_h: float,
    env_fn: Callable | None = None,
    departure_time_utc: datetime | None = None,
    heading_noise_deg: float = GUIDED_HEADING_NOISE_DEG,
    speed_noise_kts: float = GUIDED_SPEED_NOISE_KTS,
):
    """Build a destination-seeking individual that starts inside the feasible terminal cone."""
    n_free = n_segments - 1
    dt_per_seg = rta_h / n_segments
    resolved_env_fn = _normalize_env_fn(env_fn=env_fn)
    departure_time_utc = _coerce_departure_time_utc(departure_time_utc)

    genes: list[float] = []
    current_position = BUSAN_PORT
    prev_heading = compute_bearing(*BUSAN_PORT, *JEJU_PORT)

    for segment_index in range(n_free):
        remaining_dist_nm = haversine_nm(current_position, JEJU_PORT)
        remaining_time_h = max((n_segments - segment_index) * dt_per_seg, 1e-9)
        desired_heading = compute_bearing(
            current_position[0],
            current_position[1],
            JEJU_PORT[0],
            JEJU_PORT[1],
        )

        if segment_index == 0:
            heading = desired_heading + random.uniform(-heading_noise_deg, heading_noise_deg)
            heading = max(FIRST_HEADING_MIN, min(FIRST_HEADING_MAX, heading))
            target_speed = remaining_dist_nm / remaining_time_h
            env = _resolve_environment(
                resolved_env_fn,
                current_position[0],
                current_position[1],
                departure_time_utc + timedelta(hours=segment_index * dt_per_seg),
            )
            target_stw = _required_stw_for_ground_speed(target_speed, heading, env)
            speed = target_stw + random.uniform(-speed_noise_kts, speed_noise_kts)
            speed = max(V_MIN_PORT, min(V_MAX_PORT, speed))

            genes.extend([speed, heading])
            theta_deg = heading
        else:
            delta_deg = ((desired_heading - prev_heading + 180.0) % 360.0) - 180.0
            delta_deg += random.uniform(-heading_noise_deg, heading_noise_deg)
            delta_deg = max(-MAX_HEADING_DELTA, min(MAX_HEADING_DELTA, delta_deg))
            target_speed = remaining_dist_nm / remaining_time_h
            env = _resolve_environment(
                resolved_env_fn,
                current_position[0],
                current_position[1],
                departure_time_utc + timedelta(hours=segment_index * dt_per_seg),
            )
            theta_deg = (prev_heading + delta_deg) % 360.0
            target_stw = _required_stw_for_ground_speed(target_speed, theta_deg, env)
            speed = target_stw + random.uniform(-speed_noise_kts, speed_noise_kts)
            speed = max(V_MIN, min(V_MAX, speed))

            genes.extend([speed, delta_deg])

        v_sog_kts = compute_speed_over_ground_knots(speed, theta_deg, env)
        segment_dist_nm = v_sog_kts * dt_per_seg
        current_position = move_position(
            current_position[0],
            current_position[1],
            theta_deg,
            segment_dist_nm,
        )
        prev_heading = theta_deg

    return creator.Individual(genes)


def build_required_power_profile(
    route: dict,
    env_fn: Callable | None = None,
    departure_time_utc: datetime | None = None,
    weather_fn: Callable | None = None,
) -> list[dict]:
    """Build the propulsion-load profile by sampling environment at segment start nodes."""
    resolved_env_fn = _normalize_env_fn(env_fn=env_fn, weather_fn=weather_fn)
    departure_time_utc = _coerce_departure_time_utc(departure_time_utc)
    nodes = route["nodes"]
    power_profile: list[dict] = []

    for segment_index in range(len(route["speeds"])):
        node = nodes[segment_index]
        when_utc = departure_time_utc + timedelta(hours=float(node["time_h"]))
        node["time_utc"] = when_utc
        wp_from = (node["lat"], node["lon"])
        wp_to = route["waypoints"][segment_index + 1]
        heading = float(node["heading_out_deg"])
        speed_kts = float(node["speed_out_kts"])
        env = _resolve_environment(resolved_env_fn, node["lat"], node["lon"], when_utc)
        encounter_deg = compute_encounter_angle(
            heading_deg=heading,
            wind_dir_deg=float(env.wind_dir_deg or 0.0),
            ship_speed_knots=speed_kts,
            wind_speed_ms=float(env.wind_speed_ms),
        )

        if segment_index == 0:
            phase = "departure"
        elif segment_index == len(route["speeds"]) - 1:
            phase = "approach"
        else:
            phase = "cruising"

        p_service = SERVICE_LOAD[phase]
        result = compute_P_req(
            v_ship_knots=speed_kts,
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
                "when_utc": when_utc,
                "node": node,
                "waypoint_from": wp_from,
                "waypoint_to": wp_to,
                "speed_kts": speed_kts,
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
                "P_service": p_service,
                **result,
            }
        )

    if nodes:
        nodes[-1]["time_utc"] = departure_time_utc + timedelta(hours=float(nodes[-1]["time_h"]))

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
    route = decode_route(
        individual,
        n_segments=n_segments,
        rta_h=rta_h,
        env_fn=resolved_env_fn,
        departure_time_utc=departure_time_utc,
    )

    valid_penalty = 0.0
    if not route["valid_speed"]:
        v_last = route["last_speed"]
        if v_last < V_MIN_PORT:
            valid_penalty = BIG_PENALTY * (1.0 + (V_MIN_PORT - v_last) / V_MIN_PORT)
        else:
            valid_penalty = BIG_PENALTY * (1.0 + (v_last - V_MAX_PORT) / V_MAX_PORT)
    if not route["valid_heading"]:
        heading_excess = abs(route["final_heading_delta"]) - MAX_HEADING_DELTA
        valid_penalty += BIG_PENALTY * (1.0 + heading_excess / MAX_HEADING_DELTA)
    if not route["valid_departure_heading"]:
        first_heading = route["headings"][0]
        if first_heading < FIRST_HEADING_MIN:
            departure_excess = FIRST_HEADING_MIN - first_heading
        else:
            departure_excess = first_heading - FIRST_HEADING_MAX
        valid_penalty += BIG_PENALTY * (1.0 + departure_excess / 180.0)

    violation = compute_violation(route, cost_map)
    violation_penalty = 0.0
    if violation > 0.0:
        violation_penalty = BIG_PENALTY * 10.0 * (1.0 + violation / 100.0)

    if valid_penalty > 0.0 or violation_penalty > 0.0:
        return (valid_penalty + violation_penalty,)

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
        return (BIG_PENALTY,)

    fuel = float(milp_result["total_fuel_kg"])
    smoothing_penalty = compute_smoothing_penalty(p_req_list, weight=smoothing_weight)
    return (fuel + smoothing_penalty,)


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
    base_bearing = compute_bearing(*BUSAN_PORT, *JEJU_PORT)

    def init_individual():
        if random.random() < GUIDED_INIT_FRACTION:
            return _build_guided_individual(
                n_segments=n_segments,
                rta_h=rta_h,
                env_fn=resolved_env_fn,
                departure_time_utc=departure_time_utc,
            )

        genes = []
        for segment_index in range(n_free):
            if segment_index == 0:
                genes.append(random.uniform(V_MIN_PORT, V_MAX_PORT))
                genes.append(random.uniform(FIRST_HEADING_MIN, FIRST_HEADING_MAX))
            else:
                genes.append(random.uniform(V_MIN, V_MAX))
                genes.append(random.uniform(-MAX_HEADING_DELTA, MAX_HEADING_DELTA))
        return creator.Individual(genes)

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
                sfoc_path,
                5,
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
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    print(f"\n{'=' * 60}")
    print(" DEAP GA run")
    print(f" Population: {pop_size} | Generations: {n_gen} | Segments: {n_segments}")
    print(f" RTA: {rta_h}h | Speed: {V_MIN}~{V_MAX} kts")
    print(f" Base bearing: {base_bearing:.1f} deg | Smoothing weight: {smoothing_weight:.1f}")
    if n_workers > 1:
        print(f" Parallel workers: {n_workers}")
    print(f"{'=' * 60}")

    population = toolbox.population(n=pop_size)

    try:
        result_pop, logbook = algorithms.eaSimple(
            population,
            toolbox,
            cxpb=cx_prob,
            mutpb=mut_prob,
            ngen=n_gen,
            stats=stats,
            halloffame=hof,
            verbose=True,
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
        env_fn=resolved_env_fn,
        departure_time_utc=departure_time_utc,
    )
    for node in best_route["nodes"]:
        node["time_utc"] = departure_time_utc + timedelta(hours=float(node["time_h"]))

    return {
        "best_individual": list(best_individual),
        "best_fitness": float(best_individual.fitness.values[0]),
        "best_route": best_route,
        "logbook": logbook,
        "population": result_pop,
        "hall_of_fame": hof,
    }
