"""
Integrated GA + MILP optimizer for the Busan -> Jeju routing study.
"""

from __future__ import annotations

import math
import multiprocessing
import random
from functools import partial
from typing import Callable, Dict, List, Tuple

import numpy as np
from deap import algorithms, base, creator, tools

from src.grid.no_go_zone import BUSAN_PORT, JEJU_PORT
from src.resistance.kwon_method import compute_P_req
from src.ship.kcs_specs import POWER_MODEL, SERVICE_LOAD


BIG_PENALTY = 1e12
RTA_HOURS = 12.0
MAX_HEADING_DELTA = 30.0
V_MIN, V_MAX = 5.0, 24.0
GAMMA = 0.7
V_MIN_PORT = V_MIN
V_MAX_PORT = V_MAX * GAMMA
N_SEGMENTS = 24
LAMBDA_VIOLATION = 500.0
VIOLATION_THRESHOLD = 5.0
DEFAULT_INITIAL_SOC = 0.7


_worker_cost_map = None
_worker_milp = None
_worker_weather_fn = None
_worker_n_segments = N_SEGMENTS
_worker_rta_h = RTA_HOURS


def _worker_init(cost_map_data, sfoc_path, n_pwl, weather_data, n_segments, rta_h):
    global _worker_cost_map, _worker_milp, _worker_weather_fn
    global _worker_n_segments, _worker_rta_h

    from src.optimizer.milp_solver import MILPSolver

    _worker_milp = MILPSolver(sfoc_json_path=sfoc_path, n_pwl_segments=n_pwl)
    _worker_cost_map = cost_map_data
    _worker_weather_fn = weather_data
    _worker_n_segments = n_segments
    _worker_rta_h = rta_h


def _evaluate_parallel(individual):
    return evaluate(
        individual,
        cost_map=_worker_cost_map,
        milp_solver=_worker_milp,
        weather_fn=_worker_weather_fn,
        n_segments=_worker_n_segments,
        rta_h=_worker_rta_h,
    )


def move_position(
    lat: float,
    lon: float,
    bearing_deg: float,
    distance_nm: float,
) -> Tuple[float, float]:
    """Move a point along a great-circle arc."""
    radius_nm = 3440.065
    angular_distance = distance_nm / radius_nm
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    brg_r = math.radians(bearing_deg)

    new_lat_r = math.asin(
        math.sin(lat_r) * math.cos(angular_distance)
        + math.cos(lat_r) * math.sin(angular_distance) * math.cos(brg_r)
    )
    new_lon_r = lon_r + math.atan2(
        math.sin(brg_r) * math.sin(angular_distance) * math.cos(lat_r),
        math.cos(angular_distance) - math.sin(lat_r) * math.sin(new_lat_r),
    )
    return math.degrees(new_lat_r), math.degrees(new_lon_r)


def compute_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return a true heading in degrees."""
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlon = lon2_r - lon1_r
    x = math.sin(dlon) * math.cos(lat2_r)
    y = (
        math.cos(lat1_r) * math.sin(lat2_r)
        - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
    )
    return math.degrees(math.atan2(x, y)) % 360.0


def compute_heading(wp_from: tuple, wp_to: tuple) -> float:
    return compute_bearing(wp_from[0], wp_from[1], wp_to[0], wp_to[1])


def compute_encounter_angle(heading_deg: float, wind_dir_deg: float) -> float:
    """Return encounter angle in [0, 180] using meteorological wind direction."""
    relative = (wind_dir_deg - heading_deg + 180.0) % 360.0
    if relative > 180.0:
        relative = 360.0 - relative
    return relative


def haversine_nm(p1: tuple, p2: tuple) -> float:
    radius_nm = 3440.065
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return radius_nm * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def decode_route(
    individual: list,
    n_segments: int = N_SEGMENTS,
    rta_h: float = RTA_HOURS,
) -> dict:
    """
    Decode a chromosome into a route dictionary.

    Chromosome layout:
    [speed_0, delta_heading_0, ..., speed_(n-2), delta_heading_(n-2)]
    """
    n_free = n_segments - 1
    dt_per_seg = rta_h / n_segments

    speeds_free = [individual[i * 2] for i in range(n_free)]
    delta_headings_free = [individual[i * 2 + 1] for i in range(n_free)]

    waypoints = [BUSAN_PORT]
    speeds: List[float] = []
    headings: List[float] = []
    delta_headings: List[float] = []
    distances: List[float] = []
    dt_list: List[float] = []

    current_lat, current_lon = BUSAN_PORT
    prev_heading = compute_heading(BUSAN_PORT, JEJU_PORT)

    for i in range(n_free):
        v_kts = speeds_free[i]
        delta_heading = delta_headings_free[i]
        theta_deg = (prev_heading + delta_heading) % 360.0
        dist_nm = v_kts * dt_per_seg

        new_lat, new_lon = move_position(current_lat, current_lon, theta_deg, dist_nm)

        waypoints.append((new_lat, new_lon))
        speeds.append(v_kts)
        headings.append(theta_deg)
        delta_headings.append(delta_heading)
        distances.append(dist_nm)
        dt_list.append(dt_per_seg)

        current_lat, current_lon = new_lat, new_lon
        prev_heading = theta_deg

    final_heading = compute_heading((current_lat, current_lon), JEJU_PORT)
    final_dist_nm = haversine_nm((current_lat, current_lon), JEJU_PORT)
    final_speed = final_dist_nm / dt_per_seg if dt_per_seg > 0 else 0.0
    final_heading_delta = ((final_heading - prev_heading + 180.0) % 360.0) - 180.0

    waypoints.append(JEJU_PORT)
    speeds.append(final_speed)
    headings.append(final_heading)
    delta_headings.append(final_heading_delta)
    distances.append(final_dist_nm)
    dt_list.append(dt_per_seg)

    valid_speed = V_MIN_PORT <= final_speed <= V_MAX_PORT
    valid_heading = abs(final_heading_delta) <= MAX_HEADING_DELTA

    return {
        "waypoints": waypoints,
        "speeds": speeds,
        "headings": headings,
        "delta_headings": delta_headings,
        "dt": dt_list,
        "distances_nm": distances,
        "valid": valid_speed and valid_heading,
        "valid_speed": valid_speed,
        "valid_heading": valid_heading,
        "last_speed": final_speed,
        "final_heading_delta": final_heading_delta,
    }


def compute_violation(route: dict, cost_map) -> float:
    return cost_map.route_violation(route["waypoints"])


def _build_guided_individual(cost_map, n_segments: int, rta_h: float):
    """Seed a route that stays in open water and roughly respects the RTA."""
    n_free = n_segments - 1
    dt_per_seg = rta_h / n_segments
    genes: List[float] = []

    current = BUSAN_PORT
    prev_heading = compute_heading(BUSAN_PORT, JEJU_PORT)

    for seg in range(n_free):
        remaining_segments = n_segments - seg
        remaining_dist = haversine_nm(current, JEJU_PORT)
        target_speed = remaining_dist / max(remaining_segments * dt_per_seg, dt_per_seg)

        speed_low = V_MIN_PORT if seg == 0 else V_MIN
        speed_high = V_MAX_PORT if seg == 0 else V_MAX
        v_kts = min(speed_high, max(speed_low, target_speed + random.gauss(0.0, 0.25)))
        dist_nm = v_kts * dt_per_seg

        candidates = []
        for delta_heading in range(
            -int(MAX_HEADING_DELTA),
            int(MAX_HEADING_DELTA) + 1,
            5,
        ):
            theta_deg = (prev_heading + delta_heading) % 360.0
            next_point = move_position(current[0], current[1], theta_deg, dist_nm)
            seg_cost = cost_map.segment_cost(current[0], current[1], next_point[0], next_point[1])
            remain_after = haversine_nm(next_point, JEJU_PORT)
            score = seg_cost * 10_000.0 + remain_after
            candidates.append((score, float(delta_heading), next_point, theta_deg))

        candidates.sort(key=lambda item: item[0])
        _, best_delta, next_point, theta_deg = candidates[0]

        genes.extend([v_kts, best_delta])
        current = next_point
        prev_heading = theta_deg

    return genes

def build_required_power_profile(
    route: dict,
    weather_fn: Callable | None = None,
) -> List[Dict[str, float]]:
    """Build per-segment power demand inputs from a decoded route."""
    n_segments = len(route["speeds"])
    profile: List[Dict[str, float]] = []

    for i in range(n_segments):
        wp_from = route["waypoints"][i]
        wp_to = route["waypoints"][i + 1]
        v_kts = route["speeds"][i]
        heading = route["headings"][i]

        mid_lat = (wp_from[0] + wp_to[0]) / 2.0
        mid_lon = (wp_from[1] + wp_to[1]) / 2.0
        if weather_fn is None:
            wind_speed, wind_dir = 0.0, 0.0
        else:
            wind_speed, wind_dir = weather_fn(mid_lat, mid_lon)

        encounter = compute_encounter_angle(heading, wind_dir)

        if i == 0:
            phase = "departure"
        elif i == n_segments - 1:
            phase = "approach"
        else:
            phase = "cruising"

        power = compute_P_req(
            v_ship_knots=v_kts,
            v_wind_ms=wind_speed,
            encounter_angle_deg=encounter,
            a1=POWER_MODEL["a1"],
            P_service=SERVICE_LOAD[phase],
        )

        profile.append(
            {
                "segment_index": i,
                "phase": phase,
                "speed_knots": v_kts,
                "heading_deg": heading,
                "wind_speed_ms": wind_speed,
                "wind_dir_deg": wind_dir,
                "encounter_angle_deg": encounter,
                "distance_nm": route["distances_nm"][i],
                "dt_h": route["dt"][i],
                "P_req": power["P_req"],
                "P_prop": power["P_prop"],
                "P_service": power["P_service"],
            }
        )

    return profile


def evaluate(
    individual: list,
    cost_map,
    milp_solver,
    weather_fn=None,
    n_segments: int = N_SEGMENTS,
    rta_h: float = RTA_HOURS,
) -> Tuple[float]:
    route = decode_route(individual, n_segments=n_segments, rta_h=rta_h)

    valid_penalty = 0.0
    if not route["valid_speed"]:
        v_last = route["last_speed"]
        if v_last < V_MIN_PORT:
            valid_penalty += BIG_PENALTY * (1 + (V_MIN_PORT - v_last) / V_MIN_PORT)
        else:
            valid_penalty += BIG_PENALTY * (1 + (v_last - V_MAX_PORT) / V_MAX_PORT)
    if not route["valid_heading"]:
        heading_excess = abs(route["final_heading_delta"]) - MAX_HEADING_DELTA
        valid_penalty += BIG_PENALTY * (1 + heading_excess / MAX_HEADING_DELTA)

    violation = compute_violation(route, cost_map)
    violation_penalty = 0.0
    if violation > VIOLATION_THRESHOLD:
        violation_penalty = BIG_PENALTY * 10.0 * (1 + violation / 100.0)

    if valid_penalty > 0 or violation_penalty > 0:
        return (valid_penalty + violation_penalty,)

    power_profile = build_required_power_profile(route, weather_fn=weather_fn)
    p_req_list = [segment["P_req"] for segment in power_profile]

    milp_result = milp_solver.solve(
        P_req=p_req_list,
        dt=route["dt"],
        initial_SOC=DEFAULT_INITIAL_SOC,
        msg=False,
    )
    if not milp_result["feasible"]:
        return (BIG_PENALTY,)

    fuel = milp_result["total_fuel_kg"]
    return (fuel + LAMBDA_VIOLATION * violation,)


def setup_ga(
    cost_map,
    milp_solver,
    weather_fn=None,
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
    verbose: bool = True,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    n_free = n_segments - 1
    n_genes = n_free * 2

    if "FitnessMin" not in dir(creator):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in dir(creator):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    base_bearing = compute_heading(BUSAN_PORT, JEJU_PORT)

    def init_individual():
        if random.random() < 0.8:
            genes = _build_guided_individual(cost_map, n_segments=n_segments, rta_h=rta_h)
            return creator.Individual(genes)

        genes = []
        for seg in range(n_free):
            if seg == 0:
                genes.append(random.uniform(V_MIN_PORT, V_MAX_PORT))
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
        if verbose:
            print(f"  Multiprocessing: {n_workers} workers")
        pool = multiprocessing.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(cost_map, sfoc_path, 5, weather_fn, n_segments, rta_h),
        )
        toolbox.register("map", pool.map)
        toolbox.register("evaluate", _evaluate_parallel)
    else:
        if verbose:
            print("  Sequential mode (1 worker)")
        toolbox.register(
            "evaluate",
            partial(
                evaluate,
                cost_map=cost_map,
                milp_solver=milp_solver,
                weather_fn=weather_fn,
                n_segments=n_segments,
                rta_h=rta_h,
            ),
        )

    low_bounds = [V_MIN_PORT, -MAX_HEADING_DELTA] + [V_MIN, -MAX_HEADING_DELTA] * (n_free - 1)
    up_bounds = [V_MAX_PORT, MAX_HEADING_DELTA] + [V_MAX, MAX_HEADING_DELTA] * (n_free - 1)

    toolbox.register(
        "mate",
        tools.cxSimulatedBinaryBounded,
        low=low_bounds,
        up=up_bounds,
        eta=20.0,
    )
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

    if verbose:
        print("\n" + "=" * 60)
        print(" DEAP GA Run")
        print(f" Population: {pop_size} | Generations: {n_gen} | Segments: {n_segments}")
        print(f" RTA: {rta_h} h | Speed range: {V_MIN:.1f}..{V_MAX:.1f} kts")
        print(f" Base bearing: {base_bearing:.1f} deg | Heading delta: +/-{MAX_HEADING_DELTA:.1f} deg")
        print("=" * 60)

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
            verbose=verbose,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    best_ind = hof[0]
    best_route = decode_route(list(best_ind), n_segments=n_segments, rta_h=rta_h)

    return {
        "best_individual": list(best_ind),
        "best_fitness": best_ind.fitness.values[0],
        "best_route": best_route,
        "logbook": logbook,
        "population": result_pop,
        "hall_of_fame": hof,
    }
