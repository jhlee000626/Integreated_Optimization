"""
DEAP GA 엔진 — 항로 + 속도 통합 최적화 (병렬 지원)
====================================================
설계 변수:
    Chromosome = [V₁, θ₁, V₂, θ₂, ..., V_{n-1}, θ_{n-1}]
    - V_i : 구간 속도 (knots, 16~24)
    - θ_i : 구간 방위각 (degrees, 0~360)

경로 생성 방식:
    부산항 출발 → V_i, θ_i로 Δt 동안 항해 → 다음 위치 계산
    마지막 구간: 목적지(제주항)까지 직접 연결, V_n 역산 → RTA 강제 충족

제약:
    1. 육지 회피 (Cost Map 기반 Hard Penalty)
    2. 속도 범위: V_min ≤ V_i ≤ V_max
    3. RTA = 12h (Required Time of Arrival)

적합도:
    Fitness = MILP(P_req) → 총 연료 소모량 (kg) → 최소화

병렬화:
    multiprocessing.Pool로 개체별 Fitness 평가 병렬 수행.
    워커 프로세스마다 독립된 MILPSolver 인스턴스를 생성.
"""

import math
import multiprocessing
import os
import random
import sys

import numpy as np

# Legacy path bootstrap for direct script execution.
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

from typing import Tuple
from deap import base, creator, tools, algorithms

from src.resistance.kwon_method import compute_P_req
from src.ship.kcs_specs import POWER_MODEL, SERVICE_LOAD
from src.grid.no_go_zone import BUSAN_PORT, JEJU_PORT


# =============================================================================
# 상수
# =============================================================================

BIG_PENALTY = 1e12
RTA_HOURS = 12           # Required Time of Arrival (h)
MAX_HEADING_DELTA = 30.0
V_MIN, V_MAX = 5.0, 24.0    # 순항 속도 범위 (knots)
GAMMA = 0.7                 # 출/입항 감속 계수
V_MIN_PORT = V_MIN          # 출/입항 최소 속도 (감속 미적용)
V_MAX_PORT = V_MAX * GAMMA  # 출/입항 최대 속도 = 16.8 kts
N_SEGMENTS = 24             # 구간 수

LAMBDA_VIOLATION = 500.0      # 위반 비용 가중치
VIOLATION_THRESHOLD = 5.0     # 이 값 이상이면 hard penalty


# =============================================================================
# 워커 프로세스 전역 (multiprocessing용)
# =============================================================================

_worker_cost_map = None
_worker_milp = None
_worker_weather_fn = None
_worker_n_segments = N_SEGMENTS
_worker_rta_h = RTA_HOURS


def _worker_init(cost_map_data, sfoc_path, n_pwl,
                 weather_data, n_segments, rta_h):
    """
    워커 프로세스 초기화.
    각 워커가 자신만의 MILPSolver + CostMap 수신.
    """
    global _worker_cost_map, _worker_milp, _worker_weather_fn
    global _worker_n_segments, _worker_rta_h

    from src.optimizer.milp_solver import MILPSolver

    _worker_milp = MILPSolver(sfoc_json_path=sfoc_path, n_pwl_segments=n_pwl)
    _worker_cost_map = cost_map_data  # CostMap 인스턴스 (pickle 가능)
    _worker_weather_fn = weather_data
    _worker_n_segments = n_segments
    _worker_rta_h = rta_h


def _evaluate_parallel(individual):
    """워커 프로세스에서 실행되는 평가 함수."""
    return evaluate(
        individual,
        cost_map=_worker_cost_map,
        milp_solver=_worker_milp,
        weather_fn=_worker_weather_fn,
        n_segments=_worker_n_segments,
        rta_h=_worker_rta_h,
    )


# =============================================================================
# 1. 경로 디코딩 (Chromosome → Route)
# =============================================================================

def decode_route(
    individual: list,
    n_segments: int = N_SEGMENTS,
    rta_h: float = RTA_HOURS,
) -> dict:
    """
    염색체를 경로로 디코딩.

    Parameters
    ----------
    individual : list
        [V₁, θ₁, V₂, θ₂, ..., V_{n-1}, θ_{n-1}]
    n_segments : int
        총 구간 수
    rta_h : float
        Required Time of Arrival (h)

    Returns
    -------
    dict
        {
            "waypoints": [(lat, lon), ...],  # n_segments+1개 점
            "speeds": [V₁, ..., Vₙ],        # n_segments개 속도
            "headings": [θ₁, ..., θₙ],      # n_segments개 방위각
            "dt": [Δt₁, ..., Δtₙ],          # 각 구간 시간 (h)
            "distances_nm": [d₁, ..., dₙ],  # 각 구간 거리 (nm)
            "valid": bool,
            "last_speed": Vₙ,               # 마지막 구간 역산 속도
        }
    """
    n_free = n_segments - 1
    dt_per_seg = rta_h / n_segments

    speeds_free = [individual[i * 2] for i in range(n_free)]
    delta_headings_free = [individual[i * 2 + 1] for i in range(n_free)]

    waypoints = [BUSAN_PORT]
    speeds = []
    headings = []
    delta_heaadings = []
    distances = []
    dt_list = []

    current_lat, current_lon = BUSAN_PORT
    prev_heading = compute_bearing(*BUSAN_PORT, *JEJU_PORT)

    for i in range(n_free):
        v_kts = speeds_free[i]
        delta_deg = delta_headings_free[i]
        theta_deg = (prev_heading + delta_deg) % 360
        dist_nm = v_kts * dt_per_seg

        new_lat, new_lon = move_position(
            current_lat, current_lon, theta_deg, dist_nm,
        )

        waypoints.append((new_lat, new_lon))
        speeds.append(v_kts)
        headings.append(theta_deg)
        delta_heaadings.append(delta_deg)
        distances.append(dist_nm)
        dt_list.append(dt_per_seg)
        current_lat, current_lon = new_lat, new_lon

    # ── 마지막 구간: 목적지까지 강제 연결 ──
    dest_lat, dest_lon = JEJU_PORT
    final_heading = compute_bearing(current_lat, current_lon, dest_lat, dest_lon)
    final_dist_nm = haversine_nm(
        (current_lat, current_lon), (dest_lat, dest_lon),
    )
    v_last = final_dist_nm / dt_per_seg if dt_per_seg > 0 else 0
    final_heading_delta = ((final_heading - prev_heading + 180.0) % 360.0) - 180.0

    waypoints.append(JEJU_PORT)
    speeds.append(v_last)
    headings.append(final_heading)
    delta_heaadings.append(final_heading_delta)
    distances.append(final_dist_nm)
    dt_list.append(dt_per_seg)

    valid = V_MIN_PORT <= v_last <= V_MAX_PORT

    return {
        "waypoints": waypoints,
        "speeds": speeds,
        "headings": headings,
        "dt": dt_list,
        "distances_nm": distances,
        "valid": valid,
        "last_speed": v_last,
    }


# =============================================================================
# 2. 항해 보조 함수
# =============================================================================

def move_position(
    lat: float, lon: float,
    bearing_deg: float, distance_nm: float,
) -> Tuple[float, float]:
    """Great Circle 이동."""
    R_nm = 3440.065
    d = distance_nm / R_nm
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    brg_r = math.radians(bearing_deg)

    new_lat_r = math.asin(
        math.sin(lat_r) * math.cos(d) +
        math.cos(lat_r) * math.sin(d) * math.cos(brg_r)
    )
    new_lon_r = lon_r + math.atan2(
        math.sin(brg_r) * math.sin(d) * math.cos(lat_r),
        math.cos(d) - math.sin(lat_r) * math.sin(new_lat_r),
    )
    return math.degrees(new_lat_r), math.degrees(new_lon_r)


def compute_bearing(lat1, lon1, lat2, lon2) -> float:
    """방위각 (°, 0=N, 시계방향)."""
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlon = lon2_r - lon1_r
    x = math.sin(dlon) * math.cos(lat2_r)
    y = (math.cos(lat1_r) * math.sin(lat2_r) -
         math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon))
    return math.degrees(math.atan2(x, y)) % 360.0


def haversine_nm(p1, p2) -> float:
    """Haversine 거리 (nm)."""
    R_nm = 3440.065
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R_nm * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


# =============================================================================
# 3. Violation Cost (Gaussian Cost Map)
# =============================================================================

def compute_violation(route: dict, cost_map) -> float:
    """
    경로의 Gaussian Cost Map 위반 비용 계산.

    Returns
    -------
    float
        평균 violation cost (0=원양, 100=육지 내부)
    """
    return cost_map.route_violation(route["waypoints"])


# =============================================================================
# 4. Fitness 평가 함수
# =============================================================================

def evaluate(
    individual: list,
    cost_map,
    milp_solver,
    weather_fn=None,
    n_segments: int = N_SEGMENTS,
    rta_h: float = RTA_HOURS,
) -> Tuple[float,]:
    """
    GA Fitness 평가.

    Fitness = FC_total + λ × violation_cost
      - violation이 THRESHOLD 이상이면 hard penalty (BIG_PENALTY)
      - 그 이하면 soft penalty로 연료에 가산
    """

    # Chromosome을 n_segments수에 맞게 Velo, heading 나누고
    # 이때, segment 수와 rta_h를 맞춰 마지막 velo, heading까지 decoding하여 넣어줌
    route = decode_route(individual, n_segments, rta_h)

    # 1. 마지막 구간 속도 타당성 패널티
    valid_penalty = 0.0
    if not route["valid"]:
        v_last = route["last_speed"]

        # 마지막 구간이 V_min보다 작을 경우 약간의 가산 penalty
        if v_last < V_MIN_PORT:
            valid_penalty = BIG_PENALTY * (1 + (V_MIN_PORT - v_last) / V_MIN_PORT)
        else:
            valid_penalty = BIG_PENALTY * (1 + (v_last - V_MAX_PORT) / V_MAX_PORT)

    # 2. 육지 관통 (Cost Map) 패널티
    violation = compute_violation(route, cost_map)
    violation_penalty = 0.0
    if violation > VIOLATION_THRESHOLD:
        # 육지 관통 시 속도 패널티보다 10배 무거운 하드 패널티 부여
        violation_penalty = BIG_PENALTY * 10.0 * (1 + violation / 100)


    # 둘 중 하나라도 위반이면 즉시 패널티 반환
    if valid_penalty > 0 and violation_penalty > 0:
        return (valid_penalty + violation_penalty,)

    # 각 구간별 P_req (Kwon's Method)
    P_req_list = []
    for i in range(n_segments):
        wp_from = route["waypoints"][i]
        wp_to = route["waypoints"][i + 1]
        v_kts = route["speeds"][i]
        heading = route["headings"][i]

        mid_lat = (wp_from[0] + wp_to[0]) / 2
        mid_lon = (wp_from[1] + wp_to[1]) / 2

        if weather_fn is not None:
            wind_speed, wind_dir = weather_fn(mid_lat, mid_lon)
        else:
            wind_speed, wind_dir = 0.0, 0.0

        # 벡터 방식으로 encounter angle 계산(cosine 제 2법칙)
        encounter_rad = math.radians(heading - wind_dir)
        encounter_x = v_kts * 0.5144 + wind_speed * math.cos(encounter_rad)
        encounter_y = wind_speed * math.sin(encounter_rad)
        encounter = math.atan2(encounter_y, encounter_x)
        encounter_deg = math.degrees(encounter) % 360

        # 기존 방식 (절대 각도 차이)
        # encounter = abs((wind_dir - heading + 180) % 360)
        # if encounter > 180:
        #     encounter = 360 - encounter

        if i == 0:
            P_service = SERVICE_LOAD["departure"]
        elif i == n_segments - 1:
            P_service = SERVICE_LOAD["approach"]
        else:
            P_service = SERVICE_LOAD["cruising"]

        result = compute_P_req(
            v_ship_knots=v_kts,
            v_wind_ms=wind_speed,
            encounter_angle_deg=encounter,
            a1=POWER_MODEL["a1"],
            P_service=P_service,
        )
        P_req_list.append(result["P_req"])

    # MILP 풀이
    '''
    P_req_list = [p1, p2, ..., p_n]
    milp_solver.solve() → {"feasible": bool, "total_fuel_kg": float, ...}
    초기 SOC 
    '''
    milp_result = milp_solver.solve(
        P_req=P_req_list,
        dt=route["dt"],
        initial_SOC=0.7,
        msg=False,
    )

    if not milp_result["feasible"]:
        return (BIG_PENALTY,)

    # 나중에 꼭 고쳐야 함 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Fitness = 연료 + λ × violation (soft penalty)
    fuel = milp_result["total_fuel_kg"]
    return (fuel + LAMBDA_VIOLATION * violation,)


# =============================================================================
# 5. DEAP 설정 및 실행
# =============================================================================

def setup_ga(
    cost_map,
    milp_solver,
    weather_fn=None,
    n_segments: int = N_SEGMENTS,
    pop_size: int = 100,
    n_gen: int = 50,
    cx_prob: float = 0.7, # 교차 확률
    mut_prob: float = 0.3, # 변이 확률
    tournament_size: int = 3, # 토너먼트 선택 크기
    seed: int = 42, # 랜덤 시드 고정
    n_workers: int = 1, 
    sfoc_path: str = "config/sfoc.json",
):
    """
    DEAP GA 전체 설정 + 실행.

    Parameters
    ----------
    cost_map : CostMap
        Cost Map (violation 평가용)
    n_workers : int
        병렬 워커 수. 0이면 CPU 코어 수 자동 감지.
    sfoc_path : str
        SFOC JSON 경로 (병렬 워커용)
    """
    random.seed(seed)
    np.random.seed(seed)

    n_free = n_segments - 1
    n_genes = n_free * 2

    # ── DEAP 타입 생성 ──
    # DEAP의 creator는 한 번만 생성해야 하므로, 이미 존재하는지 확인 후 생성
    # FitnessMin은 weghts=(-1.0,)으로 단일 목적 최소화 문제 정의, 적합도 클래스 생성
    # Individual은 list 기반으로 fitness 속성 추가
    if "FitnessMin" not in dir(creator):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in dir(creator):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    # toolbox는 유전자 초기화, 교차, 변이, 선택 함수 등록용 DEAP 객체
    toolbox = base.Toolbox()

    # ── 유전자 초기화 ──
    base_bearing = compute_bearing(*BUSAN_PORT, *JEJU_PORT)

    # 초기 개체 생성 함수: 각 구간별로 속도와 방위각을 랜덤하게 생성
    def init_individual():
        genes = []
        for seg in range(n_free):
            if seg == 0:  # 출항 구간 → 감속 범위
                genes.append(random.uniform(V_MIN_PORT, V_MAX_PORT))
            else:
                genes.append(random.uniform(V_MIN, V_MAX))
            genes.append((base_bearing + random.uniform(-30, 30)) % 360)
        return creator.Individual(genes)

    # toolbox에 개체와 인구 생성 등록
    # tools.initRepeat은 toolbox.individual 함수를 반복 호출하여 리스트 형태의 인구 생성
    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # ── 병렬 / 순차 분기 ──
    # n_workers가 0이면 시스템의 CPU 코어 수를 자동으로 감지하여 사용
    if n_workers == 0:
        n_workers = multiprocessing.cpu_count()

    pool = None

    # 병렬 모드에서는 multiprocessing.Pool을 생성하여 워커 프로세스 초기화 함수를 지정하고, toolbox의 map과 evaluate 함수를 병렬 버전으로 등록
    if n_workers > 1:
        print(f"  ⚡ Multiprocessing: {n_workers} workers")

        pool = multiprocessing.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(
                cost_map,       # CostMap 인스턴스
                sfoc_path,      # SFOC JSON path
                5,              # n_pwl
                weather_fn,     # weather function
                n_segments,
                RTA_HOURS,
            ),
        )
        toolbox.register("map", pool.map)
        toolbox.register("evaluate", _evaluate_parallel)
    else:
        print(f"  Sequential mode (1 worker)")
        import functools
        toolbox.register(
            "evaluate",
            functools.partial(
                evaluate,
                cost_map=cost_map,
                milp_solver=milp_solver,
                weather_fn=weather_fn,
                n_segments=n_segments,
            ),
        )

    # ── 교차 / 변이 / 선택 ──
    # 유전자별 상·하한: 첫 구간(출항)은 감속 범위
    low_bounds = [V_MIN_PORT, 0.0] + [V_MIN, 0.0] * (n_free - 1)
    up_bounds = [V_MAX_PORT, 360.0] + [V_MAX, 360.0] * (n_free - 1)
    
    # Simulated Binary Crossover (SBX)와 Polynomial Mutation을 경계 내에서 적용하도록 DEAP의 도구 등록
    # eta는 분포 지수, indpb는 각 유전자별 변이 확률 (여기서는 1/n_genes로 설정하여 평균적으로 한 개의 유전자 변이)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                     low=low_bounds,
                     up=up_bounds,
                     eta=20.0)

    toolbox.register("mutate", tools.mutPolynomialBounded,
                     low=low_bounds,
                     up=up_bounds,
                     eta=20.0,
                     indpb=1.0 / n_genes) # 각 유전자별 변이 확률

    # select 함수는 tool.selTournament으로 등록하여 토너먼트 선택 방식 사용, tournament_size는 토너먼트 크기
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    # Hall of Fame은 최상의 5개 개체를 저장하도록 설정(명예의 전당쓰)
    hof = tools.HallOfFame(5)

    # GA 실행 로그 기록을 위해 각 Generation마다 최소, 평균, 최대 적합도 기록하도록 Statistics 객체 생성
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    # ── 실행 ──
    print(f"\n{'='*60}")
    print(f" DEAP GA 실행")
    print(f" 인구: {pop_size} | 세대: {n_gen} | 구간: {n_segments}")
    print(f" RTA: {RTA_HOURS}h | 속도: {V_MIN}~{V_MAX} kts")
    print(f" 기본 방위: {base_bearing:.1f}°")
    if n_workers > 1:
        print(f" 병렬: {n_workers} 코어")
    print(f"{'='*60}")

    # 초기 population 생성
    pop = toolbox.population(n=pop_size)

    # GA 실행: eaSimple은 선택, 교차, 변이 과정을 반복하여 세대를 진화시키는 DEAP의 기본 알고리즘 함수
    try:
        result_pop, logbook = algorithms.eaSimple(
            pop, toolbox,
            cxpb=cx_prob,
            mutpb=mut_prob,
            ngen=n_gen,
            stats=stats,
            halloffame=hof,
            verbose=True,
        )
    finally:
        # 워커 프로세스 종료 (병렬 프로세스)
        if pool is not None:
            pool.close()
            pool.join()

    # 최종 결과 Hall of Fame에서 최상의 개체를 추출하여 디코딩, 최적 경로와 적합도 반환
    best_ind = hof[0]
    best_route = decode_route(list(best_ind), n_segments)

    return {
        "best_individual": list(best_ind),
        "best_fitness": best_ind.fitness.values[0],
        "best_route": best_route,
        "logbook": logbook,
        "population": result_pop,
        "hall_of_fame": hof,
    }
