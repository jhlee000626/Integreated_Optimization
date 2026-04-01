"""
Legacy GA fitness helpers.

The active GA/MILP integration now lives in `src.optimizer.ga_engine`.
This module is kept primarily for older experiments and for a few geometry
helpers that are still reused by legacy verification scripts.
"""

import math
from typing import List, Tuple

# ── 프로젝트 내부 모듈 ──
from src.resistance.modified_dpm import compute_P_req
from src.ship.kcs_specs import POWER_MODEL, SERVICE_LOAD, NAV_PARAMS

# =============================================================================
# 상수
# =============================================================================

BIG_PENALTY = 1e12  # Infeasible 염색체에 부여하는 패널티 값


# =============================================================================
# 1. 염색체 디코딩
# =============================================================================

def decode_chromosome(individual: list) -> Tuple[List[int], List[float]]:
    """
    DEAP 개체(individual)를 웨이포인트 인덱스와 구간 속도로 분리.

    Chromosome 구조:
        [wp_1, v_1, wp_2, v_2, ..., wp_n, v_n]

    Parameters
    ----------
    individual : list
        DEAP individual (flat list)

    Returns
    -------
    waypoints : List[int]
        Quad-tree 셀 인덱스 리스트 [wp_1, wp_2, ..., wp_n]
    speeds : List[float]
        구간별 속도 (knots) [v_1, v_2, ..., v_n]
    """
    waypoints = [int(individual[i]) for i in range(0, len(individual), 2)]
    speeds = [float(individual[i]) for i in range(1, len(individual), 2)]
    return waypoints, speeds


# =============================================================================
# 2. 구간별 항해 정보 계산
# =============================================================================

def compute_segment_info(
    wp_from: tuple,
    wp_to: tuple,
    speed_knots: float,
) -> dict:
    """
    두 웨이포인트 간 거리(nm) 및 소요 시간(h) 계산.

    Parameters
    ----------
    wp_from : tuple
        출발점 (lat, lon)
    wp_to : tuple
        도착점 (lat, lon)
    speed_knots : float
        구간 속도 (knots)

    Returns
    -------
    dict
        {"distance_nm": float, "duration_h": float}
    """
    distance_nm = haversine_nm(wp_from, wp_to)
    duration_h = distance_nm / speed_knots if speed_knots > 0.1 else 1e6

    return {
        "distance_nm": distance_nm,
        "duration_h": duration_h,
    }


def haversine_nm(p1: tuple, p2: tuple) -> float:
    """
    Haversine 공식으로 두 좌표 사이의 거리를 해리(nm) 단위로 반환.
    """
    R_nm = 3440.065  # 지구 반경 (nautical miles)
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R_nm * c


# =============================================================================
# 3. 구간별 입사각 (Encounter Angle) 계산
# =============================================================================

def compute_encounter_angle(
    heading_deg: float,
    wind_dir_deg: float,
) -> float:
    """
    선수 방위각과 바람/파도 방향으로부터 입사각(Encounter Angle) 계산.

    Parameters
    ----------
    heading_deg : float
        선수 방위각 (°, 진북 기준 시계방향, 0°=N)
    wind_dir_deg : float
        바람이 불어오는 방향 (°, 기상학 관례: FROM direction)

    Returns
    -------
    float
        입사각 (0° = Head seas, 180° = Following seas)
    """
    # 바람이 불어오는 방향 → 선수 기준 상대 각도
    # Head seas (0°): 바람이 선수에서 불어옴
    relative = (wind_dir_deg - heading_deg + 180.0) % 360.0
    if relative > 180.0:
        relative = 360.0 - relative
    return relative


def compute_heading(wp_from: tuple, wp_to: tuple) -> float:
    """
    두 웨이포인트로부터 선수 방위각(True Heading) 계산.

    Returns
    -------
    float
        방위각 (°, 0°=N, 시계방향)
    """
    lat1, lon1 = math.radians(wp_from[0]), math.radians(wp_from[1])
    lat2, lon2 = math.radians(wp_to[0]), math.radians(wp_to[1])
    dlon = lon2 - lon1

    x = math.sin(dlon) * math.cos(lat2)
    y = (math.cos(lat1) * math.sin(lat2) -
         math.sin(lat1) * math.cos(lat2) * math.cos(dlon))

    bearing = math.degrees(math.atan2(x, y))
    return bearing % 360.0


# =============================================================================
# 4. 운항 구간(Phase) 결정
# =============================================================================

def determine_operation_phase(
    segment_index: int,
    total_segments: int,
) -> str:
    """
    구간 인덱스로부터 운항 Phase (departure/cruising/approach) 결정.

    규칙:
        - 첫 번째 구간: departure (출항)
        - 마지막 구간: approach (접안)
        - 나머지: cruising (순항)
    """
    if segment_index == 0:
        return "departure"
    elif segment_index == total_segments - 1:
        return "approach"
    else:
        return "cruising"


def get_segment_speed(
    base_speed_knots: float,
    phase: str,
    gamma: float = 0.7,
) -> float:
    """
    운항 Phase에 따른 실제 속도 결정.

    이/접안 구간에서는 γ 비율 적용.
    """
    if phase in ("departure", "approach"):
        return base_speed_knots * gamma
    return base_speed_knots


# =============================================================================
# 5. GA Fitness 평가 — 메인 함수
# =============================================================================

# Legacy evaluation path retained for older experiments.
def calculate_fitness(
    individual: list,
    quadtree_graph: object,
    weather_loader: object,
    milp_solver: object,
    departure_time=None,
) -> Tuple[float,]:
    """
    GA의 적합도(Fitness) 평가 함수.
    DEAP의 evalFitness로 등록하여 사용.

    Parameters
    ----------
    individual : list
        DEAP individual — [wp_1, v_1, wp_2, v_2, ..., wp_n, v_n]
    quadtree_graph : object
        Quad-tree 그리드 → 셀 인덱스 → (lat, lon) 변환 및 인접성 검증
    weather_loader : object
        ERA5 기상 데이터 로더 → get_weather(lat, lon, datetime)
    milp_solver : object
        MILP 발전기 스케줄링 솔버
    departure_time : datetime, optional
        출항 시각

    Returns
    -------
    tuple
        (fitness_value,) — DEAP는 tuple 반환 필요 (단일 목적)
        fitness_value = FC_total (kg) → 최소화 문제이므로 DEAP weights=(-1.0,)
    """

    # ── Step 0: 염색체 디코딩 ──
    waypoints_idx, speeds = decode_chromosome(individual)
    n_segments = len(waypoints_idx) - 1

    if n_segments <= 0:
        return (BIG_PENALTY,)

    # ── Step 1: 경로 유효성 검증 ──
    for i in range(n_segments):
        wp_from_idx = waypoints_idx[i]
        wp_to_idx = waypoints_idx[i + 1]

        # Quad-tree 인접성 확인
        if not quadtree_graph.are_adjacent(wp_from_idx, wp_to_idx):
            return (BIG_PENALTY,)  # 불연속 경로 → 페널티

    # ── Step 2: 속도 범위 검증 ──
    v_min, v_max = NAV_PARAMS["speed_range"]
    for v in speeds:
        if v < v_min or v > v_max:
            return (BIG_PENALTY,)

    # ── Step 3: 각 구간별 요구 부하(P_req) 산출 ──
    load_profile = []      # [{P_req, duration_h, ...}, ...]
    cumulative_time_h = 0.0

    for seg_i in range(n_segments):
        # 3a. 웨이포인트 좌표 변환
        wp_from = quadtree_graph.get_coordinates(waypoints_idx[seg_i])
        wp_to = quadtree_graph.get_coordinates(waypoints_idx[seg_i + 1])

        # 3b. 운항 Phase 결정 + 실속도
        phase = determine_operation_phase(seg_i, n_segments)
        base_speed = speeds[min(seg_i, len(speeds) - 1)]
        actual_speed = get_segment_speed(base_speed, phase, NAV_PARAMS["gamma"])

        # 3c. 구간 거리 & 소요 시간
        seg_info = compute_segment_info(wp_from, wp_to, actual_speed)

        # 3d. 선수 방위각
        heading = compute_heading(wp_from, wp_to)

        # 3e. ERA5 기상 데이터 조회
        mid_lat = (wp_from[0] + wp_to[0]) / 2.0
        mid_lon = (wp_from[1] + wp_to[1]) / 2.0
        seg_time = departure_time + cumulative_time_h if departure_time else None

        weather = weather_loader.get_weather(mid_lat, mid_lon, seg_time)
        # weather = {"wind_speed_ms": float, "wind_dir_deg": float, ...}

        # 3f. Encounter Angle
        encounter_angle = compute_encounter_angle(heading, weather["wind_dir_deg"])

        # 3g. ★ Kwon's Method → P_req (비선형 연산 HERE) ★
        P_service = SERVICE_LOAD[phase]
        power_result = compute_P_req(
            v_ship_knots=actual_speed,
            v_wind_ms=weather["wind_speed_ms"],
            encounter_angle_deg=encounter_angle,
            a1=POWER_MODEL["a1"],
            P_service=P_service,
            heading_deg=heading,
            wind_dir_deg=weather["wind_dir_deg"],
        )

        # 3h. 구간 정보 저장
        load_profile.append({
            "segment_index": seg_i,
            "phase": phase,
            "speed_knots": actual_speed,
            "distance_nm": seg_info["distance_nm"],
            "duration_h": seg_info["duration_h"],
            "P_prop_MW": power_result["P_prop"],
            "P_service_MW": power_result["P_service"],
            "P_req_MW": power_result["P_req"],       # ← MILP에 상수로 전달
            "a2_penalty": power_result["a2"],
            "BN": power_result["BN"],
            "L_percent": power_result["L_percent"],
        })

        cumulative_time_h += seg_info["duration_h"]

    # ── Step 4: MILP에 부하 프로파일 전달 ──
    # P_req 리스트는 '결정된 상수' → MILP 내부는 순수 선형
    P_req_list = [seg["P_req_MW"] for seg in load_profile]
    dt_list = [seg["duration_h"] for seg in load_profile]

    # ── Step 5: MILP 풀이 → 총 연료 소모량 ──
    milp_result = milp_solver.solve(
        P_req=P_req_list,
        dt=dt_list,
    )

    if not milp_result["feasible"]:
        return (BIG_PENALTY,)

    FC_total_kg = milp_result["total_fuel_kg"]

    # ── Step 6: Fitness 반환 ──
    # DEAP weights=(-1.0,) → 최소화 문제 → raw FC를 반환
    return (FC_total_kg,)


# =============================================================================
# 6. DEAP 등록 예시 (main.py에서 호출)
# =============================================================================

def setup_deap_fitness():
    """
    DEAP Toolbox에 Fitness 함수를 등록하는 예시 코드.
    실제 main.py에서 호출.
    """
    from deap import base, creator

    # 최소화 문제: weights=(-1.0,)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # ── 개체 생성 (예시) ──
    # 실제로는 Quad-tree 셀 인덱스 + 속도 범위로 초기화
    # toolbox.register("individual", ...)
    # toolbox.register("population", ...)

    # ── Evaluation 등록 ──
    # calculate_fitness는 추가 인자(quadtree, weather, milp)가 필요하므로
    # functools.partial 또는 lambda로 래핑
    # toolbox.register("evaluate", calculate_fitness,
    #                   quadtree_graph=qt, weather_loader=era5,
    #                   milp_solver=milp, departure_time=dt)

    # ── GA 연산자 등록 ──
    # toolbox.register("select", tools.selTournament, tournsize=3)
    # toolbox.register("mate", ...)
    # toolbox.register("mutate", ...)

    return toolbox
