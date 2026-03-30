"""
Manual GA integration verification script.

This file is a scenario runner, not a pytest-style unit test.
Run it directly with `python tests/test_ga.py`.
"""

import sys
import os
import time

# Project 절대 경로 만들고 모듈 임포트 절대성 주입
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.grid.no_go_zone import BUSAN_PORT, JEJU_PORT
from src.grid.cost_map import build_cost_map
from src.optimizer.milp_solver import MILPSolver
from src.optimizer.ga_engine import setup_ga, haversine_nm
from src.weather.era5_loader import ERA5Loader
from src.visualization.plotter import (
    plot_optimal_route, plot_convergence, plot_power_schedule, plot_weather_map
)


# ─── 설정 ───
ERA5_NC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "era5", "era5_wind_2024_01.nc",
)
SFOC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "sfoc.json",
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output",
)


def main():
    t0 = time.time()

    # ═══════════════════════════════════════════
    # 1. Cost Map 구축
    # ═══════════════════════════════════════════
    print("=" * 60)
    print(" [Phase 1] Cost Map 구축")
    print("=" * 60)

    cmap = build_cost_map(resolution=0.005)

    # ═══════════════════════════════════════════
    # 2. 기상 데이터 로드
    # ═══════════════════════════════════════════
    print("\n" + "=" * 60)
    print(" [Phase 2] 기상 데이터")
    print("=" * 60)

    if os.path.exists(ERA5_NC_PATH):
        print("  ERA5 실제 데이터 사용")
        era5 = ERA5Loader(ERA5_NC_PATH, time_index=0)
        weather_fn = era5.get_weather_fn()
    else:
        print("  ERA5 미확보")
        raise FileNotFoundError(f"ERA5 file not found: {ERA5_NC_PATH}")

    # 부산항 기상 확인
    ws, wd = weather_fn(*BUSAN_PORT)
    print(f"  부산항 기상: 풍속 {ws:.1f} m/s, 풍향 {wd:.0f}°")
    ws, wd = weather_fn(*JEJU_PORT)
    print(f"  제주항 기상: 풍속 {ws:.1f} m/s, 풍향 {wd:.0f}°")

    # ═══════════════════════════════════════════
    # 3. MILP 솔버
    # ═══════════════════════════════════════════
    print("\n" + "=" * 60)
    print(" [Phase 3] MILP 솔버 초기화")
    print("=" * 60)

    milp = MILPSolver(sfoc_json_path=SFOC_PATH, n_pwl_segments=5)
    print("  MILP 준비 완료")

    # ═══════════════════════════════════════════
    # 4. GA 실행
    # ═══════════════════════════════════════════
    print("\n" + "=" * 60)
    print(" [Phase 4] GA 최적화")
    print("=" * 60)

    direct_dist = haversine_nm(BUSAN_PORT, JEJU_PORT)
    print(f"  직선 거리: {direct_dist:.1f} nm")

    ga_result = setup_ga(
        cost_map=cmap,
        milp_solver=milp,
        weather_fn=weather_fn,
        n_segments=24,
        pop_size=20,
        n_gen=100,
        seed=42,
        n_workers=0,            # 0 = CPU 코어 자동 감지
        sfoc_path=SFOC_PATH,    # 병렬 워커용
    )

    # ═══════════════════════════════════════════
    # 5. 결과 출력
    # ═══════════════════════════════════════════
    elapsed = time.time() - t0
    route = ga_result["best_route"]

    print("\n" + "=" * 60)
    print(" 최적화 결과")
    print("=" * 60)
    print(f"  총 연료: {ga_result['best_fitness']:.1f} kg")
    print(f"  계산 시간: {elapsed:.1f} sec")
    print(f"  마지막 구간 속도: {route['last_speed']:.2f} kts")

    print(f"\n  구간별 상세:")
    print(f"  {'Seg':>4s} {'V(kts)':>8s} {'θ(°)':>8s} {'dist(nm)':>9s} {'lat':>8s} {'lon':>9s}")
    print(f"  {'-'*52}")
    for i in range(len(route["speeds"])):
        wp = route["waypoints"][i]
        print(
            f"  {i:>4d} {route['speeds'][i]:>8.2f} "
            f"{route['headings'][i]:>8.1f} "
            f"{route['distances_nm'][i]:>9.2f} "
            f"{wp[0]:>8.4f} {wp[1]:>9.4f}"
        )
    wp = route["waypoints"][-1]
    print(f"  {'END':>4s} {'':>8s} {'':>8s} {'':>9s} {wp[0]:>8.4f} {wp[1]:>9.4f}")

    # ═══════════════════════════════════════════
    # 6. 최적 해의 MILP 상세 결과 추출 (시각화용)
    # ═══════════════════════════════════════════
    print("\n" + "=" * 60)
    print(" [Phase 6] 시각화")
    print("=" * 60)

    # 최적 해로 MILP 재실행 (상세 스케줄 추출)
    from src.resistance.kwon_method import compute_P_req
    from src.ship.kcs_specs import POWER_MODEL, SERVICE_LOAD

    P_req_best = []
    for i in range(len(route["speeds"])):
        wp_from = route["waypoints"][i]
        wp_to = route["waypoints"][i + 1]
        heading = route["headings"][i]
        v_kts = route["speeds"][i]
        mid_lat = (wp_from[0] + wp_to[0]) / 2
        mid_lon = (wp_from[1] + wp_to[1]) / 2
        ws, wd = weather_fn(mid_lat, mid_lon)
        enc = abs((wd - heading + 180) % 360)
        if enc > 180:
            enc = 360 - enc
        if i == 0:
            ps = SERVICE_LOAD["departure"]
        elif i == len(route["speeds"]) - 1:
            ps = SERVICE_LOAD["approach"]
        else:
            ps = SERVICE_LOAD["cruising"]
        res = compute_P_req(v_kts, ws, enc, POWER_MODEL["a1"], ps)
        P_req_best.append(res["P_req"])

    milp_detail = milp.solve(P_req=P_req_best, dt=route["dt"], initial_SOC=0.8, msg=False)

    # ── 시각화 ──
    cmap.plot(save_path=os.path.join(OUTPUT_DIR, "cost_map_route.png"), route_wps=route["waypoints"])
    plot_weather_map(weather_fn, cmap, route_wps=route["waypoints"], save_dir=OUTPUT_DIR)
    plot_optimal_route(route, cmap, save_dir=OUTPUT_DIR)
    plot_convergence(ga_result["logbook"], save_dir=OUTPUT_DIR)
    if milp_detail["feasible"]:
        plot_power_schedule(milp_detail, save_dir=OUTPUT_DIR)

    print(f"\n  총 실행 시간: {elapsed:.1f} sec")
    print(f"  결과 저장: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
