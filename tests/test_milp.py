"""
Manual MILP verification script.

This file is a scenario runner, not a pytest-style unit test.
Run it directly with `python tests/test_milp.py`.
"""

import sys
import os

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimizer.milp_solver import MILPSolver


def create_synthetic_load_profile():
    """
    합성 부하 프로파일 생성.
    P_req = P_prop + P_service (MW)

    시나리오:
        출항 → 순항 (부하 변동) → 접안
    """
    # 12시간 항해, 1시간 간격
    dt = [1.0] * 12  # 각 구간 1시간

    # 요구 부하 (MW): 추진 + 서비스 합산
    P_req = [
        15.0,   # t=0:  출항 (저속 추진 5~6MW + 서비스 8.69MW)
        22.0,   # t=1:  순항 가속
        28.0,   # t=2:  순항 정속 (양호한 해상)
        30.0,   # t=3:  순항 (약간의 역풍)
        33.0,   # t=4:  순항 (역풍 증가)
        35.0,   # t=5:  순항 (BN 5~6, 피크)
        32.0,   # t=6:  순항 (역풍 감소)
        28.0,   # t=7:  순항 (양호)
        25.0,   # t=8:  순항 (양호)
        27.0,   # t=9:  순항
        20.0,   # t=10: 감속 시작
        12.0,   # t=11: 접안 (저속 + 서비스 8.69MW)
    ]

    return P_req, dt


def print_schedule(result):
    """스케줄 상세 출력."""
    if not result["feasible"]:
        print("\n❌ INFEASIBLE — 해를 찾지 못했습니다.")
        print(f"   Status: {result['status']}")
        return

    print("\n" + "=" * 100)
    print(" MILP 최적 스케줄 결과")
    print("=" * 100)

    header = (
        f"{'t':>3s} {'dt':>4s} {'P_req':>7s} │"
        f"{'DG1_P':>7s} {'ON':>3s} │"
        f"{'DG2_P':>7s} {'ON':>3s} │"
        f"{'DG3_P':>7s} {'ON':>3s} │"
        f"{'P_dc':>6s} {'P_c':>6s} {'SOC':>6s}"
    )
    print(header)
    print("-" * 100)

    for s in result["schedule"]:
        line = (
            f"{s['t']:>3d} {s['dt_h']:>4.1f} {s['P_req_MW']:>7.2f} │"
            f"{s['DG1_P_MW']:>7.2f} {s['DG1_ON']:>3d} │"
            f"{s['DG2_P_MW']:>7.2f} {s['DG2_ON']:>3d} │"
            f"{s['DG3_P_MW']:>7.2f} {s['DG3_ON']:>3d} │"
            f"{s['P_dc_MW']:>6.2f} {s['P_c_MW']:>6.2f} {s['SOC']:>6.3f}"
        )
        print(line)

    # 요약
    summary = result["summary"]
    print("\n" + "─" * 60)
    print(" 요약")
    print("─" * 60)
    print(f"  항해 시간:      {summary['total_voyage_h']:.1f} h")
    print(f"  총 연료 소모:   {summary['total_fuel_kg']:.1f} kg")
    print(f"  총 기동 비용:   ${summary['total_start_cost_usd']:.0f}")
    print(f"  목적함수 값:    {result['objective_value']:.1f}")
    print(f"  ESS 최종 SOC:   {summary['ess_final_SOC']:.3f}")

    print("\n  DG별 가동 시간:")
    for dg, hours in summary["dg_running_hours"].items():
        fuel = summary["dg_fuel_kg"][dg]
        starts = summary["dg_start_count"][dg]
        print(f"    {dg}: {hours:.1f}h / 연료 {fuel:.1f}kg / 기동 {starts}회")


def main():
    print("=" * 60)
    print(" MILP 솔버 테스트 — 합성 부하 프로파일")
    print("=" * 60)

    # SFOC JSON 경로 (프로젝트 루트 기준)
    sfoc_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "sfoc.json"
    )

    print(f"\n[1] SFOC 로드: {sfoc_path}")

    # 솔버 인스턴스 생성
    solver = MILPSolver(sfoc_json_path=sfoc_path, n_pwl_segments=5)
    print("[2] MILPSolver 인스턴스 생성 완료")

    # PWL Breakpoints 확인
    print("\n[3] PWL Breakpoints:")
    for dg in solver.dg_names:
        bp = solver.pwl_breakpoints[dg]
        print(f"    {dg}: ", end="")
        for p, fc in bp:
            print(f"({p:.1f}MW, {fc:.0f}kg/h) ", end="")
        print()

    # 합성 부하 프로파일
    P_req, dt = create_synthetic_load_profile()
    print(f"\n[4] 부하 프로파일: {len(P_req)} 스텝, 총 {sum(dt):.0f}h")
    total_cap = sum(s["P_max"] for s in solver.dg_specs.values())
    print(f"    DG 총 용량: {total_cap:.0f} MW")
    print(f"    P_req 범위: {min(P_req):.1f} ~ {max(P_req):.1f} MW")

    # 풀이
    print("\n[5] MILP 풀이 시작...")
    result = solver.solve(
        P_req=P_req,
        dt=dt,
        initial_SOC=0.8,
        initial_u={"DG1": 0, "DG2": 0, "DG3": 0},
        time_limit_sec=60,
        msg=False,
    )

    print_schedule(result)


if __name__ == "__main__":
    main()
