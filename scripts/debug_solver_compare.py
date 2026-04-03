"""
4.1 진단: 동일 MILP 모델을 LP/MPS로 export 후 CBC / CPLEX_CMD 비교.

재현 조건:
  P_req = [11.0] * 18,  dt = [0.5] * 18,  initial_SOC = 0.7
"""

from __future__ import annotations

import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pulp import LpStatus, value, listSolvers, PULP_CBC_CMD
from src.optimizer.milp_solver import MILPSolver, SafeCPLEX_CMD

SFOC_PATH = os.path.join(PROJECT_ROOT, "config", "sfoc.json")
OUT_DIR = os.path.join(PROJECT_ROOT, "output", "debug_solver")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 재현 입력 ───
T = 18
P_REQ = [11.0] * T
DT = [0.5] * T
INITIAL_SOC = 0.7


def solve_and_report(solver_name: str, export_files: bool = False):
    """주어진 solver로 풀고 결과를 출력."""
    milp = MILPSolver(
        sfoc_json_path=SFOC_PATH,
        n_pwl_segments=5,
        solver_name=solver_name,
    )

    # ── 내부 prob 객체에 접근하기 위해 solve() 코드를 일부 복제 ──
    # milp.solve()를 직접 호출하되, export가 필요하면 별도 처리
    if not export_files:
        result = milp.solve(
            P_req=P_REQ,
            dt=DT,
            initial_SOC=INITIAL_SOC,
            msg=True,  # 솔버 로그 출력
        )
        return result

    # export_files=True일 때: 내부 prob를 직접 구성
    return milp.solve(
        P_req=P_REQ,
        dt=DT,
        initial_SOC=INITIAL_SOC,
        msg=True,
    )


def solve_with_export(solver_backend_name: str):
    """
    MILP 모델을 구성하고, LP/MPS 파일로 export한 뒤, 지정된 solver로 풀기.
    milp_solver.py의 solve()를 재현하되 prob 객체를 export하는 단계를 추가.
    """
    from pulp import (
        LpProblem, LpMinimize, LpVariable, LpBinary,
        LpStatus, lpSum, value,
    )
    from src.ship.kcs_specs import DG_SPECS, DG_MIN_LOAD_RATIO, ESS_SPECS
    from src.optimizer.milp_solver import load_sfoc, generate_pwl_breakpoints

    sfoc_data = load_sfoc(SFOC_PATH)
    dg_specs = DG_SPECS
    min_load_ratio = DG_MIN_LOAD_RATIO
    dg_names = list(dg_specs.keys())
    ess = ESS_SPECS
    n_pwl = 5

    pwl_breakpoints = {}
    for dg in dg_names:
        P_min = dg_specs[dg]["P_max"] * min_load_ratio
        P_max = dg_specs[dg]["P_max"]
        sfoc = sfoc_data[dg]
        pwl_breakpoints[dg] = generate_pwl_breakpoints(
            P_min, P_max, sfoc["alpha1"], sfoc["alpha2"], sfoc["alpha3"], n_pwl,
        )

    T = len(P_REQ)
    initial_u = {dg: 0 for dg in dg_names}

    prob = LpProblem("DG_ESS_Debug", LpMinimize)

    # ── 변수 ──
    P_dg = {(dg, t): LpVariable(f"P_{dg}_{t}", lowBound=0) for dg in dg_names for t in range(T)}
    u = {(dg, t): LpVariable(f"u_{dg}_{t}", cat=LpBinary) for dg in dg_names for t in range(T)}
    y = {(dg, t): LpVariable(f"y_{dg}_{t}", cat=LpBinary) for dg in dg_names for t in range(T)}
    z = {(dg, t): LpVariable(f"z_{dg}_{t}", cat=LpBinary) for dg in dg_names for t in range(T)}
    P_c = {t: LpVariable(f"P_c_{t}", lowBound=0, upBound=ess["P_c_max"]) for t in range(T)}
    P_dc = {t: LpVariable(f"P_dc_{t}", lowBound=0, upBound=ess["P_dc_max"]) for t in range(T)}
    SOC = {t: LpVariable(f"SOC_{t}", lowBound=ess["SOC_min"], upBound=ess["SOC_max"]) for t in range(T)}
    u_ess = {t: LpVariable(f"u_ess_{t}", cat=LpBinary) for t in range(T)}
    lam = {
        (dg, t, k): LpVariable(f"lam_{dg}_{t}_{k}", lowBound=0, upBound=1)
        for dg in dg_names for t in range(T) for k in range(n_pwl + 1)
    }
    delta = {
        (dg, t, s): LpVariable(f"delta_{dg}_{t}_{s}", cat=LpBinary)
        for dg in dg_names for t in range(T) for s in range(n_pwl)
    }
    FC_pwl = {(dg, t): LpVariable(f"FC_{dg}_{t}", lowBound=0) for dg in dg_names for t in range(T)}

    # ── 목적함수 ──
    prob += (
        lpSum(FC_pwl[dg, t] * DT[t] for dg in dg_names for t in range(T))
        + lpSum(dg_specs[dg]["cost_start"] * y[dg, t] for dg in dg_names for t in range(T)),
        "Total_Cost"
    )

    # ── 제약 ──
    for t in range(T):
        prob += (lpSum(P_dg[dg, t] for dg in dg_names) + P_dc[t] - P_c[t] == P_REQ[t], f"PowerBalance_{t}")
        prob += (P_c[t] <= ess["P_c_max"] * u_ess[t], f"ESS_Charge_Mode_{t}")
        prob += (P_dc[t] <= ess["P_dc_max"] * (1 - u_ess[t]), f"ESS_Discharge_Mode_{t}")

        inv_eta_dc = 1.0 / ess["eta_dc"]
        coeff = DT[t] / ess["capacity"]
        if t == 0:
            prob += (SOC[t] == INITIAL_SOC + (ess["eta_c"] * P_c[t] - inv_eta_dc * P_dc[t]) * coeff, f"SOC_update_{t}")
        else:
            prob += (SOC[t] == SOC[t-1] + (ess["eta_c"] * P_c[t] - inv_eta_dc * P_dc[t]) * coeff, f"SOC_update_{t}")

        for dg in dg_names:
            P_max_dg = dg_specs[dg]["P_max"]
            P_min_dg = P_max_dg * min_load_ratio
            ramp = dg_specs[dg]["ramp_rate"]
            bp = pwl_breakpoints[dg]

            prob += (P_dg[dg, t] >= P_min_dg * u[dg, t], f"DG_Pmin_{dg}_{t}")
            prob += (P_dg[dg, t] <= P_max_dg * u[dg, t], f"DG_Pmax_{dg}_{t}")
            prob += (P_dg[dg, t] == lpSum(lam[dg, t, k] * bp[k][0] for k in range(n_pwl + 1)), f"PWL_power_{dg}_{t}")
            prob += (FC_pwl[dg, t] == lpSum(lam[dg, t, k] * bp[k][1] for k in range(n_pwl + 1)), f"PWL_cost_{dg}_{t}")
            prob += (lpSum(lam[dg, t, k] for k in range(n_pwl + 1)) == u[dg, t], f"PWL_sum_{dg}_{t}")

            # SOS2
            prob += (lpSum(delta[dg, t, s] for s in range(n_pwl)) == u[dg, t], f"SOS2_seg_sum_{dg}_{t}")
            prob += (lam[dg, t, 0] <= delta[dg, t, 0], f"SOS2_lam_first_{dg}_{t}")
            for k in range(1, n_pwl):
                prob += (lam[dg, t, k] <= delta[dg, t, k-1] + delta[dg, t, k], f"SOS2_lam_mid_{dg}_{t}_{k}")
            prob += (lam[dg, t, n_pwl] <= delta[dg, t, n_pwl - 1], f"SOS2_lam_last_{dg}_{t}")

            if t == 0:
                prob += (y[dg, t] - z[dg, t] == u[dg, t] - initial_u[dg], f"StartStop_{dg}_{t}")
            else:
                prob += (y[dg, t] - z[dg, t] == u[dg, t] - u[dg, t-1], f"StartStop_{dg}_{t}")
            prob += (y[dg, t] + z[dg, t] <= 1, f"NoSimultaneous_{dg}_{t}")

            if t > 0:
                ramp_limit = ramp * DT[t] * P_max_dg
                prob += (P_dg[dg, t] - P_dg[dg, t-1] <= ramp_limit, f"RampUp_{dg}_{t}")
                prob += (P_dg[dg, t-1] - P_dg[dg, t] <= ramp_limit, f"RampDown_{dg}_{t}")

    # MinUp / MinDown
    def find_covering_steps(start_t, dt_list, required_hours):
        steps = []
        accumulated = 0.0
        for tau in range(start_t, len(dt_list)):
            steps.append(tau)
            accumulated += dt_list[tau]
            if accumulated >= required_hours:
                break
        return steps

    for dg in dg_names:
        min_up_h = dg_specs[dg]["min_up"]
        min_down_h = dg_specs[dg]["min_down"]
        for t in range(T):
            up_steps = find_covering_steps(t, DT, min_up_h)
            if up_steps:
                prob += (lpSum(u[dg, tau] for tau in up_steps) >= len(up_steps) * y[dg, t], f"MinUp_{dg}_{t}")
            down_steps = find_covering_steps(t, DT, min_down_h)
            if down_steps:
                prob += (lpSum(1 - u[dg, tau] for tau in down_steps) >= len(down_steps) * z[dg, t], f"MinDown_{dg}_{t}")

    # ── Export ──
    lp_path = os.path.join(OUT_DIR, "debug_model.lp")
    mps_path = os.path.join(OUT_DIR, "debug_model.mps")
    prob.writeLP(lp_path)
    prob.writeMPS(mps_path)
    print(f"\n  Exported LP  → {lp_path}")
    print(f"  Exported MPS → {mps_path}")

    # ── 모델 통계 ──
    n_vars = len(prob.variables())
    n_binary = sum(1 for v in prob.variables() if v.cat == "Binary")
    n_continuous = n_vars - n_binary
    n_constraints = len(prob.constraints)
    lp_size_kb = os.path.getsize(lp_path) / 1024
    mps_size_kb = os.path.getsize(mps_path) / 1024

    print(f"\n  Model stats:")
    print(f"    Variables:   {n_vars} (continuous={n_continuous}, binary={n_binary})")
    print(f"    Constraints: {n_constraints}")
    print(f"    LP file:     {lp_size_kb:.1f} KB")
    print(f"    MPS file:    {mps_size_kb:.1f} KB")

    # ── LP 파일에서 계수 정밀도 검사 ──
    print(f"\n  Checking LP file coefficient precision...")
    with open(lp_path, "r") as f:
        lp_content = f.read()

    # 매우 작은 계수 찾기
    import re
    coefficients = re.findall(r'([+-]?\s*\d+\.?\d*(?:[eE][+-]?\d+)?)\s+\w+', lp_content)
    tiny_coeffs = []
    for c_str in coefficients:
        try:
            c_val = float(c_str.replace(" ", ""))
            if 0 < abs(c_val) < 1e-4:
                tiny_coeffs.append(c_val)
        except ValueError:
            continue

    if tiny_coeffs:
        print(f"    Found {len(tiny_coeffs)} tiny coefficients (|c| < 1e-4):")
        unique_tiny = sorted(set(f"{c:.10e}" for c in tiny_coeffs))
        for c in unique_tiny[:20]:
            print(f"      {c}")
    else:
        print(f"    No tiny coefficients found.")

    # ── CBC로 풀기 ──
    print(f"\n{'='*60}")
    print(f"  Solving with CBC...")
    print(f"{'='*60}")
    from copy import deepcopy
    prob_cbc = deepcopy(prob)
    solver_cbc = PULP_CBC_CMD(msg=True, timeLimit=120)
    prob_cbc.solve(solver_cbc)
    cbc_status = LpStatus[prob_cbc.status]
    cbc_obj = value(prob_cbc.objective) if prob_cbc.status == 1 else None
    print(f"  CBC result: {cbc_status}, objective={cbc_obj}")

    # ── CPLEX_CMD로 풀기 ──
    available = set(listSolvers(onlyAvailable=True))
    if "CPLEX_CMD" in available:
        print(f"\n{'='*60}")
        print(f"  Solving with CPLEX_CMD...")
        print(f"{'='*60}")
        import tempfile
        log_path = os.path.join(tempfile.gettempdir(), "pulp_cplex_debug.log")
        prob_cplex = deepcopy(prob)
        solver_cplex = SafeCPLEX_CMD(
            msg=True,
            timeLimit=120,
            threads=1,
            logPath=log_path,
        )
        prob_cplex.solve(solver_cplex)
        cplex_status = LpStatus[prob_cplex.status]
        cplex_obj = value(prob_cplex.objective) if prob_cplex.status == 1 else None
        print(f"  CPLEX_CMD result: {cplex_status}, objective={cplex_obj}")

        # CPLEX 로그에서 infeasibility 원인 추출
        if os.path.exists(log_path):
            print(f"\n  CPLEX log (last 30 lines):")
            with open(log_path, "r") as f:
                lines = f.readlines()
            for line in lines[-30:]:
                print(f"    {line.rstrip()}")

        # ── 비교 ──
        print(f"\n{'='*60}")
        print(f"  COMPARISON")
        print(f"{'='*60}")
        print(f"    CBC:       {cbc_status} | obj={cbc_obj}")
        print(f"    CPLEX_CMD: {cplex_status} | obj={cplex_obj}")
        if cbc_status == "Optimal" and cplex_status != "Optimal":
            print(f"    ⚠️  MISMATCH: CBC finds optimal but CPLEX says {cplex_status}")
        elif cbc_status == "Optimal" and cplex_status == "Optimal":
            if cbc_obj is not None and cplex_obj is not None:
                diff = abs(cbc_obj - cplex_obj)
                print(f"    Objective difference: {diff:.6e}")
    else:
        print(f"\n  CPLEX_CMD not available, skipping CPLEX comparison.")
        print(f"  Available solvers: {available}")

    # ── SOS2 없는 버전도 테스트 ──
    print(f"\n{'='*60}")
    print(f"  Testing WITHOUT SOS2 constraints (CBC)...")
    print(f"{'='*60}")

    prob_no_sos2 = LpProblem("DG_ESS_NoSOS2", LpMinimize)

    # 같은 변수/제약 재구성 (SOS2 delta 빼고)
    P_dg2 = {(dg, t): LpVariable(f"P2_{dg}_{t}", lowBound=0) for dg in dg_names for t in range(T)}
    u2 = {(dg, t): LpVariable(f"u2_{dg}_{t}", cat=LpBinary) for dg in dg_names for t in range(T)}
    y2 = {(dg, t): LpVariable(f"y2_{dg}_{t}", cat=LpBinary) for dg in dg_names for t in range(T)}
    z2 = {(dg, t): LpVariable(f"z2_{dg}_{t}", cat=LpBinary) for dg in dg_names for t in range(T)}
    P_c2 = {t: LpVariable(f"P_c2_{t}", lowBound=0, upBound=ess["P_c_max"]) for t in range(T)}
    P_dc2 = {t: LpVariable(f"P_dc2_{t}", lowBound=0, upBound=ess["P_dc_max"]) for t in range(T)}
    SOC2 = {t: LpVariable(f"SOC2_{t}", lowBound=ess["SOC_min"], upBound=ess["SOC_max"]) for t in range(T)}
    u_ess2 = {t: LpVariable(f"u_ess2_{t}", cat=LpBinary) for t in range(T)}
    lam2 = {
        (dg, t, k): LpVariable(f"lam2_{dg}_{t}_{k}", lowBound=0, upBound=1)
        for dg in dg_names for t in range(T) for k in range(n_pwl + 1)
    }
    FC_pwl2 = {(dg, t): LpVariable(f"FC2_{dg}_{t}", lowBound=0) for dg in dg_names for t in range(T)}

    prob_no_sos2 += (
        lpSum(FC_pwl2[dg, t] * DT[t] for dg in dg_names for t in range(T))
        + lpSum(dg_specs[dg]["cost_start"] * y2[dg, t] for dg in dg_names for t in range(T)),
        "Total_Cost"
    )

    for t in range(T):
        prob_no_sos2 += (lpSum(P_dg2[dg, t] for dg in dg_names) + P_dc2[t] - P_c2[t] == P_REQ[t], f"PB_{t}")
        prob_no_sos2 += (P_c2[t] <= ess["P_c_max"] * u_ess2[t], f"EC_{t}")
        prob_no_sos2 += (P_dc2[t] <= ess["P_dc_max"] * (1 - u_ess2[t]), f"ED_{t}")
        inv_eta_dc = 1.0 / ess["eta_dc"]
        coeff = DT[t] / ess["capacity"]
        if t == 0:
            prob_no_sos2 += (SOC2[t] == INITIAL_SOC + (ess["eta_c"] * P_c2[t] - inv_eta_dc * P_dc2[t]) * coeff, f"SOC_{t}")
        else:
            prob_no_sos2 += (SOC2[t] == SOC2[t-1] + (ess["eta_c"] * P_c2[t] - inv_eta_dc * P_dc2[t]) * coeff, f"SOC_{t}")

        for dg in dg_names:
            P_max_dg = dg_specs[dg]["P_max"]
            P_min_dg = P_max_dg * min_load_ratio
            ramp = dg_specs[dg]["ramp_rate"]
            bp = pwl_breakpoints[dg]
            prob_no_sos2 += (P_dg2[dg, t] >= P_min_dg * u2[dg, t], f"Pmin_{dg}_{t}")
            prob_no_sos2 += (P_dg2[dg, t] <= P_max_dg * u2[dg, t], f"Pmax_{dg}_{t}")
            prob_no_sos2 += (P_dg2[dg, t] == lpSum(lam2[dg, t, k] * bp[k][0] for k in range(n_pwl + 1)), f"Pp_{dg}_{t}")
            prob_no_sos2 += (FC_pwl2[dg, t] == lpSum(lam2[dg, t, k] * bp[k][1] for k in range(n_pwl + 1)), f"Pc_{dg}_{t}")
            prob_no_sos2 += (lpSum(lam2[dg, t, k] for k in range(n_pwl + 1)) == u2[dg, t], f"Ps_{dg}_{t}")
            # NO SOS2 delta here
            if t == 0:
                prob_no_sos2 += (y2[dg, t] - z2[dg, t] == u2[dg, t] - initial_u[dg], f"SS_{dg}_{t}")
            else:
                prob_no_sos2 += (y2[dg, t] - z2[dg, t] == u2[dg, t] - u2[dg, t-1], f"SS_{dg}_{t}")
            prob_no_sos2 += (y2[dg, t] + z2[dg, t] <= 1, f"NS_{dg}_{t}")
            if t > 0:
                ramp_limit = ramp * DT[t] * P_max_dg
                prob_no_sos2 += (P_dg2[dg, t] - P_dg2[dg, t-1] <= ramp_limit, f"RU_{dg}_{t}")
                prob_no_sos2 += (P_dg2[dg, t-1] - P_dg2[dg, t] <= ramp_limit, f"RD_{dg}_{t}")

    for dg in dg_names:
        for t in range(T):
            up_steps = find_covering_steps(t, DT, dg_specs[dg]["min_up"])
            if up_steps:
                prob_no_sos2 += (lpSum(u2[dg, tau] for tau in up_steps) >= len(up_steps) * y2[dg, t], f"MU_{dg}_{t}")
            down_steps = find_covering_steps(t, DT, dg_specs[dg]["min_down"])
            if down_steps:
                prob_no_sos2 += (lpSum(1 - u2[dg, tau] for tau in down_steps) >= len(down_steps) * z2[dg, t], f"MD_{dg}_{t}")

    solver_cbc2 = PULP_CBC_CMD(msg=False, timeLimit=120)
    prob_no_sos2.solve(solver_cbc2)
    no_sos2_status = LpStatus[prob_no_sos2.status]
    no_sos2_obj = value(prob_no_sos2.objective) if prob_no_sos2.status == 1 else None
    print(f"  Without SOS2 (CBC): {no_sos2_status}, objective={no_sos2_obj}")

    n_vars_no = len(prob_no_sos2.variables())
    n_binary_no = sum(1 for v in prob_no_sos2.variables() if v.cat == "Binary")
    n_constr_no = len(prob_no_sos2.constraints)
    print(f"  Without SOS2: vars={n_vars_no} (binary={n_binary_no}), constraints={n_constr_no}")
    print(f"  With    SOS2: vars={n_vars}, constraints={n_constraints}")

    if "CPLEX_CMD" in available:
        print(f"\n  Testing WITHOUT SOS2 (CPLEX_CMD)...")
        prob_no_sos2_cplex = deepcopy(prob_no_sos2)
        log_path2 = os.path.join(tempfile.gettempdir(), "pulp_cplex_no_sos2.log")
        solver_cplex2 = SafeCPLEX_CMD(msg=True, timeLimit=120, threads=1, logPath=log_path2)
        prob_no_sos2_cplex.solve(solver_cplex2)
        cplex_no_sos2_status = LpStatus[prob_no_sos2_cplex.status]
        cplex_no_sos2_obj = value(prob_no_sos2_cplex.objective) if prob_no_sos2_cplex.status == 1 else None
        print(f"  Without SOS2 (CPLEX_CMD): {cplex_no_sos2_status}, objective={cplex_no_sos2_obj}")

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Inputs: P_req={P_REQ[0]} MW × {T} steps, dt={DT[0]} h, SOC₀={INITIAL_SOC}")
    print(f"")
    print(f"  {'Model':<25s} {'CBC':<15s} {'CPLEX_CMD':<15s}")
    print(f"  {'-'*55}")
    print(f"  {'With SOS2':<25s} {cbc_status:<15s}", end="")
    if "CPLEX_CMD" in available:
        print(f" {cplex_status:<15s}")
    else:
        print(f" {'N/A':<15s}")
    print(f"  {'Without SOS2':<25s} {no_sos2_status:<15s}", end="")
    if "CPLEX_CMD" in available:
        print(f" {cplex_no_sos2_status:<15s}")
    else:
        print(f" {'N/A':<15s}")

    print(f"\n  Output dir: {OUT_DIR}")


if __name__ == "__main__":
    solve_with_export("cbc")
