"""
MILP 발전기 + ESS 스케줄링 솔버 (PuLP + CBC)
=============================================
3기 DG (Unit Commitment) + 1기 ESS (충방전) 통합 최적화.

수학적 구조:
    - 목적함수: min Σ_t [ Σ_i FC_i(P_it)·Δt + C_start_i·y_it ]
    - SFOC_i(P) = α₁P² + α₂P + α₃ (g/kWh)
    - FC_i(P) = SFOC_i(P) · P(MW) (kg/h) → PWL 근사
    - 모든 P_req는 GA에서 상수로 전달 → MILP 내부 완전 선형

의존성: pip install pulp
"""

import json
import os
import tempfile
from typing import List, Dict, Optional, Tuple

from pulp import (
    LpProblem, LpMinimize, LpVariable, LpBinary,
    LpStatus, lpSum, value, PULP_CBC_CMD, CPLEX_CMD, listSolvers
)


class SafeCPLEX_CMD(CPLEX_CMD):
    """Ignore shared log cleanup races when multiple CPLEX_CMD solvers run."""

    def silent_remove(self, file):
        try:
            super().silent_remove(file)
        except PermissionError:
            pass


# SFOC 데이터 로더

def load_sfoc(json_path: str) -> Dict:
    """
    SFOC JSON 파일 로드. SFOC(P) = α₁P² + α₂P + α₃ (g/kWh)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# SFOC 함수 PWL 근사
def fuel_consumption(P: float, alpha1: float, alpha2: float, alpha3: float) -> float:
    """
    연료 소비율 계산.

    Parameters
    ----------
    P : float
        DG 출력 (MW)

    Returns
    -------
    float
        연료 소비율 (kg/h)

    Notes
    -----
    SFOC 곡선 계수는 g/kWh 기준이다.
    따라서 SFOC(P)[g/kWh] * P[MW] = kg/h 가 된다.
    """
    sfoc_g_per_kwh = alpha1 * P ** 2 + alpha2 * P + alpha3
    return sfoc_g_per_kwh * P
# PWL (Piecewise Linear) 근사
# =============================================================================

def generate_pwl_breakpoints(
    P_min: float,
    P_max: float,
    alpha1: float,
    alpha2: float,
    alpha3: float,
    n_segments: int = 5,
) -> List[Tuple[float, float]]:
    """
    이차 SFOC를 n_segments 구간의 Piecewise Linear로 근사.

    Returns
    -------
    list of (P, FC) tuples
        각 breakpoint에서의 출력(MW) 및 연료 소비율(kg/h)
    """
    breakpoints = []
    for k in range(n_segments + 1):
        P = P_min + (P_max - P_min) * k / n_segments
        FC = fuel_consumption(P, alpha1, alpha2, alpha3)
        breakpoints.append((P, FC))
    return breakpoints


# =============================================================================
# MILP 솔버 클래스
# =============================================================================

class MILPSolver:
    """
    3기 DG + 1기 ESS 최적 스케줄링.

    GA에서 계산된 P_req(상수 벡터)를 받아,
    총 연료 소모량을 최소화하는 발전기 부하 분배 및
    ESS 충방전 스케줄을 결정한다.
    """

    def __init__(
        self,
        sfoc_json_path: str = "config/sfoc.json",
        n_pwl_segments: int = 5,
        solver_name: str = "cplex",
        dump_penalty_per_mwh: float = 10000.0,
    ):
        """
        Parameters
        ----------
        sfoc_json_path : str
            SFOC 계수 JSON 파일 경로
        n_pwl_segments : int
            PWL 근사 세그먼트 수 (기본 5)
        solver_name : str
            사용할 MILP solver. "cplex", "cbc", "auto" 지원.
        """
        # ── DG / ESS 사양 (kcs_specs.py에서 Import) ──
        from src.ship.kcs_specs import DG_SPECS, DG_MIN_LOAD_RATIO, ESS_SPECS

        self.dg_specs = DG_SPECS
        self.min_load_ratio = DG_MIN_LOAD_RATIO
        self.dg_names = list(self.dg_specs.keys())

        self.ess = ESS_SPECS

        # ── SFOC 로드 ──
        self.sfoc_data = load_sfoc(sfoc_json_path)
        self.n_pwl = n_pwl_segments
        self.solver_name = solver_name.lower()
        self.dump_penalty_per_mwh = dump_penalty_per_mwh
        self.available_solvers = set(listSolvers(onlyAvailable=True))

        # ── PWL Breakpoints 사전 계산 ──
        self.pwl_breakpoints = {}
        for dg in self.dg_names:
            P_min = self.dg_specs[dg]["P_max"] * self.min_load_ratio
            P_max = self.dg_specs[dg]["P_max"]
            sfoc = self.sfoc_data[dg]
            self.pwl_breakpoints[dg] = generate_pwl_breakpoints(
                P_min, P_max,
                sfoc["alpha1"], sfoc["alpha2"], sfoc["alpha3"],
                self.n_pwl,
            )

    def _create_solver(self, time_limit_sec: int, msg: bool):
        """
        Build the requested PuLP backend, preferring CPLEX when available.
        """
        solver_name = self.solver_name

        if solver_name == "auto":
            solver_name = "cplex" if "CPLEX_CMD" in self.available_solvers else "cbc"

        if solver_name == "cplex":
            if "CPLEX_CMD" in self.available_solvers:
                log_path = os.path.join(
                    tempfile.gettempdir(),
                    f"pulp_cplex_{os.getpid()}_{id(self)}.log",
                )
                return (
                    SafeCPLEX_CMD(
                        msg=msg,
                        timeLimit=time_limit_sec,
                        threads=1,
                        logPath=log_path,
                    ),
                    "CPLEX_CMD",
                )
            return PULP_CBC_CMD(msg=msg, timeLimit=time_limit_sec), "PULP_CBC_CMD"

        if solver_name == "cbc":
            return PULP_CBC_CMD(msg=msg, timeLimit=time_limit_sec), "PULP_CBC_CMD"

        raise ValueError(
            f"Unsupported solver_name: {self.solver_name}. "
            "Use one of: 'cplex', 'cbc', 'auto'."
        )

    # ─────────────────────────────────────────────────────────────────
    # 메인 풀이 함수
    # ─────────────────────────────────────────────────────────────────

    def solve(
        self,
        P_req: List[float],
        dt: List[float],
        initial_SOC: float = 0.8,
        initial_u: Optional[Dict[str, int]] = None,
        time_limit_sec: int = 120,
        msg: bool = False,
    ) -> Dict:
        """
        MILP 발전기 + ESS 스케줄링 문제 풀이.

        Parameters
        ----------
        P_req : list[float]
            각 시간 구간의 요구 부하 (MW). GA가 상수로 전달.
        dt : list[float]
            각 시간 구간의 길이 (h).
        initial_SOC : float
            ESS 초기 SOC (0~1), default 0.8
        initial_u : dict, optional
            각 DG의 초기 ON/OFF 상태. None이면 모두 OFF.
        time_limit_sec : int
            CBC 솔버 시간 제한 (초)
        msg : bool
            솔버 로그 출력 여부

        Returns
        -------
        dict
            {
                "feasible": bool,
                "status": str,
                "total_fuel_kg": float,
                "total_start_cost": float,
                "objective_value": float,
                "schedule": list[dict],  # 시간별 상세 스케줄
                "summary": dict,
            }
        """
        T = len(P_req)
        assert len(dt) == T, "P_req와 dt의 길이가 같아야 합니다."

        if initial_u is None:
            initial_u = {dg: 0 for dg in self.dg_names}

        # =================================================================
        # 문제 정의
        # =================================================================
        prob = LpProblem("DG_ESS_Unit_Commitment", LpMinimize)

        # =================================================================
        # 결정 변수 생성
        # =================================================================

        # DG 출력 (MW) — 연속
        P_dg = {
            (dg, t): LpVariable(f"P_{dg}_{t}", lowBound=0)
            for dg in self.dg_names for t in range(T)
        }

        # DG ON/OFF — 이진
        u = {
            (dg, t): LpVariable(f"u_{dg}_{t}", cat=LpBinary)
            for dg in self.dg_names for t in range(T)
        }

        # DG 기동 이벤트 — 이진
        y = {
            (dg, t): LpVariable(f"y_{dg}_{t}", cat=LpBinary)
            for dg in self.dg_names for t in range(T)
        }

        # DG 정지 이벤트 — 이진
        z = {
            (dg, t): LpVariable(f"z_{dg}_{t}", cat=LpBinary)
            for dg in self.dg_names for t in range(T)
        }

        # ESS 충전 전력 (MW) — 연속
        P_c = {
            t: LpVariable(f"P_c_{t}", lowBound=0, upBound=self.ess["P_c_max"])
            for t in range(T)
        }

        # ESS 방전 전력 (MW) — 연속
        P_dc = {
            t: LpVariable(f"P_dc_{t}", lowBound=0, upBound=self.ess["P_dc_max"])
            for t in range(T)
        }

        # ESS SOC — 연속
        SOC = {
            t: LpVariable(
                f"SOC_{t}",
                lowBound=self.ess["SOC_min"],
                upBound=self.ess["SOC_max"],
            )
            for t in range(T)
        }

        # ESS 충방전 모드 — 이진 (1: 충전, 0: 방전)
        u_ess = {
            t: LpVariable(f"u_ess_{t}", cat=LpBinary)
            for t in range(T)
        }

        # Dump Load (잉여 전력 소모 저항기) (MW) — 연속
        P_dump = {
            t: LpVariable(f"P_dump_{t}", lowBound=0)
            for t in range(T)
        }

        # PWL 보조 변수: λ (convex combination weights)
        # λ[dg, t, k] — breakpoint k의 가중치
        lam = {
            (dg, t, k): LpVariable(f"lam_{dg}_{t}_{k}", lowBound=0, upBound=1)
            for dg in self.dg_names
            for t in range(T)
            for k in range(self.n_pwl + 1)
        }

        # =================================================================
        # 목적 함수: min Σ_t [ Σ_i FC_PWL_i(t)·Δt + C_start·y_it ]
        # =================================================================

        # PWL 연료 비용 변수
        FC_pwl = {
            (dg, t): LpVariable(f"FC_{dg}_{t}", lowBound=0)
            for dg in self.dg_names for t in range(T)
        }

        # 목적함수
        prob += (
            lpSum(
                FC_pwl[dg, t] * dt[t]
                for dg in self.dg_names for t in range(T)
            )
            + lpSum(
                self.dump_penalty_per_mwh * P_dump[t] * dt[t]
                for t in range(T)
            )
            + lpSum(
                self.dg_specs[dg]["cost_start"] * y[dg, t]
                for dg in self.dg_names for t in range(T)
            ),
            "Total_Cost"
        )

        # =================================================================
        # 제약 조건
        # =================================================================

        for t in range(T):

            # ─── (1) 전력 수급 균형 ───
            # Σ P_DG_it + P_dc_t - P_c_t - P_dump_t = P_req_t
            prob += (
                lpSum(P_dg[dg, t] for dg in self.dg_names)
                + P_dc[t] - P_c[t] - P_dump[t] == P_req[t],
                f"PowerBalance_{t}"
            )

            # ─── (1.5) ESS 동시 충방전 방지 ───
            prob += (
                P_c[t] <= self.ess["P_c_max"] * u_ess[t],
                f"ESS_Charge_Mode_{t}"
            )
            prob += (
                P_dc[t] <= self.ess["P_dc_max"] * (1 - u_ess[t]),
                f"ESS_Discharge_Mode_{t}"
            )

            # ─── (2) ESS SOC 업데이트 ───
            inv_eta_dc = 1.0 / self.ess["eta_dc"]
            coeff = dt[t] / self.ess["capacity"]

            if t == 0:
                prob += (
                    SOC[t] == initial_SOC
                    + (self.ess["eta_c"] * P_c[t]
                       - inv_eta_dc * P_dc[t])
                    * coeff,
                    f"SOC_update_{t}"
                )
            else:
                prob += (
                    SOC[t] == SOC[t - 1]
                    + (self.ess["eta_c"] * P_c[t]
                       - inv_eta_dc * P_dc[t])
                    * coeff,
                    f"SOC_update_{t}"
                )

            for dg in self.dg_names:
                P_max_dg = self.dg_specs[dg]["P_max"]
                P_min_dg = P_max_dg * self.min_load_ratio
                ramp = self.dg_specs[dg]["ramp_rate"]

                # ─── (3) DG 출력 상·하한 ───
                prob += (
                    P_dg[dg, t] >= P_min_dg * u[dg, t],
                    f"DG_Pmin_{dg}_{t}"
                )
                prob += (
                    P_dg[dg, t] <= P_max_dg * u[dg, t],
                    f"DG_Pmax_{dg}_{t}"
                )

                # ─── (4) PWL 제약 (Convex Combination) ───
                bp = self.pwl_breakpoints[dg]

                # P_DG = Σ_k λ_k · p_k
                prob += (
                    P_dg[dg, t] == lpSum(
                        lam[dg, t, k] * bp[k][0]
                        for k in range(self.n_pwl + 1)
                    ),
                    f"PWL_power_{dg}_{t}"
                )

                # FC_PWL = Σ_k λ_k · fc_k
                prob += (
                    FC_pwl[dg, t] == lpSum(
                        lam[dg, t, k] * bp[k][1]
                        for k in range(self.n_pwl + 1)
                    ),
                    f"PWL_cost_{dg}_{t}"
                )

                # Σ_k λ_k = u_it (ON이면 1, OFF면 0)
                prob += (
                    lpSum(
                        lam[dg, t, k]
                        for k in range(self.n_pwl + 1)
                    ) == u[dg, t],
                    f"PWL_sum_{dg}_{t}"
                )

                # ─── (5) 기동/정지 연결 ───
                # y_it - z_it = u_it - u_i,t-1
                if t == 0:
                    prob += (
                        y[dg, t] - z[dg, t] == u[dg, t] - initial_u[dg],
                        f"StartStop_{dg}_{t}"
                    )
                else:
                    prob += (
                        y[dg, t] - z[dg, t] == u[dg, t] - u[dg, t - 1],
                        f"StartStop_{dg}_{t}"
                    )

                # y + z <= 1 (동시에 기동+정지 불가)
                prob += (
                    y[dg, t] + z[dg, t] <= 1,
                    f"NoSimultaneous_{dg}_{t}"
                )

                # ─── (6) 증감발률 (Ramp Rate) ───
                # |P_it - P_i,t-1| ≤ ramp × Δt × P_max
                if t > 0:
                    ramp_limit = ramp * dt[t] * P_max_dg
                    prob += (
                        P_dg[dg, t] - P_dg[dg, t - 1] <= ramp_limit,
                        f"RampUp_{dg}_{t}"
                    )
                    prob += (
                        P_dg[dg, t - 1] - P_dg[dg, t] <= ramp_limit,
                        f"RampDown_{dg}_{t}"
                    )

        # ─── (7) 최소 기동/정지 시간 ───
        for dg in self.dg_names:
            min_up_h = self.dg_specs[dg]["min_up"]
            min_down_h = self.dg_specs[dg]["min_down"]

            for t in range(T):
                # 최소 기동 시간: 기동 후 min_up_h 동안 ON 유지
                up_steps = self._find_covering_steps(t, dt, min_up_h)
                if up_steps:
                    prob += (
                        lpSum(u[dg, tau] for tau in up_steps)
                        >= len(up_steps) * y[dg, t],
                        f"MinUp_{dg}_{t}"
                    )

                # 최소 정지 시간: 정지 후 min_down_h 동안 OFF 유지
                down_steps = self._find_covering_steps(t, dt, min_down_h)
                if down_steps:
                    prob += (
                        lpSum(1 - u[dg, tau] for tau in down_steps)
                        >= len(down_steps) * z[dg, t],
                        f"MinDown_{dg}_{t}"
                    )

        # =================================================================
        # 풀이
        # =================================================================
        solver, solver_backend = self._create_solver(
            time_limit_sec=time_limit_sec,
            msg=msg,
        )
        prob.solve(solver)

        # =================================================================
        # 결과 추출
        # =================================================================
        status = LpStatus[prob.status]
        feasible = prob.status == 1  # Optimal

        if not feasible:
            return {
                "feasible": False,
                "status": status,
                "total_fuel_kg": float("inf"),
                "total_start_cost": float("inf"),
                "objective_value": float("inf"),
                "solver_backend": solver_backend,
                "schedule": [],
                "summary": {},
            }

        # 상세 스케줄 추출
        schedule = []
        total_fuel = 0.0
        total_start_cost = 0.0
        total_dump_energy = 0.0
        total_dump_penalty = 0.0

        for t in range(T):
            step = {
                "t": t,
                "dt_h": dt[t],
                "P_req_MW": P_req[t],
                "P_dc_MW": value(P_dc[t]),
                "P_c_MW": value(P_c[t]),
                "P_dump_MW": value(P_dump[t]),
                "SOC": value(SOC[t]),
            }
            total_dump_energy += step["P_dump_MW"] * dt[t]
            total_dump_penalty += self.dump_penalty_per_mwh * step["P_dump_MW"] * dt[t]

            for dg in self.dg_names:
                p_val = value(P_dg[dg, t])
                u_val = value(u[dg, t])
                y_val = value(y[dg, t])
                fc_val = value(FC_pwl[dg, t])

                step[f"{dg}_P_MW"] = p_val
                step[f"{dg}_ON"] = int(round(u_val))
                step[f"{dg}_Start"] = int(round(y_val))
                step[f"{dg}_FC_kgh"] = fc_val

                # 연료 적산
                total_fuel += fc_val * dt[t]
                if int(round(y_val)) == 1:
                    total_start_cost += self.dg_specs[dg]["cost_start"]

            schedule.append(step)

        return {
            "feasible": True,
            "status": status,
            "total_fuel_kg": total_fuel,
            "total_start_cost": total_start_cost,
            "total_dump_energy_mwh": total_dump_energy,
            "total_dump_penalty": total_dump_penalty,
            "objective_value": value(prob.objective),
            "solver_backend": solver_backend,
            "schedule": schedule,
            "summary": self._build_summary(
                schedule,
                total_fuel,
                total_start_cost,
                total_dump_energy,
                total_dump_penalty,
            ),
        }

    # ─────────────────────────────────────────────────────────────────
    # 헬퍼 함수
    # ─────────────────────────────────────────────────────────────────

    def _find_covering_steps(
        self,
        start_t: int,
        dt_list: List[float],
        required_hours: float,
    ) -> List[int]:
        """
        시점 start_t부터 required_hours를 커버하는 시간 스텝 인덱스 리스트 반환.
        범위를 초과하면 가능한 범위까지만 반환.
        """
        T = len(dt_list)
        steps = []
        accumulated = 0.0

        for tau in range(start_t, T):
            steps.append(tau)
            accumulated += dt_list[tau]
            if accumulated >= required_hours:
                break

        return steps

    def _build_summary(
        self,
        schedule: List[dict],
        total_fuel: float,
        total_start_cost: float,
        total_dump_energy: float,
        total_dump_penalty: float,
    ) -> dict:
        """결과 요약 생성."""
        T = len(schedule)
        dg_hours = {dg: 0.0 for dg in self.dg_names}
        dg_fuel = {dg: 0.0 for dg in self.dg_names}
        dg_starts = {dg: 0 for dg in self.dg_names}

        total_hours = sum(s["dt_h"] for s in schedule)

        for s in schedule:
            for dg in self.dg_names:
                if s[f"{dg}_ON"]:
                    dg_hours[dg] += s["dt_h"]
                    dg_fuel[dg] += s[f"{dg}_FC_kgh"] * s["dt_h"]
                dg_starts[dg] += s[f"{dg}_Start"]

        return {
            "total_voyage_h": total_hours,
            "total_fuel_kg": total_fuel,
            "total_start_cost_usd": total_start_cost,
            "total_dump_energy_mwh": total_dump_energy,
            "total_dump_penalty": total_dump_penalty,
            "dg_running_hours": dg_hours,
            "dg_fuel_kg": dg_fuel,
            "dg_start_count": dg_starts,
            "ess_final_SOC": schedule[-1]["SOC"] if schedule else None,
        }


# =============================================================================
