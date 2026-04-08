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
    LpStatus, lpSum, value, CPLEX_CMD,
    PulpSolverError,
)


DEFAULT_CPLEX_CMD_PATH = (
    r"C:\Program Files\IBM\ILOG\CPLEX_Studio2212\cplex\bin\x64_win64\cplex.exe"
)


class SafeCPLEX_CMD(CPLEX_CMD):
    """유전 알고리즘 멀티 프로세싱으로 항로 평가 시
    CPLEX_CMD 솔버가 로그 파일에 접근 권한 문제로 실패하는 경우가 있음.
    이 클래스는 CPLEX_CMD 솔버의 로그 파일 제거 시 PermissionError가 발생해도 무시하도록 오버라이드
    """

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
    SFOC 계수식은 출력 kW를 입력으로 사용한다.
    따라서 SFOC(P_kW)[g/kWh] * P_kW / 1000 = kg/h 가 된다.
    """
    P_kW = 1000.0 * P
    sfoc_g_per_kwh = alpha1 * P_kW ** 2 + alpha2 * P_kW + alpha3
    return sfoc_g_per_kwh * P_kW / 1000.0 # kg/h로 연료 소모율

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
    이차 SFOC를 n_segments 구간의 Piecewise Linear로 근사
    MILP 솔버에서 각 구간의 breakpoint (P, FC) 계산에 사용
    Returns
    -------
    list of (P, FC) tuples
        각 breakpoint에서의 출력(MW) 및 연료 소비율(kg/h)
    """
    breakpoints = []

    # P_min에서 P_max까지 n_segments 구간으로 균등 분할하여 breakpoint 계산
    for k in range(n_segments + 1):
        P = P_min + (P_max - P_min) * k / n_segments # MW
        FC = fuel_consumption(P, alpha1, alpha2, alpha3) # kg/h
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
        solver_name: str = "cplex_cmd",
    ):
        """
        Parameters
        ----------
        sfoc_json_path : str
            SFOC 계수 JSON 파일 경로
        n_pwl_segments : int
            PWL 근사 세그먼트 수 (기본 5)
        solver_name : str
            사용할 MILP solver. "cplex", "cbc" 지원.
        """
        # ── DG / ESS 사양 (kcs_specs.py에서 Import) ──
        from src.ship.kcs_specs import DG_SPECS, DG_MIN_LOAD_RATIO, ESS_SPECS

        self.dg_specs = DG_SPECS # DG별 P_max, min_up/down, cost_start
        self.min_load_ratio = DG_MIN_LOAD_RATIO # DG 최소 부하 비율 
        self.dg_names = list(self.dg_specs.keys()) # DG Key

        self.ess = ESS_SPECS
        self.ess_power_limit = float(self.ess["capacity"]) * float(self.ess["c_rate"])

        # ── SFOC 로드 ──
        self.sfoc_data = load_sfoc(sfoc_json_path)
        self.n_pwl = n_pwl_segments
        self.solver_name = solver_name.lower()

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

    # 입력받은 Solver 이름에 따라 pulp 솔버 객체 생성
    def _create_solver(self, time_limit_sec: int, msg: bool):
        """
        Build the fixed PuLP CPLEX_CMD backend.
        """
        solver_name = self.solver_name

        if solver_name not in {"cplex", "cplex_cmd"}:
            raise ValueError(
                f"Unsupported solver_name: {self.solver_name}. "
                "Only 'cplex_cmd' is supported."
            )

        if not os.path.exists(DEFAULT_CPLEX_CMD_PATH):
            raise RuntimeError(
                "CPLEX_CMD executable was not found at the configured path: "
                f"{DEFAULT_CPLEX_CMD_PATH}"
            )

        log_path = os.path.join(
            tempfile.gettempdir(),
            f"pulp_cplex_{os.getpid()}_{id(self)}.log",
        )
        return (
            SafeCPLEX_CMD(
                path=DEFAULT_CPLEX_CMD_PATH,
                msg=msg,
                timeLimit=time_limit_sec,
                threads=1,
                logPath=log_path,
            ),
            "CPLEX_CMD",
        )

    @staticmethod # 인스턴스 정보 안쓰는 독립 함수로 정의
    # Infeasible 상태일 때 반환할 결과 딕셔너리 생성 > Feasible결과와 형식 통일
    def _build_infeasible_result(status: str, solver_backend: str) -> Dict:
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

    @staticmethod
    def _is_expected_cplex_infeasible_error(exc: Exception, solver_backend: str) -> bool:
        if solver_backend != "CPLEX_CMD":
            return False
        message = str(exc).lower()
        return "infeasible" in message and "cplex" in message

    # ─────────────────────────────────────────────────────────────────
    # 메인 풀이 함수
    # ─────────────────────────────────────────────────────────────────

    def solve(
        self,
        P_req: List[float],
        dt: List[float],
        initial_SOC: float = 0.7,
        initial_u: Optional[Dict[str, int]] = None,
        time_limit_sec: int = 120,
        msg: bool = False,
        enable_sos2: bool = False,
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
        enable_sos2 : bool
            SOS2 인접성 제약 활성화 여부. False(기본)이면 PWL
            convex combination만 사용하여 빠르게 풀고, True이면
            δ 이진 변수를 추가하여 엄밀한 PWL 근사를 보장.
            SFOC가 볼록 함수일 때는 False에서도 최적해 동일.

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
        T = len(P_req) # 요구 부하의 수가 전체 시간창의 시간 스텝 수
        assert len(dt) == T, "P_req와 dt의 길이가 같아야 합니다."

        # dict 형태로 전달된 초기 ON/OFF 상태가 없으면 모두 OFF로 간주
        if initial_u is None:
            initial_u = {dg: 0 for dg in self.dg_names}

        # =================================================================
        # 문제 정의
        # LPProblem 객체 생성, 최소화하는 문제로 설정
        # =================================================================
        prob = LpProblem("DG_ESS_Unit_Commitment", LpMinimize)

        # =================================================================
        # 결정 변수 생성
        # =================================================================
        
        # DG 출력 (MW) — 연속
        # 결정 변수의 발전기, 시간 인덱스에서의 출력 전력을 딕셔너리로 정의
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
            t: LpVariable(f"P_c_{t}", lowBound=0, upBound=self.ess_power_limit)
            for t in range(T)
        }

        # ESS 방전 전력 (MW) — 연속
        P_dc = {
            t: LpVariable(f"P_dc_{t}", lowBound=0, upBound=self.ess_power_limit)
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


        # PWL 보조 변수: λ (convex combination weights)
        # λ[dg, t, k] — breakpoint k의 가중치
        lam = {
            (dg, t, k): LpVariable(f"lam_{dg}_{t}_{k}", lowBound=0, upBound=1)
            for dg in self.dg_names
            for t in range(T)
            for k in range(self.n_pwl + 1)
        }

        # SOS2 구간 선택 이진 변수: δ[dg, t, s]
        # s = 0, ..., n_pwl-1 (breakpoint 사이 구간)
        # 인접한 두 breakpoint의 λ만 양수가 되도록 강제
        # enable_sos2=False이면 생성하지 않아 변수/제약 수 절감
        delta = {}
        if enable_sos2:
            delta = {
                (dg, t, s): LpVariable(f"delta_{dg}_{t}_{s}", cat=LpBinary)
                for dg in self.dg_names
                for t in range(T)
                for s in range(self.n_pwl)
            }

        # =================================================================
        # 목적 함수: min Σ_t [ Σ_i FC_PWL_i(t)·Δt + C_start·y_it ]
        # =================================================================

        # PWL 연료 비용 변수 kg/h 단위로 각 DG의 연료 소비율을 나타내는 보조 변수
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
            # Σ P_DG_it + P_dc_t - P_c_t = P_req_t
            prob += (
                lpSum(P_dg[dg, t] for dg in self.dg_names)
                + P_dc[t] - P_c[t] == P_req[t],
                f"PowerBalance_{t}"
            )

            # ─── (1.5) ESS 동시 충방전 방지 ───
            prob += (
                P_c[t] <= self.ess_power_limit * u_ess[t],
                f"ESS_Charge_Mode_{t}"
            )

            prob += (
                P_dc[t] <= self.ess_power_limit * (1 - u_ess[t]),
                f"ESS_Discharge_Mode_{t}"
            )

            # ─── (2) ESS SOC 업데이트 ───
            inv_eta_dc = 1.0 / self.ess["eta_dc"]
            coeff = dt[t] / self.ess["capacity"] # SOC 변화량 계산을 위한 계수 (시간 구간 길이 / ESS 용량)

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
                # P_MIN부터 P_MAX까지 n_segments 구간의 breakpoint로 SFOC와 추진 부하를 곱함
                # 각 breakpoint에서의 (P, FC) 값이 pwl_breakpoints에 저장되어 있음
                # P = Σ_k λ_k · p_k
                # Σ_k λ_k = u_it (ON이면 1, OFF면 0)
                # 연립방정식으로도 풀 수 있음 
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

                # ─── (4b) SOS2 인접성 제약 (선택적) ───
                if enable_sos2:
                    # 최대 2개의 인접 breakpoint λ만 양수 허용
                    # Σ_s δ_s = u_it (ON이면 정확히 1개 구간 선택)
                    prob += (
                        lpSum(
                            delta[dg, t, s]
                            for s in range(self.n_pwl)
                        ) == u[dg, t],
                        f"SOS2_seg_sum_{dg}_{t}"
                    )

                    # λ_0 ≤ δ_0 (첫 breakpoint는 첫 구간에만 속함)
                    prob += (
                        lam[dg, t, 0] <= delta[dg, t, 0],
                        f"SOS2_lam_first_{dg}_{t}"
                    )

                    # λ_k ≤ δ_{k-1} + δ_k (중간 breakpoint는 인접 2개 구간)
                    for k in range(1, self.n_pwl):
                        prob += (
                            lam[dg, t, k] <= delta[dg, t, k - 1] + delta[dg, t, k],
                            f"SOS2_lam_mid_{dg}_{t}_{k}"
                        )

                    # λ_{n_pwl} ≤ δ_{n_pwl-1} (마지막 breakpoint는 마지막 구간에만)
                    prob += (
                        lam[dg, t, self.n_pwl] <= delta[dg, t, self.n_pwl - 1],
                        f"SOS2_lam_last_{dg}_{t}"
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

        # ─── (7) 최소 기동/정지 시간 ───
        for dg in self.dg_names:
            min_up_h = self.dg_specs[dg]["min_up"]
            min_down_h = self.dg_specs[dg]["min_down"]

            for t in range(T):
                # 최소 기동 시간: 기동 후 min_up_h 동안 ON 유지
                # find_covering_steps 함수로 t부터 시작해서 min_up_h를 커버하는 시간 스텝 리스트 반환
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
        try:
            prob.solve(solver)
        except PulpSolverError as exc:
            if self._is_expected_cplex_infeasible_error(exc, solver_backend):
                return self._build_infeasible_result("Infeasible", solver_backend)
            raise

        # =================================================================
        # 결과 추출
        # =================================================================
        status = LpStatus[prob.status]
        feasible = prob.status == 1  # Optimal

        if not feasible:
            return self._build_infeasible_result(status, solver_backend)

        # 상세 스케줄 추출
        schedule = []
        total_fuel = 0.0
        total_start_cost = 0.0

        for t in range(T):
            step = {
                "t": t,
                "dt_h": dt[t],
                "P_req_MW": P_req[t],
                "P_dc_MW": value(P_dc[t]),
                "P_c_MW": value(P_c[t]),
                "SOC": value(SOC[t]),
            }

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
            "objective_value": value(prob.objective),
            "solver_backend": solver_backend,
            "schedule": schedule,
            "summary": self._build_summary(
                schedule,
                total_fuel,
                total_start_cost,
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
            "dg_running_hours": dg_hours,
            "dg_fuel_kg": dg_fuel,
            "dg_start_count": dg_starts,
            "ess_final_SOC": schedule[-1]["SOC"] if schedule else None,
        }


# =============================================================================
