import pandas as pd
import numpy as np
from pulp import LpMinimize, LpProblem, LpStatus, LpVariable, lpSum, value, LpBinary, LpContinuous
import math
from spec import(DG_SPECS, DT, ESS_PARAMS, PROP_SPECS, RESERVE_PROPULSION, RESERVE_SERVICE, SERVICE_LOAD, T_TOTAL, V_MAX)

def piecewise_linear(x, breakpoints, values):
    if x <= breakpoints[0]:
        return values[0]
    if x >= breakpoints[-1]:
        return values[-1]

    for i in range(1, len(breakpoints)):
        if x < breakpoints[i]:
            y1 = values[i-1]
            y2 = values[i]
            value = y1 + (y2 - y1) * (x - breakpoints[i-1]) / (breakpoints[i] - breakpoints[i-1])
            return value

    return values[-1]


def breakpoints_and_values(V_min, V_max, num_points=5):
    breakpoints = np.linspace(V_min, V_max, num_points)
    values = [
        PROP_SPECS["C1"] * np.power(V, PROP_SPECS["C2"]) + PROP_SPECS["C3"] * V
        for V in breakpoints
    ]
    return breakpoints, values

class MILPSolver:
    def __init__(self, n_pwl_segments=5, solver_name="CBC", profile="profile.csv"):
        self.dg_specs = DG_SPECS
        self.dg_names = list(DG_SPECS.keys())
        self.ess_params = ESS_PARAMS
        self.service_load = SERVICE_LOAD
        self.reserve_propulsion = RESERVE_PROPULSION
        self.reserve_service = RESERVE_SERVICE
        self.n_pwl_segments = n_pwl_segments
        self.solver_name = solver_name
        self.dt = DT
        self.t_total = T_TOTAL
        self.ess_dch_max = self.ess_params['P_max'] * self.ess_params['eta_dch'] # ESS 최대 방전 전력 (MW)
        self.ess_ch_max = self.ess_params['P_max'] / self.ess_params['eta_ch'] # ESS 최대 충전 전력 (MW)
        self.ess_rating = self.ess_params['E_rat'] # ESS 정격 용량 (MWh)
        self.SOC_MAX = self.ess_params['SOC_max']
        self.SOC_MIN = self.ess_params['SOC_min']
        self.initial_SOC = self.ess_params['SOC_0']
        self.eta_ch = self.ess_params['eta_ch']
        self.eta_dch = self.ess_params['eta_dch']
        self.eta_rt = self.ess_params['eta_rt']
        self.profile = pd.read_csv(profile)
        self.V_MAX = self.profile['Max Speed']
        self.V_MIN = self.profile['Min Speed']
        breakpoints, values = breakpoints_and_values(0, 30, self.n_pwl_segments)
        self.propulsion_breakpoints = {
            "breakpoints": breakpoints.tolist(),
            "values": values,
        }

        self.service_load = SERVICE_LOAD

    # ---------------------------------------------------------
    # 연료비 선형화 계수 추출 메서드 
    # ---------------------------------------------------------
    def _get_fuel_pwl_lines(self, dg_name, num_segments=4):
        specs = self.dg_specs[dg_name]
        a1, a2, a3 = specs['alpha1'], specs['alpha2'], specs['alpha3']
        P_max = specs['P_max']
        
        breakpoints = np.linspace(0, P_max, num_segments + 1)
        lines = []
        
        for i in range(num_segments):
            x1, x2 = breakpoints[i], breakpoints[i+1]
            y1 = a1*(x1/P_max)**2 + a2*(x1/P_max) + a3
            y2 = a1*(x2/P_max)**2 + a2*(x2/P_max) + a3
            
            slope = (y2 - y1) / (x2 - x1)  
            intercept = y1 - slope * x1    
            lines.append({'slope': slope, 'intercept': intercept})
            
        return lines
    
    #---------------------------------------------------------
    # 연료비
    #---------------------------------------------------------
    def _add_fuel_cost_constraints(self, prob, P_dg, Fuel_cost, u, T):
        """Fuel_cost 변수를 실제 P_dg 발전량과 묶어주는 제약조건"""
        for dg in self.dg_names:
            pwl_lines = self._get_fuel_pwl_lines(dg, num_segments=4)
            for t in range(T):
                for line in pwl_lines:
                    # 발전기가 켜져있을 때(u=1)만 직선 절편이 활성화됨
                    prob += Fuel_cost[dg, t] >= line['slope'] * P_dg[dg, t] + line['intercept'] * u[dg, t], f"PWL_Fuel_{dg}_{t}_{line['slope']:.2f}"


    def solve(self, P_propulsion, P_service, dt, initial_SOC, initial_DG_Status):
        # MILP 모델 구축 및 최적화 로직 구현

        T = T
        if initial_DG_Status is None:
            initial_DG_Status = {dg: 0 for dg in self.dg_names}
        
        # 문제 정의
        prob = LpProblem("Optimal Scheduling", LpMinimize)

        # 결정 변수

        # -----------------------------------------------
        # 이진 변수
        # -----------------------------------------------
        # DG ON/OFF 상태 변수
        u = {(dg, t): LpVariable(f"u_{dg}_{t}", cat=LpBinary)
            for dg in self.dg_names for t in range(T)}
        
        # DG 기동 변수(Start-up, event 변수)
        y = {(dg,t) : LpVariable(f"y_{dg}_{t}", cat=LpBinary)
            for dg in self.dg_names for t in range(T)}

        u_ch = {t: LpVariable(f"u_ch_{t}", cat=LpBinary) for t in range(T)} # ESS 충전 상태 변수

        u_dch = {t: LpVariable(f"u_dch_{t}", cat=LpBinary) for t in range(T)} # ESS 방전 상태 변수

        # -----------------------------------------------
        # 연속 변수
        # -----------------------------------------------
        # DG 출력 변수
        P_dg = {(dg, t): LpVariable(f"P_{dg}_{t}", lowBound=0)
                for dg in self.dg_names for t in range(T)}
        
        # ESS 방전량 변수
        P_dc = {t: LpVariable(f"P_dch_{t}", lowBound=0, upBound=self.ess_dch_max, cat=LpContinuous) for t in range(T)} 

        # ESS 충전량 변수
        P_c = {t: LpVariable(f"P_ch_{t}", lowBound=0, upBound=self.ess_ch_max, cat=LpContinuous) for t in range(T)}

        # SOC 변수
        SOC = {t: LpVariable(f"SOC_{t}", lowBound=self.ess_params['SOC_min'], upBound=self.ess_params['SOC_max'], cat=LpContinuous) for t in range(T)} 

        P_dg_avail = {(dg, t): LpVariable(f"P_avail_{dg}_{t}", lowBound=0) for dg in self.dg_names for t in range(T)}

        # 속도 변수
        Vel = {t: LpVariable(f"Vel_{t}", lowBound=self.V_MIN[t], upBound=self.V_MAX[t], cat=LpContinuous) for t in range(T)}

        # 추진 부하 변수
        P_prop = {t: LpVariable(f"P_prop_{t}", lowBound=0, cat=LpContinuous) for t in range(T)}
        
        # 람다 변수 (P_propulsion과 Vel의 곱을 선형화하기 위한 변수, SO2 제약식에서 사용)
        K = len(self.propulsion_breakpoints['breakpoints'])
        Lambda = {(t, k): LpVariable(f"Lambda_{t}_{k}", lowBound=0, upBound=1, cat=LpContinuous) for t in range(T) for k in range(K)}

        # 연료 비용 변수
        Fuel_cost = {(dg,t): LpVariable(f"Fuel_cost_{dg}_{t}", lowBound=0, cat=LpContinuous) for dg in self.dg_names for t in range(T)}
        # -----------------------------------------------
        # 목적 함수
        # -----------------------------------------------
        # 마모 비용 미리계산
        C_ESS_Deg = self.ess_params['C_rep'] / (self.ess_params['L_cycle'] * np.sqrt(self.ess_params['eta_rt']))  

        # 연료 비용 + DG 기동 비용 + ESS 사이클 비용 => 연료비용 수정 필요
        prob += (
            lpSum(Fuel_cost[dg, t] * self.dt for dg in self.dg_names for t in range(T))
            + lpSum(y[dg, t] * self.dg_specs[dg]['C_SU'] for dg in self.dg_names for t in range(T))
            + lpSum(C_ESS_Deg * P_dc[t]/self.eta_dch * self.dt for t in range(T))
        )

        # -----------------------------------------------
        # 제약 조건
        # -----------------------------------------------

        # 발전기 출력과 연료비를 연결하는 제약식 추가
        self._add_fuel_cost_constraints(prob, P_dg, Fuel_cost, u, T)

        # 전력 수급 균형
        for t in range(T):
            prob += (lpSum(P_dg[dg, t] for dg in self.dg_names) + P_dc[t] - P_c[t] == P_propulsion[t] + P_service[t])

        # DG ON/OFF 상태와 기동 변수 연결
        for dg in self.dg_names:
            for t in range(T):
                if t == 0:
                    prob += (y[dg, t] >= u[dg, t] - initial_DG_Status[dg])
                else:
                    prob += (y[dg, t] >= u[dg, t] - u[dg, t-1])
        
        # 최소/최대 기동 제약
        for dg in self.dg_names:
            min_up = math.ceil(self.dg_specs[dg]['T_ON'] / self.dt)
            min_down = math.ceil(self.dg_specs[dg]['T_OFF'] / self.dt)
            for t in range(T):
                # 시점이 min_up보다 작을 때는 초기 상태를 고려하여 제약식 설정, 그렇지 않으면 일반적인 최소 ON 시간 제약식 적용
                # t시점에 발전기가 켜져있다면, 과거 min_up 시점까지 발전기가 켜져있어야 함
                if t < min_up:
                    prob += (lpSum(y[dg, tau] for tau in range(t+1)) <= u[dg, t])
                else:
                    prob += (lpSum(y[dg, tau] for tau in range(t-min_up+1, t+1)) <= u[dg, t])
                # t시점에 발전기가 꺼져있다면, 과거 min_down 시점까지 발전기가 꺼져있어야 함
                if t < min_down:
                    prob += (lpSum(y[dg, tau] for tau in range(t+1)) <= 1 - initial_DG_Status[dg])
                else:
                    prob += (lpSum(y[dg, tau] for tau in range(t-min_down+1, t+1)) <= 1 - u[dg, t-min_down])

        # 발전 출력 및 증감률 제약
        for dg in self.dg_names:
            P_min = self.dg_specs[dg]['P_min']
            P_max = self.dg_specs[dg]['P_max']
            for t in range(T):
                # 발전 출력 범위 제약
                prob += P_dg[dg, t] >= P_min * u[dg, t]
                prob += P_dg[dg, t] <= P_dg_avail[dg, t]  # 가용 출력 제약
                prob += P_dg_avail[dg, t] <= P_max * u[dg, t]

                Ramp_up_MW = P_max * self.dg_specs[dg]['Ramp_up'] 
                Ramp_down_MW = P_max * self.dg_specs[dg]['Ramp_down']
                Ramp_start_MW = P_min

                if t > 0:
                    # Ramp Up 제약 
                    prob += P_dg_avail[dg, t] - P_dg[dg, t-1] <= Ramp_up_MW * u[dg, t-1] + Ramp_start_MW * y[dg, t]
                    # Ramp Down 제약
                    prob += P_dg[dg, t-1] - P_dg[dg, t] <= Ramp_down_MW * u[dg, t-1]

        for t in range(T):
            # ESS 충방전 변수 제약
            prob += u_ch[t] + u_dch[t] == 1
            # ESS 충전 출력 제약
            # ESS 충전 시, 배터리의 충전 효율(eta_ch)과 왕복 효율(eta_rt)을 고려하여 충전량은 최대 충전 전력보다 크게 충전됨
            prob += P_c[t] <= self.ess_ch_max * u_ch[t]/(self.eta_ch * np.sqrt(self.eta_rt))
            # ESS 방전 출력 제약
            # ESS 방전 시, 배터리의 왕복 효율, 방전 효율을 고려하여 방전량은 최대량보다 작음
            prob += P_dc[t] <= self.ess_dch_max * u_dch[t] * np.sqrt(self.eta_rt) * self.eta_dch

            # 식 21의 SOC 업데이트 제약식(식에서는 ESS_Level로 표현, 코드에서는 SOC로 표현)
            if t == 0:
                prob += SOC[t] == initial_SOC + (P_c[t] * np.sqrt(self.eta_rt) * self.eta_ch - P_dc[t] / (self.eta_dch * np.sqrt(self.eta_rt))) * self.dt / self.ess_rating
            else:
                prob += SOC[t] == SOC[t-1] + (P_c[t] * np.sqrt(self.eta_rt) * self.eta_ch - P_dc[t] / (self.eta_dch * np.sqrt(self.eta_rt))) * self.dt / self.ess_rating

            # SOC 범위 제약
            prob += SOC[t] >= self.SOC_MIN
            prob += SOC[t] <= self.SOC_MAX

        # SOC 초기값 == 마지막 SOC제약 준수
        prob += SOC[T-1] == self.initial_SOC 

        # 속도, 추진 부하 SO2 제약
        for t in range(T):
            # 해당 t 시점의 람다 변수의 총합은 1
            prob += lpSum(Lambda[t, k] for k in range(K)) == 1, f"Lambda_sum_{t}"

            # 현재 속도는 람다 변수에 따른 가중 평균
            prob += Vel[t] == lpSum(Lambda[t, k] * self.propulsion_breakpoints['breakpoints'][k] for k in range(K))

            # 현재 추진 부하는 람다 변수에 따른 가중 평균
            prob += P_prop[t] == lpSum(Lambda[t, k] * self.propulsion_breakpoints['values'][k] for k in range(K))

        # 거리 제약

        # 부하 예비력 제약

        # 고장 예비력 제약
        

if __name__ == "__main__":
    test = MILPSolver()
    print(test.propulsion_breakpoints)
