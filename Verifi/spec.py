"""
Verifi/spec.py
논문의 DG SPEC과 추진 계수, 매개 변수 테이블
"""
V_MAX = 30 #Knots

PROP_SPECS ={
    "C1" : 0.1018,
    "C2" : 3.83,
    "C3" : 1160,
    'D' : 5.6,
    'rho' : 1.025, # 해수 밀도 (g/cm^3)
    'K_Q' : 0.0666 # 토크 효율 계수
}

DT = 0.5
T_TOTAL = 30.0

RESERVE_SERVICE = 0.25
RESERVE_PROPULSION = 0.191

ESS_PARAMS = {
    'E_rat': 8.75,          # 배터리 정격 용량 (MWh) 
    'P_max': 7.0,       # 최대 충전 전력 (MW)
    'SOC_0': 0.5,          # 초기 SOC (State of Charge)
    'SOC_min': 0.2,        # 최소 허용 SOC
    'SOC_max': 0.95,       # 최대 허용 SOC
    'eta_ch': 0.92,        # 충전 효율
    'eta_dch': 0.95,       # 방전 효율
    'eta_rt': 0.90,        # 왕복 효율 (Round-trip) = thermal efficiency
    'C_rep': 300,          # 배터리 교체 비용 (m.u / kWh) 
    'L_cycle': 3000        # 배터리 수명 동안 처리되는 전력 용량 (KWh)
}

SERVICE_LOAD = [2.7, 3.7, 3.8, 2.4, 3.0, 2.6, 2.9, 3.0, 2.5, 3.6, 3.4, 3.5, 3.4, 3.7, 2.5, 3.7, 3.1, 2.7, 3.6, 3.6, 3.4, 3.0, 3.9, 3.4, 2.8, 3.4, 3.9, 2.7, 4.4, 4.4]

DG_SPECS = {
    "DG1" : {
        "P_max" : 12.5,
        "P_min" : 5,
        "T_ON" : 3,
        "T_OFF" : 2,
        "Ramp_up" : 0.6,
        "Ramp_down" : 0.6,
        "alpha1" : 23,
        "alpha2" : 2158,
        "alpha3" : 300,
        "C_SU" : 800
    },

    "DG2" : {
        "P_max" : 12.5,
        "P_min" : 5,
        "T_ON" : 3,
        "T_OFF" : 2,
        "Ramp_up" : 0.6,
        "Ramp_down" : 0.6,
        "alpha1" : 23,
        "alpha2" : 2158,
        "alpha3" : 300,
        "C_SU" : 800
    },

    "DG3" : {
        "P_max" : 7,
        "P_min" : 3,
        "T_ON" : 2,
        "T_OFF" : 1,
        "Ramp_up" : 0.6,
        "Ramp_down" : 0.6,
        "alpha1" : 10,
        "alpha2" : 1623,
        "alpha3" : 210,
        "C_SU" : 400
    },

    "DG4" : {
        "P_max" : 7,
        "P_min" : 3,
        "T_ON" : 2,
        "T_OFF" : 1,
        "Ramp_up" : 0.6,
        "Ramp_down" : 0.6,
        "alpha1" : 10,
        "alpha2" : 1623,
        "alpha3" : 210,
        "C_SU" : 400
    },

    "DG5" : {
        "P_max" : 5,
        "P_min" : 1,
        "T_ON" : 1,
        "T_OFF" : 1,
        "Ramp_up" : 0.6,
        "Ramp_down" : 0.6,
        "alpha1" : 30,
        "alpha2" : 1500,
        "alpha3" : 150,
        "C_SU" : 300
    },

    "DG6" : {
        "P_max" : 5,
        "P_min" : 1,
        "T_ON" : 1,
        "T_OFF" : 1,
        "Ramp_up" : 0.6,
        "Ramp_down" : 0.6,
        "alpha1" : 30,
        "alpha2" : 1500,
        "alpha3" : 150,
        "C_SU" : 300
    }
}