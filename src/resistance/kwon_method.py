"""
Kwon's Method (2008) — 기상 할증 계수 산출
==========================================
"Speed loss due to added resistance in wind and waves"

GA의 Evaluation 단계에서 호출되어, ERA5 기상 데이터로부터
속도 저하율(L) 및 마력 할증 계수(a₂)를 계산한다.

수식 체계:
    L (%)  = C_β × C_U × C_Form
    a₂     = (1 / (1 - L/100))³
    P_req  = a₁ × V³ × a₂
"""

import math
import numpy as np


# =============================================================================
# Beaufort Number ↔ Wind Speed 변환
# =============================================================================

# WMO Beaufort Scale 상한 풍속 (m/s)
_BN_UPPER_BOUNDS = [
    0.2,   # BN 0
    1.5,   # BN 1
    3.3,   # BN 2
    5.4,   # BN 3
    7.9,   # BN 4
    10.7,  # BN 5
    13.8,  # BN 6
    17.1,  # BN 7
    20.7,  # BN 8
    24.4,  # BN 9
    28.4,  # BN 10
    32.6,  # BN 11
    36.9,  # BN 12
]


def wind_speed_to_beaufort(v_wind_ms: float) -> int:
    """
    풍속(m/s)으로부터 Beaufort Number를 반환.

    Parameters
    ----------
    v_wind_ms : float
        풍속 (m/s), ERA5의 u10/v10으로부터 합성.

    Returns
    -------
    int
        Beaufort Number (0 ~ 12)
    """
    for bn, upper in enumerate(_BN_UPPER_BOUNDS):
        if v_wind_ms <= upper:
            return bn
    return 12


def compute_P_req(
    v_ship_knots: float,
    v_wind_ms: float,
    encounter_angle_deg: float,
    a1: float = 0.003,
    p_service: float = 9.845,
    **kwargs,
) -> dict:
    """
    최종 요구 부하 P_req 계산 (GA Evaluation에서 호출).

    P_prop = a₁ × V³ × a₂
    P_req  = P_prop + P_service

    Parameters
    ----------
    v_ship_knots : float
        선박 속도 (knots)
    v_wind_ms : float
        풍속 (m/s)
    encounter_angle_deg : float
        파도/바람 입사각 (°)
    a1 : float
        추진 계수 (default=0.003)
    P_service : float
        서비스 부하 (MW), default=9.845 (cruising)

    Returns
    -------
    dict
        {
            "P_prop": float,    # 추진 출력 (MW)
            "P_service": float, # 서비스 부하 (MW)
            "P_req": float,     # 총 요구 부하 (MW)
            "a2": float,        # 마력 할증 계수
            "L_percent": float, # 속도 저하율 (%)
            ...                 # Kwon 세부 계수
        }
    """

    # 1. Beaufort Number 계산
    bn = wind_speed_to_beaufort(v_wind_ms)
    
    # 2. 입사각 계수
    mu = 0.5 * (1 + math.cos(math.radians(encounter_angle_deg)))

    # 3. 속도 저하율
    if bn > 0:
        denominator = 2.7 * (disp_vol ** (2/3))
        numerator = 0.7 * bn + (bn ** 6.5)
        dV_V = a1 * mu * (numerator / denominator)
    else:
        dV_V = 0.0
    
    dV_V = min(dV_V, 0.9)

    a2 = 1.0 / ((1.0 - dV_V) ** 3)

    # 추진 출력: P_prop = a₁ × V³ × a₂
    P_prop = a1 * (v_ship_knots ** 3) * a2

    # 총 요구 부하
    P_req = P_prop + p_service

    return {
        "P_prop": P_prop,
        "P_service": p_service,
        "P_req": P_req
    }
