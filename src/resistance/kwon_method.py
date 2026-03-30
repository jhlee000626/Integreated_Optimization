"""
Kwon-style added-resistance approximation used by the routing optimizer.
"""

from __future__ import annotations

import math

from src.ship.kcs_specs import KCS_HULL


_BN_UPPER_BOUNDS = [
    0.2,
    1.5,
    3.3,
    5.4,
    7.9,
    10.7,
    13.8,
    17.1,
    20.7,
    24.4,
    28.4,
    32.6,
    36.9,
]


def wind_speed_to_beaufort(v_wind_ms: float) -> int:
    """Map wind speed in m/s to the WMO Beaufort number."""
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
    Estimate propulsion and hotel load power demand.

    `P_service` is accepted as an alias for backward compatibility with
    existing callers in the optimizer and verification scripts.
    """
    if "P_service" in kwargs:
        p_service = kwargs["P_service"]

    bn = wind_speed_to_beaufort(v_wind_ms)
    mu = 0.5 * (1.0 + math.cos(math.radians(encounter_angle_deg)))

    displacement_vol = KCS_HULL["displacement_vol"]
    if bn > 0:
        denominator = 2.7 * (displacement_vol ** (2.0 / 3.0))
        numerator = 0.7 * bn + (bn ** 6.5)
        speed_loss_ratio = a1 * mu * (numerator / denominator)
    else:
        speed_loss_ratio = 0.0

    speed_loss_ratio = min(speed_loss_ratio, 0.9)
    added_resistance_factor = 1.0 / ((1.0 - speed_loss_ratio) ** 3)

    p_prop = a1 * (v_ship_knots ** 3) * added_resistance_factor
    p_req = p_prop + p_service

    return {
        "P_prop": p_prop,
        "P_service": p_service,
        "P_req": p_req,
        "a2": added_resistance_factor,
        "L_percent": speed_loss_ratio * 100.0,
    }
