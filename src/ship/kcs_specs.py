"""
Ship and power-system reference data for the KCS test vessel.
"""

from __future__ import annotations

import math


KCS_HULL = {
    "Lpp": 230.0,
    "Lbp": 230.0,
    "Lwl": 232.5,
    "B": 32.2,
    "Beam": 32.2,
    "T": 10.8,
    "Draft": 10.8,
    "Depth": 19.0,
    "displacement_vol": 52030.0,
    "VolumeOfDisplacement": 52030.0,
    "Cb": 0.651,
    "CM": 0.985,
    "Cm": 0.985,
    "Surface_area": 9530.0,
    "S": 9530.0,
    "LCB": -1.48,
    "LCB_Percent": -1.48,
    "g": 9.81,
}

KCS_PROPULSION = {
    "PropellerDiameter": 7.9,
    "AEAO": 0.65,
    "NumberOfBlades": 5,
    "N_MCR": 104.0,
    "DesignSpeed": 24.0,
    "eta_drive": 0.98,
    "ContainerHeight": 2.6,
    "ContainerTiersOnDeck": 9,
    "SternShapeCoefficient": 10.0,
}

KCS_ISO15016 = {
    **KCS_HULL,
    **KCS_PROPULSION,
}

POWER_MODEL = {
    "a1": 0.00238,
    "a2_exp": 3,
}

DG_SPECS = {
    "DG1": {
        "P_max": 14.4,
        "ramp_rate": 0.5,
        "min_up": 0.5,
        "min_down": 0.5,
        "cost_start": 170,
    },
    "DG2": {
        "P_max": 14.4,
        "ramp_rate": 0.5,
        "min_up": 1.0,
        "min_down": 1.0,
        "cost_start": 170,
    },
    "DG3": {
        "P_max": 10.8,
        "ramp_rate": 0.5,
        "min_up": 0.5,
        "min_down": 0.5,
        "cost_start": 120,
    },
}

DG_MIN_LOAD_RATIO = 0.25

ESS_SPECS = {
    "P_c_max": 15.0,
    "P_dc_max": 15.0,
    "capacity": 30.0,
    "SOC_max": 0.90,
    "SOC_min": 0.30,
    "eta_c": 0.95,
    "eta_dc": 0.97,
}

SERVICE_LOAD = {
    "departure": 0.5,
    "approach": 0.5,
    "berthing": 0.5,
    "cruising": 0.5,
}

# SERVICE_LOAD = {
#     "departure": 8.69,
#     "approach": 8.69,
#     "berthing": 3.50,
#     "cruising": 9.845,
# }

NAV_PARAMS = {
    "gamma": 0.7,
    "speed_range": (5.0, 24.0),
}


def get_froude_number(v_knots: float) -> float:
    """Return the Froude number for the reference hull."""
    v_ms = v_knots * 0.514444
    return v_ms / math.sqrt(KCS_HULL["g"] * KCS_HULL["Lpp"])
