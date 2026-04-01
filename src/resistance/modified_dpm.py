"""
ISO 15016 + reverse-DPM propulsion load estimator.
"""

from __future__ import annotations

import math
from typing import Any

from src.resistance.models import EnvironmentData, ShipResistanceSpecs
from src.resistance.propulsion_helper import PropulsionHelper
from src.resistance.resistance_estimator import ResistanceEstimator
from src.ship.kcs_specs import KCS_ISO15016


_RESISTANCE_ESTIMATOR = ResistanceEstimator()
_PROPULSION_HELPER = PropulsionHelper()
MAX_PROPULSION_LOAD_KW = 36000.0


def _relative_wind_speed_from_encounter(
    v_ship_knots: float,
    v_wind_ms: float,
    encounter_angle_deg: float,
    current_speed_ms: float = 0.0,
) -> float:
    v_ship_ms = max(0.0, v_ship_knots * 0.514444 + current_speed_ms)
    encounter_rad = math.radians(encounter_angle_deg)
    return math.sqrt(
        v_ship_ms**2
        + v_wind_ms**2
        + 2.0 * v_ship_ms * v_wind_ms * math.cos(encounter_rad)
    )


class ModifiedDPMCalculator:
    """Facade that bundles resistance estimation and reverse-DPM power solve."""

    def __init__(
        self,
        resistance_estimator: ResistanceEstimator | None = None,
        propulsion_helper: PropulsionHelper | None = None,
    ):
        self.resistance_estimator = resistance_estimator or _RESISTANCE_ESTIMATOR
        self.propulsion_helper = propulsion_helper or _PROPULSION_HELPER

    def compute_P_req(
        self,
        v_ship_knots: float,
        v_wind_ms: float,
        encounter_angle_deg: float,
        a1: float | None = None,
        p_service: float = 9.845,
        **kwargs: Any,
    ) -> dict:
        if "P_service" in kwargs:
            p_service = kwargs["P_service"]

        ship_data = kwargs.get("ship_specs", KCS_ISO15016)
        ship = ship_data if isinstance(ship_data, ShipResistanceSpecs) else ShipResistanceSpecs.from_mapping(ship_data)

        heading_deg = float(kwargs.get("heading_deg", 0.0))
        eta_drive = float(kwargs.get("eta_drive", ship.eta_drive))
        env = kwargs.get("env")
        if env is not None and not isinstance(env, EnvironmentData):
            raise TypeError("env must be an EnvironmentData instance when provided.")

        if env is None:
            wind_dir_deg = kwargs.get("wind_dir_deg")
            current_speed_ms = float(kwargs.get("current_speed_ms", 0.0))
            current_dir_deg = float(kwargs.get("current_dir_deg", heading_deg))
            wave_height_m = float(kwargs.get("wave_height_m", 0.0))
            wave_period_s = float(kwargs.get("wave_period_s", 0.0))
            wave_dir_deg = float(kwargs.get("wave_dir_deg", 0.0))

            if wind_dir_deg is None:
                rel_wind_speed_ms = _relative_wind_speed_from_encounter(
                    v_ship_knots=v_ship_knots,
                    v_wind_ms=v_wind_ms,
                    encounter_angle_deg=encounter_angle_deg,
                    current_speed_ms=current_speed_ms,
                )
                env = EnvironmentData(
                    current_speed_ms=current_speed_ms,
                    current_dir_deg=current_dir_deg,
                    wave_height_m=wave_height_m,
                    wave_period_s=wave_period_s,
                    wave_dir_deg=wave_dir_deg,
                    rel_wind_speed_ms=rel_wind_speed_ms,
                    rel_wind_dir_deg=abs(encounter_angle_deg),
                )
            else:
                env = EnvironmentData(
                    wind_speed_ms=v_wind_ms,
                    wind_dir_deg=float(wind_dir_deg),
                    current_speed_ms=current_speed_ms,
                    current_dir_deg=current_dir_deg,
                    wave_height_m=wave_height_m,
                    wave_period_s=wave_period_s,
                    wave_dir_deg=wave_dir_deg,
                )
        elif env.wind_dir_deg is None and env.rel_wind_speed_ms is None:
            env = EnvironmentData(
                wind_speed_ms=v_wind_ms,
                wind_dir_deg=None,
                current_speed_ms=env.current_speed_ms,
                current_dir_deg=env.current_dir_deg,
                wave_height_m=env.wave_height_m,
                wave_period_s=env.wave_period_s,
                wave_dir_deg=env.wave_dir_deg,
                rel_wind_speed_ms=_relative_wind_speed_from_encounter(
                    v_ship_knots=v_ship_knots,
                    v_wind_ms=v_wind_ms,
                    encounter_angle_deg=encounter_angle_deg,
                    current_speed_ms=env.current_speed_ms,
                ),
                rel_wind_dir_deg=abs(encounter_angle_deg),
            )

        resistance = self.resistance_estimator.estimate_resistance(
            ship=ship,
            env=env,
            v_stw_knots=v_ship_knots,
            heading_deg=heading_deg,
        )
        propulsion = self.propulsion_helper.calculate_propulsion(
            v_knots=v_ship_knots,
            res_data=resistance,
            ship=ship,
        )

        pd_kw_raw = propulsion.power_kw
        bhp_kw_raw = pd_kw_raw / eta_drive if eta_drive > 1e-9 else pd_kw_raw
        bhp_kw = min(bhp_kw_raw, MAX_PROPULSION_LOAD_KW)
        pd_kw = bhp_kw * eta_drive if eta_drive > 1e-9 else bhp_kw
        p_prop = bhp_kw / 1000.0
        p_req = p_prop + p_service

        beaufort_wind_speed = env.wind_speed_ms if env.wind_speed_ms > 0.0 else v_wind_ms

        return {
            "P_prop": p_prop,
            "P_service": p_service,
            "P_req": p_req,
            "PD_kW": pd_kw,
            "BHP_kW": bhp_kw,
            "PD_kW_raw": pd_kw_raw,
            "BHP_kW_raw": bhp_kw_raw,
            "total_resistance_N": resistance.total_resistance_n,
            "R_calm_N": resistance.calm_resistance_n,
            "R_wind_N": resistance.wind_resistance_n,
            "R_wave_N": resistance.wave_resistance_n,
            "v_sog_knots": resistance.speed_over_ground_knots,
            "wake_fraction": resistance.wake_fraction,
            "thrust_deduction": resistance.thrust_deduction,
            "eta_R": resistance.relative_rotation_efficiency,
            "eta_D": propulsion.eta_d,
            "eta_O": propulsion.eta_o,
            "RPM": propulsion.rpm,
            "Torque_kNm": propulsion.torque_knm,
            "J_op": propulsion.j_op,
            "V_ideal_ms": propulsion.v_ideal,
            "relative_wind_speed_ms": resistance.relative_wind_speed_ms,
            "relative_wind_dir_deg": resistance.relative_wind_dir_deg,
            "a2": 1.0,
            "L_percent": 0.0,
        }


_DEFAULT_CALCULATOR = ModifiedDPMCalculator()


def compute_P_req(
    v_ship_knots: float,
    v_wind_ms: float,
    encounter_angle_deg: float,
    a1: float | None = None,
    p_service: float = 9.845,
    **kwargs: Any,
) -> dict:
    """Module-level compatibility wrapper for the default Modified DPM calculator."""
    return _DEFAULT_CALCULATOR.compute_P_req(
        v_ship_knots=v_ship_knots,
        v_wind_ms=v_wind_ms,
        encounter_angle_deg=encounter_angle_deg,
        a1=a1,
        p_service=p_service,
        **kwargs,
    )
