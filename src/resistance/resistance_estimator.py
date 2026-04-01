"""
Combined calm, wind, and wave resistance estimator.
"""

from __future__ import annotations

import math

from src.resistance.calm_water_resistance import CalmWaterResistanceCalculator
from src.resistance.models import EnvironmentData, ResistanceResult, ShipResistanceSpecs
from src.resistance.wave_resistance import WaveResistanceCalculator
from src.resistance.wind_resistance import WindResistanceCalculator


class ResistanceEstimator:
    def __init__(self):
        self._calm_calc = CalmWaterResistanceCalculator()
        self._wind_calc = WindResistanceCalculator()
        self._wave_calc = WaveResistanceCalculator()

    @staticmethod
    def _angle_delta_deg(angle_a: float, angle_b: float) -> float:
        return ((angle_a - angle_b + 180.0) % 360.0) - 180.0

    def estimate_resistance(
        self,
        ship: ShipResistanceSpecs,
        env: EnvironmentData,
        v_stw_knots: float,
        heading_deg: float,
    ) -> ResistanceResult:
        v_stw_ms = v_stw_knots * 0.514444
        if v_stw_ms < 0.1:
            return ResistanceResult(
                total_resistance_n=0.0,
                calm_resistance_n=0.0,
                wind_resistance_n=0.0,
                wave_resistance_n=0.0,
                speed_over_ground_knots=0.0,
                wake_fraction=0.0,
                thrust_deduction=0.0,
                relative_rotation_efficiency=0.0,
            )

        current_component = env.current_speed_ms * math.cos(
            math.radians(self._angle_delta_deg(env.current_dir_deg, heading_deg))
        )
        v_sog_ms = max(0.1, v_stw_ms + current_component)

        if env.rel_wind_speed_ms is not None and env.rel_wind_dir_deg is not None:
            rel_wind_speed_ms = env.rel_wind_speed_ms
            rel_wind_dir_deg = abs(env.rel_wind_dir_deg)
        elif env.wind_dir_deg is not None and env.wind_speed_ms > 0.0:
            wind_angle_diff = abs(self._angle_delta_deg(env.wind_dir_deg, heading_deg))
            rel_wind_speed_ms = math.sqrt(
                v_sog_ms**2
                + env.wind_speed_ms**2
                + 2.0 * v_sog_ms * env.wind_speed_ms * math.cos(math.radians(wind_angle_diff))
            )
            rel_wind_dir_deg = abs(
                math.degrees(
                    math.atan2(
                        env.wind_speed_ms * math.sin(math.radians(wind_angle_diff)),
                        v_sog_ms + env.wind_speed_ms * math.cos(math.radians(wind_angle_diff)),
                    )
                )
            )
        else:
            rel_wind_speed_ms = 0.0
            rel_wind_dir_deg = 0.0

        calm = self._calm_calc.calculate(ship, v_stw_ms)
        r_wind = self._wind_calc.calculate(ship, rel_wind_speed_ms, rel_wind_dir_deg)
        rel_wave_dir_deg = abs(self._angle_delta_deg(env.wave_dir_deg, heading_deg))
        r_wave = self._wave_calc.calculate_irregular(ship, env, v_stw_ms, rel_wave_dir_deg)
        total = calm.calm_resistance_n + r_wind + r_wave

        return ResistanceResult(
            total_resistance_n=total,
            calm_resistance_n=calm.calm_resistance_n,
            wind_resistance_n=r_wind,
            wave_resistance_n=r_wave,
            speed_over_ground_knots=v_sog_ms / 0.514444,
            wake_fraction=calm.wake_fraction,
            thrust_deduction=calm.thrust_deduction,
            relative_rotation_efficiency=calm.relative_rotation_efficiency,
            relative_wind_speed_ms=rel_wind_speed_ms,
            relative_wind_dir_deg=rel_wind_dir_deg,
        )
