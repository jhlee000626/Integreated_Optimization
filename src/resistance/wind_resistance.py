"""
ISO 15016 / Kristensen-style wind resistance approximation.
"""

from __future__ import annotations

from src.resistance.models import ShipResistanceSpecs


WIND_COEFFICIENT_LADEN = {
    0: 0.67,
    15: 0.76,
    30: 0.66,
    45: 0.40,
    60: 0.25,
    75: 0.21,
    90: 0.26,
    105: 0.04,
    120: -0.36,
    135: -0.69,
    150: -0.90,
    165: -0.85,
    180: -0.67,
}


class WindResistanceCalculator:
    RHO_AIR = 1.225

    def calculate(self, ship: ShipResistanceSpecs, v_rel_wind_ms: float, v_rel_wind_dir_deg: float) -> float:
        if v_rel_wind_ms <= 0.0:
            return 0.0
        freeboard = ship.Depth - ship.Draft
        hull_area = ship.Beam * freeboard
        superstructure_area = ship.Beam * ship.ContainerHeight * ship.ContainerTiersOnDeck
        total_projected_area = hull_area + superstructure_area
        c_aa = self.get_interpolated_coefficient(v_rel_wind_dir_deg)
        c_0 = self.get_interpolated_coefficient(0.0)
        dynamic_pressure = 0.5 * self.RHO_AIR * total_projected_area * (v_rel_wind_ms**2)
        return dynamic_pressure * (c_aa - c_0)

    def get_interpolated_coefficient(self, angle_deg: float) -> float:
        norm_angle = abs(angle_deg)
        while norm_angle > 180.0:
            norm_angle = abs(norm_angle - 360.0)
        keys = sorted(WIND_COEFFICIENT_LADEN)
        if norm_angle in WIND_COEFFICIENT_LADEN:
            return WIND_COEFFICIENT_LADEN[int(norm_angle)]
        lower = max(key for key in keys if key <= norm_angle)
        upper = min(key for key in keys if key >= norm_angle)
        if upper == lower:
            return WIND_COEFFICIENT_LADEN[lower]
        y1 = WIND_COEFFICIENT_LADEN[lower]
        y2 = WIND_COEFFICIENT_LADEN[upper]
        factor = (norm_angle - lower) / (upper - lower)
        return y1 + factor * (y2 - y1)
