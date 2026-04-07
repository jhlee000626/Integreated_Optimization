"""
Reverse-DPM propulsion power estimator.
"""

from __future__ import annotations

import math

from src.resistance.models import PropulsionResult, ResistanceResult, ShipResistanceSpecs


KT_COEFFS = (-0.0299, -0.4763, 0.54)
KQ_COEFFS = (-0.0116, -0.05077, 0.07448)


class PropulsionHelper:
    RHO_SEA_WATER = 1025.0

    @staticmethod
    def _solve_quadratic_positive(a: float, b: float, c: float, fallback: float) -> float:
        if abs(a) < 1e-12:
            if abs(b) < 1e-12:
                return fallback
            sol = -c / b
            return sol if sol >= 0.0 else fallback
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return fallback
        sqrt_disc = math.sqrt(disc)
        sol1 = (-b + sqrt_disc) / (2.0 * a)
        sol2 = (-b - sqrt_disc) / (2.0 * a)
        positive = [sol for sol in (sol1, sol2) if sol >= 0.0]
        if not positive:
            return fallback
        
        # 양수 해, 음수 해중에 기존 전진비에 가까운 해 선택
        return min(positive, key=lambda value: abs(value - fallback))

    @staticmethod
    def _evaluate_quadratic(coeffs: tuple[float, float, float], x: float) -> float:
        a, b, c = coeffs
        return a * x * x + b * x + c

    def calculate_propulsion(
        self,
        v_knots: float,
        res_data: ResistanceResult,
        ship: ShipResistanceSpecs,
    ) -> PropulsionResult:
        """Estimate delivered power using V_STW (= V_C) as the DPM speed basis."""
        n_rps = ship.N_MCR / 60.0 * v_knots / ship.DesignSpeed
        if n_rps <= 1e-9 or v_knots <= 0.0:
            return PropulsionResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        kt_coeffs = KT_COEFFS
        kq_coeffs = KQ_COEFFS

        v_c = v_knots * 0.514444
        if v_c < 0.1:
            return PropulsionResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        total_resistance_n = res_data.total_resistance_n
        w = res_data.wake_fraction
        t = res_data.thrust_deduction
        eta_r = res_data.relative_rotation_efficiency # 상대 회전 효율 > Holtrop
        eta_h = (1.0 - t) / (1.0 - w) if abs(1.0 - w) > 1e-6 else 1.0 # 선각 효율

        v_id = v_c
        tolerance = 1e-4
        alpha = 0.5
        j_0 = 0.0
        j_1 = 0.0

        for _ in range(100):
            j_0 = 0.0 if n_rps <= 1e-6 else (v_id * (1.0 - w)) / (n_rps * ship.PropellerDiameter)
            denom = (1.0 - t) * (1.0 - w) ** 2 * self.RHO_SEA_WATER * ship.PropellerDiameter**2 * v_id**2
            tau_op = total_resistance_n / denom if denom > 1e-9 else 0.0
            a_poly, b_poly, c_poly = kt_coeffs
            j_1 = self._solve_quadratic_positive(a_poly - tau_op, b_poly, c_poly, j_0)
            v_sc = (n_rps * j_1 * ship.PropellerDiameter) / (1.0 - w)
            diff = v_sc - v_c
            if abs(diff) < tolerance:
                break
            v_id = max(0.1, v_id - alpha * diff)

        j_final = v_id * (1.0 - w) / (n_rps * ship.PropellerDiameter)
        kt_final = self._evaluate_quadratic(kt_coeffs, j_final)
        kq_final = self._evaluate_quadratic(kq_coeffs, j_final)
        eta_o = 0.0 if abs(kq_final) < 1e-12 else j_final / (2.0 * math.pi) * (kt_final / kq_final)
        eta_d = eta_o * eta_r * eta_h
        power_kw = total_resistance_n * v_id / eta_d / 1000.0 if eta_d > 1e-4 else 0.0
        torque_knm = (power_kw * 1000.0) / (2.0 * math.pi * n_rps) / 1000.0 if n_rps > 1e-6 else 0.0

        return PropulsionResult(
            power_kw=power_kw,
            rpm=n_rps * 60.0,
            torque_knm=torque_knm,
            j_op=j_final,
            v_ideal=v_id,
            eta_d=eta_d,
            eta_o=eta_o,
        )
