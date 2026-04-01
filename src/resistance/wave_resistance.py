"""
Irregular-wave added resistance model ported from the C# implementation.
"""

from __future__ import annotations

import math

from src.resistance.models import EnvironmentData, ShipResistanceSpecs


class WaveResistanceCalculator:
    G = 9.81
    RHO_SEAWATER = 1025.0

    def calculate_irregular(
        self,
        ship: ShipResistanceSpecs,
        env: EnvironmentData,
        v_ms: float,
        main_wave_dir_deg: float,
    ) -> float:
        if env.wave_height_m < 0.5 or env.wave_period_s <= 0.0:
            return 0.0

        w_start, w_end, dw = 0.1, 3.0, 0.05
        mu_start, mu_end, d_mu = -math.pi / 2.0, math.pi / 2.0, math.pi / 18.0
        total = 0.0

        w = w_start
        while w <= w_end + 1e-12:
            s_w = self._ittc1978_spectrum(w, env.wave_height_m, env.wave_period_s)
            if s_w >= 1e-6:
                mu = mu_start
                while mu <= mu_end + 1e-12:
                    spread = self._spreading_function(mu)
                    if spread >= 1e-6:
                        energy_component = s_w * spread * dw * d_mu
                        component_amp = math.sqrt(2.0 * energy_component)
                        if component_amp >= 0.01:
                            current_wave_dir_deg = main_wave_dir_deg + math.degrees(mu)
                            total += self.calculate_regular(
                                ship=ship,
                                wave_amp=component_amp,
                                omega=w,
                                rel_wave_dir_deg=current_wave_dir_deg,
                                v_ms=v_ms,
                            )
                    mu += d_mu
            w += dw

        return total

    def calculate_regular(
        self,
        ship: ShipResistanceSpecs,
        wave_amp: float,
        omega: float,
        rel_wave_dir_deg: float,
        v_ms: float,
    ) -> float:
        if rel_wave_dir_deg > 180.0:
            rel_wave_dir_deg = 360.0 - rel_wave_dir_deg

        cb = ship.Cb
        alpha = math.radians(rel_wave_dir_deg)
        k = omega**2 / self.G
        wavelength = 2.0 * math.pi / k
        fr = v_ms / math.sqrt(self.G * ship.Lbp)
        kyy = 0.25
        group_velocity = self.G / (2.0 * omega)
        fr_rel = (v_ms - group_velocity) / math.sqrt(self.G * ship.Lbp)

        def total_resistance_at_angle(calc_alpha: float) -> float:
            w_term1 = 2.142 * (kyy ** (1.0 / 3.0)) * math.sqrt(ship.Lbp / (2.0 * math.pi * self.G)) * (cb / 0.65) ** 0.17
            w_term2 = 1.0 - (0.111 / ship.Cb) * (math.log(ship.Beam / ship.Draft) - math.log(2.75))
            poly_fr = (-1.377 * fr * fr + 1.157 * fr) * abs(math.cos(calc_alpha))
            poly_head = 0.618 * (13.0 + math.cos(2.0 * calc_alpha)) / 14.0
            omega_bar = w_term1 * w_term2 * (poly_fr + poly_head) * omega
            # Negative omega_bar is non-physical in this response model and
            # can produce complex values for non-integer powers. Clamp it into
            # the positive real domain so the motion-induced term smoothly
            # vanishes instead of crashing.
            omega_bar_safe = max(float(omega_bar), 1e-9)

            if calc_alpha <= math.pi / 2.0 + 1e-6:
                term_base = (ship.Beam / ship.Draft) ** -1 * (1.0 + 2.0 * math.cos(calc_alpha)) / 3.0
                term_cb = (0.87 / cb) ** ((1.0 + fr) * math.cos(calc_alpha))
                a1 = term_cb * term_base
                a2 = 0.0072 + 0.1676 * fr if fr < 0.12 else (fr**1.5) * math.exp(-3.5 * fr)
            else:
                term_base_pi = 1.0 / math.log(ship.Beam / ship.Draft)
                term_cb_pi = 0.87 / cb
                if v_ms > group_velocity and fr_rel >= 0.12:
                    a1 = term_cb_pi ** (1.0 + fr_rel) * term_base_pi
                else:
                    a1 = term_cb_pi * term_base_pi
                if v_ms <= group_velocity:
                    a2 = 0.0072 * (2.0 * v_ms / group_velocity - 1.0)
                elif fr_rel < 0.12:
                    a2 = 0.0072 + 0.1676 * fr_rel
                else:
                    a2 = (fr_rel**1.5) * math.exp(-3.5 * fr_rel)

            shape_param = ship.Lbp * ship.Cb / ship.Beam
            b1 = 11.0 if omega_bar_safe < 1.0 else -8.5
            d1 = 566.0 * (shape_param ** -2.66)
            if omega_bar_safe >= 1.0:
                d1 *= -4.0
            exponent_val = (b1 / d1) * (1.0 - omega_bar_safe**d1)
            if math.isnan(exponent_val) or math.isinf(exponent_val):
                exponent_val = 0.0
            shape_func = (omega_bar_safe**b1) * math.exp(exponent_val)
            r_awm = (
                3859.2
                * self.RHO_SEAWATER
                * self.G
                * (wave_amp**2)
                * ((ship.Beam**2) / ship.Lbp)
                * (cb**1.34)
                * (kyy**2)
                * a1
                * a2
                * shape_func
            )
            r_awr = self._calculate_reflection(ship, wave_amp, omega, calc_alpha, fr, v_ms)
            if wavelength / ship.Lbp > 0.5:
                r_awr = 0.0
            return r_awm + r_awr

        if alpha <= math.pi / 2.0:
            return total_resistance_at_angle(alpha)
        if alpha < math.pi:
            res_beam = total_resistance_at_angle(math.pi / 2.0)
            res_following = total_resistance_at_angle(math.pi)
            ratio = (alpha - math.pi / 2.0) / (math.pi - math.pi / 2.0)
            return res_beam + ratio * (res_following - res_beam)
        return total_resistance_at_angle(math.pi)

    def _spreading_function(self, mu: float) -> float:
        if abs(mu) > math.pi / 2.0:
            return 0.0
        return (2.0 / math.pi) * (math.cos(mu) ** 2)

    def _ittc1978_spectrum(self, omega: float, hs: float, t1: float) -> float:
        if omega <= 0.0 or t1 <= 0.0:
            return 0.0
        a_fw = 173.0 * (hs**2) / (t1**4)
        b_fw = 691.0 / (t1**4)
        return a_fw / (omega**5) * math.exp(-b_fw / (omega**4))

    def _calculate_reflection(
        self,
        ship: ShipResistanceSpecs,
        wave_amp: float,
        omega: float,
        alpha: float,
        fr: float,
        v_ms: float,
    ) -> float:
        cp = ship.Cb / ship.Cm
        lcb = -10.0 * fr - (cp - 0.64) / 0.05 + 3.0
        lr = ship.Lwl * (1.0 - cp + 0.06 * cp * lcb / (4.0 * cp - 1.0))
        l_parallel = ship.Lbp * (1.0 - cp)
        le = l_parallel - lr
        if le <= 0.1 * ship.Lbp:
            le = lr

        e1 = math.atan(0.495 * ship.Beam / le)
        e2 = math.atan(0.495 * ship.Beam / lr)
        base_factor = (2.25 / 4.0) * self.RHO_SEAWATER * self.G * ship.Beam * (wave_amp**2)
        term_speed = 2.0 * omega * v_ms / self.G
        f_alpha = math.cos(alpha) if 0.0 <= alpha <= e1 else 0.0
        total = 0.0

        if 0.0 <= alpha <= (math.pi - e1):
            alpha_t = 1.0 - math.exp(-4.0 * math.pi * (ship.Draft / ship.Lbp - ship.Draft / (2.5 * ship.Lbp)))
            term_cb = (0.87 / ship.Cb) ** ((1.0 + 4.0 * math.sqrt(fr)) * f_alpha)
            term_geom = math.sin(e1 + alpha) ** 2 + term_speed * (
                math.cos(alpha) - math.cos(e1) * math.cos(e1 + alpha)
            )
            total += base_factor * alpha_t * term_geom * term_cb

        if 0.0 <= alpha <= e1:
            alpha_t = 1.0 - math.exp(-4.0 * math.pi * (ship.Draft / ship.Lbp - ship.Draft / (2.5 * ship.Lbp)))
            term_cb = (0.87 / ship.Cb) ** ((1.0 + 4.0 * math.sqrt(fr)) * f_alpha)
            term_geom = math.sin(e1 - alpha) ** 2 + term_speed * (
                math.cos(alpha) - math.cos(e1) * math.cos(e1 - alpha)
            )
            total += base_factor * alpha_t * term_geom * term_cb

        if e2 <= alpha <= math.pi:
            t_star = self._get_effective_draft_stern(ship.Draft, ship.Cb, alpha)
            alpha_t = 1.0 - math.exp(-4.0 * math.pi * (t_star / ship.Lbp - t_star / (2.5 * ship.Lbp)))
            term_geom = math.sin(e2 - alpha) ** 2 + term_speed * (
                math.cos(alpha) - math.cos(e2) * math.cos(e2 - alpha)
            )
            total -= base_factor * alpha_t * term_geom

        if alpha <= math.pi and (math.pi - e2) <= alpha:
            t_star = self._get_effective_draft_stern(ship.Draft, ship.Cb, alpha)
            alpha_t = 1.0 - math.exp(-4.0 * math.pi * (t_star / ship.Lbp - t_star / (2.5 * ship.Lbp)))
            term_geom = math.sin(e2 + alpha) ** 2 + term_speed * (math.cos(e2) * math.cos(e2 + alpha))
            total -= base_factor * alpha_t * term_geom

        return total

    def _get_effective_draft_stern(self, draft: float, cb: float, alpha_rad: float) -> float:
        if cb <= 0.75:
            return draft * (4.0 + math.sqrt(abs(math.cos(alpha_rad)))) / 5.0
        return draft * (2.0 + math.sqrt(abs(math.cos(alpha_rad)))) / 3.0
