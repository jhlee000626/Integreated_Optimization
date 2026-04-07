"""
Holtrop-based calm-water resistance and interaction coefficients.(Completed)
"""

from __future__ import annotations

import math

from src.resistance.models import CalmWaterResult, ShipResistanceSpecs


class CalmWaterResistanceCalculator:
    RHO_SEA_WATER = 1025.0
    G = 9.81
    NU = 1.188e-6

    def calculate(self, ship: ShipResistanceSpecs, v_ms: float) -> CalmWaterResult:
        l_wl = ship.Lwl
        b = ship.Beam
        t = ship.Draft
        d_prop = ship.PropellerDiameter
        volume = ship.VolumeOfDisplacement
        cb = ship.Cb
        cm = ship.Cm
        cp = cb / cm
        lr = l_wl * (1.0 - cp + 0.06 * cp * ship.LCB_Percent / (4.0 * cp - 1.0))
        c_stern = ship.SternShapeCoefficient
        lcb_percent = ship.LCB_Percent
        ae_ao = ship.AEAO
        wetted_surface = ship.S

        fn = v_ms / math.sqrt(self.G * l_wl)
        rn = (v_ms * l_wl) / self.NU
        cf = 0.075 / ((math.log10(rn) - 2.0) ** 2) # ITTC-1957 마찰 계수

        if b / l_wl < 0.11:
            c7 = 0.229577 * (b / l_wl) ** (1.0 / 3.0)
        elif b / l_wl < 0.25:
            c7 = b / l_wl
        else:
            c7 = 0.5 - 0.0625 * (l_wl / b)

        c14 = 1.0 + 0.011 * c_stern

        # 형상 계수 (1 + k1)
        one_plus_k1 = (
            0.93
            + 0.487118
            * c14
            * (b / l_wl) ** 1.06806
            * (t / l_wl) ** 0.46106
            * (l_wl / lr) ** 0.121563
            * ((l_wl**3) / volume) ** 0.36486
            * (1.0 - cp) ** -0.604247
        )

        # 점성 저항
        rv = 0.5 * self.RHO_SEA_WATER * (v_ms**2) * wetted_surface * cf * one_plus_k1

        # 조파 저항
        rw = self._calculate_wave_resistance(ship, fn, c7, lcb_percent)

        # A_BT 추정식
        a_bt = 0.08 * cm * b * t

        # 구상선수의 중심 높이 추정식
        h_b = 0.3 * t

        # 침수에 대한 Froude수
        f_ni = v_ms / math.sqrt(self.G * (t - h_b - 0.25 * math.sqrt(a_bt) + 0.15 * (v_ms**2)))

        # 선수 출현부 저항
        p_b = 0.56 * math.sqrt(a_bt) / (t - 1.5 * h_b)

        # 구상선수의 부가압력저항
        rb = (
            0.11
            * math.exp(-3.0 * (p_b**-2))
            * (f_ni**3)
            * (a_bt**1.5)
            * self.RHO_SEA_WATER
            * self.G
        ) / (1.0 + f_ni**2)
        
        # 잠수된 트랜섬의 면적
        a_t = 0.051 * cm * b * t

        # 트랜섬 잠수에 기초한 프루드 수 (C_WP = (1 + 2*CB)/3)
        fn_t = v_ms / math.sqrt(2.0 * self.G * a_t / (b + b * ((1.0 + 2.0 * cb) / 3.0)))

        # 트랜섬 저항 계수
        c6 = 0.2 * (1.0 - 0.2 * fn_t) if fn_t < 5.0 else 0.0

        # 잠수된 트랜섬에 의한 부가저항
        rtr = 0.5 * self.RHO_SEA_WATER * (v_ms**2) * a_t * c6

        c4 = t / l_wl if t / l_wl <= 0.04 else 0.04
        c3 = 0.56 * (a_bt**1.5) / (b * t * (0.31 * math.sqrt(a_bt) + t - h_b))
        c2 = math.exp(-1.89 * math.sqrt(c3))

        # 실선 상관 수정계수
        ca = (
            0.006 * (l_wl + 100.0) ** -0.16
            - 0.00205
            + 0.003 * math.sqrt(l_wl / 7.5) * (cb**4) * c2 * (0.04 - c4)
        )

        # 실선 상관수정 저항
        ra = 0.5 * self.RHO_SEA_WATER * (v_ms**2) * wetted_surface * ca
        r_total = rv + rw + rb + rtr + ra

        if b / t < 5.0:
            c8 = b * wetted_surface / (l_wl * d_prop * t)
        else:
            c8 = wetted_surface * (7.0 * b / t - 25.0) / (l_wl * d_prop * (b / t - 3.0))
        c9 = c8 if c8 < 28.0 else 32.0 - 16.0 / (c8 - 24.0)
        c11 = t / d_prop if t / d_prop < 2.0 else 0.0833333 * (t / d_prop) ** 3 + 1.33333
        if cp < 0.7:
            c19 = 0.12997 / (0.95 - cb) - 0.11056 / (0.95 - cp)
        else:
            c19 = 0.18567 / (0.13571 - cm) - 0.71276 + 0.38648 * cp
        c20 = 1.0 + 0.0015 * c_stern
        cp1 = 1.45 * cp - 0.315 - 0.0225 * lcb_percent
        cv = one_plus_k1 * cf + ca

        wake_fraction = (
            c9
            * c20
            * cv
            * l_wl
            / t
            * (0.050776 + 0.93405 * c11 * cv / (1.0 - cp1))
            + 0.27915 * c20 * math.sqrt(b / (l_wl * (1.0 - cp1)))
            + c19 * c20
        )
        thrust_deduction = (
            0.25014
            * (b / l_wl) ** 0.28956
            * (math.sqrt(b * t) / d_prop) ** 0.2624
            / (1.0 - cp + 0.0225 * lcb_percent) ** 0.01762
            + 0.0015 * c_stern
        )
        eta_r = 0.9922 - 0.05908 * ae_ao + 0.07424 * (cp - 0.0225 * lcb_percent)

        return CalmWaterResult(
            total_resistance_kn=r_total / 1000.0,
            calm_resistance_n=r_total,
            wake_fraction=wake_fraction,
            thrust_deduction=thrust_deduction,
            relative_rotation_efficiency=eta_r,
        )

    def _calculate_wave_resistance(self, ship: ShipResistanceSpecs, fn: float, c7: float, lcb: float) -> float:
        l_wl = ship.Lwl
        b = ship.Beam
        t = ship.Draft
        volume = ship.VolumeOfDisplacement
        cp = ship.Cb / ship.Cm
        lr = l_wl * (1.0 - cp + 0.06 * cp * lcb / (4.0 * cp - 1.0))
        cwp = (1.0 + 2.0 * ship.Cb) / 3.0
        cm = ship.Cm
        a_bt = 0.08 * cm * b * t
        h_b = 0.4 * t

        lam = 1.446 * cp - (0.03 * (l_wl / b) if l_wl / b < 12.0 else 0.36)
        c17 = 6919.3 * cm ** -1.3346 * (volume / (l_wl**3)) ** 2.00977 * (l_wl / b - 2.0) ** 1.40692
        m3 = -7.2035 * (b / l_wl) ** 0.326869 * (t / b) ** 0.605375
        c16 = (
            8.07981 * cp - 13.8673 * (cp**2) + 6.984388 * (cp**3)
            if cp < 0.8
            else 1.73014 - 0.7067 * cp
        )
        m1 = 0.0140407 * (l_wl / t) - 1.75254 * (volume ** (1.0 / 3.0)) / l_wl - 4.79323 * (b / l_wl) - c16
        slenderness = (l_wl**3) / volume
        if slenderness < 512.0:
            c15 = -1.69385
        elif slenderness < 1726.91:
            c15 = -1.69385 + (l_wl / (volume ** (1.0 / 3.0)) - 8.0) / 2.36
        else:
            c15 = 0.0
        m4 = c15 * 0.4 * math.exp(-0.034 * fn ** -3.29)
        c3 = 0.56 * (a_bt**1.5) / (b * t * (0.31 * math.sqrt(a_bt) + t - h_b))
        c2 = math.exp(-1.89 * math.sqrt(c3))
        i_e = 1.0 + 89.0 * math.exp(
            -(l_wl / b) ** 0.80856
            * (1.0 - cwp) ** 0.30484
            * (1.0 - cp - 0.0225 * lcb) ** 0.6367
            * (lr / b) ** 0.34574
            * (100.0 * volume / (l_wl**3)) ** 0.16302
        )
        c1 = 2223105.0 * (c7**3.78613) * (t / b) ** 1.07961 * (90.0 - i_e) ** -1.37565
        c5 = 1.0 - 0.8 * (0.051 * cm * b * t) / (b * t * 0.99)

        if fn > 0.55:
            return c17 * c2 * c5 * volume * self.RHO_SEA_WATER * self.G * math.exp(
                m3 * (fn**-0.9) + m4 * math.cos(lam * fn ** -2)
            )
        if fn < 0.4:
            return c1 * c2 * c5 * volume * self.RHO_SEA_WATER * self.G * math.exp(
                m1 * (fn**-0.9) + m4 * math.cos(lam * fn ** -2)
            )
        rw_low = c1 * c2 * c5 * volume * self.RHO_SEA_WATER * self.G * math.exp(
            m1 * (0.4**-0.9) + m4 * math.cos(lam * (0.4 ** -2))
        )
        rw_high = c17 * c2 * c5 * volume * self.RHO_SEA_WATER * self.G * math.exp(
            m3 * (0.55**-0.9) + m4 * math.cos(lam * (0.55 ** -2))
        )
        return rw_low + (10.0 * fn - 4.0) * (rw_high - rw_low) / 1.5
