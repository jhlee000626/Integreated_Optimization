"""
Wageningen B-series helper functions.
"""

from __future__ import annotations

import math


class _WageningenTerm:
    __slots__ = ("coeff", "s", "t", "u", "v")

    def __init__(self, coeff: float, s: int, t: int, u: int, v: int):
        self.coeff = coeff
        self.s = s
        self.t = t
        self.u = u
        self.v = v


class _RnTerm:
    __slots__ = ("coeff", "r", "u", "t", "s", "v")

    def __init__(self, coeff: float, r: int, u: int, t: int, s: int, v: int):
        self.coeff = coeff
        self.r = r
        self.u = u
        self.t = t
        self.s = s
        self.v = v


KT_TERMS = [
    _WageningenTerm(+0.00880496, 0, 0, 0, 0),
    _WageningenTerm(-0.204554, 1, 0, 0, 0),
    _WageningenTerm(+0.166351, 0, 1, 0, 0),
    _WageningenTerm(+0.158114, 0, 2, 0, 0),
    _WageningenTerm(-0.147581, 2, 0, 1, 0),
    _WageningenTerm(-0.481497, 1, 1, 1, 0),
    _WageningenTerm(+0.415437, 0, 2, 1, 0),
    _WageningenTerm(+0.0144043, 0, 0, 0, 1),
    _WageningenTerm(-0.0530054, 2, 0, 0, 1),
    _WageningenTerm(+0.0143481, 0, 1, 0, 1),
    _WageningenTerm(+0.0606826, 1, 1, 0, 1),
    _WageningenTerm(-0.0125894, 0, 0, 1, 1),
    _WageningenTerm(+0.0109689, 1, 0, 1, 1),
    _WageningenTerm(-0.133698, 0, 3, 0, 0),
    _WageningenTerm(+0.00638407, 0, 6, 0, 0),
    _WageningenTerm(-0.00132718, 2, 6, 0, 0),
    _WageningenTerm(+0.168496, 3, 0, 1, 0),
    _WageningenTerm(-0.0507214, 0, 0, 2, 0),
    _WageningenTerm(+0.0854559, 2, 0, 2, 0),
    _WageningenTerm(-0.0504475, 3, 0, 2, 0),
    _WageningenTerm(+0.010465, 1, 6, 2, 0),
    _WageningenTerm(-0.00648272, 2, 6, 2, 0),
    _WageningenTerm(-0.00841728, 0, 3, 0, 1),
    _WageningenTerm(+0.0168424, 1, 3, 0, 1),
    _WageningenTerm(-0.00102296, 3, 3, 0, 1),
    _WageningenTerm(-0.0317791, 0, 3, 1, 1),
    _WageningenTerm(+0.018604, 1, 0, 2, 1),
    _WageningenTerm(-0.00410798, 0, 2, 2, 1),
    _WageningenTerm(-0.000606848, 0, 0, 0, 2),
    _WageningenTerm(-0.0049819, 1, 0, 0, 2),
    _WageningenTerm(+0.0025983, 2, 0, 0, 2),
    _WageningenTerm(-0.000560528, 3, 0, 0, 2),
    _WageningenTerm(-0.00163652, 1, 2, 0, 2),
    _WageningenTerm(-0.000328787, 1, 6, 0, 2),
    _WageningenTerm(+0.000116502, 2, 6, 0, 2),
    _WageningenTerm(+0.000690904, 0, 0, 1, 2),
    _WageningenTerm(+0.00421749, 0, 3, 1, 2),
    _WageningenTerm(+0.0000565229, 3, 6, 1, 2),
    _WageningenTerm(-0.00146564, 0, 3, 2, 2),
]

KQ_TERMS = [
    _WageningenTerm(+0.00379368, 0, 0, 0, 0),
    _WageningenTerm(+0.00886523, 2, 0, 0, 0),
    _WageningenTerm(-0.032241, 1, 1, 0, 0),
    _WageningenTerm(+0.00344778, 0, 2, 0, 0),
    _WageningenTerm(-0.0408811, 0, 1, 1, 0),
    _WageningenTerm(-0.108009, 1, 1, 1, 0),
    _WageningenTerm(-0.0885381, 2, 1, 1, 0),
    _WageningenTerm(+0.188561, 0, 2, 1, 0),
    _WageningenTerm(-0.00370871, 1, 0, 0, 1),
    _WageningenTerm(+0.00513696, 0, 1, 0, 1),
    _WageningenTerm(+0.0209449, 1, 1, 0, 1),
    _WageningenTerm(+0.00474319, 2, 1, 0, 1),
    _WageningenTerm(-0.00723408, 2, 0, 1, 1),
    _WageningenTerm(+0.00438388, 1, 1, 1, 1),
    _WageningenTerm(-0.0269403, 0, 2, 1, 1),
    _WageningenTerm(+0.0558082, 3, 0, 1, 0),
    _WageningenTerm(+0.0161886, 0, 3, 1, 0),
    _WageningenTerm(+0.00318086, 1, 3, 1, 0),
    _WageningenTerm(+0.015896, 0, 0, 2, 0),
    _WageningenTerm(+0.0471729, 1, 0, 2, 0),
    _WageningenTerm(+0.0196283, 3, 0, 2, 0),
    _WageningenTerm(-0.0502782, 0, 1, 2, 0),
    _WageningenTerm(-0.030055, 3, 1, 2, 0),
    _WageningenTerm(+0.0417122, 2, 2, 2, 0),
    _WageningenTerm(-0.0397722, 0, 3, 2, 0),
    _WageningenTerm(-0.00350024, 0, 6, 2, 0),
    _WageningenTerm(-0.0106854, 3, 0, 0, 1),
    _WageningenTerm(+0.00110903, 3, 3, 0, 1),
    _WageningenTerm(-0.000313912, 0, 6, 0, 1),
    _WageningenTerm(+0.0035985, 3, 0, 1, 1),
    _WageningenTerm(-0.00142121, 0, 6, 1, 1),
    _WageningenTerm(-0.00383637, 1, 0, 2, 1),
    _WageningenTerm(+0.0126803, 0, 2, 2, 1),
    _WageningenTerm(-0.00318278, 2, 3, 2, 1),
    _WageningenTerm(+0.00334268, 0, 6, 2, 1),
    _WageningenTerm(-0.00183491, 1, 1, 0, 2),
    _WageningenTerm(+0.000112451, 3, 2, 0, 2),
    _WageningenTerm(-0.0000297228, 3, 6, 0, 2),
    _WageningenTerm(+0.000269551, 1, 0, 1, 2),
    _WageningenTerm(+0.00083265, 2, 0, 1, 2),
    _WageningenTerm(+0.00155334, 0, 2, 1, 2),
    _WageningenTerm(+0.000302683, 0, 6, 1, 2),
    _WageningenTerm(-0.0001843, 0, 0, 2, 2),
    _WageningenTerm(-0.000425399, 0, 3, 2, 2),
    _WageningenTerm(+0.0000869243, 3, 3, 2, 2),
    _WageningenTerm(-0.0004659, 0, 6, 2, 2),
    _WageningenTerm(+0.0000554194, 1, 6, 2, 2),
]

DELTA_KT_TERMS = [
    _RnTerm(+0.000353485, 0, 0, 0, 0, 0),
    _RnTerm(-0.00333758, 0, 1, 0, 2, 0),
    _RnTerm(-0.00478125, 0, 1, 1, 1, 0),
    _RnTerm(+0.000257792, 2, 1, 0, 2, 0),
    _RnTerm(+0.0000643192, 1, 0, 6, 2, 0),
    _RnTerm(-0.0000110636, 2, 0, 6, 2, 0),
    _RnTerm(-0.0000276305, 2, 1, 0, 2, 1),
    _RnTerm(+0.0000954, 1, 1, 1, 1, 1),
    _RnTerm(+0.0000032049, 1, 1, 3, 1, 2),
]

DELTA_KQ_TERMS = [
    _RnTerm(-0.000591412, 0, 0, 0, 0, 0),
    _RnTerm(+0.00696898, 0, 0, 1, 0, 0),
    _RnTerm(-0.0000666654, 0, 0, 6, 0, 1),
    _RnTerm(+0.0160818, 0, 2, 0, 0, 0),
    _RnTerm(-0.000938091, 1, 0, 1, 0, 0),
    _RnTerm(-0.00059593, 1, 0, 2, 0, 0),
    _RnTerm(+0.0000782099, 2, 0, 2, 0, 0),
    _RnTerm(+0.0000052199, 1, 1, 0, 2, 1),
    _RnTerm(-0.00000088528, 2, 1, 1, 1, 1),
    _RnTerm(+0.0000230171, 1, 0, 6, 0, 1),
    _RnTerm(-0.00000184341, 2, 0, 6, 0, 1),
    _RnTerm(-0.00400252, 1, 2, 0, 0, 0),
    _RnTerm(+0.000220915, 2, 2, 0, 0, 0),
]


def _evaluate_terms(terms, j: float, p_d: float, ae_ao: float, z: float) -> float:
    value = 0.0
    for term in terms:
        contrib = term.coeff
        if term.s:
            contrib *= j ** term.s
        if term.t:
            contrib *= p_d ** term.t
        if term.u:
            contrib *= ae_ao ** term.u
        if term.v:
            contrib *= z ** term.v
        value += contrib
    return value


def calculate_kt(j: float, p_d: float, ae_ao: float, z: float) -> float:
    return _evaluate_terms(KT_TERMS, j, p_d, ae_ao, z)


def calculate_kq(j: float, p_d: float, ae_ao: float, z: float) -> float:
    return _evaluate_terms(KQ_TERMS, j, p_d, ae_ao, z)


def _calculate_delta(terms, j: float, p_d: float, ae_ao: float, z: float, rn: float) -> float:
    log_rn_term = math.log10(rn) - 0.301
    value = 0.0
    for term in terms:
        contrib = term.coeff
        if term.r:
            contrib *= log_rn_term ** term.r
        if term.u:
            contrib *= ae_ao ** term.u
        if term.t:
            contrib *= p_d ** term.t
        if term.s:
            contrib *= j ** term.s
        if term.v:
            contrib *= z ** term.v
        value += contrib
    return value


def calculate_total_kt(j: float, p_d: float, ae_ao: float, z: float, rn: float = 0.0) -> float:
    base = calculate_kt(j, p_d, ae_ao, z)
    if rn <= 0.0:
        return base
    return base + _calculate_delta(DELTA_KT_TERMS, j, p_d, ae_ao, z, rn)


def calculate_total_kq(j: float, p_d: float, ae_ao: float, z: float, rn: float = 0.0) -> float:
    base = calculate_kq(j, p_d, ae_ao, z)
    if rn <= 0.0:
        return base
    return base + _calculate_delta(DELTA_KQ_TERMS, j, p_d, ae_ao, z, rn)


def fit_quadratic(x1: float, x2: float, x3: float, y1: float, y2: float, y3: float) -> tuple[float, float, float]:
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
    c = (
        x2 * x3 * (x2 - x3) * y1
        + x3 * x1 * (x3 - x1) * y2
        + x1 * x2 * (x1 - x2) * y3
    ) / denom
    return (a, b, c)


def get_quadratic_coefficients(
    j_min: float,
    j_max: float,
    p_d: float,
    ae_ao: float,
    z: float,
    rn: float = 0.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    j1 = j_min
    j2 = (j_min + j_max) / 2.0
    j3 = j_max
    kt1 = calculate_total_kt(j1, p_d, ae_ao, z, rn)
    kt2 = calculate_total_kt(j2, p_d, ae_ao, z, rn)
    kt3 = calculate_total_kt(j3, p_d, ae_ao, z, rn)
    kq1 = calculate_total_kq(j1, p_d, ae_ao, z, rn)
    kq2 = calculate_total_kq(j2, p_d, ae_ao, z, rn)
    kq3 = calculate_total_kq(j3, p_d, ae_ao, z, rn)
    return (
        fit_quadratic(j1, j2, j3, kt1, kt2, kt3),
        fit_quadratic(j1, j2, j3, kq1, kq2, kq3),
    )
