import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimizer.milp_solver import fuel_consumption


def test_fuel_consumption_interprets_sfoc_as_g_per_kwh():
    alpha1 = 3e-7
    alpha2 = -0.0073
    alpha3 = 220.07
    power_mw = 10.0

    expected_sfoc_g_per_kwh = alpha1 * power_mw**2 + alpha2 * power_mw + alpha3
    expected_fuel_kgph = expected_sfoc_g_per_kwh * power_mw

    assert fuel_consumption(power_mw, alpha1, alpha2, alpha3) == expected_fuel_kgph
