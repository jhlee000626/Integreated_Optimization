import numpy as np
import pandas as pd
from spec import PROP_SPECS, DG_SPECS
import matplotlib.pyplot as plt

V_Space = np.linspace(0, 30, 31)

def compute_propulsion_power(V):
    C1 = PROP_SPECS["C1"]
    C2 = PROP_SPECS["C2"]
    C3 = PROP_SPECS["C3"]

    P_propulsion = (C1 * V**C2 + C3)/ 1000 # MW 단위로 변환
    return P_propulsion

def piecewise_linear(breakpoints):
    y = compute_propulsion_power(breakpoints)
    return list(y)

def fuel_cost(P):
    fuel_costs = {}
    for dg in DG_SPECS:
        p_space = np.linspace(0, DG_SPECS[dg]["P_max"], 100)
        p_max = DG_SPECS[dg]["P_max"]  
        alpha1 = DG_SPECS[dg]["alpha1"]
        alpha2 = DG_SPECS[dg]["alpha2"]
        alpha3 = DG_SPECS[dg]["alpha3"]
        fuel_costs[dg] = alpha1 * (p_space/p_max)**2 + alpha2 * (p_space/p_max) + alpha3

    return fuel_costs

if __name__ == "__main__":
    # propulsion_power = compute_propulsion_power(V_Space)
    # breakpoints = np.linspace(0,30,5)
    # values = piecewise_linear(breakpoints)
    # plt.figure(figsize=(10, 6))
    # plt.plot(V_Space, propulsion_power, marker='o', markersize=5, label='Propulsion Power')
    # plt.plot(breakpoints, values, marker='x', markersize=10, label='Piecewise Linear Approximation')
    # plt.xlabel("Velocity (m/s)")
    # plt.ylabel("Propulsion Power (W)")
    # plt.title("Propulsion Power vs. Velocity")
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    fuel_costs = fuel_cost(np.linspace(5, 12.5, 10))
    for dg in fuel_costs:
        plt.plot(np.linspace(0, DG_SPECS[dg]["P_max"], 100), fuel_costs[dg], marker='o', label=f'Fuel Cost {dg}')
    plt.xlabel("Power Output (MW)")
    plt.ylabel("Fuel Cost (m.u.)")
    plt.title("Fuel Cost vs. Power Output for DGs")
    plt.grid(True)
    plt.legend()
    plt.show()
