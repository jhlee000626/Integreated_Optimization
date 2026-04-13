import numpy as np
import pandas as pd
from spec import PROP_SPECS
import matplotlib.pyplot as plt

V_Space = np.linspace(0, 30, 31)

def compute_propulsion_power(V):
    C1 = PROP_SPECS["C1"]
    C2 = PROP_SPECS["C2"]
    C3 = PROP_SPECS["C3"]

    P_propulsion = (C1 * V**C2 + C3)
    return P_propulsion

def piecewise_linear(breakpoints):
    y = compute_propulsion_power(breakpoints)
    return list(y)

if __name__ == "__main__":
    propulsion_power = compute_propulsion_power(V_Space)
    breakpoints = np.linspace(0,30,5)
    values = piecewise_linear(breakpoints)
    plt.figure(figsize=(10, 6))
    plt.plot(V_Space, propulsion_power, marker='o', markersize=5, label='Propulsion Power')
    plt.plot(breakpoints, values, marker='x', markersize=10, label='Piecewise Linear Approximation')
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Propulsion Power (W)")
    plt.title("Propulsion Power vs. Velocity")
    plt.grid(True)
    plt.legend()
    plt.show()
