"""Quick CPLEX_PY vs CBC comparison (T=22, no SOS2)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimizer.milp_solver import MILPSolver

T = 22
P_REQ = [11.0] * T
DT = [0.5] * T
SOC0 = 0.7

print(f"Test: P_req={P_REQ[0]} MW x {T} steps, dt={DT[0]} h, SOC0={SOC0}")
print(f"{'Solver':<15s} {'Status':<12s} {'Fuel(kg)':<12s} {'Time(s)':<10s}")
print("-" * 50)

for name in ["cbc", "cplex_py"]:
    milp = MILPSolver(sfoc_json_path="config/sfoc.json", n_pwl_segments=5, solver_name=name)
    t0 = time.time()
    r = milp.solve(P_req=P_REQ, dt=DT, initial_SOC=SOC0, msg=False, enable_sos2=False)
    elapsed = time.time() - t0
    fuel = f"{r['total_fuel_kg']:.1f}" if r["feasible"] else "N/A"
    print(f"{name:<15s} {r['status']:<12s} {fuel:<12s} {elapsed:<10.2f}")

# SOS2 enabled test
print(f"\n--- With SOS2 ---")
print(f"{'Solver':<15s} {'Status':<12s} {'Fuel(kg)':<12s} {'Time(s)':<10s}")
print("-" * 50)

for name in ["cbc", "cplex_py"]:
    milp = MILPSolver(sfoc_json_path="config/sfoc.json", n_pwl_segments=5, solver_name=name)
    t0 = time.time()
    r = milp.solve(P_req=P_REQ, dt=DT, initial_SOC=SOC0, msg=False, enable_sos2=True)
    elapsed = time.time() - t0
    fuel = f"{r['total_fuel_kg']:.1f}" if r["feasible"] else "N/A"
    print(f"{name:<15s} {r['status']:<12s} {fuel:<12s} {elapsed:<10.2f}")
