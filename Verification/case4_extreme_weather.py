"""
Verification Case 4: Extreme Weather Blackout Validation
========================================================
강력한 악천후 발생 시(풍속 25m/s 이상), 요구 전력(P_req)이 디젤 발전기 
최대 총용량(30MW)을 초과하는 상황을 가정합니다.
이 때 ESS의 방전 제어가 피크를 깎아주어, 배 전체가 Blackout(정전) 
위험에 처하는 것을 막아낼 수 있는지(Feasibility)를 검증합니다.
"""

import os
import sys
import time
import multiprocessing
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from deap import tools, algorithms

from src.grid.cost_map import build_cost_map
from src.optimizer.ga_engine import setup_ga, decode_route, evaluate, N_SEGMENTS
from src.optimizer.milp_solver import MILPSolver
from src.weather.era5_loader import create_synthetic_weather
from src.resistance.kwon_method import compute_P_req
from src.optimizer.fitness import compute_heading, compute_encounter_angle
from src.ship.kcs_specs import POWER_MODEL, SERVICE_LOAD
from src.visualization.plotter import plot_power_schedule

RTA_HOURS = 12.0
DT_HOURS = 0.5
SFOC_PATH = os.path.join(PROJECT_ROOT, "config", "sfoc.json")

def main():
    print("=" * 60)
    print(" [Verification Case 4] Extreme Weather Blackout Test ")
    print("=" * 60)

    cmap = build_cost_map(resolution=0.01)
    
    # [핵심] 풍속 25m/s 이상의 태풍급 날씨 모델링
    EXTREME_WIND_SPEED = 25.0
    weather_fn = create_synthetic_weather(base_wind_speed=EXTREME_WIND_SPEED, base_wind_dir=315.0)
    print(f"\n[Environment] Weather generated with Base Wind Speed = {EXTREME_WIND_SPEED} m/s")
    
    toolbox, pop = setup_ga(pop_size=40, seed=42)
    milp = MILPSolver(sfoc_json_path=SFOC_PATH, n_pwl_segments=5)
    
    toolbox.register("evaluate", evaluate, cmap=cmap, weather_fn=weather_fn, milp=milp)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    toolbox.register("map", pool.map)
    
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    
    print("\n[Integrated GA] Searching resilient optimal route against extreme weather...")
    pop, logbook = algorithms.eaSimple(
        pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=50,
        stats=stats, halloffame=hof, verbose=True
    )
    pool.close()
    
    best_ind = hof[0]
    best_route, best_speed = decode_route(best_ind)
    
    p_req_list = []
    print("\n[Power] Recreating best profile for final output...")
    for k in range(N_SEGMENTS):
        wp_from = best_route[k]
        wp_to = best_route[k+1]
        v = best_speed[k]
        phase = 'cruising'
        if k == 0: phase = 'departure'
        elif k == N_SEGMENTS - 1: phase = 'approach'
        
        v_actual = v * 0.7 if phase != 'cruising' else v
        heading = compute_heading(wp_from, wp_to)
        mid_lat, mid_lon = (wp_from[0] + wp_to[0]) / 2.0, (wp_from[1] + wp_to[1]) / 2.0
        ws, wd = weather_fn(mid_lat, mid_lon)
        
        enc_angle = compute_encounter_angle(heading, wd)
        power_res = compute_P_req(v_actual, ws, enc_angle, POWER_MODEL["a1"], SERVICE_LOAD[phase])
        p_req_list.append(power_res["P_req"])
        
    print(f"  [Power] Min P_req: {min(p_req_list):.2f} MW, Max P_req: {max(p_req_list):.2f} MW")
    
    dt_list = [DT_HOURS] * N_SEGMENTS
    final_res = milp.solve(P_req=p_req_list, dt=dt_list)
    
    out_dir = os.path.join(PROJECT_ROOT, "output", "verification")
    os.makedirs(out_dir, exist_ok=True)
    
    if final_res["feasible"]:
        print("\n  [MILP] System SURVIVED Extreme Weather. ESS covered the peaks successfully!")
        print(f"         Total Fuel: {final_res['total_fuel_kg']:.2f} kg")
        plot_power_schedule(final_res, dt_list, p_req_list, milp.dg_names, num_hours=RTA_HOURS, 
                            save_path=os.path.join(out_dir, "case4_power_schedule_survived.png"))
        print("[Saved] Case 4 survival schedule generated.")
    else:
        print("\n  [MILP] System FAILED (BLACKOUT)! ")
        print("         The combined generator and ESS capacity could not cover the extreme P_req.")
        
        # Even if infeasible, we save the plot if `p_req_list` can be plotted against max capacity.
        # But MILP solver won't have DG/ESS keys for infeasible.
        # So we can just save a bare line chart of P_req vs Capacity threshold 
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10,5))
        times = np.linspace(0, RTA_HOURS, N_SEGMENTS)
        ax.plot(times, p_req_list, 'r-', label="P_req (Extreme)")
        ax.axhline(sum(milp.dg_max), color='k', linestyle='--', label="Total DG Capacity")
        ax.axhline(sum(milp.dg_max) + 15.0, color='b', linestyle='-.', label="With ESS Max Discharging")
        ax.set_title("Blackout Scenario: P_req vs Capacity", fontweight='bold')
        ax.legend()
        plt.savefig(os.path.join(out_dir, "case4_blackout_limit.png"))
        plt.close(fig)
        print("[Saved] Case 4 Blackout limit plot generated.")

if __name__ == "__main__":
    main()
