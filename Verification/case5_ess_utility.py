"""
Verification Case 5: ESS Utility (ESS 유무에 따른 DG 최적 효율 운전 방어)
========================================================================
ESS 모듈의 핵심 가치인 'Load Leveling / Energy Shifting' 기능을 검증합니다.
배터리 용량을 0으로 강제 설정한 채로 통합 최적화 모델(Case 3 동일 조건)을 구동하면,
발전기가 P_req의 자잘한 변동에 일일이 끌려다니며 최적 SFOC(Sweet Spot) 구간을 
이탈하여 연료 효율이 크게 악화됨을 시각적/수치적으로 증명합니다.
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
from src.visualization.plotter import plot_power_schedule, plot_optimal_route, plot_convergence

# ---------- [핵심] 스펙 강제 오버라이드 ----------
import src.ship.kcs_specs as kcs_specs

# ESS 작동 완전 정지
kcs_specs.ESS_SPECS["E_cap"] = 0.0
kcs_specs.ESS_SPECS["P_c_max"] = 0.0
kcs_specs.ESS_SPECS["P_dc_max"] = 0.0

POWER_MODEL = kcs_specs.POWER_MODEL
SERVICE_LOAD = kcs_specs.SERVICE_LOAD
# -----------------------------------------------

RTA_HOURS = 12.0
DT_HOURS = 0.5
SFOC_PATH = os.path.join(PROJECT_ROOT, "config", "sfoc.json")

def main():
    print("=" * 60)
    print(" [Verification Case 5] ESS Utility & Energy Shifting ")
    print("=" * 60)
    print("  => Overriding ESS capacity to 0.0 MWh.")

    cmap = build_cost_map(resolution=0.01)
    weather_fn = create_synthetic_weather(base_wind_speed=10.0, base_wind_dir=315.0)
    
    toolbox, pop = setup_ga(pop_size=40, seed=42)
    milp = MILPSolver(sfoc_json_path=SFOC_PATH, n_pwl_segments=5)
    
    toolbox.register("evaluate", evaluate, cmap=cmap, weather_fn=weather_fn, milp=milp)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    toolbox.register("map", pool.map)
    
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    
    print("\n[Integrated GA without ESS] Running Optimization...")
    t0 = time.time()
    pop, logbook = algorithms.eaSimple(
        pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=50,
        stats=stats, halloffame=hof, verbose=True
    )
    pool.close()
    
    print(f"\n[Finished] Optimization complete in {time.time() - t0:.1f} seconds.")
    
    best_ind = hof[0]
    best_route, best_speed = decode_route(best_ind)
    
    p_req_list = []
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
        
    dt_list = [DT_HOURS] * N_SEGMENTS
    final_res = milp.solve(P_req=p_req_list, dt=dt_list)
    
    out_dir = os.path.join(PROJECT_ROOT, "output", "verification")
    os.makedirs(out_dir, exist_ok=True)
    
    if final_res["feasible"]:
        print(f"  [Output] DG Only Fuel Consumption: {final_res['total_fuel_kg']:.2f} kg")
        plot_power_schedule(final_res, dt_list, p_req_list, milp.dg_names, num_hours=RTA_HOURS, 
                            save_path=os.path.join(out_dir, "case5_power_schedule_no_ess.png"))
        plot_optimal_route(best_route, save_path=os.path.join(out_dir, "case5_route.png"))
        plot_convergence(logbook, save_path=os.path.join(out_dir, "case5_convergence.png"))
        print("[Saved] Case 5 Output files generated.")
        print("\n=> COMPARE THIS RESULT (case5_power_schedule_no_ess.png) WITH CASE 3 (Base)")
    else:
        print("  [ERROR] Infeasible without ESS.")

if __name__ == "__main__":
    main()
