"""
Verification Case 3: Integrated Optimization (GA + MILP)
========================================================
가장 핵심이 되는 제안 모델입니다.
GA의 적합도(Fitness) 평가 단계마다 MILP 솔버를 호출하여, 
순수한 기계적/유체 역학적 에너지가 아닌 실제 '연료 소모량 최소화'를 
목표로 경로와 속도를 동시에 최적화합니다.
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
from src.visualization.plotter import plot_power_schedule, plot_optimal_route, plot_convergence

RTA_HOURS = 12.0
DT_HOURS = 0.5
SFOC_PATH = os.path.join(PROJECT_ROOT, "config", "sfoc.json")

def main():
    print("=" * 60)
    print(" [Verification Case 3] GA + MILP Integrated Optimization ")
    print("=" * 60)

    # 1. 환경 및 날씨 로드 (강한 풍속)
    cmap = build_cost_map(resolution=0.01)
    weather_fn = create_synthetic_weather(base_wind_speed=15.0, base_wind_dir=315.0)
    
    # 2. GA 셋업
    toolbox, pop = setup_ga(pop_size=40, seed=42)
    
    # 3. MILP 인스턴스 (Fitness Evaluator 내부에 전달)
    milp = MILPSolver(sfoc_json_path=SFOC_PATH, n_pwl_segments=5)
    
    # Fitness 평가 함수 바인딩 (Integrated)
    toolbox.register("evaluate", evaluate, cmap=cmap, weather_fn=weather_fn, milp=milp)
    
    # 병렬 처리
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    toolbox.register("map", pool.map)
    
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    
    print("\n[Integrated GA] Running Evolutionary Algorithm with embedded MILP...")
    t0 = time.time()
    pop, logbook = algorithms.eaSimple(
        pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=50,
        stats=stats, halloffame=hof, verbose=True
    )
    pool.close()
    
    print(f"\n[Integrated GA] Finished in {time.time() - t0:.1f} seconds.")
    
    # 4. 결과 분석
    best_ind = hof[0]
    best_route, best_speed = decode_route(best_ind)
    
    # 최우수 객체 P_req 재생산 및 최종 MILP 결과 획득
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
        print(f"  [Output] Optimal Fuel Consumption: {final_res['total_fuel_kg']:.2f} kg")
        plot_power_schedule(final_res, dt_list, p_req_list, milp.dg_names, num_hours=RTA_HOURS, 
                            save_path=os.path.join(out_dir, "case3_power_schedule.png"))
        plot_optimal_route(best_route, save_path=os.path.join(out_dir, "case3_route.png"))
        plot_convergence(logbook, save_path=os.path.join(out_dir, "case3_convergence.png"))
        print("[Saved] Case 3 Output files generated.")
    else:
        print("  [ERROR] Final Best Individual is Infeasible.")

if __name__ == "__main__":
    main()
