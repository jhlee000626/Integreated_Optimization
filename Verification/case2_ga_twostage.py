"""
Verification Case 2: 2-Stage Optimization (GA standalone -> MILP 분리)
=====================================================================
GA가 발전기 제원을 모른 체 오직 요구 에너지 합계($\sum P_{req} \times dt$)만을 
최소화하는 경로/속도 세트를 찾은 뒤, 이렇게 결정된 $P_{req}$ 프로파일을 바탕으로 
단 한 번의 MILP 스케줄링을 수행하여 최종 연료 소모량을 산출합니다.
"""

import os
import sys
import numpy as np
import multiprocessing

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from deap import base, creator, tools, algorithms

from src.grid.cost_map import build_cost_map
from src.optimizer.milp_solver import MILPSolver
from src.weather.era5_loader import create_synthetic_weather
from src.optimizer.ga_engine import setup_ga, decode_route, N_SEGMENTS, N_WAYPOINTS
from src.optimizer.fitness import compute_heading, compute_encounter_angle, haversine_nm
from src.resistance.kwon_method import compute_P_req
from src.ship.kcs_specs import POWER_MODEL, SERVICE_LOAD
from src.visualization.plotter import plot_power_schedule, plot_optimal_route, plot_convergence

# ─── 런타임 설정 ───
RTA_HOURS = 12.0
DT_HOURS = 0.5
SFOC_PATH = os.path.join(PROJECT_ROOT, "config", "sfoc.json")

def evaluate_energy_only(individual, weather_fn):
    """
    MILP 호출 없이 오직 총 요구 에너지(P_req * dt)만 계산하는 Fitness 함수.
    육지 침범 시 치명적인 페널티를 부과합니다.
    """
    route, speed_profile = decode_route(individual)
    
    # 1. 제약 조건 (거리 및 시간)
    total_dist = 0.0
    for i in range(len(route)-1):
        total_dist += haversine_nm(route[i], route[i+1])
        
    avg_speed = np.mean(speed_profile)
    est_time = total_dist / max(avg_speed, 1.0)
    
    # RTA 위반 페널티 (시간 오차)
    time_penalty = abs(est_time - RTA_HOURS) * 1000.0
    
    raw_p_reqs = []
    land_penalty = 0.0
    
    cmap = build_cost_map(resolution=0.01) # Fitness 평가용
    
    for k in range(N_SEGMENTS):
        wp_from = route[k]
        wp_to = route[k+1]
        
        # Segment 육지 페널티
        seg_cost = cmap.segment_cost(wp_from[0], wp_from[1], wp_to[0], wp_to[1])
        if seg_cost > 0:
            land_penalty += seg_cost * 500.0
            
        v = speed_profile[k]
        phase = 'cruising'
        if k == 0: phase = 'departure'
        elif k == N_SEGMENTS - 1: phase = 'approach'
        
        v_actual = v * 0.7 if phase != 'cruising' else v
        heading = compute_heading(wp_from, wp_to)
        
        mid_lat = (wp_from[0] + wp_to[0]) / 2.0
        mid_lon = (wp_from[1] + wp_to[1]) / 2.0
        ws, wd = weather_fn(mid_lat, mid_lon)
        
        enc_angle = compute_encounter_angle(heading, wd)
        power_res = compute_P_req(v_actual, ws, enc_angle, POWER_MODEL["a1"], SERVICE_LOAD[phase])
        
        raw_p_reqs.append(power_res["P_req"])
        
    # 총 에너지(MW*h) 
    total_energy = sum(p * DT_HOURS for p in raw_p_reqs)
    
    fitness_val = total_energy + time_penalty + land_penalty
    return (fitness_val,)

def main():
    print("=" * 60)
    print(" [Verification Case 2] GA Standalone -> MILP (2-Stage) ")
    print("=" * 60)
    
    weather_fn = create_synthetic_weather(base_wind_speed=15.0, base_wind_dir=315.0)
    
    # GA 환경 셋업
    toolbox, pop = setup_ga(pop_size=40, seed=42)
    
    # 기존 Fitness 함수 덮어쓰기 (Energy Minimization)
    toolbox.register("evaluate", evaluate_energy_only, weather_fn=weather_fn)
    
    # 병렬 처리
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
    toolbox.register("map", pool.map)
    
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    
    print("\n[GA] Running Evolutionary Algorithm (Energy Optimization)...")
    pop, logbook = algorithms.eaSimple(
        pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=50, 
        stats=stats, halloffame=hof, verbose=True
    )
    
    pool.close()
    
    best_ind = hof[0]
    best_route, best_speed = decode_route(best_ind)
    
    # 최우수 객체의 P_req 프로파일 추출
    p_req_list = []
    print("\n[Power] Re-evaluating best route for P_req profile...")
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
    
    # MILP "1회" 실행
    print("\n[MILP] Running Unit Commitment on Best Profile...")
    milp = MILPSolver(sfoc_json_path=SFOC_PATH, n_pwl_segments=5)
    dt_list = [DT_HOURS] * N_SEGMENTS
    result = milp.solve(P_req=p_req_list, dt=dt_list)
    
    out_dir = os.path.join(PROJECT_ROOT, "output", "verification")
    os.makedirs(out_dir, exist_ok=True)
    
    if result["feasible"]:
        print(f"  [MILP] Success! Total Fuel: {result['total_fuel_kg']:.2f} kg")
        plot_power_schedule(result, dt_list, p_req_list, milp.dg_names, 
                            num_hours=RTA_HOURS, 
                            save_path=os.path.join(out_dir, "case2_power_schedule.png"))
        plot_optimal_route(best_route, 
                           save_path=os.path.join(out_dir, "case2_route.png"))
        plot_convergence(logbook, 
                         save_path=os.path.join(out_dir, "case2_convergence.png"))
        print("[Saved] Case 2 Output files generated.")
    else:
        print("  [MILP] INFEASIBLE! Failed to schedule power for GA requests.")

if __name__ == "__main__":
    main()
