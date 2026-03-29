"""
Verification Case 1: Baseline (A* Router + Fixed Speed + MILP)
==============================================================
최단거리(A*) 주행 시 고정 속도로 운항하는 기본 모드에서의 파워 스케줄링 검증.
"""

import sys
import os
import time
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.grid.no_go_zone import BUSAN_PORT, JEJU_PORT
from src.grid.cost_map import build_cost_map
from src.optimizer.milp_solver import MILPSolver
from src.weather.era5_loader import ERA5Loader, create_synthetic_weather
from src.resistance.kwon_method import compute_P_req
from src.optimizer.fitness import compute_heading, compute_encounter_angle, haversine_nm
from src.ship.kcs_specs import POWER_MODEL, SERVICE_LOAD
from src.visualization.plotter import plot_power_schedule

# ─── 상수 설정 ───
RTA_HOURS = 12.0
DT_HOURS = 0.5
N_SEGMENTS = int(RTA_HOURS / DT_HOURS)
DEPARTURE_TIME = datetime(2024, 1, 1, 12, 0)
SFOC_PATH = os.path.join(PROJECT_ROOT, "config", "sfoc.json")

def build_astar_graph(cmap):
    print("[A*] Building graph from CostMap...")
    G = nx.Graph()
    n_lat, n_lon = cmap.n_lat, cmap.n_lon
    
    # 노드 추가 (육지가 아닌 곳만)
    for i in range(n_lat):
        for j in range(n_lon):
            if not cmap.land_mask[i, j]:
                G.add_node((i, j))
                
    # 엣지 추가 (8방향)
    dirs = [
        (0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),
        (1, 1, 1.414), (-1, -1, 1.414), (1, -1, 1.414), (-1, 1, 1.414)
    ]
    
    for i in range(n_lat):
        for j in range(n_lon):
            if cmap.land_mask[i, j]: continue
            for di, dj, w in dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < n_lat and 0 <= nj < n_lon:
                    if not cmap.land_mask[ni, nj]:
                        G.add_edge((i, j), (ni, nj), weight=w * cmap.resolution)
                        
    print(f"[A*] Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def get_closest_node(cmap, point, G):
    lat, lon = point
    best_node = None
    min_d = float('inf')
    
    for (i, j) in G.nodes():
        n_lat, n_lon = cmap.lats[i], cmap.lons[j]
        d = math.hypot(n_lat - lat, n_lon - lon)
        if d < min_d:
            min_d = d
            best_node = (i, j)
            
    return best_node

def interpolate_path(path_coords, num_segments):
    """주어진 경로를 동일한 거리 간격의 N개 세그먼트(N+1개 웨이포인트)로 분할"""
    # 누적 거리 계산
    cum_dist = [0.0]
    for i in range(1, len(path_coords)):
        d = haversine_nm(path_coords[i-1], path_coords[i])
        cum_dist.append(cum_dist[-1] + d)
        
    total_dist = cum_dist[-1]
    target_dists = np.linspace(0, total_dist, num_segments + 1)
    
    lats = [p[0] for p in path_coords]
    lons = [p[1] for p in path_coords]
    
    interp_lats = np.interp(target_dists, cum_dist, lats)
    interp_lons = np.interp(target_dists, cum_dist, lons)
    
    return list(zip(interp_lats, interp_lons)), total_dist

def main():
    log_file = open("case1_internal.log", "w", encoding="utf-8")
    def log(msg):
        log_file.write(msg + "\n")
        log_file.flush()
        
    try:
        log("=" * 60)
        log(" [Verification Case 1] A* Baseline & Fixed Speed ")
        log("=" * 60)
    
        # 1. 환경 준비
        log("Building CostMap...")
        cmap = build_cost_map(resolution=0.05) # A* Search space 대폭 축소 (5.5km 격자, 9천 노드)
        G = build_astar_graph(cmap)
        
        start_node = get_closest_node(cmap, BUSAN_PORT, G)
        goal_node = get_closest_node(cmap, JEJU_PORT, G)
        
        # 2. A* 경로 탐색
        log("\n[A*] Searching for shortest path...")
        def heuristic(n1, n2):
            # 유클리디안 거리 베이스에 101% 가중치(Greedy heuristic)를 주어 노드 확장 방어
            return math.hypot(n2[0] - n1[0], n2[1] - n1[1]) * cmap.resolution * 1.01
            
            
        path_idx = nx.astar_path(G, start_node, goal_node, heuristic=heuristic, weight='weight')
        raw_path_coords = [(cmap.lats[i], cmap.lons[j]) for (i, j) in path_idx]
        log(f"  [A*] Found path with {len(raw_path_coords)} points.")
        
        # 3. 경로 보간 및 속도 할당
        log(f"\n[Path] Interpolating to {N_SEGMENTS} segments (dt={DT_HOURS}h)...")
        waypoints, total_dist_nm = interpolate_path(raw_path_coords, N_SEGMENTS)
        
        fixed_base_speed = total_dist_nm / (DT_HOURS * (N_SEGMENTS - 0.6))
        log(f"  [Path] Total Distance: {total_dist_nm:.2f} nm, Fixed Base Speed: {fixed_base_speed:.2f} knots")
    
        # 4. 기상 정보 및 P_req 계산
        weather_fn = create_synthetic_weather(base_wind_speed=15.0, base_wind_dir=315.0) # 강한 풍속
        
        p_req_list = []
        log("\n[Power] Computing P_req for each segment...")
        
        for k in range(N_SEGMENTS):
            wp_from = waypoints[k]
            wp_to = waypoints[k+1]
            
            phase = "cruising"
            if k == 0: phase = "departure"
            elif k == N_SEGMENTS - 1: phase = "approach"
            
            v_actual = fixed_base_speed * 0.7 if phase != "cruising" else fixed_base_speed
            heading = compute_heading(wp_from, wp_to)
            
            mid_lat = (wp_from[0] + wp_to[0]) / 2.0
            mid_lon = (wp_from[1] + wp_to[1]) / 2.0
            ws, wd = weather_fn(mid_lat, mid_lon)
            
            enc_angle = compute_encounter_angle(heading, wd)
            power_res = compute_P_req(v_actual, ws, enc_angle, POWER_MODEL["a1"], SERVICE_LOAD[phase])
            
            p_req_list.append(power_res["P_req"])
            
        log(f"  [Power] Min P_req: {min(p_req_list):.2f} MW, Max P_req: {max(p_req_list):.2f} MW")
    
        # 5. MILP 실행
        log("\n[MILP] Running Unit Commitment & Economic Dispatch...")
        milp = MILPSolver(sfoc_json_path=SFOC_PATH, n_pwl_segments=5)
        
        dt_list = [DT_HOURS] * N_SEGMENTS
        result = milp.solve(P_req=p_req_list, dt=dt_list)
        
        if result["feasible"]:
            log(f"  [MILP] Success! Total Fuel: {result['total_fuel_kg']:.2f} kg")
            out_dir = os.path.join(PROJECT_ROOT, "output", "verification")
            os.makedirs(out_dir, exist_ok=True)
            img_path = os.path.join(out_dir, "case1_power_schedule.png")
            plot_power_schedule(result, dt_list, p_req_list, milp.dg_names, num_hours=RTA_HOURS, save_path=img_path)
            log(f"Saved: {img_path}")
        else:
            log("  [MILP] INFEASIBLE! Failed to schedule power for demands.")
            
    except Exception as e:
        import traceback
        log("ERROR:\n" + traceback.format_exc())
    finally:
        log_file.close()

if __name__ == "__main__":
    main()
