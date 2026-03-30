"""
Verification Case 1

Fixed-speed A* baseline:
- build an A* route on the land-mask grid
- interpolate to the voyage segment count
- run MILP scheduling on the fixed route
"""

from __future__ import annotations

import math
import os
import sys

import networkx as nx
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Verification.common import ensure_output_dir, load_era5_weather, make_milp_solver
from src.grid.cost_map import build_cost_map
from src.grid.no_go_zone import BUSAN_PORT, JEJU_PORT
from src.optimizer.ga_engine import N_SEGMENTS, RTA_HOURS, compute_heading, haversine_nm
from src.resistance.kwon_method import compute_P_req
from src.ship.kcs_specs import POWER_MODEL, SERVICE_LOAD
from src.visualization.plotter import plot_optimal_route, plot_power_schedule, plot_weather_map


DT_HOURS = RTA_HOURS / N_SEGMENTS


def build_astar_graph(cost_map):
    graph = nx.Graph()
    directions = [
        (0, 1, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (-1, 0, 1.0),
        (1, 1, math.sqrt(2.0)),
        (-1, -1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
    ]

    for i in range(cost_map.n_lat):
        for j in range(cost_map.n_lon):
            if cost_map.land_mask[i, j]:
                continue
            graph.add_node((i, j))

    for i in range(cost_map.n_lat):
        for j in range(cost_map.n_lon):
            if cost_map.land_mask[i, j]:
                continue
            for di, dj, weight in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < cost_map.n_lat and 0 <= nj < cost_map.n_lon:
                    if not cost_map.land_mask[ni, nj]:
                        graph.add_edge((i, j), (ni, nj), weight=weight * cost_map.resolution)

    return graph


def get_closest_node(cost_map, point, graph):
    lat, lon = point
    best_node = None
    best_dist = float("inf")
    for node in graph.nodes():
        node_lat = cost_map.lats[node[0]]
        node_lon = cost_map.lons[node[1]]
        dist = math.hypot(node_lat - lat, node_lon - lon)
        if dist < best_dist:
            best_dist = dist
            best_node = node
    return best_node


def interpolate_path(path_coords, num_segments):
    cumulative = [0.0]
    for idx in range(1, len(path_coords)):
        cumulative.append(cumulative[-1] + haversine_nm(path_coords[idx - 1], path_coords[idx]))

    target = np.linspace(0.0, cumulative[-1], num_segments + 1)
    lats = [point[0] for point in path_coords]
    lons = [point[1] for point in path_coords]
    interp_lats = np.interp(target, cumulative, lats)
    interp_lons = np.interp(target, cumulative, lons)
    return list(zip(interp_lats, interp_lons)), cumulative[-1]


def build_fixed_route(waypoints, base_speed_knots):
    route = {
        "waypoints": waypoints,
        "speeds": [],
        "headings": [],
        "delta_headings": [],
        "dt": [DT_HOURS] * N_SEGMENTS,
        "distances_nm": [],
        "valid": True,
        "valid_speed": True,
        "valid_heading": True,
        "last_speed": 0.0,
        "final_heading_delta": 0.0,
    }

    prev_heading = compute_heading(BUSAN_PORT, JEJU_PORT)
    for idx in range(N_SEGMENTS):
        wp_from = waypoints[idx]
        wp_to = waypoints[idx + 1]
        heading = compute_heading(wp_from, wp_to)
        if idx == 0 or idx == N_SEGMENTS - 1:
            speed = base_speed_knots * 0.7
        else:
            speed = base_speed_knots
        delta = ((heading - prev_heading + 180.0) % 360.0) - 180.0

        route["speeds"].append(speed)
        route["headings"].append(heading)
        route["delta_headings"].append(delta)
        route["distances_nm"].append(haversine_nm(wp_from, wp_to))
        prev_heading = heading

    route["last_speed"] = route["speeds"][-1]
    route["final_heading_delta"] = route["delta_headings"][-1]
    return route


def build_power_profile(route, weather_fn):
    p_req_list = []
    for idx in range(N_SEGMENTS):
        wp_from = route["waypoints"][idx]
        wp_to = route["waypoints"][idx + 1]
        heading = route["headings"][idx]
        speed = route["speeds"][idx]

        mid_lat = (wp_from[0] + wp_to[0]) / 2.0
        mid_lon = (wp_from[1] + wp_to[1]) / 2.0
        wind_speed, wind_dir = weather_fn(mid_lat, mid_lon)
        encounter = (wind_dir - heading + 180.0) % 360.0
        if encounter > 180.0:
            encounter = 360.0 - encounter

        if idx == 0:
            phase = "departure"
        elif idx == N_SEGMENTS - 1:
            phase = "approach"
        else:
            phase = "cruising"

        power = compute_P_req(
            v_ship_knots=speed,
            v_wind_ms=wind_speed,
            encounter_angle_deg=encounter,
            a1=POWER_MODEL["a1"],
            P_service=SERVICE_LOAD[phase],
        )
        p_req_list.append(power["P_req"])

    return p_req_list


def main():
    print("=" * 60)
    print(" [Verification Case 1] A* Fixed Route + MILP Scheduling ")
    print("=" * 60)

    out_dir = ensure_output_dir("case1")
    weather_loader, weather_fn = load_era5_weather(time_index=0)
    try:
        cost_map = build_cost_map(resolution=0.1)
        graph = build_astar_graph(cost_map)
        start_node = get_closest_node(cost_map, BUSAN_PORT, graph)
        goal_node = get_closest_node(cost_map, JEJU_PORT, graph)

        def heuristic(node_a, node_b):
            return math.hypot(node_b[0] - node_a[0], node_b[1] - node_a[1]) * cost_map.resolution

        path_idx = nx.astar_path(graph, start_node, goal_node, heuristic=heuristic, weight="weight")
        raw_path = [(cost_map.lats[i], cost_map.lons[j]) for i, j in path_idx]
        waypoints, total_dist_nm = interpolate_path(raw_path, N_SEGMENTS)
        base_speed = total_dist_nm / (DT_HOURS * (N_SEGMENTS - 0.6))

        route = build_fixed_route(waypoints, base_speed)
        p_req_list = build_power_profile(route, weather_fn)

        milp = make_milp_solver()
        milp_result = milp.solve(P_req=p_req_list, dt=route["dt"], initial_SOC=0.7, msg=False)

        print(f"  Path points: {len(raw_path)}")
        print(f"  Interpolated distance: {total_dist_nm:.2f} nm")
        print(f"  Base speed: {base_speed:.2f} kts")

        if milp_result["feasible"]:
            print(f"  MILP fuel: {milp_result['total_fuel_kg']:.2f} kg")
        else:
            print("  MILP infeasible")

        plot_optimal_route(route, cost_map, save_dir=out_dir)
        plot_weather_map(weather_fn, cost_map, route_wps=route["waypoints"], save_dir=out_dir)
        if milp_result["feasible"]:
            plot_power_schedule(milp_result, save_dir=out_dir)
    finally:
        weather_loader.close()


if __name__ == "__main__":
    main()
