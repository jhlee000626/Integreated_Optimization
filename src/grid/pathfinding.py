"""
Grid-based pathfinding helpers shared by verification baselines and GA seeding.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np


def haversine_nm(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    radius_nm = 3440.065
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return radius_nm * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def build_astar_graph(cost_map):
    graph = nx.Graph()
    directions = [
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, 0),
        (1, 1),
        (-1, -1),
        (1, -1),
        (-1, 1),
    ]

    for lat_index in range(cost_map.n_lat):
        for lon_index in range(cost_map.n_lon):
            if cost_map.land_mask[lat_index, lon_index]:
                continue
            graph.add_node((lat_index, lon_index))

    for lat_index in range(cost_map.n_lat):
        for lon_index in range(cost_map.n_lon):
            if cost_map.land_mask[lat_index, lon_index]:
                continue
            node_from = (lat_index, lon_index)
            point_from = (cost_map.lats[lat_index], cost_map.lons[lon_index])
            for delta_lat, delta_lon in directions:
                next_lat = lat_index + delta_lat
                next_lon = lon_index + delta_lon
                if 0 <= next_lat < cost_map.n_lat and 0 <= next_lon < cost_map.n_lon:
                    if cost_map.land_mask[next_lat, next_lon]:
                        continue
                    node_to = (next_lat, next_lon)
                    point_to = (cost_map.lats[next_lat], cost_map.lons[next_lon])
                    graph.add_edge(node_from, node_to, weight=haversine_nm(point_from, point_to))

    return graph


def get_closest_node(cost_map, point, graph):
    best_node = None
    best_dist = float("inf")
    for node in graph.nodes():
        node_point = (cost_map.lats[node[0]], cost_map.lons[node[1]])
        dist = haversine_nm(node_point, point)
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


def compress_path_to_waypoints(raw_path, cost_map, num_segments: int):
    if len(raw_path) <= num_segments + 1:
        return raw_path, sum(
            haversine_nm(raw_path[idx - 1], raw_path[idx]) for idx in range(1, len(raw_path))
        )

    cumulative = [0.0]
    for idx in range(1, len(raw_path)):
        cumulative.append(cumulative[-1] + haversine_nm(raw_path[idx - 1], raw_path[idx]))

    selected_indices = [0]
    current_index = 0
    total_distance_nm = cumulative[-1]

    for segment_index in range(1, num_segments):
        remaining_segments = num_segments - segment_index
        min_next_index = current_index + 1
        max_next_index = len(raw_path) - 1 - remaining_segments
        if min_next_index > max_next_index:
            min_next_index = max_next_index

        target_distance = total_distance_nm * (segment_index / num_segments)
        candidates = []
        for next_index in range(min_next_index, max_next_index + 1):
            if cost_map.segment_cost(
                raw_path[current_index][0],
                raw_path[current_index][1],
                raw_path[next_index][0],
                raw_path[next_index][1],
            ) <= 0.0:
                candidates.append(next_index)

        if not candidates:
            chosen_index = min_next_index
        else:
            chosen_index = min(candidates, key=lambda idx: abs(cumulative[idx] - target_distance))

        selected_indices.append(chosen_index)
        current_index = chosen_index

    selected_indices.append(len(raw_path) - 1)
    waypoints = [raw_path[index] for index in selected_indices]
    return waypoints, total_distance_nm


def build_astar_route_points(cost_map, start_point, goal_point, num_segments: int):
    graph = build_astar_graph(cost_map)
    start_node = get_closest_node(cost_map, start_point, graph)
    goal_node = get_closest_node(cost_map, goal_point, graph)
    goal_coords = (cost_map.lats[goal_node[0]], cost_map.lons[goal_node[1]])

    def heuristic(node_a, node_b):
        del node_b
        point_a = (cost_map.lats[node_a[0]], cost_map.lons[node_a[1]])
        return haversine_nm(point_a, goal_coords)

    path_idx = nx.astar_path(graph, start_node, goal_node, heuristic=heuristic, weight="weight")
    raw_path = [(cost_map.lats[i], cost_map.lons[j]) for i, j in path_idx]
    waypoints, total_dist_nm = compress_path_to_waypoints(raw_path, cost_map, num_segments)
    return raw_path, waypoints, total_dist_nm
