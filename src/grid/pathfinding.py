"""
Grid-based A* pathfinding utilities for the Busan-Jeju cost map.
"""

from __future__ import annotations

import math

import networkx as nx


def haversine_nm(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Great-circle distance in nautical miles between two (lat, lon) points."""
    radius_nm = 3440.065
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return radius_nm * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def planar_map_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Planar distance on the map grid without great-circle correction."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def _distance(
    p1: tuple[float, float],
    p2: tuple[float, float],
    distance_mode: str,
) -> float:
    if distance_mode == "earth":
        return haversine_nm(p1, p2)
    if distance_mode == "grid":
        return planar_map_distance(p1, p2)
    raise ValueError(f"Unsupported distance_mode: {distance_mode}")


def build_astar_graph(cost_map, distance_mode: str = "earth") -> nx.Graph:
    """Build an 8-connected water-only graph with selectable edge distance metric."""
    if cost_map.cost_grid is None or cost_map.land_mask is None:
        raise RuntimeError("CostMap must be built before pathfinding.")

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

                if next_lat < 0 or next_lat >= cost_map.n_lat:
                    continue
                if next_lon < 0 or next_lon >= cost_map.n_lon:
                    continue
                if cost_map.land_mask[next_lat, next_lon]:
                    continue

                node_to = (next_lat, next_lon)
                point_to = (cost_map.lats[next_lat], cost_map.lons[next_lon])
                graph.add_edge(
                    node_from,
                    node_to,
                    weight=_distance(point_from, point_to, distance_mode=distance_mode),
                )

    return graph


def get_closest_node(
    cost_map,
    point: tuple[float, float],
    graph: nx.Graph,
    distance_mode: str = "earth",
):
    """Return the graph node closest to the given (lat, lon) point."""
    best_node = None
    best_dist = float("inf")

    for node in graph.nodes():
        node_point = (cost_map.lats[node[0]], cost_map.lons[node[1]])
        dist = _distance(node_point, point, distance_mode=distance_mode)
        if dist < best_dist:
            best_dist = dist
            best_node = node

    if best_node is None:
        raise RuntimeError("No water node available for A* pathfinding.")
    return best_node


def compress_path_to_waypoints(
    raw_path: list[tuple[float, float]],
    cost_map,
    num_segments: int,
    distance_mode: str = "earth",
) -> tuple[list[tuple[float, float]], float]:
    """
    Compress a raw A* path into exactly `num_segments + 1` waypoints while preserving
    water-only feasibility between consecutive selected points.
    """
    if len(raw_path) < 2:
        return list(raw_path), 0.0

    cumulative = [0.0]
    for idx in range(1, len(raw_path)):
        cumulative.append(
            cumulative[-1]
            + _distance(raw_path[idx - 1], raw_path[idx], distance_mode=distance_mode)
        )

    total_distance_nm = cumulative[-1]
    n_points = len(raw_path)
    target_segment_nm = total_distance_nm / num_segments if num_segments > 0 else 0.0

    feasible_successors: list[set[int]] = [set() for _ in range(n_points)]
    for start_index in range(n_points - 1):
        max_end_index = n_points - 1
        for end_index in range(start_index + 1, max_end_index + 1):
            if (
                cost_map.segment_cost(
                    raw_path[start_index][0],
                    raw_path[start_index][1],
                    raw_path[end_index][0],
                    raw_path[end_index][1],
                )
                <= 0.0
            ):
                feasible_successors[start_index].add(end_index)

    inf = float("inf")
    dp = [[inf] * n_points for _ in range(num_segments + 1)]
    prev = [[None] * n_points for _ in range(num_segments + 1)]
    dp[0][0] = 0.0

    for segment_count in range(1, num_segments + 1):
        for end_index in range(segment_count, n_points):
            remaining_segments = num_segments - segment_count
            best_cost = inf
            best_prev = None

            for start_index in range(segment_count - 1, end_index):
                if dp[segment_count - 1][start_index] == inf:
                    continue
                if end_index not in feasible_successors[start_index]:
                    continue
                if n_points - 1 - end_index < remaining_segments:
                    continue

                segment_distance = cumulative[end_index] - cumulative[start_index]
                deviation = segment_distance - target_segment_nm
                candidate_cost = dp[segment_count - 1][start_index] + deviation * deviation
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_prev = start_index

            dp[segment_count][end_index] = best_cost
            prev[segment_count][end_index] = best_prev

    selected_indices = [n_points - 1]
    current_index = n_points - 1
    for segment_count in range(num_segments, 0, -1):
        current_index = prev[segment_count][current_index]
        if current_index is None:
            break
        selected_indices.append(current_index)

    if len(selected_indices) != num_segments + 1:
        selected_indices = [0]
        current_index = 0
        for segment_index in range(1, num_segments):
            remaining_segments = num_segments - segment_index
            min_next_index = current_index + 1
            max_next_index = len(raw_path) - 1 - remaining_segments
            if min_next_index > max_next_index:
                min_next_index = max_next_index

            target_distance = total_distance_nm * segment_index / num_segments
            candidates: list[int] = []
            for next_index in range(min_next_index, max_next_index + 1):
                if next_index in feasible_successors[current_index]:
                    candidates.append(next_index)

            if not candidates:
                chosen_index = min_next_index
            else:
                chosen_index = min(candidates, key=lambda idx: abs(cumulative[idx] - target_distance))

            selected_indices.append(chosen_index)
            current_index = chosen_index

        selected_indices.append(n_points - 1)
    else:
        selected_indices.reverse()

    waypoints = [raw_path[index] for index in selected_indices]
    return waypoints, total_distance_nm


def build_astar_route_points(
    cost_map,
    start_point: tuple[float, float],
    goal_point: tuple[float, float],
    num_segments: int,
    distance_mode: str = "earth",
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], float]:
    """Build the raw A* water path and compress it into route waypoints."""
    graph = build_astar_graph(cost_map, distance_mode=distance_mode)
    start_node = get_closest_node(cost_map, start_point, graph, distance_mode=distance_mode)
    goal_node = get_closest_node(cost_map, goal_point, graph, distance_mode=distance_mode)
    goal_coords = (cost_map.lats[goal_node[0]], cost_map.lons[goal_node[1]])

    def heuristic(node_a, node_b):
        del node_b
        point_a = (cost_map.lats[node_a[0]], cost_map.lons[node_a[1]])
        return _distance(point_a, goal_coords, distance_mode=distance_mode)

    path_idx = nx.astar_path(
        graph,
        start_node,
        goal_node,
        heuristic=heuristic,
        weight="weight",
    )
    raw_path = [(cost_map.lats[i], cost_map.lons[j]) for i, j in path_idx]
    waypoints, total_dist_nm = compress_path_to_waypoints(
        raw_path,
        cost_map,
        num_segments,
        distance_mode=distance_mode,
    )
    return raw_path, waypoints, total_dist_nm
