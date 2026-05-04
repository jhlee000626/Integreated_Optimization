"""
Grid-based A* pathfinding utilities for the Busan-Shanghai cost map.
"""

from __future__ import annotations

import heapq
import math

import numpy as np


DEFAULT_ASTAR_CONNECTIVITY = 32
ASTAR_ALGORITHM_VERSION = "queue_v1"
SUPPORTED_ASTAR_CONNECTIVITY = (8, 16, 32)


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


def _build_direction_offsets(stride: int, connectivity: int) -> list[tuple[int, int]]:
    """Return neighbor offsets for the requested A* connectivity."""
    if connectivity not in SUPPORTED_ASTAR_CONNECTIVITY:
        raise ValueError(
            f"Unsupported A* connectivity: {connectivity}. "
            f"Expected one of {SUPPORTED_ASTAR_CONNECTIVITY}."
        )

    directions = [
        (0, stride), (stride, 0), (0, -stride), (-stride, 0),
        (stride, stride), (-stride, -stride), (stride, -stride), (-stride, stride),
    ]

    if connectivity >= 16:
        directions.extend(
            [
                (stride, 2 * stride), (2 * stride, stride), (-stride, 2 * stride), (-2 * stride, stride),
                (stride, -2 * stride), (2 * stride, -stride), (-stride, -2 * stride), (-2 * stride, -stride),
            ]
        )

    if connectivity >= 32:
        directions.extend(
            [
                (stride, 3 * stride), (3 * stride, stride), (-stride, 3 * stride), (-3 * stride, stride),
                (stride, -3 * stride), (3 * stride, -stride), (-stride, -3 * stride), (-3 * stride, -stride),
                (2 * stride, 3 * stride), (3 * stride, 2 * stride), (-2 * stride, 3 * stride), (-3 * stride, 2 * stride),
                (2 * stride, -3 * stride), (3 * stride, -2 * stride), (-2 * stride, -3 * stride), (-3 * stride, -2 * stride),
            ]
        )

    return directions


def build_astar_graph(
    cost_map,
    distance_mode: str = "grid",
    search_resolution: float = 0.01,
    connectivity: int = DEFAULT_ASTAR_CONNECTIVITY,
):
    """Build a water-only graph with selectable edge distance metric and neighbor connectivity."""
    import networkx as nx

    if cost_map.cost_grid is None or cost_map.land_mask is None:
        raise RuntimeError("CostMap must be built before pathfinding.")

    stride = max(1, int(round(search_resolution / cost_map.resolution)))
    directions = _build_direction_offsets(stride=stride, connectivity=connectivity)

    graph = nx.Graph()
    # 16방향(8-connected + Knight moves) 탐색으로 계단 현상 완화

    for lat_index in range(0, cost_map.n_lat, stride):
        for lon_index in range(0, cost_map.n_lon, stride):
            if cost_map.land_mask[lat_index, lon_index]:
                continue
            graph.add_node((lat_index, lon_index))

    for lat_index in range(0, cost_map.n_lat, stride):
        for lon_index in range(0, cost_map.n_lon, stride):
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
                
                # Knight moves처럼 건너뛰는 간선이나 stride > 1일 때는 간선 전체가 육지를 가로지르는지 확인
                if stride > 1 or abs(delta_lat) > stride or abs(delta_lon) > stride:
                    if cost_map.segment_cost(point_from[0], point_from[1], point_to[0], point_to[1]) > 0.0:
                        continue

                graph.add_edge(
                    node_from,
                    node_to,
                    weight=_distance(point_from, point_to, distance_mode=distance_mode),
                )

    return graph


def get_closest_node(
    cost_map,
    point: tuple[float, float],
    graph,
    distance_mode: str = "grid",
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


def _closest_water_node(
    cost_map,
    point: tuple[float, float],
    stride: int,
    distance_mode: str,
) -> tuple[int, int]:
    """Return the closest stride-aligned water grid node without building a graph."""
    lat_indices = np.arange(0, cost_map.n_lat, stride, dtype=np.int32)
    lon_indices = np.arange(0, cost_map.n_lon, stride, dtype=np.int32)
    water_mask = ~cost_map.land_mask[np.ix_(lat_indices, lon_indices)]
    candidate_rows, candidate_cols = np.where(water_mask)
    if len(candidate_rows) == 0:
        raise RuntimeError("No water node available for A* pathfinding.")

    candidate_i = lat_indices[candidate_rows]
    candidate_j = lon_indices[candidate_cols]
    if distance_mode == "grid":
        d_lat = cost_map.lats[candidate_i] - point[0]
        d_lon = cost_map.lons[candidate_j] - point[1]
        best = int(np.argmin(d_lat * d_lat + d_lon * d_lon))
        return int(candidate_i[best]), int(candidate_j[best])

    best_node = None
    best_dist = float("inf")
    for i, j in zip(candidate_i, candidate_j):
        node_point = (float(cost_map.lats[i]), float(cost_map.lons[j]))
        dist = _distance(node_point, point, distance_mode=distance_mode)
        if dist < best_dist:
            best_dist = dist
            best_node = (int(i), int(j))
    if best_node is None:
        raise RuntimeError("No water node available for A* pathfinding.")
    return best_node


def _sample_offsets_for_direction(delta_lat: int, delta_lon: int, stride: int) -> list[tuple[int, int]]:
    """Grid cells sampled along a candidate edge for land-mask crossing checks."""
    if stride <= 1 and abs(delta_lat) <= stride and abs(delta_lon) <= stride:
        return [(delta_lat, delta_lon)]

    dist_cells = math.hypot(delta_lat, delta_lon)
    n_samples = max(2, int(dist_cells / 0.5) + 1)
    offsets: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for sample_idx in range(n_samples + 1):
        frac = sample_idx / n_samples
        offset = (int(round(frac * delta_lat)), int(round(frac * delta_lon)))
        if offset not in seen:
            offsets.append(offset)
            seen.add(offset)
    if (delta_lat, delta_lon) not in seen:
        offsets.append((delta_lat, delta_lon))
    return offsets


def _edge_is_water(cost_map, start_i: int, start_j: int, offsets: list[tuple[int, int]]) -> bool:
    for offset_i, offset_j in offsets:
        sample_i = start_i + offset_i
        sample_j = start_j + offset_j
        if sample_i < 0 or sample_i >= cost_map.n_lat:
            return False
        if sample_j < 0 or sample_j >= cost_map.n_lon:
            return False
        if cost_map.land_mask[sample_i, sample_j]:
            return False
    return True


def _direction_step_cost(
    cost_map,
    start_i: int,
    start_j: int,
    delta_i: int,
    delta_j: int,
    distance_mode: str,
) -> float:
    if distance_mode == "grid":
        return math.hypot(
            float(cost_map.lats[start_i + delta_i] - cost_map.lats[start_i]),
            float(cost_map.lons[start_j + delta_j] - cost_map.lons[start_j]),
        )
    point_from = (float(cost_map.lats[start_i]), float(cost_map.lons[start_j]))
    point_to = (float(cost_map.lats[start_i + delta_i]), float(cost_map.lons[start_j + delta_j]))
    return _distance(point_from, point_to, distance_mode=distance_mode)


def _queue_astar_path_indices(
    cost_map,
    start_point: tuple[float, float],
    goal_point: tuple[float, float],
    distance_mode: str = "grid",
    search_resolution: float = 0.01,
    connectivity: int = DEFAULT_ASTAR_CONNECTIVITY,
) -> list[tuple[int, int]]:
    """Run A* directly from a heapq priority queue, without materializing a graph."""
    if cost_map.cost_grid is None or cost_map.land_mask is None:
        raise RuntimeError("CostMap must be built before pathfinding.")

    stride = max(1, int(round(search_resolution / cost_map.resolution)))
    directions = _build_direction_offsets(stride=stride, connectivity=connectivity)
    direction_offsets = [
        (delta_i, delta_j, _sample_offsets_for_direction(delta_i, delta_j, stride))
        for delta_i, delta_j in directions
    ]

    start_node = _closest_water_node(cost_map, start_point, stride=stride, distance_mode=distance_mode)
    goal_node = _closest_water_node(cost_map, goal_point, stride=stride, distance_mode=distance_mode)
    goal_coords = (float(cost_map.lats[goal_node[0]]), float(cost_map.lons[goal_node[1]]))

    def heuristic(node: tuple[int, int]) -> float:
        point = (float(cost_map.lats[node[0]]), float(cost_map.lons[node[1]]))
        return _distance(point, goal_coords, distance_mode=distance_mode)

    g_score = np.full((cost_map.n_lat, cost_map.n_lon), np.inf, dtype=float)
    prev_i = np.full((cost_map.n_lat, cost_map.n_lon), -1, dtype=np.int32)
    prev_j = np.full((cost_map.n_lat, cost_map.n_lon), -1, dtype=np.int32)
    closed = np.zeros((cost_map.n_lat, cost_map.n_lon), dtype=bool)

    g_score[start_node] = 0.0
    heap_counter = 0
    open_heap: list[tuple[float, int, tuple[int, int]]] = [(heuristic(start_node), heap_counter, start_node)]

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        current_i, current_j = current
        if closed[current_i, current_j]:
            continue
        closed[current_i, current_j] = True

        if current == goal_node:
            break

        current_g = float(g_score[current_i, current_j])
        for delta_i, delta_j, offsets in direction_offsets:
            next_i = current_i + delta_i
            next_j = current_j + delta_j
            if next_i < 0 or next_i >= cost_map.n_lat:
                continue
            if next_j < 0 or next_j >= cost_map.n_lon:
                continue
            if closed[next_i, next_j]:
                continue
            if not _edge_is_water(cost_map, current_i, current_j, offsets):
                continue

            step_cost = _direction_step_cost(
                cost_map,
                current_i,
                current_j,
                delta_i,
                delta_j,
                distance_mode=distance_mode,
            )
            candidate_g = current_g + step_cost
            if candidate_g >= g_score[next_i, next_j]:
                continue

            g_score[next_i, next_j] = candidate_g
            prev_i[next_i, next_j] = current_i
            prev_j[next_i, next_j] = current_j
            heap_counter += 1
            heapq.heappush(
                open_heap,
                (candidate_g + heuristic((next_i, next_j)), heap_counter, (next_i, next_j)),
            )

    if not math.isfinite(float(g_score[goal_node])):
        raise RuntimeError("No water-only A* path found between the selected ports.")

    path_indices = [goal_node]
    current = goal_node
    while current != start_node:
        current_i, current_j = current
        previous = (int(prev_i[current_i, current_j]), int(prev_j[current_i, current_j]))
        if previous[0] < 0 or previous[1] < 0:
            raise RuntimeError("A* predecessor chain is broken.")
        path_indices.append(previous)
        current = previous
    path_indices.reverse()
    return path_indices


# 방위각 체크
def check_heading_delta(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    lat1_r, lon1_r = math.radians(p1[0]), math.radians(p1[1])
    lat2_r, lon2_r = math.radians(p2[0]), math.radians(p2[1])
    dlon = lon2_r - lon1_r
    x = math.sin(dlon) * math.cos(lat2_r)
    y = (
        math.cos(lat1_r) * math.sin(lat2_r)
        - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
    )
    return math.degrees(math.atan2(x, y)) % 360.0

# 라플라시안스무딩
def smooth_compressed_waypoints(
    waypoints: list[tuple[float, float]],
    cost_map,
    max_heading_delta: float = 30.0,
    iterations: int = 150,
) -> list[tuple[float, float]]:
    """
    Apply Laplacian smoothing to waypoints to ensure heading deltas are minimized
    below max_heading_delta, while maintaining water-only feasibility.
    """
    n = len(waypoints)
    if n <= 2:
        return waypoints

    smoothed = list(waypoints)
    
    # 각도 차이 계산 
    def _angle_delta_deg(angle_a, angle_b):
        return ((angle_a - angle_b + 180.0) % 360.0) - 180.0
        
    print(f"  [A* Smooth] Starting Laplacian smoothing for max {max_heading_delta}° turns")
    for loop_idx in range(iterations):

        new_smoothed = list(smoothed)

        # 각 구간의 방위각 계산
        headings = [check_heading_delta(smoothed[i], smoothed[i+1]) for i in range(n-1)]
        # 인접한 두 구간의 방위각 차이 계산
        deltas = [abs(_angle_delta_deg(headings[i+1], headings[i])) for i in range(n-2)]
        
        # 최대 각도 차이가 설정값보다 작으면 종료
        if max(deltas) <= max_heading_delta:
            print(f"  [A* Smooth] Converged after {loop_idx} iterations. Max delta: {max(deltas):.1f}°")
            break
        
        # 각 웨이포인트(내부점)를 주변 3개 점의 가중평균으로 이동
        for i in range(1, n - 1):
            w_prev = new_smoothed[i-1] # Use newly updated prev for faster cascade
            w_curr = smoothed[i]
            w_next = smoothed[i+1]
            
            # 가중평균 계산
            # 현재 점을 이전 점과 다음 점의 가중평균으로 이동
            # 예시로, 현재 점이 (3,5), 이전 점이 (2,4), 다음 점이 (6,9)이면, 
            # 가중평균은 (0.5*3 + 0.25*2 + 0.25*6, 0.5*5 + 0.25*4 + 0.25*9) = (3.5, 5.5)가 됨
            new_lat = 0.5 * w_curr[0] + 0.25 * w_prev[0] + 0.25 * w_next[0]
            new_lon = 0.5 * w_curr[1] + 0.25 * w_prev[1] + 0.25 * w_next[1]
            
            # 육지 통과 방지
            c1 = cost_map.segment_cost(w_prev[0], w_prev[1], new_lat, new_lon)
            c2 = cost_map.segment_cost(new_lat, new_lon, w_next[0], w_next[1])
            if c1 <= 0.0 and c2 <= 0.0:
                new_smoothed[i] = (new_lat, new_lon)
                
        smoothed = new_smoothed
    else:
        # 최대 반복 횟수 도달 시
        headings = [check_heading_delta(smoothed[i], smoothed[i+1]) for i in range(n-1)]
        deltas = [abs(_angle_delta_deg(headings[i+1], headings[i])) for i in range(n-2)]
        print(f"  [A* Smooth] Max iterations reached. Max delta: {max(deltas):.1f}°")
        
    return smoothed


# A_star 경로 waypoints 균등 분배
def compress_path_to_waypoints(
    raw_path: list[tuple[float, float]],
    cost_map,
    num_segments: int,
    distance_mode: str = "grid",
) -> tuple[list[tuple[float, float]], float]:
    """
    Compress a raw A* path into exactly `num_segments + 1` waypoints while preserving
    water-only feasibility between consecutive selected points.
    """
    if len(raw_path) < 2:
        return list(raw_path), 0.0

    # 누적 거리 계산
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
        # ── O(N^2) 병목 해결: 1500개를 다 검사하지 않고, 합리적인 최대 탐색 거리로 자름
        # 총 경로 점 개수를 목표 구간 수로 나눈 평균 길이의 3배 정도까지만 내다봄
        lookahead = max(10, int((n_points / num_segments) * 3.0))
        max_end_index = min(n_points - 1, start_index + lookahead)
        
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
    distance_mode: str = "grid",
    search_resolution: float = 0.01,
    apply_smoothing: bool = True,
    connectivity: int = DEFAULT_ASTAR_CONNECTIVITY,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], float]:
    """Build the raw A* water path and compress it into route waypoints."""
    path_idx = _queue_astar_path_indices(
        cost_map,
        start_point,
        goal_point,
        distance_mode=distance_mode,
        search_resolution=search_resolution,
        connectivity=connectivity,
    )
    raw_path = [(cost_map.lats[i], cost_map.lons[j]) for i, j in path_idx]
    raw_path[0] = start_point
    raw_path[-1] = goal_point
    waypoints, total_dist_nm = compress_path_to_waypoints(
        raw_path,
        cost_map,
        num_segments,
        distance_mode=distance_mode,
    )

    if apply_smoothing:
        waypoints = smooth_compressed_waypoints(waypoints, cost_map, max_heading_delta=30.0)

    return raw_path, waypoints, total_dist_nm

    # 맨 처음과 끝을 무조건 정확한 입력 Port 좌표로 바꿔치기 (오차 방지)
