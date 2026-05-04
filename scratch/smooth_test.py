import math

def compute_bearing(lat1, lon1, lat2, lon2):
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlon = lon2_r - lon1_r
    x = math.sin(dlon) * math.cos(lat2_r)
    y = (
        math.cos(lat1_r) * math.sin(lat2_r)
        - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
    )
    return math.degrees(math.atan2(x, y)) % 360.0

def _angle_delta_deg(angle_a, angle_b):
    return ((angle_a - angle_b + 180.0) % 360.0) - 180.0

def check_angles(waypoints):
    headings = []
    deltas = []
    for i in range(len(waypoints)-1):
        heading = compute_bearing(*waypoints[i], *waypoints[i+1])
        headings.append(heading)
        if i > 0:
            deltas.append(_angle_delta_deg(heading, headings[-2]))
    return deltas

def apply_laplacian_smoothing(waypoints, cost_map=None, iterations=50, max_delta=30.0):
    smoothed = list(waypoints)
    n = len(smoothed)
    if n <= 2: return smoothed

    # We do rounds of smoothing
    for _ in range(iterations):
        new_smoothed = list(smoothed)
        for i in range(1, n - 1):
            w_prev = smoothed[i-1]
            w_curr = smoothed[i]
            w_next = smoothed[i+1]
            
            # Simple relaxation
            new_lat = 0.5 * w_curr[0] + 0.25 * w_prev[0] + 0.25 * w_next[0]
            new_lon = 0.5 * w_curr[1] + 0.25 * w_prev[1] + 0.25 * w_next[1]
            
            # If cost_map, check land. For scratch, mock cost_map check True
            new_smoothed[i] = (new_lat, new_lon)
            
        smoothed = new_smoothed
        
    return smoothed

wp = [
    (35.0, 129.0),
    (34.0, 129.0),  # heading 180
    (34.0, 128.0),  # heading 270 (90 deg turn)
]
print("Orig deltas:", check_angles(wp))
sm = apply_laplacian_smoothing(wp, iterations=2)
print(sm)
print("Smoothed deltas:", check_angles(sm))
