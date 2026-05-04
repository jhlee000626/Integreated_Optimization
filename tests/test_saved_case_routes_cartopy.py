from __future__ import annotations

import os
import sys

from shapely.geometry import MultiPolygon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.visualize_saved_case_routes_cartopy import render_saved_case_routes_map


def test_render_saved_case_routes_map_smoke(tmp_path):
    case_artifacts = [
        {
            "case_name": "case1",
            "route": {
                "waypoints": [(34.95, 129.15), (33.5, 126.3), (31.0, 122.0)],
            },
            "map_bounds": {"lat_min": 30.0, "lat_max": 36.0, "lon_min": 121.0, "lon_max": 131.0},
        },
        {
            "case_name": "case2",
            "route": {
                "waypoints": [(34.95, 129.15), (33.8, 127.2), (31.0, 122.0)],
            },
            "map_bounds": {"lat_min": 30.0, "lat_max": 36.0, "lon_min": 121.0, "lon_max": 131.0},
        },
    ]
    output_path = tmp_path / "saved_case_routes_cartopy.png"

    saved_path = render_saved_case_routes_map(
        case_artifacts=case_artifacts,
        output_path=str(output_path),
        land_geometry=MultiPolygon(),
        use_cartopy=False,
    )

    assert saved_path == str(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
