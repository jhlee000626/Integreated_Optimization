from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import matplotlib
import pytest
from shapely.geometry import MultiPolygon

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Validation.common import (
    build_case_metrics,
    load_case2_seed_cache,
    resolve_case3_seed_individual,
    save_case2_seed_cache,
    save_case_artifacts,
)
from scripts.visualize_case_route_environment import render_wind_route_map
from src.weather.marine_environment import create_synthetic_environment


def _sample_route():
    departure_time_utc = datetime(2025, 3, 25, 6, 0, tzinfo=timezone.utc)
    route = {
        "waypoints": [(34.9, 129.1), (33.8, 126.6), (31.4, 122.4)],
        "speeds": [12.0, 11.5],
        "speeds_sog": [12.0, 11.5],
        "speeds_stw": [11.6, 11.1],
        "headings": [245.0, 238.0],
        "delta_headings": [-5.0, -7.0],
        "dt": [1.0, 1.0],
        "distances_nm": [12.0, 11.5],
        "valid": True,
        "valid_departure_heading": True,
        "valid_turning": True,
        "valid_speed": True,
        "valid_heading": True,
        "land_violation": 0.0,
        "last_speed": 11.1,
        "last_speed_sog": 11.5,
        "final_heading_delta": -7.0,
        "nodes": [
            {
                "lat": 34.9,
                "lon": 129.1,
                "time_h": 0.0,
                "time_utc": departure_time_utc,
                "speed_out_kts": 12.0,
                "speed_over_ground_kts": 12.0,
                "speed_through_water_kts": 11.6,
                "heading_out_deg": 245.0,
            },
            {
                "lat": 33.8,
                "lon": 126.6,
                "time_h": 1.0,
                "time_utc": departure_time_utc.replace(hour=7),
                "speed_out_kts": 11.5,
                "speed_over_ground_kts": 11.5,
                "speed_through_water_kts": 11.1,
                "heading_out_deg": 238.0,
            },
            {
                "lat": 31.4,
                "lon": 122.4,
                "time_h": 2.0,
                "time_utc": departure_time_utc.replace(hour=8),
                "speed_out_kts": None,
                "speed_over_ground_kts": None,
                "speed_through_water_kts": None,
                "heading_out_deg": None,
            },
        ],
    }
    return route, departure_time_utc


def _sample_power_profile(departure_time_utc: datetime):
    return [
        {
            "segment": 0,
            "phase": "cruise",
            "when_utc": departure_time_utc,
            "waypoint_from": (34.9, 129.1),
            "waypoint_to": (33.8, 126.6),
            "heading_deg": 245.0,
            "encounter_angle_deg": 35.0,
            "speed_sog_kts": 12.0,
            "speed_stw_kts": 11.6,
            "speed_stw_power_kts": 11.6,
            "current_component_kts": 0.4,
            "wind_speed_ms": 9.0,
            "wind_dir_deg": 310.0,
            "current_speed_ms": 0.5,
            "current_dir_deg": 110.0,
            "wave_height_m": 1.6,
            "wave_period_s": 7.2,
            "wave_dir_deg": 300.0,
            "P_prop": 15.2,
            "P_service": 3.0,
            "P_req": 18.2,
            "PD_kW": 13200.0,
            "BHP_kW": 14500.0,
            "total_resistance_N": 225000.0,
            "R_calm_N": 180000.0,
            "R_wind_N": 23000.0,
            "R_wave_N": 22000.0,
            "relative_wind_speed_ms": 12.0,
            "relative_wind_dir_deg": 42.0,
        },
        {
            "segment": 1,
            "phase": "cruise",
            "when_utc": departure_time_utc.replace(hour=7),
            "waypoint_from": (33.8, 126.6),
            "waypoint_to": (31.4, 122.4),
            "heading_deg": 238.0,
            "encounter_angle_deg": 30.0,
            "speed_sog_kts": 11.5,
            "speed_stw_kts": 11.1,
            "speed_stw_power_kts": 11.1,
            "current_component_kts": 0.4,
            "wind_speed_ms": 8.4,
            "wind_dir_deg": 300.0,
            "current_speed_ms": 0.6,
            "current_dir_deg": 105.0,
            "wave_height_m": 1.3,
            "wave_period_s": 7.0,
            "wave_dir_deg": 295.0,
            "P_prop": 14.5,
            "P_service": 3.0,
            "P_req": 17.5,
            "PD_kW": 12700.0,
            "BHP_kW": 13900.0,
            "total_resistance_N": 214000.0,
            "R_calm_N": 172000.0,
            "R_wind_N": 21000.0,
            "R_wave_N": 21000.0,
            "relative_wind_speed_ms": 11.5,
            "relative_wind_dir_deg": 39.0,
        },
    ]


def _sample_milp_result():
    return {
        "feasible": True,
        "total_fuel_kg": 70353.0,
        "summary": {"dg_running_hours": {"DG1": 2.0}},
        "schedule": [
            {
                "t": 0,
                "dt_h": 1.0,
                "P_req_MW": 18.2,
                "P_dc_MW": 0.5,
                "P_c_MW": 0.0,
                "SOC": 0.7,
                "DG1_P_MW": 18.2,
                "DG1_ON": 1,
                "DG1_Start": 1,
                "DG1_FC_kgh": 90.0,
            },
            {
                "t": 1,
                "dt_h": 1.0,
                "P_req_MW": 17.5,
                "P_dc_MW": 0.0,
                "P_c_MW": 0.2,
                "SOC": 0.68,
                "DG1_P_MW": 17.3,
                "DG1_ON": 1,
                "DG1_Start": 0,
                "DG1_FC_kgh": 86.0,
            },
        ],
    }


def test_case2_seed_cache_roundtrip_and_mismatch(tmp_path):
    route, departure_time_utc = _sample_route()
    seed = [12.0, 245.0, 11.5, -7.0]

    cache_path = save_case2_seed_cache(
        best_individual=seed,
        best_fitness=70353.0,
        route=route,
        start_port=(34.95, 129.15),
        end_port=(31.0, 122.0),
        n_segments=3,
        rta_h=2.0,
        ga_cost_resolution=0.1,
        departure_time_utc=departure_time_utc,
        data_dir=str(tmp_path),
    )
    payload = load_case2_seed_cache(
        start_port=(34.95, 129.15),
        end_port=(31.0, 122.0),
        n_segments=3,
        rta_h=2.0,
        ga_cost_resolution=0.1,
        departure_time_utc=departure_time_utc,
        data_dir=str(tmp_path),
    )

    assert payload is not None
    assert payload["best_individual"] == seed

    with open(cache_path, "r", encoding="utf-8") as handle:
        corrupted = json.load(handle)
    corrupted["metadata"]["rta_h"] = 99.0
    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump(corrupted, handle, indent=2)

    assert (
        load_case2_seed_cache(
            start_port=(34.95, 129.15),
            end_port=(31.0, 122.0),
            n_segments=3,
            rta_h=2.0,
            ga_cost_resolution=0.1,
            departure_time_utc=departure_time_utc,
            data_dir=str(tmp_path),
        )
        is None
    )


def test_resolve_case3_seed_prefers_same_run_and_honors_opt_out(tmp_path):
    route, departure_time_utc = _sample_route()
    cached_seed = [12.0, 245.0, 11.5, -7.0]
    same_run_seed = [13.0, 250.0, 12.0, -6.0]

    save_case2_seed_cache(
        best_individual=cached_seed,
        best_fitness=70353.0,
        route=route,
        start_port=(34.95, 129.15),
        end_port=(31.0, 122.0),
        n_segments=3,
        rta_h=2.0,
        ga_cost_resolution=0.1,
        departure_time_utc=departure_time_utc,
        data_dir=str(tmp_path),
    )

    seed, source = resolve_case3_seed_individual(
        use_case2_seed=True,
        start_port=(34.95, 129.15),
        end_port=(31.0, 122.0),
        n_segments=3,
        rta_h=2.0,
        ga_cost_resolution=0.1,
        departure_time_utc=departure_time_utc,
        same_run_seed_individual=same_run_seed,
        data_dir=str(tmp_path),
    )
    assert seed == same_run_seed
    assert source == "same_run"

    seed, source = resolve_case3_seed_individual(
        use_case2_seed=False,
        start_port=(34.95, 129.15),
        end_port=(31.0, 122.0),
        n_segments=3,
        rta_h=2.0,
        ga_cost_resolution=0.1,
        departure_time_utc=departure_time_utc,
        same_run_seed_individual=same_run_seed,
        data_dir=str(tmp_path),
    )
    assert seed is None
    assert source == "disabled"

    seed, source = resolve_case3_seed_individual(
        use_case2_seed=True,
        start_port=(34.95, 129.15),
        end_port=(31.0, 122.0),
        n_segments=3,
        rta_h=2.0,
        ga_cost_resolution=0.1,
        departure_time_utc=departure_time_utc,
        same_run_seed_individual=None,
        data_dir=str(tmp_path),
    )
    assert seed == cached_seed
    assert source == "cache"


def test_save_case_artifacts_writes_json_and_csv(tmp_path):
    route, departure_time_utc = _sample_route()
    power_profile = _sample_power_profile(departure_time_utc)
    milp_result = _sample_milp_result()
    metrics = build_case_metrics(
        case_name="case2_ga_twostage",
        route=route,
        power_profile=power_profile,
        objective=70353.0,
        objective_kind="energy_objective",
        milp_result=milp_result,
        route_violation=0.0,
        runtime_sec=12.5,
        note="unit-test",
    )
    cost_map = SimpleNamespace(
        bounds={"lat_min": 30.0, "lat_max": 36.0, "lon_min": 121.0, "lon_max": 131.0},
        resolution=0.1,
    )

    paths = save_case_artifacts(
        case_name="case2_ga_twostage",
        output_dir=str(tmp_path),
        route=route,
        power_profile=power_profile,
        milp_result=milp_result,
        departure_time_utc=departure_time_utc,
        cost_map=cost_map,
        metrics=metrics,
        ga_best_individual=[12.0, 245.0, 11.5, -7.0],
        ga_best_fitness=70353.0,
        ga_logbook=None,
    )

    assert os.path.exists(paths["case_artifact"])
    assert os.path.exists(paths["route_nodes"])
    assert os.path.exists(paths["route_segments"])
    assert os.path.exists(paths["schedule"])
    assert os.path.exists(paths["ga_seed"])

    with open(paths["case_artifact"], "r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    assert artifact["departure_time_utc"] == departure_time_utc.isoformat()
    assert artifact["cost_map_resolution"] == pytest.approx(0.1)
    assert artifact["ga"]["best_individual"] == [12.0, 245.0, 11.5, -7.0]
    assert artifact["map_bounds"]["lon_min"] == pytest.approx(121.0)

    with open(paths["route_segments"], "r", encoding="utf-8", newline="") as handle:
        segment_rows = list(csv.DictReader(handle))
    assert len(segment_rows) == 2
    assert "wind_speed_ms" in segment_rows[0]
    assert segment_rows[0]["milp_DG1_P_MW"] == "18.2"

    with open(paths["schedule"], "r", encoding="utf-8", newline="") as handle:
        schedule_rows = list(csv.DictReader(handle))
    assert len(schedule_rows) == 2
    assert "DG1_P_MW" in schedule_rows[0]


def test_render_wind_route_map_smoke(tmp_path):
    route, departure_time_utc = _sample_route()
    case_artifact = {
        "departure_time_utc": departure_time_utc.isoformat(),
        "route": {"waypoints": route["waypoints"]},
        "map_bounds": {"lat_min": 31.0, "lat_max": 35.5, "lon_min": 122.0, "lon_max": 130.0},
        "cost_map_resolution": 0.5,
    }
    env_fn = create_synthetic_environment(base_wind_speed=10.0, base_wind_dir=315.0)
    output_path = tmp_path / "wind_route_map.png"

    saved_path = render_wind_route_map(
        case_artifact=case_artifact,
        output_path=str(output_path),
        env_fn=env_fn,
        land_geometry=MultiPolygon(),
    )

    assert saved_path == str(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
