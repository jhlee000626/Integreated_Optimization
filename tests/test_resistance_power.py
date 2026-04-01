from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimizer.ga_engine import build_required_power_profile, decode_route
from src.resistance.models import EnvironmentData
from src.resistance.modified_dpm import compute_P_req


def test_compute_p_req_accepts_environment_object():
    env = EnvironmentData(
        wind_speed_ms=8.0,
        wind_dir_deg=300.0,
        current_speed_ms=0.6,
        current_dir_deg=90.0,
        wave_height_m=1.8,
        wave_period_s=7.5,
        wave_dir_deg=315.0,
    )
    result = compute_P_req(
        v_ship_knots=12.0,
        v_wind_ms=env.wind_speed_ms,
        encounter_angle_deg=40.0,
        env=env,
        heading_deg=250.0,
        P_service=9.0,
    )

    assert result["P_req"] > result["P_service"]
    assert result["R_wave_N"] >= 0.0
    assert result["v_sog_knots"] >= 0.1


def test_current_only_changes_v_sog_not_propulsion():
    aided = compute_P_req(
        v_ship_knots=12.0,
        v_wind_ms=0.0,
        encounter_angle_deg=0.0,
        env=EnvironmentData(current_speed_ms=1.5, current_dir_deg=90.0),
        heading_deg=90.0,
        P_service=0.0,
    )
    opposed = compute_P_req(
        v_ship_knots=12.0,
        v_wind_ms=0.0,
        encounter_angle_deg=0.0,
        env=EnvironmentData(current_speed_ms=1.5, current_dir_deg=270.0),
        heading_deg=90.0,
        P_service=0.0,
    )

    assert aided["v_sog_knots"] > opposed["v_sog_knots"]
    assert aided["PD_kW"] == pytest.approx(opposed["PD_kW"], rel=1e-9)
    assert aided["P_prop"] == pytest.approx(opposed["P_prop"], rel=1e-9)


def test_build_required_power_profile_uses_segment_start_nodes():
    route = decode_route([10.0, 60.0, 10.0, 0.0], n_segments=3, rta_h=3.0)
    departure_time_utc = datetime(2025, 3, 25, 0, 0, tzinfo=timezone.utc)
    calls = []

    def env_fn(lat, lon, when_utc):
        calls.append((lat, lon, when_utc))
        return EnvironmentData(wind_speed_ms=0.0, wind_dir_deg=0.0)

    profile = build_required_power_profile(
        route,
        env_fn=env_fn,
        departure_time_utc=departure_time_utc,
    )

    assert len(profile) == len(route["speeds"])
    for index, (lat, lon, when_utc) in enumerate(calls):
        node = route["nodes"][index]
        expected_time = departure_time_utc + timedelta(hours=float(node["time_h"]))
        assert lat == pytest.approx(node["lat"])
        assert lon == pytest.approx(node["lon"])
        assert when_utc == expected_time


def test_decode_route_uses_sog_for_ground_track():
    individual = [12.0, 90.0, 12.0, 0.0]
    route = decode_route(
        individual,
        n_segments=3,
        rta_h=3.0,
    )

    assert route["distances_nm"][0] == pytest.approx(12.0)
    assert route["distances_nm"][1] == pytest.approx(12.0)
    assert route["speeds"][0] == pytest.approx(12.0)
    assert route["speeds_sog"][0] == pytest.approx(12.0)
    assert route["headings"][0] == pytest.approx(90.0)
    assert route["headings"][1] == pytest.approx(90.0)


def test_build_required_power_profile_derives_stw_from_sog_and_current():
    route = decode_route([12.0, 90.0, 12.0, 0.0], n_segments=3, rta_h=3.0)
    departure_time_utc = datetime(2025, 3, 25, 0, 0, tzinfo=timezone.utc)

    def env_fn(lat, lon, when_utc):
        del lat, lon, when_utc
        return EnvironmentData(
            wind_speed_ms=0.0,
            wind_dir_deg=0.0,
            current_speed_ms=1.5,
            current_dir_deg=90.0,
        )

    profile = build_required_power_profile(route, env_fn=env_fn, departure_time_utc=departure_time_utc)

    expected_stw = 12.0 - (1.5 / 0.514444)
    assert profile[0]["speed_sog_kts"] == pytest.approx(12.0)
    assert profile[0]["speed_stw_kts"] == pytest.approx(expected_stw)
    assert route["nodes"][0]["speed_through_water_kts"] == pytest.approx(expected_stw)


def test_decode_route_uses_first_absolute_then_delta_heading():
    route = decode_route([12.0, 180.0, 12.0, 10.0], n_segments=3, rta_h=3.0)

    assert route["headings"][0] == pytest.approx(180.0)
    assert route["headings"][1] == pytest.approx(190.0)
