from __future__ import annotations

from datetime import datetime, timezone
import os
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.weather.marine_environment import MarineEnvironmentLoader


def _build_era5_dataset(times):
    lats = np.array([32.0, 34.0])
    lons = np.array([125.0, 127.0])
    t = np.arange(len(times), dtype=float)[:, None, None]
    i = np.arange(len(lats), dtype=float)[None, :, None]
    j = np.arange(len(lons), dtype=float)[None, None, :]

    return xr.Dataset(
        data_vars={
            "u10": (("valid_time", "latitude", "longitude"), 10.0 + 2.0 * t + i + 0.5 * j),
            "v10": (("valid_time", "latitude", "longitude"), np.zeros((len(times), len(lats), len(lons)))),
            "swh": (("valid_time", "latitude", "longitude"), 2.0 + t + 0.1 * i + 0.2 * j),
            "mwp": (("valid_time", "latitude", "longitude"), 8.0 + 0.5 * t + 0.0 * i + 0.0 * j),
            "mwd": (("valid_time", "latitude", "longitude"), np.full((len(times), len(lats), len(lons)), 300.0)),
        },
        coords={
            "valid_time": np.asarray(times, dtype="datetime64[ns]"),
            "latitude": lats,
            "longitude": lons,
        },
    )


def _build_cmems_dataset(times):
    lats = np.array([32.0, 34.0])
    lons = np.array([125.0, 127.0])
    t = np.arange(len(times), dtype=float)[:, None, None, None]
    depth = np.array([0.5], dtype=float)

    return xr.Dataset(
        data_vars={
            "uo": (("time", "depth", "latitude", "longitude"), 0.4 + 0.2 * t + np.zeros((len(times), 1, 2, 2))),
            "vo": (("time", "depth", "latitude", "longitude"), np.zeros((len(times), 1, 2, 2))),
        },
        coords={
            "time": np.asarray(times, dtype="datetime64[ns]"),
            "depth": depth,
            "latitude": lats,
            "longitude": lons,
        },
    )


def test_marine_environment_loader_interpolates_time_and_space():
    times = [np.datetime64("2025-03-25T00:00:00"), np.datetime64("2025-03-25T01:00:00")]
    loader = MarineEnvironmentLoader.from_datasets(
        _build_era5_dataset(times),
        _build_cmems_dataset(times),
    )
    env = loader.get_environment(33.0, 126.0, datetime(2025, 3, 25, 0, 30, tzinfo=timezone.utc))

    assert env.wind_speed_ms == pytest.approx(11.75, rel=1e-6)
    assert env.wind_dir_deg == pytest.approx(270.0, rel=1e-6)
    assert env.current_speed_ms == pytest.approx(0.5, rel=1e-6)
    assert env.current_dir_deg == pytest.approx(90.0, rel=1e-6)
    assert env.wave_height_m == pytest.approx(2.65, rel=1e-6)
    assert env.wave_period_s == pytest.approx(8.25, rel=1e-6)
    assert env.wave_dir_deg == pytest.approx(300.0, rel=1e-6)


def test_marine_environment_loader_rejects_non_overlapping_time_windows():
    with pytest.raises(ValueError, match="do not overlap"):
        MarineEnvironmentLoader.from_datasets(
            _build_era5_dataset([np.datetime64("2024-01-01T00:00:00")]),
            _build_cmems_dataset([np.datetime64("2025-03-25T00:00:00")]),
        )
