"""
Shared helpers for maintained verification scenarios.
"""

from __future__ import annotations

import os
from typing import Callable, Tuple

from src.optimizer.ga_engine import build_required_power_profile
from src.optimizer.milp_solver import MILPSolver
from src.weather.era5_loader import ERA5Loader


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ERA5_NC_PATH = os.path.join(PROJECT_ROOT, "data", "era5", "era5_wind_2024_01.nc")
SFOC_PATH = os.path.join(PROJECT_ROOT, "config", "sfoc.json")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output", "verification")


def ensure_output_dir(case_name: str) -> str:
    path = os.path.join(OUTPUT_ROOT, case_name)
    os.makedirs(path, exist_ok=True)
    return path


def load_era5_weather(time_index: int = 0) -> Tuple[ERA5Loader, Callable]:
    if not os.path.exists(ERA5_NC_PATH):
        raise FileNotFoundError(f"ERA5 file not found: {ERA5_NC_PATH}")
    loader = ERA5Loader(ERA5_NC_PATH, time_index=time_index)
    return loader, loader.get_weather_fn()


def make_milp_solver() -> MILPSolver:
    return MILPSolver(sfoc_json_path=SFOC_PATH, n_pwl_segments=5)


def solve_route_schedule(route: dict, weather_fn: Callable, initial_soc: float = 0.7):
    milp = make_milp_solver()
    power_profile = build_required_power_profile(route, weather_fn=weather_fn)
    result = milp.solve(
        P_req=[segment["P_req"] for segment in power_profile],
        dt=route["dt"],
        initial_SOC=initial_soc,
        msg=False,
    )
    return milp, power_profile, result
