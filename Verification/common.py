"""
Shared helpers for maintained verification scenarios.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Callable, Tuple

from src.optimizer.ga_engine import apply_route_validation, build_required_power_profile, evaluate_route_validity
from src.optimizer.milp_solver import MILPSolver
from src.weather import MarineEnvironmentLoader, resolve_marine_dataset_paths


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SFOC_PATH = os.path.join(PROJECT_ROOT, "config", "sfoc.json")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output", "verification")
VERIFICATION_COST_MAP_RESOLUTION = 0.01


def ensure_output_dir(case_name: str) -> str:
    path = os.path.join(OUTPUT_ROOT, case_name)
    os.makedirs(path, exist_ok=True)
    return path


def load_marine_environment() -> Tuple[MarineEnvironmentLoader, Callable, datetime]:
    era5_path, cmems_path = resolve_marine_dataset_paths(PROJECT_ROOT)
    loader = MarineEnvironmentLoader(era5_path, cmems_path)
    departure_time_utc = loader.available_times_utc[0]
    return loader, loader.get_environment_fn(), departure_time_utc


def make_milp_solver() -> MILPSolver:
    return MILPSolver(sfoc_json_path=SFOC_PATH, n_pwl_segments=5)


def solve_route_schedule(
    route: dict,
    env_fn: Callable,
    departure_time_utc: datetime,
    initial_soc: float = 0.7,
):
    milp = make_milp_solver()
    power_profile = build_required_power_profile(
        route,
        env_fn=env_fn,
        departure_time_utc=departure_time_utc,
    )
    result = milp.solve(
        P_req=[segment["P_req"] for segment in power_profile],
        dt=route["dt"],
        initial_SOC=initial_soc,
        msg=False,
    )
    return milp, power_profile, result


def validate_route(
    route: dict,
    cost_map,
    env_fn: Callable,
    departure_time_utc: datetime,
):
    validation = evaluate_route_validity(
        route,
        cost_map=cost_map,
        env_fn=env_fn,
        departure_time_utc=departure_time_utc,
    )
    apply_route_validation(route, validation)
    return validation
