"""
Diagnose why the two-stage Case 2 route becomes MILP infeasible.

This script reruns Verification Case 2 sequentially, builds the final power
profile, solves the real MILP, and writes a diagnosis report that breaks down
likely infeasibility causes such as power-cap violations, energy deficit,
ramp deficit, and stepwise reachable-power gaps.
"""

from __future__ import annotations

import json
import os
import sys
import copy
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Verification.common import (
    VERIFICATION_COST_MAP_RESOLUTION,
    ensure_output_dir,
    load_marine_environment,
    make_milp_solver,
    solve_route_schedule,
)
from src.grid.cost_map import build_cost_map
from src.optimizer.ga_engine import (
    N_SEGMENTS,
    RTA_HOURS,
    compute_milp_infeasible_surrogate,
    setup_ga,
)


class EnergyObjectiveSolver:
    """MILP-shaped adapter used by Verification Case 2."""

    @staticmethod
    def solve(P_req, dt, initial_SOC=0.7, msg=False):
        del initial_SOC, msg
        total_energy = sum(power * duration for power, duration in zip(P_req, dt))
        return {
            "feasible": True,
            "total_fuel_kg": total_energy,
            "schedule": [],
            "summary": {},
        }


@dataclass
class ReachableRange:
    dg_on: list[str]
    net_min_mw: float
    net_max_mw: float


def _generator_subset_ranges(milp_solver) -> list[ReachableRange]:
    ranges: list[ReachableRange] = []
    dg_names = list(milp_solver.dg_names)
    p_c_max = float(milp_solver.ess["P_c_max"])
    p_dc_max = float(milp_solver.ess["P_dc_max"])

    # All-off case: only ESS discharge can supply load.
    ranges.append(ReachableRange(dg_on=[], net_min_mw=0.0, net_max_mw=p_dc_max))

    for count in range(1, len(dg_names) + 1):
        for subset in combinations(dg_names, count):
            p_min = sum(
                float(milp_solver.dg_specs[dg]["P_max"]) * float(milp_solver.min_load_ratio)
                for dg in subset
            )
            p_max = sum(float(milp_solver.dg_specs[dg]["P_max"]) for dg in subset)
            ranges.append(
                ReachableRange(
                    dg_on=list(subset),
                    net_min_mw=max(0.0, p_min - p_c_max),
                    net_max_mw=p_max + p_dc_max,
                )
            )
    return ranges


def _reachable_ranges_for_load(load_mw: float, ranges: list[ReachableRange]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for info in ranges:
        if info.net_min_mw - 1e-9 <= load_mw <= info.net_max_mw + 1e-9:
            matches.append(asdict(info))
    return matches


def build_case2_diagnosis(
    route: dict,
    power_profile: list[dict],
    milp,
    milp_result: dict,
    initial_soc: float,
) -> dict[str, Any]:
    p_req_list = [float(step["P_req"]) for step in power_profile]
    dt = [float(value) for value in route["dt"]]

    total_dg_power_cap = sum(float(spec["P_max"]) for spec in milp.dg_specs.values())
    total_supply_cap = total_dg_power_cap + float(milp.ess["P_dc_max"])
    total_required_energy = sum(power * duration for power, duration in zip(p_req_list, dt))
    total_dg_energy_cap = total_dg_power_cap * sum(dt)
    ess_deliverable_energy = (
        float(milp.ess["capacity"])
        * max(0.0, float(initial_soc) - float(milp.ess["SOC_min"]))
        * float(milp.ess["eta_dc"])
    )
    energy_deficit_mwh = max(0.0, total_required_energy - total_dg_energy_cap - ess_deliverable_energy)

    power_excess_steps: list[dict[str, Any]] = []
    for step_index, (power, duration) in enumerate(zip(p_req_list, dt)):
        excess_mw = max(0.0, power - total_supply_cap)
        if excess_mw > 0.0:
            power_excess_steps.append(
                {
                    "step": step_index,
                    "p_req_mw": power,
                    "duration_h": duration,
                    "excess_mw": excess_mw,
                }
            )

    ramp_steps: list[dict[str, Any]] = []
    ess_net_ramp_cap = float(milp.ess["P_dc_max"]) + float(milp.ess["P_c_max"])
    for step_index in range(1, len(p_req_list)):
        dg_ramp_cap = sum(
            float(spec["ramp_rate"]) * float(spec["P_max"]) * dt[step_index]
            for spec in milp.dg_specs.values()
        )
        total_ramp_cap = dg_ramp_cap + ess_net_ramp_cap
        load_delta = abs(p_req_list[step_index] - p_req_list[step_index - 1])
        excess_mw = max(0.0, load_delta - total_ramp_cap)
        ramp_steps.append(
            {
                "from_step": step_index - 1,
                "to_step": step_index,
                "load_delta_mw": load_delta,
                "ramp_cap_mw": total_ramp_cap,
                "excess_mw": excess_mw,
            }
        )

    reachable_ranges = _generator_subset_ranges(milp)
    unreachable_steps: list[dict[str, Any]] = []
    for step_index, power in enumerate(p_req_list):
        matches = _reachable_ranges_for_load(power, reachable_ranges)
        if not matches:
            unreachable_steps.append(
                {
                    "step": step_index,
                    "p_req_mw": power,
                }
            )

    top_power_steps = sorted(
        (
            {
                "step": step["segment"],
                "phase": step["phase"],
                "p_req_mw": float(step["P_req"]),
                "speed_sog_kts": float(step["speed_sog_kts"]),
                "speed_stw_kts": float(step["speed_stw_kts"]),
                "heading_deg": float(step["heading_deg"]),
            }
            for step in power_profile
        ),
        key=lambda item: item["p_req_mw"],
        reverse=True,
    )[:5]

    top_ramp_steps = sorted(ramp_steps, key=lambda item: item["load_delta_mw"], reverse=True)[:5]
    surrogate = compute_milp_infeasible_surrogate(p_req_list, dt, milp, initial_soc=initial_soc)

    likely_causes: list[str] = []
    if power_excess_steps:
        likely_causes.append("One or more time steps exceed total DG+ESS discharge power capacity.")
    if unreachable_steps:
        likely_causes.append("Some time steps are outside the instantaneous reachable net-power range given DG minimum loads and ESS charge/discharge bounds.")
    if any(step["excess_mw"] > 0.0 for step in ramp_steps):
        likely_causes.append("One or more step-to-step load changes exceed the combined DG ramp and ESS swing capability.")
    if energy_deficit_mwh > 0.0:
        likely_causes.append("Total voyage energy demand exceeds DG energy plus deliverable ESS energy.")
    if route.get("valid_speed") is False:
        likely_causes.append("The final approach speed is outside the allowed port-speed range, which often corresponds to a distorted final segment and an extreme last-step load.")
    if not likely_causes and not milp_result["feasible"]:
        likely_causes.append("Infeasibility is likely driven by coupled commitment constraints such as min up/down time and start-stop transitions rather than simple aggregate caps.")

    return {
        "route_summary": {
            "valid": bool(route.get("valid")),
            "valid_departure_heading": bool(route.get("valid_departure_heading")),
            "valid_turning": bool(route.get("valid_turning")),
            "valid_speed": bool(route.get("valid_speed")),
            "valid_heading": bool(route.get("valid_heading")),
            "land_violation": float(route.get("land_violation", 0.0)),
            "last_speed_sog_kts": float(route.get("last_speed_sog", float("nan"))),
            "last_speed_stw_kts": float(route.get("last_speed", float("nan"))),
            "final_heading_delta_deg": float(route.get("final_heading_delta", float("nan"))),
        },
        "power_profile_summary": {
            "n_steps": len(p_req_list),
            "p_req_min_mw": min(p_req_list),
            "p_req_max_mw": max(p_req_list),
            "p_req_mean_mw": sum(p_req_list) / len(p_req_list),
            "top_power_steps": top_power_steps,
            "top_ramp_steps": top_ramp_steps,
        },
        "capacity_checks": {
            "total_dg_power_cap_mw": total_dg_power_cap,
            "total_supply_cap_mw": total_supply_cap,
            "total_required_energy_mwh": total_required_energy,
            "total_dg_energy_cap_mwh": total_dg_energy_cap,
            "ess_deliverable_energy_mwh": ess_deliverable_energy,
            "energy_deficit_mwh": energy_deficit_mwh,
            "power_excess_steps": power_excess_steps,
            "ramp_excess_steps": [step for step in ramp_steps if step["excess_mw"] > 0.0],
            "surrogate_penalty_value": surrogate,
        },
        "instantaneous_feasibility": {
            "reachable_ranges": [asdict(info) for info in reachable_ranges],
            "unreachable_steps": unreachable_steps,
        },
        "milp_result": {
            "feasible": bool(milp_result["feasible"]),
            "status": milp_result["status"],
            "solver_backend": milp_result.get("solver_backend"),
        },
        "likely_causes": likely_causes,
    }


def solve_variant(
    p_req_list: list[float],
    dt: list[float],
    *,
    initial_soc: float,
    initial_u: dict[str, int] | None = None,
    ramp_rate_scale: float | None = None,
    min_up_h: float | None = None,
    min_down_h: float | None = None,
    min_load_ratio: float | None = None,
) -> dict[str, Any]:
    milp = make_milp_solver(solver_name="cplex")
    milp.dg_specs = copy.deepcopy(milp.dg_specs)

    if ramp_rate_scale is not None:
        for dg in milp.dg_names:
            milp.dg_specs[dg]["ramp_rate"] = float(milp.dg_specs[dg]["ramp_rate"]) * float(ramp_rate_scale)

    if min_up_h is not None:
        for dg in milp.dg_names:
            milp.dg_specs[dg]["min_up"] = float(min_up_h)

    if min_down_h is not None:
        for dg in milp.dg_names:
            milp.dg_specs[dg]["min_down"] = float(min_down_h)

    if min_load_ratio is not None:
        milp.min_load_ratio = float(min_load_ratio)
        for dg in milp.dg_names:
            p_max = float(milp.dg_specs[dg]["P_max"])
            sfoc = milp.sfoc_data[dg]
            milp.pwl_breakpoints[dg] = []
            p_min = p_max * milp.min_load_ratio
            for k in range(milp.n_pwl + 1):
                p = p_min + (p_max - p_min) * k / milp.n_pwl
                sfoc_g_per_kwh = sfoc["alpha1"] * p ** 2 + sfoc["alpha2"] * p + sfoc["alpha3"]
                milp.pwl_breakpoints[dg].append((p, sfoc_g_per_kwh * p))

    result = milp.solve(
        P_req=p_req_list,
        dt=dt,
        initial_SOC=initial_soc,
        initial_u=initial_u,
        msg=False,
    )
    return {
        "feasible": bool(result["feasible"]),
        "status": result["status"],
        "solver_backend": result.get("solver_backend"),
    }


def diagnose_constraint_variants(
    p_req_list: list[float],
    dt: list[float],
    initial_soc: float,
    dg_names: list[str],
) -> dict[str, Any]:
    initial_all_on = {dg: 1 for dg in dg_names}
    variants = {
        "base": solve_variant(p_req_list, dt, initial_soc=initial_soc),
        "initial_all_on": solve_variant(p_req_list, dt, initial_soc=initial_soc, initial_u=initial_all_on),
        "no_min_up_down": solve_variant(
            p_req_list,
            dt,
            initial_soc=initial_soc,
            min_up_h=0.0,
            min_down_h=0.0,
        ),
        "no_ramp": solve_variant(
            p_req_list,
            dt,
            initial_soc=initial_soc,
            ramp_rate_scale=1000.0,
        ),
        "zero_min_load": solve_variant(
            p_req_list,
            dt,
            initial_soc=initial_soc,
            min_load_ratio=0.0,
        ),
        "fully_relaxed_commitment": solve_variant(
            p_req_list,
            dt,
            initial_soc=initial_soc,
            initial_u=initial_all_on,
            ramp_rate_scale=1000.0,
            min_up_h=0.0,
            min_down_h=0.0,
            min_load_ratio=0.0,
        ),
    }
    return variants


def main() -> None:
    out_dir = ensure_output_dir("case2")
    env_loader, env_fn, departure_time_utc = load_marine_environment()
    try:
        cost_map = build_cost_map(resolution=VERIFICATION_COST_MAP_RESOLUTION)
        ga_result = setup_ga(
            cost_map=cost_map,
            milp_solver=EnergyObjectiveSolver(),
            env_fn=env_fn,
            departure_time_utc=departure_time_utc,
            n_segments=N_SEGMENTS,
            rta_h=RTA_HOURS,
            pop_size=100,
            n_gen=100,
            seed=42,
            n_workers=1,
            smoothing_weight=0.0,
        )

        route = ga_result["best_route"]
        initial_soc = 0.7
        milp, power_profile, milp_result = solve_route_schedule(
            route,
            env_fn,
            departure_time_utc,
            initial_soc=initial_soc,
        )
        diagnosis = build_case2_diagnosis(
            route=route,
            power_profile=power_profile,
            milp=milp,
            milp_result=milp_result,
            initial_soc=initial_soc,
        )
        variants = diagnose_constraint_variants(
            p_req_list=[float(step["P_req"]) for step in power_profile],
            dt=[float(value) for value in route["dt"]],
            initial_soc=initial_soc,
            dg_names=list(milp.dg_names),
        )
        diagnosis["constraint_variants"] = variants

        if not diagnosis["milp_result"]["feasible"]:
            if variants["zero_min_load"]["feasible"]:
                diagnosis["likely_causes"].append("Relaxing generator minimum load makes the problem feasible, so minimum-load discreteness is a primary driver.")
            if variants["no_min_up_down"]["feasible"]:
                diagnosis["likely_causes"].append("Relaxing minimum up/down time makes the problem feasible, so commitment-duration constraints are a primary driver.")
            if variants["no_ramp"]["feasible"]:
                diagnosis["likely_causes"].append("Relaxing ramp limits makes the problem feasible, so ramping is a primary driver.")
            if variants["initial_all_on"]["feasible"]:
                diagnosis["likely_causes"].append("Starting with all generators already on makes the problem feasible, so the initial off-state and startup sequence are a primary driver.")
            if variants["fully_relaxed_commitment"]["feasible"] and not any(
                variants[name]["feasible"] for name in ("zero_min_load", "no_min_up_down", "no_ramp", "initial_all_on")
            ):
                diagnosis["likely_causes"].append("Feasibility is recovered only when several commitment constraints are relaxed together, indicating a combined interaction effect.")

        save_path = os.path.join(out_dir, "case2_milp_diagnosis.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(diagnosis, f, indent=2, ensure_ascii=False)

        print("=" * 60)
        print(" Case 2 MILP diagnosis")
        print("=" * 60)
        print(f"  MILP feasible: {diagnosis['milp_result']['feasible']}")
        print(f"  MILP status: {diagnosis['milp_result']['status']}")
        print(f"  Last speed SOG: {diagnosis['route_summary']['last_speed_sog_kts']:.3f} kts")
        print(f"  P_req max: {diagnosis['power_profile_summary']['p_req_max_mw']:.3f} MW")
        print(f"  Energy deficit: {diagnosis['capacity_checks']['energy_deficit_mwh']:.3f} MWh")
        print(f"  Power-cap violation steps: {len(diagnosis['capacity_checks']['power_excess_steps'])}")
        print(f"  Ramp violation steps: {len(diagnosis['capacity_checks']['ramp_excess_steps'])}")
        print(f"  Instantaneously unreachable steps: {len(diagnosis['instantaneous_feasibility']['unreachable_steps'])}")
        print("  Constraint variants:")
        for name, result in variants.items():
            print(f"    - {name}: feasible={result['feasible']} status={result['status']}")
        if diagnosis["likely_causes"]:
            print("  Likely causes:")
            for cause in diagnosis["likely_causes"]:
                print(f"    - {cause}")
        print(f"  Saved: {save_path}")
    finally:
        del env_loader


if __name__ == "__main__":
    main()
