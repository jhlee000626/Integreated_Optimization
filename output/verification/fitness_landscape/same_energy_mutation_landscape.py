"""
Same-energy profile mutation test for the GA+MILP fitness landscape.

Purpose
-------
Compare two different seed-derived power profiles under the same total required
energy, then perturb each timestep while preserving that total energy. This
tests whether MILP fuel fitness depends only on total energy or also on the
temporal load distribution and commitment-regime changes.

The experiment is deliberately simple:
  1. Use Case2 as the reference profile.
  2. Scale Case3's P_req profile to exactly the same total energy.
  3. For each timestep t, apply +/- DELTA_MW to P_req[t].
  4. Distribute the opposite energy change uniformly over all other timesteps.
  5. Solve N-1 MILP and record fuel, DG-count changes, and commitment changes.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


# ---------------------------------------------------------------------------
# Edit these values only.
# ---------------------------------------------------------------------------
OUTPUT_DIR_NAME = "same_energy_mutation"

REFERENCE_ARTIFACT = Path("output/verification/case2/case_artifact.json")
COMPARISON_ARTIFACT = Path("output/verification/case3/case_artifact.json")

DELTA_MW = 0.25
INITIAL_SOC = 0.7
TIME_LIMIT_SEC = 15


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
OUTPUT_DIR = SCRIPT_DIR / OUTPUT_DIR_NAME
SFOC_PATH = PROJECT_ROOT / "config" / "sfoc.json"
DG_NAMES = ["DG1", "DG2", "DG3", "DG4"]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.optimizer.milp_solver import MILPSolver  # noqa: E402


def configure_fonts() -> None:
    for font_path in font_manager.findSystemFonts():
        if "malgun" in font_path.lower():
            font_manager.fontManager.addfont(font_path)
            plt.rcParams["font.family"] = "Malgun Gothic"
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_artifact(path: Path) -> dict[str, Any]:
    full_path = PROJECT_ROOT / path
    with full_path.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    artifact["_source_path"] = str(full_path)
    return artifact


def profile_from_artifact(artifact: dict[str, Any]) -> tuple[list[float], list[float]]:
    p_req = [float(segment["P_req"]) for segment in artifact["power_profile"]]
    dt = [float(value) for value in artifact["route"]["dt"]]
    return p_req, dt


def total_energy_mwh(p_req: list[float], dt: list[float]) -> float:
    return float(sum(p * duration for p, duration in zip(p_req, dt)))


def scale_to_energy(p_req: list[float], dt: list[float], target_energy: float) -> list[float]:
    current_energy = total_energy_mwh(p_req, dt)
    if current_energy <= 0.0:
        raise ValueError("Cannot scale a non-positive energy profile.")
    scale = target_energy / current_energy
    return [float(p * scale) for p in p_req]


def mutate_preserve_energy(
    p_req: list[float],
    dt: list[float],
    t: int,
    delta_mw: float,
) -> tuple[list[float], float]:
    mutated = list(p_req)
    other_dt = sum(duration for idx, duration in enumerate(dt) if idx != t)
    if other_dt <= 0.0:
        raise ValueError("Need at least two timesteps for energy-preserving mutation.")
    compensation_mw = delta_mw * dt[t] / other_dt
    mutated[t] += delta_mw
    for idx in range(len(mutated)):
        if idx != t:
            mutated[idx] -= compensation_mw
    return mutated, compensation_mw


def solve_profile(solver: MILPSolver, p_req: list[float], dt: list[float]) -> dict[str, Any]:
    return solver.solve(
        P_req=[float(value) for value in p_req],
        dt=[float(value) for value in dt],
        initial_SOC=INITIAL_SOC,
        msg=False,
        time_limit_sec=TIME_LIMIT_SEC,
        enable_output_n_minus_1=True,
    )


def dg_count(step: dict[str, Any]) -> int:
    return sum(int(step.get(f"{dg}_ON", 0)) for dg in DG_NAMES)


def commitment_label(step: dict[str, Any]) -> str:
    online = [dg for dg in DG_NAMES if int(step.get(f"{dg}_ON", 0)) == 1]
    return "+".join(online) if online else "OFF"


def schedule_counts(result: dict[str, Any]) -> list[int]:
    if not result.get("feasible"):
        return []
    return [dg_count(step) for step in result["schedule"]]


def schedule_labels(result: dict[str, Any]) -> list[str]:
    if not result.get("feasible"):
        return []
    return [commitment_label(step) for step in result["schedule"]]


def profile_stats(p_req: list[float], dt: list[float]) -> dict[str, float]:
    values = np.asarray(p_req, dtype=float)
    return {
        "energy_mwh": total_energy_mwh(p_req, dt),
        "mean_mw": float(np.mean(values)),
        "std_mw": float(np.std(values)),
        "min_mw": float(np.min(values)),
        "max_mw": float(np.max(values)),
        "mean_abs_ramp_mw": float(np.mean(np.abs(np.diff(values)))) if values.size > 1 else 0.0,
    }


def row_for_mutation(
    profile_name: str,
    base_p_req: list[float],
    mutated_p_req: list[float],
    dt: list[float],
    t: int,
    delta_mw: float,
    compensation_mw: float,
    base_result: dict[str, Any],
    mutated_result: dict[str, Any],
) -> dict[str, Any]:
    base_counts = schedule_counts(base_result)
    mut_counts = schedule_counts(mutated_result)
    base_labels = schedule_labels(base_result)
    mut_labels = schedule_labels(mutated_result)

    count_changes = (
        sum(1 for a, b in zip(base_counts, mut_counts) if a != b)
        if base_counts and mut_counts
        else None
    )
    commitment_changes = (
        sum(1 for a, b in zip(base_labels, mut_labels) if a != b)
        if base_labels and mut_labels
        else None
    )

    row: dict[str, Any] = {
        "profile": profile_name,
        "t": int(t),
        "delta_mw": float(delta_mw),
        "compensation_mw_on_other_steps": float(compensation_mw),
        "base_P_req_t_MW": float(base_p_req[t]),
        "mutated_P_req_t_MW": float(mutated_p_req[t]),
        "base_energy_mwh": total_energy_mwh(base_p_req, dt),
        "mutated_energy_mwh": total_energy_mwh(mutated_p_req, dt),
        "energy_error_mwh": total_energy_mwh(mutated_p_req, dt) - total_energy_mwh(base_p_req, dt),
        "feasible": bool(mutated_result.get("feasible")),
        "base_fuel_kg": float(base_result["total_fuel_kg"]) if base_result.get("feasible") else None,
        "mutated_fuel_kg": float(mutated_result["total_fuel_kg"]) if mutated_result.get("feasible") else None,
        "delta_fuel_kg": None,
        "abs_delta_fuel_kg": None,
        "dg_count_changes": count_changes,
        "commitment_pattern_changes": commitment_changes,
        "base_count_at_t": base_counts[t] if base_counts else None,
        "mutated_count_at_t": mut_counts[t] if mut_counts else None,
        "base_commitment_at_t": base_labels[t] if base_labels else None,
        "mutated_commitment_at_t": mut_labels[t] if mut_labels else None,
    }
    if base_result.get("feasible") and mutated_result.get("feasible"):
        delta_fuel = float(mutated_result["total_fuel_kg"]) - float(base_result["total_fuel_kg"])
        row["delta_fuel_kg"] = delta_fuel
        row["abs_delta_fuel_kg"] = abs(delta_fuel)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, dict):
        return {str(key): json_ready(sub_value) for key, sub_value in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def summarize_mutations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for profile in sorted({str(row["profile"]) for row in rows}):
        subset = [row for row in rows if row["profile"] == profile and row["feasible"]]
        deltas = [float(row["delta_fuel_kg"]) for row in subset if row["delta_fuel_kg"] is not None]
        abs_deltas = [abs(value) for value in deltas]
        commitment_changes = [
            int(row["commitment_pattern_changes"])
            for row in subset
            if row["commitment_pattern_changes"] is not None
        ]
        dg_count_changes = [
            int(row["dg_count_changes"])
            for row in subset
            if row["dg_count_changes"] is not None
        ]
        top = sorted(
            subset,
            key=lambda row: -float(row["abs_delta_fuel_kg"] or -1.0),
        )[:8]
        summary[profile] = {
            "feasible_mutations": len(subset),
            "mean_abs_delta_fuel_kg": float(np.mean(abs_deltas)) if abs_deltas else None,
            "max_abs_delta_fuel_kg": float(np.max(abs_deltas)) if abs_deltas else None,
            "std_delta_fuel_kg": float(np.std(deltas)) if deltas else None,
            "mean_commitment_pattern_changes": float(np.mean(commitment_changes)) if commitment_changes else None,
            "max_commitment_pattern_changes": int(np.max(commitment_changes)) if commitment_changes else None,
            "mean_dg_count_changes": float(np.mean(dg_count_changes)) if dg_count_changes else None,
            "max_dg_count_changes": int(np.max(dg_count_changes)) if dg_count_changes else None,
            "top_mutations": top,
        }
    return summary


def plot_results(
    profiles: dict[str, dict[str, Any]],
    mutation_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    configure_fonts()
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=300)

    ax = axes[0, 0]
    for name, payload in profiles.items():
        p_req = payload["p_req"]
        ax.plot(range(len(p_req)), p_req, linewidth=2.0, label=name)
    ax.set_title("Same-energy profiles with different temporal shapes")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("P_req (MW)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    names = list(profiles)
    fuels = [profiles[name]["fuel_kg"] for name in names]
    stds = [profiles[name]["stats"]["std_mw"] for name in names]
    bars = ax.bar(names, fuels, color=["#1565c0", "#c62828"])
    ax.set_title("Same total energy, different MILP fuel")
    ax.set_ylabel("MILP total fuel incl. starts (kg)")
    ax2 = ax.twinx()
    ax2.plot(names, stds, color="#6d4c41", marker="o", linewidth=2.0, label="P_req std")
    ax2.set_ylabel("P_req std (MW)")
    for bar, fuel in zip(bars, fuels):
        ax.text(bar.get_x() + bar.get_width() / 2, fuel, f"{fuel:.1f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1, 0]
    for name in profiles:
        for delta, linestyle in [(DELTA_MW, "-"), (-DELTA_MW, "--")]:
            rows = [
                row for row in mutation_rows
                if row["profile"] == name and abs(float(row["delta_mw"]) - delta) < 1e-12 and row["feasible"]
            ]
            rows.sort(key=lambda row: int(row["t"]))
            ax.plot(
                [int(row["t"]) for row in rows],
                [float(row["delta_fuel_kg"]) for row in rows],
                linestyle=linestyle,
                linewidth=1.8,
                label=f"{name} {delta:+.2f} MW",
            )
    ax.axhline(0.0, color="#424242", linewidth=1.0)
    ax.set_title("Energy-preserving one-timestep mutation response")
    ax.set_xlabel("Mutated timestep")
    ax.set_ylabel("Delta MILP fuel (kg)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for name in profiles:
        rows = [row for row in mutation_rows if row["profile"] == name and row["feasible"]]
        ax.scatter(
            [int(row["commitment_pattern_changes"]) for row in rows],
            [float(row["abs_delta_fuel_kg"]) for row in rows],
            s=18,
            alpha=0.75,
            label=name,
        )
    ax.set_title("Fuel sensitivity vs temporal commitment disruption")
    ax.set_xlabel("Number of timesteps with changed commitment pattern")
    ax.set_ylabel("|Delta fuel| (kg)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.suptitle("Same-Energy Mutation Landscape under N-1 MILP", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUTPUT_DIR / "same_energy_mutation_landscape.png")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ref_artifact = load_artifact(REFERENCE_ARTIFACT)
    cmp_artifact = load_artifact(COMPARISON_ARTIFACT)
    ref_p_req, ref_dt = profile_from_artifact(ref_artifact)
    cmp_p_req_raw, cmp_dt = profile_from_artifact(cmp_artifact)

    if len(ref_p_req) != len(cmp_p_req_raw):
        raise ValueError("Profiles must have the same number of timesteps.")
    if len(ref_dt) != len(cmp_dt) or any(abs(a - b) > 1e-9 for a, b in zip(ref_dt, cmp_dt)):
        raise ValueError("Profiles must share the same dt vector for this controlled test.")

    target_energy = total_energy_mwh(ref_p_req, ref_dt)
    cmp_p_req_equal = scale_to_energy(cmp_p_req_raw, cmp_dt, target_energy)

    solver = MILPSolver(
        sfoc_json_path=str(SFOC_PATH),
        n_pwl_segments=5,
        solver_name="cplex_cmd",
        enable_output_n_minus_1=True,
    )

    profiles: dict[str, dict[str, Any]] = {
        "case2_seed_profile": {
            "artifact": ref_artifact,
            "p_req": ref_p_req,
            "dt": ref_dt,
        },
        "case3_seed_profile_scaled_to_case2_energy": {
            "artifact": cmp_artifact,
            "p_req": cmp_p_req_equal,
            "dt": cmp_dt,
        },
    }

    mutation_rows: list[dict[str, Any]] = []
    for name, payload in profiles.items():
        p_req = payload["p_req"]
        dt = payload["dt"]
        base_result = solve_profile(solver, p_req, dt)
        payload["base_result"] = base_result
        payload["fuel_kg"] = float(base_result["total_fuel_kg"]) if base_result.get("feasible") else None
        payload["stats"] = profile_stats(p_req, dt)
        payload["feasible"] = bool(base_result.get("feasible"))
        print(f"Base solve: {name} feasible={payload['feasible']} fuel={payload['fuel_kg']}")

        for t in range(len(p_req)):
            for delta_mw in (DELTA_MW, -DELTA_MW):
                mutated, compensation = mutate_preserve_energy(p_req, dt, t, delta_mw)
                if min(mutated) <= 0.0:
                    continue
                result = solve_profile(solver, mutated, dt)
                mutation_rows.append(
                    row_for_mutation(
                        profile_name=name,
                        base_p_req=p_req,
                        mutated_p_req=mutated,
                        dt=dt,
                        t=t,
                        delta_mw=delta_mw,
                        compensation_mw=compensation,
                        base_result=base_result,
                        mutated_result=result,
                    )
                )
            if t == 0 or (t + 1) % 8 == 0 or t + 1 == len(p_req):
                print(f"  {name}: timestep {t + 1:>2}/{len(p_req)} mutation solved")

    summary = {
        "settings": {
            "reference_artifact": str(PROJECT_ROOT / REFERENCE_ARTIFACT),
            "comparison_artifact": str(PROJECT_ROOT / COMPARISON_ARTIFACT),
            "target_energy_mwh": target_energy,
            "delta_mw": DELTA_MW,
            "initial_soc": INITIAL_SOC,
            "time_limit_sec": TIME_LIMIT_SEC,
            "n_minus_1_enabled": True,
        },
        "profiles": {
            name: {
                "case_name": payload["artifact"].get("case_name"),
                "source_path": payload["artifact"].get("_source_path"),
                "feasible": payload["feasible"],
                "fuel_kg": payload["fuel_kg"],
                "stats": payload["stats"],
                "dg_start_count": payload["base_result"].get("summary", {}).get("dg_start_count", {}),
            }
            for name, payload in profiles.items()
        },
        "mutation_summary": summarize_mutations(mutation_rows),
    }

    write_csv(OUTPUT_DIR / "same_energy_profiles.csv", [
        {
            "profile": name,
            "t": idx,
            "P_req_MW": float(value),
            "dt_h": float(payload["dt"][idx]),
        }
        for name, payload in profiles.items()
        for idx, value in enumerate(payload["p_req"])
    ])
    write_csv(OUTPUT_DIR / "same_energy_mutations.csv", mutation_rows)
    write_json(OUTPUT_DIR / "same_energy_mutation_summary.json", summary)
    plot_results(profiles, mutation_rows, summary)

    print(json.dumps(json_ready(summary), indent=2))


if __name__ == "__main__":
    main()
