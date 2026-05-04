"""
Same-energy crossover evidence for the integrated GA+MILP fitness map.

This is a controlled P_req-profile-level proxy, not a full route-gene GA rerun.
The purpose is to isolate the MILP black-box response after crossover-like
recombination while keeping total required energy fixed.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
OUTPUT_DIR = SCRIPT_DIR / "crossover_disruption_evidence"
CASE2_ARTIFACT = PROJECT_ROOT / "output" / "verification" / "case2" / "case_artifact.json"
CASE3_ARTIFACT = PROJECT_ROOT / "output" / "verification" / "case3" / "case_artifact.json"
SFOC_PATH = PROJECT_ROOT / "config" / "sfoc.json"

INITIAL_SOC = 0.7
TIME_LIMIT_SEC = 15
N_ARITHMETIC = 41
N_UNIFORM = 60
N_SBX = 60
RANDOM_SEED = 20260428
SBX_ETA = 20.0
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


def read_artifact(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, dict):
        return {str(key): json_ready(sub_value) for key, sub_value in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    return value


def power_profile(artifact: dict[str, Any]) -> np.ndarray:
    return np.asarray([float(item["P_req"]) for item in artifact["power_profile"]], dtype=float)


def route_dt(artifact: dict[str, Any]) -> np.ndarray:
    return np.asarray([float(value) for value in artifact["route"]["dt"]], dtype=float)


def energy_mwh(p_req: np.ndarray, dt: np.ndarray) -> float:
    return float(np.dot(p_req, dt))


def scale_to_energy(p_req: np.ndarray, dt: np.ndarray, target_energy: float) -> np.ndarray:
    current = energy_mwh(p_req, dt)
    if current <= 0.0:
        raise ValueError("Cannot scale a non-positive energy profile.")
    return p_req * (target_energy / current)


def compensate_to_energy(p_req: np.ndarray, dt: np.ndarray, target_energy: float) -> np.ndarray:
    error = target_energy - energy_mwh(p_req, dt)
    return p_req + error / float(np.sum(dt))


def sbx_beta(rng: random.Random, eta: float) -> float:
    u = rng.random()
    if u <= 0.5:
        return (2.0 * u) ** (1.0 / (eta + 1.0))
    return (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))


def make_children(parent_a: np.ndarray, parent_b: np.ndarray, dt: np.ndarray) -> list[dict[str, Any]]:
    target_energy = energy_mwh(parent_a, dt)
    rng = random.Random(RANDOM_SEED)
    children: list[dict[str, Any]] = [
        {"method": "parent", "child_id": "case2_seed", "alpha": 1.0, "p_req": parent_a},
        {"method": "parent", "child_id": "case3_scaled", "alpha": 0.0, "p_req": parent_b},
    ]

    for idx, alpha in enumerate(np.linspace(0.0, 1.0, N_ARITHMETIC)):
        profile = alpha * parent_a + (1.0 - alpha) * parent_b
        children.append(
            {
                "method": "arithmetic",
                "child_id": f"arithmetic_{idx:03d}",
                "alpha": float(alpha),
                "p_req": compensate_to_energy(profile, dt, target_energy),
            }
        )

    for idx in range(N_UNIFORM):
        mask = np.asarray([rng.random() < 0.5 for _ in range(len(parent_a))], dtype=bool)
        profile = np.where(mask, parent_a, parent_b)
        children.append(
            {
                "method": "uniform",
                "child_id": f"uniform_{idx:03d}",
                "alpha": float(np.mean(mask)),
                "p_req": compensate_to_energy(profile, dt, target_energy),
            }
        )

    for idx in range(N_SBX):
        betas = np.asarray([sbx_beta(rng, SBX_ETA) for _ in range(len(parent_a))], dtype=float)
        profile = 0.5 * ((1.0 + betas) * parent_a + (1.0 - betas) * parent_b)
        children.append(
            {
                "method": "sbx_profile_proxy",
                "child_id": f"sbx_{idx:03d}",
                "alpha": float(np.mean((1.0 + betas) / 2.0)),
                "p_req": compensate_to_energy(profile, dt, target_energy),
            }
        )

    return children


def commitment_signature(schedule: list[dict[str, Any]]) -> str:
    counts = []
    for step in schedule:
        counts.append(str(sum(int(step.get(f"{dg}_ON", 0)) for dg in DG_NAMES)))
    return "".join(counts)


def dg_count_switches(schedule: list[dict[str, Any]]) -> int:
    counts = [sum(int(step.get(f"{dg}_ON", 0)) for dg in DG_NAMES) for step in schedule]
    return int(sum(1 for prev, curr in zip(counts, counts[1:]) if prev != curr))


def evaluate_children(children: list[dict[str, Any]], dt: np.ndarray) -> list[dict[str, Any]]:
    solver = MILPSolver(
        sfoc_json_path=str(SFOC_PATH),
        n_pwl_segments=5,
        solver_name="cplex_cmd",
        enable_output_n_minus_1=True,
    )
    rows: list[dict[str, Any]] = []
    for idx, child in enumerate(children):
        p_req = np.asarray(child["p_req"], dtype=float)
        result = solver.solve(
            P_req=p_req.tolist(),
            dt=dt.tolist(),
            initial_SOC=INITIAL_SOC,
            msg=False,
            time_limit_sec=TIME_LIMIT_SEC,
            enable_output_n_minus_1=True,
        )
        row: dict[str, Any] = {
            "method": child["method"],
            "child_id": child["child_id"],
            "alpha": child["alpha"],
            "energy_mwh": energy_mwh(p_req, dt),
            "p_req_mean_mw": float(np.mean(p_req)),
            "p_req_std_mw": float(np.std(p_req)),
            "p_req_min_mw": float(np.min(p_req)),
            "p_req_max_mw": float(np.max(p_req)),
            "mean_abs_ramp_mw": float(np.mean(np.abs(np.diff(p_req)))),
            "feasible": bool(result.get("feasible")),
            "status": result.get("status"),
            "fuel_kg": None,
            "operating_fuel_kg": None,
            "start_fuel_kg": None,
            "dg_count_switches": None,
            "commitment_signature": None,
        }
        if result.get("feasible"):
            schedule = result["schedule"]
            row.update(
                {
                    "fuel_kg": float(result["total_fuel_kg"]),
                    "operating_fuel_kg": float(result["total_operating_fuel_kg"]),
                    "start_fuel_kg": float(result["total_start_fuel_kg"]),
                    "dg_count_switches": dg_count_switches(schedule),
                    "commitment_signature": commitment_signature(schedule),
                }
            )
        rows.append(row)
        if idx == 0 or (idx + 1) % 20 == 0 or idx + 1 == len(children):
            print(f"  crossover proxy MILP evaluations: {idx + 1:>3}/{len(children)}")
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [row for row in rows if row["feasible"] and row["fuel_kg"] is not None]
    by_method: dict[str, dict[str, Any]] = {}
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in feasible if row["method"] == method]
        fuels = np.asarray([float(row["fuel_kg"]) for row in method_rows], dtype=float)
        starts = np.asarray([float(row["start_fuel_kg"]) for row in method_rows], dtype=float)
        switches = np.asarray([float(row["dg_count_switches"]) for row in method_rows], dtype=float)
        if len(method_rows) == 0:
            by_method[method] = {"feasible_count": 0}
            continue
        by_method[method] = {
            "feasible_count": len(method_rows),
            "fuel_min_kg": float(np.min(fuels)),
            "fuel_max_kg": float(np.max(fuels)),
            "fuel_range_kg": float(np.max(fuels) - np.min(fuels)),
            "fuel_std_kg": float(np.std(fuels)),
            "start_fuel_min_kg": float(np.min(starts)),
            "start_fuel_max_kg": float(np.max(starts)),
            "dg_count_switches_mean": float(np.mean(switches)),
            "dg_count_switches_max": float(np.max(switches)),
        }

    parent_rows = {row["child_id"]: row for row in feasible if row["method"] == "parent"}
    parent_gap = None
    if "case2_seed" in parent_rows and "case3_scaled" in parent_rows:
        parent_gap = float(parent_rows["case3_scaled"]["fuel_kg"] - parent_rows["case2_seed"]["fuel_kg"])

    best = min(feasible, key=lambda row: float(row["fuel_kg"]))
    worst = max(feasible, key=lambda row: float(row["fuel_kg"]))
    return {
        "settings": {
            "initial_soc": INITIAL_SOC,
            "n_arithmetic": N_ARITHMETIC,
            "n_uniform": N_UNIFORM,
            "n_sbx_profile_proxy": N_SBX,
            "random_seed": RANDOM_SEED,
            "sbx_eta": SBX_ETA,
            "n1_contingency": True,
            "note": "Same-energy P_req-profile recombination proxy; not a full route-gene SBX rerun.",
        },
        "parent_gap_case3_scaled_minus_case2_kg": parent_gap,
        "by_method": by_method,
        "best_child": best,
        "worst_child": worst,
    }


def plot(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    configure_fonts()
    feasible = [row for row in rows if row["feasible"] and row["fuel_kg"] is not None]
    methods = ["parent", "arithmetic", "uniform", "sbx_profile_proxy"]
    colors = {
        "parent": "#111111",
        "arithmetic": "#2a9d8f",
        "uniform": "#e76f51",
        "sbx_profile_proxy": "#457b9d",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    ax0, ax1, ax2 = axes

    for method in methods:
        method_rows = [row for row in feasible if row["method"] == method]
        if not method_rows:
            continue
        x = [float(row["alpha"]) for row in method_rows]
        y = [float(row["fuel_kg"]) for row in method_rows]
        ax0.scatter(x, y, s=30 if method == "parent" else 18, alpha=0.82, label=method, color=colors[method])
    ax0.set_xlabel("case2 mixing share / mean SBX share")
    ax0.set_ylabel("N-1 MILP total fuel (kg)")
    ax0.set_title("Same-energy crossover response")
    ax0.grid(True, alpha=0.25)
    ax0.legend(fontsize=8)

    box_data = [[float(row["fuel_kg"]) for row in feasible if row["method"] == method] for method in methods]
    ax1.boxplot(box_data, labels=methods, showfliers=True)
    ax1.set_ylabel("N-1 MILP total fuel (kg)")
    ax1.set_title("Fuel dispersion by recombination type")
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(True, axis="y", alpha=0.25)

    for method in methods:
        method_rows = [row for row in feasible if row["method"] == method]
        if not method_rows:
            continue
        ax2.scatter(
            [float(row["p_req_std_mw"]) for row in method_rows],
            [float(row["fuel_kg"]) for row in method_rows],
            s=30 if method == "parent" else 18,
            alpha=0.82,
            color=colors[method],
            label=method,
        )
    ax2.set_xlabel("P_req profile std (MW)")
    ax2.set_ylabel("N-1 MILP total fuel (kg)")
    ax2.set_title("Same energy, different temporal structure")
    ax2.grid(True, alpha=0.25)

    text = (
        f"Parent fuel gap: {summary['parent_gap_case3_scaled_minus_case2_kg']:.1f} kg\n"
        f"Uniform range: {summary['by_method']['uniform']['fuel_range_kg']:.1f} kg\n"
        f"SBX-proxy range: {summary['by_method']['sbx_profile_proxy']['fuel_range_kg']:.1f} kg"
    )
    ax2.text(
        0.02,
        0.98,
        text,
        transform=ax2.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fffde7", "edgecolor": "#8d6e63"},
    )

    fig.suptitle("Crossover Disruption Evidence: energy is fixed, MILP fitness is not", fontsize=13)
    fig.savefig(OUTPUT_DIR / "crossover_disruption_evidence.png", dpi=300)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    case2 = read_artifact(CASE2_ARTIFACT)
    case3 = read_artifact(CASE3_ARTIFACT)
    dt = route_dt(case2)
    parent_a = power_profile(case2)
    parent_b_raw = power_profile(case3)
    parent_b = scale_to_energy(parent_b_raw, dt, energy_mwh(parent_a, dt))

    children = make_children(parent_a, parent_b, dt)
    rows = evaluate_children(children, dt)
    summary = summarize(rows)

    write_csv(OUTPUT_DIR / "crossover_disruption_children.csv", rows)
    (OUTPUT_DIR / "crossover_disruption_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2),
        encoding="utf-8",
    )
    plot(rows, summary)
    print(json.dumps(json_ready(summary), indent=2))


if __name__ == "__main__":
    main()
