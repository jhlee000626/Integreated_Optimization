"""
Evidence for why integrated GA+MILP converges more slowly than sequential GA.

This script focuses on the computational bottleneck:
  sequential evaluation = route physics + energy objective
  integrated evaluation = route physics + N-1 MILP scheduling

It also reuses the same-energy mutation result to report whether equal-energy
perturbations can cause non-proportional MILP fuel responses.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


# ---------------------------------------------------------------------------
# Edit these values only.
# ---------------------------------------------------------------------------
OUTPUT_DIR_NAME = "convergence_cost_evidence"
N_REPEATS = 8
INITIAL_SOC = 0.7
TIME_LIMIT_SEC = 15

CASE_ARTIFACTS = {
    "case2_twostage": Path("output/verification/case2/case_artifact.json"),
    "case3_integrated": Path("output/verification/case3/case_artifact.json"),
}


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
OUTPUT_DIR = SCRIPT_DIR / OUTPUT_DIR_NAME
SFOC_PATH = PROJECT_ROOT / "config" / "sfoc.json"
SAME_ENERGY_SUMMARY = SCRIPT_DIR / "same_energy_mutation" / "same_energy_mutation_summary.json"
SAME_ENERGY_MUTATIONS = SCRIPT_DIR / "same_energy_mutation" / "same_energy_mutations.csv"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.optimizer.ga_engine import build_required_power_profile  # noqa: E402
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


def load_environment() -> tuple[Callable | None, Any | None]:
    try:
        from Validation.common import load_marine_environment

        loader, env_fn, departure_time_utc = load_marine_environment()
        return env_fn, departure_time_utc
    except Exception as exc:
        print(f"Environment load failed; using cached power profiles only: {exc}")
        return None, None


def cached_profile(artifact: dict[str, Any]) -> tuple[list[float], list[float]]:
    p_req = [float(segment["P_req"]) for segment in artifact["power_profile"]]
    dt = [float(value) for value in artifact["route"]["dt"]]
    return p_req, dt


def sequential_energy_eval(
    artifact: dict[str, Any],
    env_fn: Callable | None,
    departure_time_utc: Any | None,
) -> tuple[float, list[float], list[float]]:
    if env_fn is None:
        p_req, dt = cached_profile(artifact)
    else:
        profile = build_required_power_profile(
            artifact["route"],
            env_fn=env_fn,
            departure_time_utc=departure_time_utc,
        )
        p_req = [float(segment["P_req"]) for segment in profile]
        dt = [float(value) for value in artifact["route"]["dt"]]
    energy = float(sum(p * duration for p, duration in zip(p_req, dt)))
    return energy, p_req, dt


def solve_n1_milp(solver: MILPSolver, p_req: list[float], dt: list[float]) -> dict[str, Any]:
    return solver.solve(
        P_req=p_req,
        dt=dt,
        initial_SOC=INITIAL_SOC,
        msg=False,
        time_limit_sec=TIME_LIMIT_SEC,
        enable_output_n_minus_1=True,
    )


def timed_call(fn):
    start = time.perf_counter()
    value = fn()
    return time.perf_counter() - start, value


def summarize_times(values: list[float]) -> dict[str, float]:
    return {
        "mean_sec": float(statistics.mean(values)),
        "median_sec": float(statistics.median(values)),
        "min_sec": float(min(values)),
        "max_sec": float(max(values)),
        "std_sec": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mutation_metrics() -> dict[str, Any]:
    if not SAME_ENERGY_SUMMARY.exists():
        return {"available": False}
    summary = json.loads(SAME_ENERGY_SUMMARY.read_text(encoding="utf-8"))
    rows = read_csv(SAME_ENERGY_MUTATIONS)
    compact: dict[str, Any] = {"available": True, "profiles": {}}
    for profile, payload in summary["mutation_summary"].items():
        compact["profiles"][profile] = {
            "mean_abs_delta_fuel_kg": payload["mean_abs_delta_fuel_kg"],
            "max_abs_delta_fuel_kg": payload["max_abs_delta_fuel_kg"],
            "mean_dg_count_changes": payload["mean_dg_count_changes"],
            "max_dg_count_changes": payload["max_dg_count_changes"],
        }
    compact["rows"] = rows
    return compact


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def json_ready(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, dict):
        return {str(key): json_ready(sub_value) for key, sub_value in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def plot_results(results: dict[str, Any]) -> None:
    configure_fonts()
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=300)

    names = list(results["cases"])
    seq_times = [results["cases"][name]["sequential_energy_eval"]["median_sec"] for name in names]
    int_times = [results["cases"][name]["integrated_milp_eval"]["median_sec"] for name in names]
    ratios = [results["cases"][name]["eval_time_ratio_milp_over_energy"] for name in names]

    ax = axes[0, 0]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, seq_times, width, label="Sequential energy eval", color="#1565c0")
    ax.bar(x + width / 2, int_times, width, label="Integrated MILP eval", color="#c62828")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10)
    ax.set_ylabel("Median time per evaluation (s)")
    ax.set_title("A. Per-individual evaluation cost")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    ax.bar(names, ratios, color="#6d4c41")
    ax.set_ylabel("MILP eval / energy eval time ratio")
    ax.set_title("Integrated fitness is an expensive black-box call")
    ax.grid(True, axis="y", alpha=0.25)
    for idx, value in enumerate(ratios):
        ax.text(idx, value, f"{value:.1f}x", ha="center", va="bottom")

    mut = results.get("same_energy_mutation", {})
    ax = axes[1, 0]
    if mut.get("available") and mut.get("rows"):
        for profile in sorted({row["profile"] for row in mut["rows"]}):
            subset = [
                row for row in mut["rows"]
                if row["profile"] == profile and row["feasible"] == "True"
            ]
            ax.scatter(
                [int(row["dg_count_changes"]) for row in subset],
                [float(row["abs_delta_fuel_kg"]) for row in subset],
                s=18,
                alpha=0.75,
                label=profile,
            )
        ax.set_xlabel("DG-count changed timesteps")
        ax.set_ylabel("|Delta fuel| under equal-energy mutation (kg)")
        ax.set_title("B. Mutation response depends on commitment disruption")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "same-energy mutation data not available", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[1, 1]
    ax.axis("off")
    text = (
        "Interpretation\n\n"
        "Sequential GA evaluates a smooth continuous surrogate: route physics + required energy.\n"
        "Integrated GA evaluates route physics plus an N-1 MILP unit-commitment problem.\n\n"
        "Therefore the slow convergence mechanism is two-layered:\n"
        "1. each fitness call is much more expensive;\n"
        "2. near commitment boundaries, equal-size mutations can produce non-proportional fuel responses."
    )
    ax.text(
        0.02,
        0.98,
        text,
        va="top",
        ha="left",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#fffde7", "edgecolor": "#8d6e63"},
    )

    fig.suptitle("Convergence Bottleneck Evidence: Cost + Regime-Switching Fitness", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUTPUT_DIR / "convergence_cost_evidence.png")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    env_fn, departure_time_utc = load_environment()
    solver = MILPSolver(
        sfoc_json_path=str(SFOC_PATH),
        n_pwl_segments=5,
        solver_name="cplex_cmd",
        enable_output_n_minus_1=True,
    )

    cases: dict[str, Any] = {}
    for case_name, path in CASE_ARTIFACTS.items():
        artifact = load_artifact(path)
        seq_times: list[float] = []
        milp_times: list[float] = []
        last_energy = None
        last_fuel = None
        last_feasible = None
        p_req = []
        dt = []

        for idx in range(N_REPEATS):
            elapsed, seq_value = timed_call(lambda: sequential_energy_eval(artifact, env_fn, departure_time_utc))
            energy, p_req, dt = seq_value
            seq_times.append(elapsed)
            last_energy = energy

            elapsed, result = timed_call(lambda: solve_n1_milp(solver, p_req, dt))
            milp_times.append(elapsed)
            last_feasible = bool(result.get("feasible"))
            last_fuel = float(result["total_fuel_kg"]) if result.get("feasible") else None
            print(f"{case_name}: repeat {idx + 1}/{N_REPEATS} seq={seq_times[-1]:.4f}s milp={milp_times[-1]:.4f}s")

        seq_summary = summarize_times(seq_times)
        milp_summary = summarize_times(milp_times)
        cases[case_name] = {
            "artifact_path": str(PROJECT_ROOT / path),
            "energy_mwh": last_energy,
            "milp_feasible": last_feasible,
            "milp_fuel_kg": last_fuel,
            "p_req_min_mw": float(np.min(p_req)),
            "p_req_max_mw": float(np.max(p_req)),
            "p_req_std_mw": float(np.std(p_req)),
            "sequential_energy_eval": seq_summary,
            "integrated_milp_eval": milp_summary,
            "eval_time_ratio_milp_over_energy": (
                milp_summary["median_sec"] / seq_summary["median_sec"]
                if seq_summary["median_sec"] > 0
                else None
            ),
        }

    results = {
        "settings": {
            "n_repeats": N_REPEATS,
            "initial_soc": INITIAL_SOC,
            "time_limit_sec": TIME_LIMIT_SEC,
            "n_minus_1_enabled": True,
        },
        "cases": cases,
        "same_energy_mutation": mutation_metrics(),
    }
    write_json(OUTPUT_DIR / "convergence_cost_evidence.json", results)
    plot_results(results)
    print(json.dumps(json_ready(results), indent=2))


if __name__ == "__main__":
    main()
