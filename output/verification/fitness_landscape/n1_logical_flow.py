"""
N-1-only logical-flow evidence for the GA+MILP fitness landscape.

The output is intentionally organized in the order used in the paper narrative:

1. Single-slot P_req -> MILP fuel.
2. Fixed full-profile section: one timestep P_req -> total route fuel.
3. 3D required-energy surface vs MILP fuel surface.
4. DG ON/OFF boundary overlay on the 3D MILP fuel surface.
5. SFOC/FC explanation of the largest step.

The single-slot scan is reused because it is route-independent. The fixed-route
section and 2D/3D surface are computed from the selected case artifact and
cached after the first run.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
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
OUTPUT_DIR_NAME = "n1_logical_flow"

INITIAL_SOC = 0.7
TIME_LIMIT_SEC = 15
GRID_N = 50
ROUTE_T_STAR = None  # None: use first DG-count transition, else max P_req.
SURFACE_T_A = None  # None: use ROUTE_T_STAR.
SURFACE_T_B = None  # None: use SURFACE_T_A + 1.
ROUTE_SECTION_N_POINTS = 160
SINGLE_SLOT_P_MIN_MW = 5.0
SINGLE_SLOT_P_MAX_MW = 40.0
SINGLE_SLOT_N_POINTS = 141
SINGLE_SLOT_DT_H = 1.0


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
OUTPUT_DIR = SCRIPT_DIR / OUTPUT_DIR_NAME
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "output"
    / "verification"
    / "case2"
    / "case_artifact.json"
)
SFOC_PATH = PROJECT_ROOT / "config" / "sfoc.json"
DG_NAMES = ["DG1", "DG2", "DG3", "DG4"]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.optimizer.milp_solver import MILPSolver, fuel_consumption  # noqa: E402
from src.ship.kcs_specs import DG_SPECS  # noqa: E402


def configure_fonts() -> None:
    for font_path in font_manager.findSystemFonts():
        if "malgun" in font_path.lower():
            font_manager.fontManager.addfont(font_path)
            plt.rcParams["font.family"] = "Malgun Gothic"
            break
    plt.rcParams["axes.unicode_minus"] = False


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def n1_only_rows(rows: list[dict[str, Any]], p_key: str) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        item = {p_key: to_float(row[p_key])}
        for key, value in row.items():
            if key.startswith("on_"):
                item[key.removeprefix("on_")] = value
        cleaned.append(item)
    return cleaned


def dg_count(step: dict[str, Any]) -> int:
    return sum(int(step.get(f"{dg}_ON", 0)) for dg in DG_NAMES)


def commitment_label(step: dict[str, Any]) -> str:
    online = [dg for dg in DG_NAMES if int(step.get(f"{dg}_ON", 0)) == 1]
    return "+".join(online) if online else "OFF"


def transition_rows(rows: list[dict[str, Any]], p_key: str) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    for prev, curr in zip(rows, rows[1:]):
        if str(prev["feasible"]) != "True" or str(curr["feasible"]) != "True":
            continue
        prev_count = int(to_float(prev["DG_count"]))
        curr_count = int(to_float(curr["DG_count"]))
        if prev_count == curr_count:
            continue
        transitions.append(
            {
                "P_star_MW": 0.5 * (to_float(prev[p_key]) + to_float(curr[p_key])),
                "from_count": prev_count,
                "to_count": curr_count,
                "from_commitment": prev["commitment"],
                "to_commitment": curr["commitment"],
                "fuel_before_kg": to_float(prev["total_fuel_kg"]),
                "fuel_after_kg": to_float(curr["total_fuel_kg"]),
                "fuel_jump_kg": to_float(curr["total_fuel_kg"]) - to_float(prev["total_fuel_kg"]),
                "operating_jump_kg": to_float(curr["total_operating_fuel_kg"]) - to_float(prev["total_operating_fuel_kg"]),
                "start_jump_kg": to_float(curr["total_start_fuel_kg"]) - to_float(prev["total_start_fuel_kg"]),
                "n1_margin_before_mw": to_float(prev["N1_min_margin_MW"]),
                "n1_margin_after_mw": to_float(curr["N1_min_margin_MW"]),
            }
        )
    return transitions


def load_artifact() -> dict[str, Any]:
    with ARTIFACT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_timestep_pair(artifact: dict[str, Any]) -> tuple[int, int]:
    p_req = [float(segment["P_req"]) for segment in artifact["power_profile"]]
    schedule = artifact.get("milp_result", {}).get("schedule", [])

    if ROUTE_T_STAR is not None:
        t_star = int(ROUTE_T_STAR)
    else:
        t_star = int(np.argmax(p_req))
        for idx in range(1, len(schedule)):
            prev_count = dg_count(schedule[idx - 1])
            curr_count = dg_count(schedule[idx])
            if prev_count != curr_count:
                t_star = idx
                break

    if SURFACE_T_A is not None:
        t_a = int(SURFACE_T_A)
    else:
        t_a = t_star

    if SURFACE_T_B is not None:
        t_b = int(SURFACE_T_B)
    else:
        t_b = min(t_a + 1, len(p_req) - 1)
        if t_b == t_a and t_a > 0:
            t_b = t_a - 1

    return t_a, t_b


def solve_route_section(artifact: dict[str, Any], t_star: int) -> list[dict[str, Any]]:
    base_p_req = [float(segment["P_req"]) for segment in artifact["power_profile"]]
    dt = [float(value) for value in artifact["route"]["dt"]]
    p_scan = np.linspace(min(base_p_req), max(base_p_req), ROUTE_SECTION_N_POINTS)
    solver = MILPSolver(
        sfoc_json_path=str(SFOC_PATH),
        n_pwl_segments=5,
        solver_name="cplex_cmd",
        enable_output_n_minus_1=True,
    )
    rows: list[dict[str, Any]] = []
    for idx, p_value in enumerate(p_scan):
        p_req = list(base_p_req)
        p_req[t_star] = float(p_value)
        result = solver.solve(
            P_req=p_req,
            dt=dt,
            initial_SOC=INITIAL_SOC,
            msg=False,
            time_limit_sec=TIME_LIMIT_SEC,
            enable_output_n_minus_1=True,
        )
        row: dict[str, Any] = {
            "idx": idx,
            "route_P_req_MW": float(p_value),
            "t_star": int(t_star),
            "feasible": bool(result.get("feasible")),
            "status": result.get("status"),
            "total_fuel_kg": None,
            "total_operating_fuel_kg": None,
            "total_start_fuel_kg": None,
            "objective_value": None,
            "DG_count": None,
            "commitment": "INFEASIBLE",
            "N1_min_margin_MW": None,
        }
        for dg in DG_NAMES:
            row[f"{dg}_ON"] = None
            row[f"{dg}_P_MW"] = None

        if result.get("feasible"):
            schedule = result["schedule"]
            step = schedule[t_star]
            row.update(
                {
                    "total_fuel_kg": float(result["total_fuel_kg"]),
                    "total_operating_fuel_kg": float(result["total_operating_fuel_kg"]),
                    "total_start_fuel_kg": float(result["total_start_fuel_kg"]),
                    "objective_value": float(result["objective_value"]),
                    "DG_count": dg_count(step),
                    "commitment": commitment_label(step),
                    "N1_min_margin_MW": step.get("N1_min_margin_MW"),
                }
            )
            for dg in DG_NAMES:
                row[f"{dg}_ON"] = int(step.get(f"{dg}_ON", 0))
                row[f"{dg}_P_MW"] = float(step.get(f"{dg}_P_MW", 0.0))
        rows.append(row)

        if idx == 0 or (idx + 1) % 20 == 0 or idx + 1 == len(p_scan):
            print(f"  N-1 fixed-profile section: {idx + 1:>3}/{len(p_scan)} solved")
    return rows


def solve_single_slot_scan() -> list[dict[str, Any]]:
    """Route-independent N-1 scan: one constant load, one one-hour slot."""
    cached_csv = OUTPUT_DIR / "step1_single_slot_n1.csv"
    if cached_csv.exists():
        return read_csv(cached_csv)

    p_scan = np.linspace(SINGLE_SLOT_P_MIN_MW, SINGLE_SLOT_P_MAX_MW, SINGLE_SLOT_N_POINTS)
    solver = MILPSolver(
        sfoc_json_path=str(SFOC_PATH),
        n_pwl_segments=5,
        solver_name="cplex_cmd",
        enable_output_n_minus_1=True,
    )

    rows: list[dict[str, Any]] = []
    for idx, p_value in enumerate(p_scan):
        result = solver.solve(
            P_req=[float(p_value)],
            dt=[SINGLE_SLOT_DT_H],
            initial_SOC=INITIAL_SOC,
            msg=False,
            time_limit_sec=TIME_LIMIT_SEC,
            enable_output_n_minus_1=True,
        )
        row: dict[str, Any] = {
            "idx": idx,
            "single_P_req_MW": float(p_value),
            "dt_h": SINGLE_SLOT_DT_H,
            "feasible": bool(result.get("feasible")),
            "status": result.get("status"),
            "total_fuel_kg": None,
            "total_operating_fuel_kg": None,
            "total_start_fuel_kg": None,
            "objective_value": None,
            "DG_count": None,
            "commitment": "INFEASIBLE",
            "N1_min_margin_MW": None,
        }
        for dg in DG_NAMES:
            row[f"{dg}_ON"] = None
            row[f"{dg}_P_MW"] = None

        if result.get("feasible"):
            step = result["schedule"][0]
            row.update(
                {
                    "total_fuel_kg": float(result["total_fuel_kg"]),
                    "total_operating_fuel_kg": float(result["total_operating_fuel_kg"]),
                    "total_start_fuel_kg": float(result["total_start_fuel_kg"]),
                    "objective_value": float(result["objective_value"]),
                    "DG_count": dg_count(step),
                    "commitment": commitment_label(step),
                    "N1_min_margin_MW": step.get("N1_min_margin_MW"),
                }
            )
            for dg in DG_NAMES:
                row[f"{dg}_ON"] = int(step.get(f"{dg}_ON", 0))
                row[f"{dg}_P_MW"] = float(step.get(f"{dg}_P_MW", 0.0))
        rows.append(row)

        if idx == 0 or (idx + 1) % 20 == 0 or idx + 1 == len(p_scan):
            print(f"  N-1 single-slot scan: {idx + 1:>3}/{len(p_scan)} solved")

    return rows


def cache_is_current(summary: dict[str, Any], artifact: dict[str, Any], t_a: int, t_b: int) -> bool:
    base_p_req = [float(segment["P_req"]) for segment in artifact["power_profile"]]
    expected = {
        "artifact_path": str(ARTIFACT_PATH),
        "grid_n": GRID_N,
        "t_a": t_a,
        "t_b": t_b,
        "p_lo_mw": min(base_p_req),
        "p_hi_mw": max(base_p_req),
    }
    for key, value in expected.items():
        if key not in summary:
            return False
        if isinstance(value, float):
            if abs(float(summary[key]) - value) > 1e-9:
                return False
        elif summary[key] != value:
            return False
    return True


def solve_2d_surface(artifact: dict[str, Any], t_a: int, t_b: int) -> dict[str, Any]:
    cache_npz = OUTPUT_DIR / "step3_2d_n1_surface.npz"
    cache_json = OUTPUT_DIR / "step3_2d_n1_surface_summary.json"
    if cache_npz.exists() and cache_json.exists():
        loaded = np.load(cache_npz, allow_pickle=False)
        summary = json.loads(cache_json.read_text(encoding="utf-8"))
        if cache_is_current(summary, artifact, t_a, t_b):
            return {
                "p1_grid": loaded["p1_grid"],
                "p2_grid": loaded["p2_grid"],
                "z_energy": loaded["z_energy"],
                "z_fuel": loaded["z_fuel"],
                "z_dg": loaded["z_dg"],
                "z_feasible": loaded["z_feasible"],
                "summary": summary,
            }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_p_req = [float(segment["P_req"]) for segment in artifact["power_profile"]]
    dt = [float(value) for value in artifact["route"]["dt"]]
    p_lo = min(base_p_req)
    p_hi = max(base_p_req)
    p1_grid = np.linspace(p_lo, p_hi, GRID_N)
    p2_grid = np.linspace(p_lo, p_hi, GRID_N)
    z_energy = np.full((GRID_N, GRID_N), np.nan)
    z_fuel = np.full((GRID_N, GRID_N), np.nan)
    z_dg = np.full((GRID_N, GRID_N), np.nan)
    z_feasible = np.zeros((GRID_N, GRID_N), dtype=bool)

    solver = MILPSolver(
        sfoc_json_path=str(SFOC_PATH),
        n_pwl_segments=5,
        solver_name="cplex_cmd",
        enable_output_n_minus_1=True,
    )

    started = time.time()
    for i, p1 in enumerate(p1_grid):
        for j, p2 in enumerate(p2_grid):
            p_req = list(base_p_req)
            p_req[t_a] = float(p1)
            p_req[t_b] = float(p2)
            z_energy[i, j] = float(sum(p * duration for p, duration in zip(p_req, dt)))
            result = solver.solve(
                P_req=p_req,
                dt=dt,
                initial_SOC=INITIAL_SOC,
                msg=False,
                time_limit_sec=TIME_LIMIT_SEC,
                enable_output_n_minus_1=True,
            )
            if result.get("feasible"):
                z_feasible[i, j] = True
                schedule = result["schedule"]
                step_a = schedule[t_a]
                step_b = schedule[t_b]
                z_fuel[i, j] = float(result["total_fuel_kg"])
                z_dg[i, j] = max(dg_count(step_a), dg_count(step_b))
        print(f"  N-1 2D surface: row {i + 1:>3}/{GRID_N} solved")

    boundary = boundary_mask(z_dg)
    boundary_jumps = boundary_jump_values(z_fuel, z_dg)
    finite_fuel = z_fuel[np.isfinite(z_fuel)]
    summary = {
        "artifact_path": str(ARTIFACT_PATH),
        "grid_n": GRID_N,
        "t_a": t_a,
        "t_b": t_b,
        "p_lo_mw": p_lo,
        "p_hi_mw": p_hi,
        "feasible_count": int(np.count_nonzero(z_feasible)),
        "boundary_pixel_count": int(np.count_nonzero(boundary)),
        "boundary_jump_mean_kg": float(np.mean(boundary_jumps)) if boundary_jumps else None,
        "boundary_jump_max_kg": float(np.max(boundary_jumps)) if boundary_jumps else None,
        "fuel_min_kg": float(np.min(finite_fuel)) if finite_fuel.size else None,
        "fuel_max_kg": float(np.max(finite_fuel)) if finite_fuel.size else None,
        "elapsed_sec": time.time() - started,
    }
    np.savez_compressed(
        cache_npz,
        p1_grid=p1_grid,
        p2_grid=p2_grid,
        z_energy=z_energy,
        z_fuel=z_fuel,
        z_dg=z_dg,
        z_feasible=z_feasible,
    )
    write_json(cache_json, summary)
    return {
        "p1_grid": p1_grid,
        "p2_grid": p2_grid,
        "z_energy": z_energy,
        "z_fuel": z_fuel,
        "z_dg": z_dg,
        "z_feasible": z_feasible,
        "summary": summary,
    }


def boundary_mask(z_dg: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(z_dg, dtype=bool)
    valid = np.isfinite(z_dg)
    vertical = valid[1:, :] & valid[:-1, :] & (z_dg[1:, :] != z_dg[:-1, :])
    horizontal = valid[:, 1:] & valid[:, :-1] & (z_dg[:, 1:] != z_dg[:, :-1])
    mask[1:, :] |= vertical
    mask[:-1, :] |= vertical
    mask[:, 1:] |= horizontal
    mask[:, :-1] |= horizontal
    return mask


def boundary_jump_values(z_fuel: np.ndarray, z_dg: np.ndarray) -> list[float]:
    jumps: list[float] = []
    for i in range(z_dg.shape[0] - 1):
        for j in range(z_dg.shape[1]):
            if np.isfinite(z_dg[i, j]) and np.isfinite(z_dg[i + 1, j]) and z_dg[i, j] != z_dg[i + 1, j]:
                jumps.append(abs(float(z_fuel[i + 1, j] - z_fuel[i, j])))
    for i in range(z_dg.shape[0]):
        for j in range(z_dg.shape[1] - 1):
            if np.isfinite(z_dg[i, j]) and np.isfinite(z_dg[i, j + 1]) and z_dg[i, j] != z_dg[i, j + 1]:
                jumps.append(abs(float(z_fuel[i, j + 1] - z_fuel[i, j])))
    return jumps


def build_transition_explanation(
    rows: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    p_key: str,
) -> dict[str, Any]:
    largest = max(transitions, key=lambda item: abs(item["fuel_jump_kg"]))
    before = None
    after = None
    for prev, curr in zip(rows, rows[1:]):
        p_star = 0.5 * (to_float(prev[p_key]) + to_float(curr[p_key]))
        if abs(p_star - largest["P_star_MW"]) < 1e-9:
            before = prev
            after = curr
            break
    if before is None or after is None:
        raise RuntimeError("Could not locate transition rows.")
    return {"transition": largest, "before": before, "after": after}


def sfoc_value(load_ratio: np.ndarray | float, coeffs: dict[str, float]) -> np.ndarray | float:
    return (
        coeffs["alpha1"] * np.asarray(load_ratio) ** 2
        + coeffs["alpha2"] * np.asarray(load_ratio)
        + coeffs["alpha3"]
    )


def plot_logical_flow(
    single_rows: list[dict[str, Any]],
    single_transitions: list[dict[str, Any]],
    route_rows: list[dict[str, Any]],
    route_transitions: list[dict[str, Any]],
    surface: dict[str, Any],
    summary: dict[str, Any],
    t_a: int,
    t_b: int,
) -> None:
    configure_fonts()
    fig = plt.figure(figsize=(16, 10), dpi=300)

    ax = fig.add_subplot(2, 2, 1)
    plot_fuel_scan(
        ax,
        rows=single_rows,
        transitions=single_transitions,
        p_key="single_P_req_MW",
        title="1. Single-slot P_req -> MILP fuel (N-1 ON)",
        y_key="total_fuel_kg",
    )

    ax = fig.add_subplot(2, 2, 2)
    plot_fuel_scan(
        ax,
        rows=route_rows,
        transitions=route_transitions,
        p_key="route_P_req_MW",
        title=f"2. Fixed full profile, perturb one timestep t*={t_a}",
        y_key="total_fuel_kg",
    )

    p1, p2 = np.meshgrid(surface["p1_grid"], surface["p2_grid"], indexing="ij")
    ax = fig.add_subplot(2, 2, 3, projection="3d")
    surface_energy = ax.plot_surface(
        p1,
        p2,
        surface["z_energy"],
        cmap="plasma",
        linewidth=0,
        antialiased=True,
        alpha=0.95,
    )
    fig.colorbar(surface_energy, ax=ax, shrink=0.65, pad=0.08, label="MWh")
    ax.set_title("3. Required-energy surface")
    ax.set_xlabel(f"P_req at t={t_a} (MW)")
    ax.set_ylabel(f"P_req at t={t_b} (MW)")
    ax.set_zlabel("Required energy (MWh)")
    ax.view_init(elev=28, azim=-135)

    ax = fig.add_subplot(2, 2, 4, projection="3d")
    surface_fuel = ax.plot_surface(
        p1,
        p2,
        surface["z_fuel"],
        cmap="inferno",
        linewidth=0,
        antialiased=True,
        alpha=0.92,
    )
    fig.colorbar(surface_fuel, ax=ax, shrink=0.65, pad=0.08, label="kg")
    boundary = boundary_mask(surface["z_dg"])
    ax.scatter(
        p1[boundary],
        p2[boundary],
        surface["z_fuel"][boundary],
        color="#00e5ff",
        s=8,
        depthshade=False,
        label="DG-count boundary",
    )
    ax.set_title("4. MILP fuel surface")
    ax.set_xlabel(f"P_req at t={t_a} (MW)")
    ax.set_ylabel(f"P_req at t={t_b} (MW)")
    ax.set_zlabel("MILP total fuel (kg)")
    ax.view_init(elev=28, azim=-135)
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("N-1 Contingency-Only Evidence Flow", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUTPUT_DIR / "n1_logical_flow.png")
    plt.close(fig)


def plot_fuel_scan(
    ax,
    rows: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    p_key: str,
    title: str,
    y_key: str,
) -> None:
    x = [to_float(row[p_key]) for row in rows if str(row["feasible"]) == "True"]
    y = [to_float(row[y_key]) for row in rows if str(row["feasible"]) == "True"]
    ax.plot(x, y, color="#c62828", linewidth=2.0)
    for item in transitions:
        ax.axvline(item["P_star_MW"], color="#263238", linestyle="--", linewidth=1.1, alpha=0.7)
        ax.scatter(
            [item["P_star_MW"]],
            [item["fuel_after_kg"]],
            s=90,
            facecolors="none",
            edgecolors="#d50000",
            linewidths=2.0,
            zorder=5,
        )
        ax.text(
            item["P_star_MW"],
            item["fuel_after_kg"],
            f" {item['from_count']}->{item['to_count']}",
            fontsize=8,
            va="bottom",
        )
    ax.set_title(title)
    ax.set_xlabel("Required power (MW)")
    ax.set_ylabel("MILP total fuel incl. starts (kg)")
    ax.grid(True, alpha=0.25)


def plot_sfoc_explanation(
    single_explain: dict[str, Any],
    route_explain: dict[str, Any],
) -> None:
    configure_fonts()
    coeffs = json.loads(SFOC_PATH.read_text(encoding="utf-8"))["DG1"]
    load = np.linspace(0.25, 1.0, 200)
    sfoc = sfoc_value(load, coeffs)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=300)
    plot_sfoc_case(axes[0, 0], single_explain, "Single-slot largest step: SFOC load points")
    plot_fc_case(axes[1, 0], single_explain, "Single-slot largest step: FC(P) points")
    plot_sfoc_case(axes[0, 1], route_explain, "Fixed-profile step: SFOC load points")
    plot_fc_case(axes[1, 1], route_explain, "Fixed-profile step: FC(P) points")

    for ax in axes[0, :]:
        ax.plot(load, sfoc, color="#263238", linewidth=2.0, label="SFOC curve")
        ax.set_xlabel("Load ratio P/Pmax")
        ax.set_ylabel("SFOC (g/kWh)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Why the Step Rises: Dispatch Moves on SFOC/FC Curves + Start Fuel", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUTPUT_DIR / "sfoc_transition_explanation.png")
    plt.close(fig)


def plot_sfoc_case(ax, explanation: dict[str, Any], title: str) -> None:
    before = explanation["before"]
    after = explanation["after"]
    for state, marker, color in [("before", "o", "#1565c0"), ("after", "s", "#c62828")]:
        row = before if state == "before" else after
        for dg in DG_NAMES:
            p = to_float(row[f"{dg}_P_MW"])
            if p <= 1e-9:
                continue
            p_max = float(DG_SPECS[dg]["P_max"])
            coeffs = json.loads(SFOC_PATH.read_text(encoding="utf-8"))[dg]
            load_ratio = p / p_max
            ax.scatter(
                [load_ratio],
                [sfoc_value(load_ratio, coeffs)],
                marker=marker,
                s=70,
                color=color,
                label=f"{state} {dg}",
            )
            ax.text(load_ratio, sfoc_value(load_ratio, coeffs), f" {dg}", fontsize=8)
    add_transition_box(ax, explanation)
    ax.set_title(title)


def plot_fc_case(ax, explanation: dict[str, Any], title: str) -> None:
    coeffs = json.loads(SFOC_PATH.read_text(encoding="utf-8"))["DG1"]
    for capacity, color, label in [(10.8, "#455a64", "10.8 MW DG"), (8.4, "#8d6e63", "8.4 MW DG")]:
        p_values = np.linspace(0.25 * capacity, capacity, 200)
        fc_values = [fuel_consumption(float(p), capacity, coeffs["alpha1"], coeffs["alpha2"], coeffs["alpha3"]) for p in p_values]
        ax.plot(p_values, fc_values, color=color, linewidth=1.8, label=label)
    for state, marker, color in [("before", "o", "#1565c0"), ("after", "s", "#c62828")]:
        row = explanation["before"] if state == "before" else explanation["after"]
        for dg in DG_NAMES:
            p = to_float(row[f"{dg}_P_MW"])
            if p <= 1e-9:
                continue
            p_max = float(DG_SPECS[dg]["P_max"])
            coeffs_dg = json.loads(SFOC_PATH.read_text(encoding="utf-8"))[dg]
            fc = fuel_consumption(p, p_max, coeffs_dg["alpha1"], coeffs_dg["alpha2"], coeffs_dg["alpha3"])
            ax.scatter([p], [fc], marker=marker, s=70, color=color, label=f"{state} {dg}")
            ax.text(p, fc, f" {dg}", fontsize=8)
    add_transition_box(ax, explanation)
    ax.set_title(title)
    ax.set_xlabel("DG output (MW)")
    ax.set_ylabel("Fuel rate (kg/h)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)


def add_transition_box(ax, explanation: dict[str, Any]) -> None:
    item = explanation["transition"]
    text = (
        f"P*={item['P_star_MW']:.1f} MW\n"
        f"DG {item['from_count']}->{item['to_count']}\n"
        f"Δtotal={item['fuel_jump_kg']:.1f} kg\n"
        f"Δoper={item['operating_jump_kg']:.1f} kg\n"
        f"Δstart={item['start_jump_kg']:.1f} kg"
    )
    text = (
        f"P*={item['P_star_MW']:.1f} MW\n"
        f"DG {item['from_count']}->{item['to_count']}\n"
        f"d_total={item['fuel_jump_kg']:.1f} kg\n"
        f"d_oper={item['operating_jump_kg']:.1f} kg\n"
        f"d_start={item['start_jump_kg']:.1f} kg"
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fffde7", "edgecolor": "#8d6e63"},
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    artifact = load_artifact()
    t_a, t_b = select_timestep_pair(artifact)

    single_rows = solve_single_slot_scan()
    route_rows = solve_route_section(artifact, t_a)
    single_transitions = transition_rows(single_rows, "single_P_req_MW")
    route_transitions = transition_rows(route_rows, "route_P_req_MW")
    write_csv(OUTPUT_DIR / "step1_single_slot_n1.csv", single_rows)
    write_csv(OUTPUT_DIR / "step2_fixed_profile_n1.csv", route_rows)
    write_csv(OUTPUT_DIR / "step1_single_slot_transitions_n1.csv", single_transitions)
    write_csv(OUTPUT_DIR / "step2_fixed_profile_transitions_n1.csv", route_transitions)

    surface = solve_2d_surface(artifact, t_a=t_a, t_b=t_b)

    single_explain = build_transition_explanation(single_rows, single_transitions, "single_P_req_MW")
    route_explain = build_transition_explanation(route_rows, route_transitions, "route_P_req_MW")

    summary = {
        "artifact_path": str(ARTIFACT_PATH),
        "narrative_order": [
            "single-slot P_req scan",
            "fixed-profile one-timestep P_req scan",
            "3D required-energy vs MILP-fuel surface",
            "DG ON/OFF boundary tracking on the MILP fuel surface",
            "SFOC/FC transition explanation",
        ],
        "selected_timesteps": {
            "route_t_star": t_a,
            "surface_t_a": t_a,
            "surface_t_b": t_b,
        },
        "step1_single_slot": {
            "transition_count": len(single_transitions),
            "max_jump_kg": max(abs(item["fuel_jump_kg"]) for item in single_transitions),
            "transitions": single_transitions,
        },
        "step2_fixed_profile": {
            "transition_count": len(route_transitions),
            "max_jump_kg": max(abs(item["fuel_jump_kg"]) for item in route_transitions),
            "transitions": route_transitions,
        },
        "step3_step4_surface": surface["summary"],
        "largest_single_step_explanation": single_explain,
        "largest_route_step_explanation": route_explain,
    }
    write_json(OUTPUT_DIR / "n1_logical_flow_summary.json", summary)
    plot_logical_flow(single_rows, single_transitions, route_rows, route_transitions, surface, summary, t_a=t_a, t_b=t_b)
    plot_sfoc_explanation(single_explain, route_explain)

    print(json.dumps(json_ready(summary), indent=2))


if __name__ == "__main__":
    main()
