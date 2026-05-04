"""
Diagnose whether the N-1 MILP surface is truly step-like or mostly linear.

This script uses the cached Case2 N-1 surface from n1_logical_flow and removes
the best-fit affine plane from MILP fuel. The residual/jump maps make the DG
commitment seam visible without exaggerating the claim that the whole raw
surface is a staircase.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


BASE_DIR = Path(__file__).resolve().parent / "n1_logical_flow"
SURFACE_NPZ = BASE_DIR / "step3_2d_n1_surface.npz"
SUMMARY_JSON = BASE_DIR / "step3_2d_n1_surface_summary.json"
OUTPUT_PNG = BASE_DIR / "n1_boundary_residual.png"
OUTPUT_JSON = BASE_DIR / "n1_boundary_residual_metrics.json"


def configure_fonts() -> None:
    for font_path in font_manager.findSystemFonts():
        if "malgun" in font_path.lower():
            font_manager.fontManager.addfont(font_path)
            plt.rcParams["font.family"] = "Malgun Gothic"
            break
    plt.rcParams["axes.unicode_minus"] = False


def boundary_edges(z_dg: np.ndarray) -> tuple[list[float], list[float]]:
    boundary_jumps: list[float] = []
    nonboundary_jumps: list[float] = []
    z_fuel = _GLOBAL_Z_FUEL
    for i in range(z_dg.shape[0] - 1):
        for j in range(z_dg.shape[1]):
            if not (np.isfinite(z_dg[i, j]) and np.isfinite(z_dg[i + 1, j])):
                continue
            jump = abs(float(z_fuel[i + 1, j] - z_fuel[i, j]))
            if z_dg[i, j] != z_dg[i + 1, j]:
                boundary_jumps.append(jump)
            else:
                nonboundary_jumps.append(jump)
    for i in range(z_dg.shape[0]):
        for j in range(z_dg.shape[1] - 1):
            if not (np.isfinite(z_dg[i, j]) and np.isfinite(z_dg[i, j + 1])):
                continue
            jump = abs(float(z_fuel[i, j + 1] - z_fuel[i, j]))
            if z_dg[i, j] != z_dg[i, j + 1]:
                boundary_jumps.append(jump)
            else:
                nonboundary_jumps.append(jump)
    return boundary_jumps, nonboundary_jumps


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


def best_fit_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    valid = np.isfinite(z)
    design = np.column_stack([x[valid], y[valid], np.ones(np.count_nonzero(valid))])
    coeff, *_ = np.linalg.lstsq(design, z[valid], rcond=None)
    fitted = coeff[0] * x + coeff[1] * y + coeff[2]
    residual = z - fitted
    ss_res = float(np.nansum((z - fitted) ** 2))
    z_mean = float(np.nanmean(z))
    ss_tot = float(np.nansum((z - z_mean) ** 2))
    return residual, {
        "plane_a_kg_per_mw": float(coeff[0]),
        "plane_b_kg_per_mw": float(coeff[1]),
        "plane_c_kg": float(coeff[2]),
        "r2_affine_plane": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "residual_min_kg": float(np.nanmin(residual)),
        "residual_max_kg": float(np.nanmax(residual)),
        "residual_std_kg": float(np.nanstd(residual)),
    }


def finite_jump_map(z: np.ndarray) -> np.ndarray:
    jump = np.zeros_like(z)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            candidates: list[float] = []
            if i > 0 and np.isfinite(z[i, j]) and np.isfinite(z[i - 1, j]):
                candidates.append(abs(float(z[i, j] - z[i - 1, j])))
            if i + 1 < z.shape[0] and np.isfinite(z[i, j]) and np.isfinite(z[i + 1, j]):
                candidates.append(abs(float(z[i + 1, j] - z[i, j])))
            if j > 0 and np.isfinite(z[i, j]) and np.isfinite(z[i, j - 1]):
                candidates.append(abs(float(z[i, j] - z[i, j - 1])))
            if j + 1 < z.shape[1] and np.isfinite(z[i, j]) and np.isfinite(z[i, j + 1]):
                candidates.append(abs(float(z[i, j + 1] - z[i, j])))
            jump[i, j] = max(candidates) if candidates else np.nan
    return jump


def json_ready(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, dict):
        return {key: json_ready(sub_value) for key, sub_value in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def main() -> None:
    configure_fonts()
    loaded = np.load(SURFACE_NPZ, allow_pickle=False)
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    p1_grid = loaded["p1_grid"]
    p2_grid = loaded["p2_grid"]
    z_fuel = loaded["z_fuel"]
    z_dg = loaded["z_dg"]
    x, y = np.meshgrid(p1_grid, p2_grid, indexing="ij")

    global _GLOBAL_Z_FUEL
    _GLOBAL_Z_FUEL = z_fuel

    residual, plane_metrics = best_fit_plane(x, y, z_fuel)
    jump_map = finite_jump_map(z_fuel)
    mask = boundary_mask(z_dg)
    boundary_jumps, nonboundary_jumps = boundary_edges(z_dg)

    metrics = {
        "source_summary": summary,
        **plane_metrics,
        "boundary_jump_mean_kg": float(np.mean(boundary_jumps)) if boundary_jumps else None,
        "boundary_jump_max_kg": float(np.max(boundary_jumps)) if boundary_jumps else None,
        "nonboundary_jump_mean_kg": float(np.mean(nonboundary_jumps)) if nonboundary_jumps else None,
        "nonboundary_jump_max_kg": float(np.max(nonboundary_jumps)) if nonboundary_jumps else None,
        "boundary_to_nonboundary_mean_ratio": (
            float(np.mean(boundary_jumps) / np.mean(nonboundary_jumps))
            if boundary_jumps and nonboundary_jumps and np.mean(nonboundary_jumps) > 0
            else None
        ),
        "interpretation": (
            "The raw Case2 N-1 MILP fuel surface is dominated by an affine trend. "
            "It should be described as near-linear with DG-boundary seams/kinks, "
            "not as a globally staircase-shaped surface."
        ),
    }

    fig = plt.figure(figsize=(16, 10), dpi=300)
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    surf = ax.plot_surface(x, y, z_fuel, cmap="inferno", linewidth=0, alpha=0.95)
    fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08, label="kg")
    ax.set_title("Raw MILP fuel surface: visually near-affine")
    ax.set_xlabel("P_req(t_a) MW")
    ax.set_ylabel("P_req(t_b) MW")
    ax.set_zlabel("Fuel kg")
    ax.view_init(elev=28, azim=-135)

    ax = fig.add_subplot(2, 2, 2, projection="3d")
    surf = ax.plot_surface(x, y, residual, cmap="coolwarm", linewidth=0, alpha=0.95)
    ax.scatter(x[mask], y[mask], residual[mask], color="#00e5ff", s=8, depthshade=False)
    fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08, label="kg")
    ax.set_title("After removing best-fit plane: boundary seam")
    ax.set_xlabel("P_req(t_a) MW")
    ax.set_ylabel("P_req(t_b) MW")
    ax.set_zlabel("Residual kg")
    ax.view_init(elev=30, azim=-135)

    ax = fig.add_subplot(2, 2, 3)
    contour = ax.contourf(x, y, jump_map, levels=25, cmap="magma")
    ax.contour(x, y, mask.astype(float), levels=[0.5], colors="#00e5ff", linewidths=1.6)
    fig.colorbar(contour, ax=ax, label="max adjacent |delta fuel| kg")
    ax.set_title("Finite-difference jump map + DG boundary")
    ax.set_xlabel("P_req(t_a) MW")
    ax.set_ylabel("P_req(t_b) MW")

    ax = fig.add_subplot(2, 2, 4)
    ax.boxplot(
        [nonboundary_jumps, boundary_jumps],
        labels=["non-boundary", "DG boundary"],
        showfliers=False,
    )
    ax.set_title("Boundary jumps are small but systematically larger")
    ax.set_ylabel("Adjacent |delta fuel| kg")
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(
        0.05,
        0.95,
        f"affine R2={metrics['r2_affine_plane']:.4f}\n"
        f"boundary mean={metrics['boundary_jump_mean_kg']:.1f} kg\n"
        f"non-boundary mean={metrics['nonboundary_jump_mean_kg']:.1f} kg\n"
        f"ratio={metrics['boundary_to_nonboundary_mean_ratio']:.2f}x",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fffde7", "edgecolor": "#8d6e63"},
    )

    fig.suptitle("Case2 N-1 Surface Diagnosis: Seam/Kink, Not Global Staircase", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUTPUT_PNG)
    plt.close(fig)

    OUTPUT_JSON.write_text(json.dumps(json_ready(metrics), indent=2), encoding="utf-8")
    print(json.dumps(json_ready(metrics), indent=2))


_GLOBAL_Z_FUEL = np.array([])


if __name__ == "__main__":
    main()
