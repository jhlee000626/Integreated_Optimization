from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.optimizer.ga_engine import compute_speed_through_water_knots
from src.resistance.models import EnvironmentData
from src.resistance.modified_dpm import compute_P_req
from src.ship.kcs_specs import SERVICE_LOAD


BASE = Path(__file__).resolve().parent
OUTPUT_PNG = BASE / "stepwise_ga_preq_surface.png"
OUTPUT_JSON = BASE / "stepwise_ga_preq_surface_metrics.json"
PLOT_CHOICES = ("all", "surface", "arrows", "gradient", "slices", "each")

SOG_MIN_KTS = 12.0
SOG_MAX_KTS = 22.0
AZ_MIN_DEG = -30.0
AZ_MAX_DEG = 30.0
N_SOG = 121
N_AZ = 121

# Synthetic marine-weather scenario for visualizing heading sensitivity.
# The values are intentionally moderate so the requested GA search window stays
# smooth while delta heading clearly changes P_req.
ENVIRONMENT = EnvironmentData(
    current_speed_ms=1.2,
    current_dir_deg=35.0,
    wind_speed_ms=2.0,
    wind_dir_deg=20.0,
    wave_height_m=0.6,
    wave_period_s=7.0,
    wave_dir_deg=180.0,
)
P_SERVICE_MW = float(SERVICE_LOAD.get("cruising", 9.845))


def configure_fonts() -> None:
    for font_path in font_manager.findSystemFonts():
        if "malgun" in font_path.lower():
            font_manager.fontManager.addfont(font_path)
            plt.rcParams["font.family"] = "Malgun Gothic"
            break
    plt.rcParams["axes.unicode_minus"] = False


def compute_surface() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sog_grid = np.linspace(SOG_MIN_KTS, SOG_MAX_KTS, N_SOG)
    az_grid = np.linspace(AZ_MIN_DEG, AZ_MAX_DEG, N_AZ)
    p_req_grid = np.zeros((N_AZ, N_SOG), dtype=float)
    stw_grid = np.zeros_like(p_req_grid)

    for i, azimuth_deg in enumerate(az_grid):
        for j, sog_kts in enumerate(sog_grid):
            stw_kts = max(
                0.1,
                compute_speed_through_water_knots(
                    v_sog_knots=float(sog_kts),
                    heading_deg=float(azimuth_deg),
                    env=ENVIRONMENT,
                ),
            )
            result = compute_P_req(
                v_ship_knots=stw_kts,
                v_wind_ms=ENVIRONMENT.wind_speed_ms,
                encounter_angle_deg=0.0,
                env=ENVIRONMENT,
                heading_deg=float(azimuth_deg),
                P_service=P_SERVICE_MW,
            )
            p_req_grid[i, j] = float(result["P_req"])
            stw_grid[i, j] = stw_kts

    return sog_grid, az_grid, p_req_grid, stw_grid


def finite_difference_diagnostics(
    sog_grid: np.ndarray,
    az_grid: np.ndarray,
    p_req_grid: np.ndarray,
) -> dict[str, float | bool | int]:
    ds = float(sog_grid[1] - sog_grid[0])
    da = float(az_grid[1] - az_grid[0])

    dp_daz, dp_dsog = np.gradient(p_req_grid, da, ds)

    h_ss = (p_req_grid[1:-1, 2:] - 2.0 * p_req_grid[1:-1, 1:-1] + p_req_grid[1:-1, :-2]) / (ds**2)
    h_aa = (p_req_grid[2:, 1:-1] - 2.0 * p_req_grid[1:-1, 1:-1] + p_req_grid[:-2, 1:-1]) / (da**2)
    h_sa = (
        p_req_grid[2:, 2:]
        - p_req_grid[2:, :-2]
        - p_req_grid[:-2, 2:]
        + p_req_grid[:-2, :-2]
    ) / (4.0 * ds * da)

    h_trace = h_ss + h_aa
    min_hessian_eigen = 0.5 * (
        h_trace - np.sqrt((h_ss - h_aa) ** 2 + 4.0 * h_sa**2)
    )
    h_det = h_ss * h_aa - h_sa**2

    grad_finite = bool(np.isfinite(dp_dsog).all() and np.isfinite(dp_daz).all())
    p_finite = bool(np.isfinite(p_req_grid).all())

    return {
        "p_req_min_mw": float(np.min(p_req_grid)),
        "p_req_max_mw": float(np.max(p_req_grid)),
        "min_d2_dsog2": float(np.min(h_ss)),
        "min_d2_daz2": float(np.min(h_aa)),
        "min_hessian_determinant": float(np.min(h_det)),
        "min_hessian_eigenvalue": float(np.min(min_hessian_eigen)),
        "all_p_req_finite": p_finite,
        "all_gradient_components_finite": grad_finite,
        "grid_points": int(p_req_grid.size),
    }


def build_quiver_components(
    sog_grid: np.ndarray,
    az_grid: np.ndarray,
    p_req_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ds = float(sog_grid[1] - sog_grid[0])
    da = float(az_grid[1] - az_grid[0])
    dp_daz, dp_dsog = np.gradient(p_req_grid, da, ds)

    sog_span = SOG_MAX_KTS - SOG_MIN_KTS
    az_span = AZ_MAX_DEG - AZ_MIN_DEG
    grad_sog_norm = dp_dsog * sog_span
    grad_az_norm = dp_daz * az_span
    grad_mag_norm = np.hypot(grad_sog_norm, grad_az_norm)

    with np.errstate(divide="ignore", invalid="ignore"):
        unit_sog = np.divide(grad_sog_norm, grad_mag_norm, out=np.zeros_like(grad_sog_norm), where=grad_mag_norm > 0.0)
        unit_az = np.divide(grad_az_norm, grad_mag_norm, out=np.zeros_like(grad_az_norm), where=grad_mag_norm > 0.0)

    arrow_sog = unit_sog * 0.45
    arrow_az = unit_az * 2.7
    return dp_dsog, dp_daz, grad_mag_norm, arrow_sog, arrow_az


def output_path_for(plot_kind: str) -> Path:
    if plot_kind == "all":
        return OUTPUT_PNG
    return BASE / f"stepwise_ga_preq_surface_{plot_kind}.png"


def draw_surface_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    sog_mesh: np.ndarray,
    az_mesh: np.ndarray,
    p_req_grid: np.ndarray,
) -> None:
    surf = ax.plot_surface(
        sog_mesh,
        az_mesh,
        p_req_grid,
        cmap="viridis",
        linewidth=0.0,
        antialiased=True,
        alpha=0.96,
    )
    # ax.contour(
    #     sog_mesh,
    #     az_mesh,
    #     p_req_grid,
    #     zdir="z",
    #     offset=float(np.min(p_req_grid)) - 0.8,
    #     levels=12,
    #     cmap="viridis",
    #     linewidths=0.8,
    # )
    ax.set_xlabel("SOG (knots)", labelpad=10)
    ax.set_ylabel("delta heading (deg)", labelpad=10)
    ax.set_zlabel("P_prop (MW)", labelpad=10)
    ax.set_title("Modified DPM P_prop surface", pad=14, fontweight="bold")
    ax.view_init(elev=28, azim=-132)
    ax.set_zlim(float(np.min(p_req_grid)) - 0.8, float(np.max(p_req_grid)) + 0.8)
    fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.1, label="P_prop (MW)")


def draw_arrows_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    sog_mesh: np.ndarray,
    az_mesh: np.ndarray,
    p_req_grid: np.ndarray,
    grad_mag_norm: np.ndarray,
    arrow_sog: np.ndarray,
    arrow_az: np.ndarray,
) -> None:
    contour = ax.contourf(sog_mesh, az_mesh, p_req_grid, levels=28, cmap="viridis")
    ax.contour(sog_mesh, az_mesh, p_req_grid, levels=12, colors="white", linewidths=0.6, alpha=0.55)
    q_step_y = 10
    q_step_x = 10
    quiver = ax.quiver(
        sog_mesh[::q_step_y, ::q_step_x],
        az_mesh[::q_step_y, ::q_step_x],
        arrow_sog[::q_step_y, ::q_step_x],
        arrow_az[::q_step_y, ::q_step_x],
        grad_mag_norm[::q_step_y, ::q_step_x],
        cmap="magma",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0038,
        headwidth=3.8,
        headlength=5.0,
        headaxislength=4.5,
    )
    ax.set_xlabel("SOG (knots)")
    ax.set_ylabel("Azimuth delta (deg)")
    ax.set_title("Local dP_req arrows: finite-difference gradient field", fontweight="bold")
    ax.set_xlim(SOG_MIN_KTS, SOG_MAX_KTS)
    ax.set_ylim(AZ_MIN_DEG, AZ_MAX_DEG)
    ax.grid(True, alpha=0.16, ls="--")
    fig.colorbar(contour, ax=ax, label="P_req (MW)")
    fig.colorbar(quiver, ax=ax, label="|dP_req| in normalized GA domain")


def draw_gradient_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    sog_mesh: np.ndarray,
    az_mesh: np.ndarray,
    p_req_grid: np.ndarray,
    grad_mag_norm: np.ndarray,
) -> None:
    grad_plot = ax.contourf(sog_mesh, az_mesh, grad_mag_norm, levels=28, cmap="magma")
    ax.contour(sog_mesh, az_mesh, p_req_grid, levels=10, colors="white", linewidths=0.55, alpha=0.55)
    ax.set_xlabel("SOG (knots)")
    ax.set_ylabel("Azimuth delta (deg)")
    ax.set_title("Gradient magnitude is finite on every sampled point", fontweight="bold")
    ax.grid(True, alpha=0.16, ls="--")
    fig.colorbar(grad_plot, ax=ax, label="|dP_req| in normalized GA domain")


def draw_slices_panel(
    ax: plt.Axes,
    sog_grid: np.ndarray,
    az_grid: np.ndarray,
    p_req_grid: np.ndarray,
    stw_grid: np.ndarray,
    diagnostics: dict[str, object],
) -> None:
    for azimuth in [-30.0, -15.0, 0.0, 15.0, 30.0]:
        idx = int(np.argmin(np.abs(az_grid - azimuth)))
        ax.plot(
            sog_grid,
            p_req_grid[idx, :],
            lw=2.2,
            label=f"az={az_grid[idx]:.0f} deg",
        )
    ax.set_xlabel("SOG (knots)")
    ax.set_ylabel("P_req (MW)")
    ax.set_title("Speed slices remain smooth and convex", fontweight="bold")
    ax.grid(True, alpha=0.2, ls="--")
    ax.legend(loc="upper left", fontsize=9)

    note = (
        f"Grid points: {diagnostics['grid_points']}\n"
        f"P_req range: {diagnostics['p_req_min_mw']:.3f}-{diagnostics['p_req_max_mw']:.3f} MW\n"
        f"STW range: {float(np.min(stw_grid)):.3f}-{float(np.max(stw_grid)):.3f} knots\n"
        f"min d2/dSOG2: {diagnostics['min_d2_dsog2']:.3e}\n"
        f"min d2/dAz2: {diagnostics['min_d2_daz2']:.3e}\n"
        f"min Hessian eigenvalue: {diagnostics['min_hessian_eigenvalue']:.3e}\n"
        f"finite gradient components: {diagnostics['all_gradient_components_finite']}"
    )
    ax.text(
        0.98,
        0.04,
        note,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#8a8f98",
            "alpha": 0.92,
        },
    )


def add_main_title(fig: plt.Figure) -> None:
    current_kts = ENVIRONMENT.current_speed_ms / 0.514444
    fig.suptitle(
        "Stepwise GA verification: Modified DPM P_req with synthetic marine weather\n"
        f"SOG {SOG_MIN_KTS:.0f}-{SOG_MAX_KTS:.0f} knots, azimuth {AZ_MIN_DEG:.0f} to {AZ_MAX_DEG:.0f} deg, "
        f"current {ENVIRONMENT.current_speed_ms:.1f} m/s ({current_kts:.2f} knots) at {ENVIRONMENT.current_dir_deg:.0f} deg, "
        f"wind {ENVIRONMENT.wind_speed_ms:.1f} m/s at {ENVIRONMENT.wind_dir_deg:.0f} deg, "
        f"wave Hs {ENVIRONMENT.wave_height_m:.1f} m at {ENVIRONMENT.wave_dir_deg:.0f} deg",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )


def plot_surface(
    sog_grid: np.ndarray,
    az_grid: np.ndarray,
    p_req_grid: np.ndarray,
    stw_grid: np.ndarray,
    diagnostics: dict[str, object],
    plot_kind: str = "all",
) -> Path:
    sog_mesh, az_mesh = np.meshgrid(sog_grid, az_grid)
    _, _, grad_mag_norm, arrow_sog, arrow_az = build_quiver_components(sog_grid, az_grid, p_req_grid)
    out_path = output_path_for(plot_kind)

    if plot_kind == "all":
        fig = plt.figure(figsize=(18, 13))
        grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.28, wspace=0.22)
        draw_surface_panel(fig, fig.add_subplot(grid[0, 0], projection="3d"), sog_mesh, az_mesh, p_req_grid)
        draw_arrows_panel(
            fig,
            fig.add_subplot(grid[0, 1]),
            sog_mesh,
            az_mesh,
            p_req_grid,
            grad_mag_norm,
            arrow_sog,
            arrow_az,
        )
        draw_gradient_panel(fig, fig.add_subplot(grid[1, 0]), sog_mesh, az_mesh, p_req_grid, grad_mag_norm)
        draw_slices_panel(fig.add_subplot(grid[1, 1]), sog_grid, az_grid, p_req_grid, stw_grid, diagnostics)
        add_main_title(fig)
    elif plot_kind == "surface":
        fig = plt.figure(figsize=(10, 8))
        draw_surface_panel(fig, fig.add_subplot(111, projection="3d"), sog_mesh, az_mesh, p_req_grid)
    elif plot_kind == "arrows":
        fig, ax = plt.subplots(figsize=(10, 8))
        draw_arrows_panel(fig, ax, sog_mesh, az_mesh, p_req_grid, grad_mag_norm, arrow_sog, arrow_az)
    elif plot_kind == "gradient":
        fig, ax = plt.subplots(figsize=(10, 8))
        draw_gradient_panel(fig, ax, sog_mesh, az_mesh, p_req_grid, grad_mag_norm)
    elif plot_kind == "slices":
        fig, ax = plt.subplots(figsize=(10, 8))
        draw_slices_panel(ax, sog_grid, az_grid, p_req_grid, stw_grid, diagnostics)
    else:
        raise ValueError(f"Unsupported plot kind: {plot_kind}")

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Modified DPM P_req surface over SOG and azimuth.")
    parser.add_argument(
        "--plot",
        choices=PLOT_CHOICES,
        default="all",
        help=(
            "Plot to save: all=2x2 summary, surface=3D surface, arrows=delta P_req arrows, "
            "gradient=gradient magnitude, slices=1D speed slices, each=all single panels."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_fonts()
    sog_grid, az_grid, p_req_grid, stw_grid = compute_surface()
    diagnostics = finite_difference_diagnostics(sog_grid, az_grid, p_req_grid)
    diagnostics["stw_min_kts"] = float(np.min(stw_grid))
    diagnostics["stw_max_kts"] = float(np.max(stw_grid))
    diagnostics["sog_range_kts"] = [SOG_MIN_KTS, SOG_MAX_KTS]
    diagnostics["azimuth_range_deg"] = [AZ_MIN_DEG, AZ_MAX_DEG]
    diagnostics["current_speed_ms"] = ENVIRONMENT.current_speed_ms
    diagnostics["current_dir_deg"] = ENVIRONMENT.current_dir_deg
    diagnostics["wind_speed_ms"] = ENVIRONMENT.wind_speed_ms
    diagnostics["wind_dir_deg"] = ENVIRONMENT.wind_dir_deg
    diagnostics["wave_height_m"] = ENVIRONMENT.wave_height_m
    diagnostics["wave_period_s"] = ENVIRONMENT.wave_period_s
    diagnostics["wave_dir_deg"] = ENVIRONMENT.wave_dir_deg
    diagnostics["p_service_mw"] = P_SERVICE_MW

    plot_kinds = ["surface", "arrows", "gradient", "slices"] if args.plot == "each" else [args.plot]
    saved_figures = [
        plot_surface(sog_grid, az_grid, p_req_grid, stw_grid, diagnostics, plot_kind=plot_kind)
        for plot_kind in plot_kinds
    ]

    with OUTPUT_JSON.open("w", encoding="utf-8") as fp:
        json.dump(diagnostics, fp, indent=2)

    for saved_figure in saved_figures:
        print(f"Saved figure: {saved_figure}")
    print(f"Saved metrics: {OUTPUT_JSON}")
    print(
        "Gradient finite:",
        diagnostics["all_gradient_components_finite"],
        "| min Hessian eigenvalue:",
        f"{diagnostics['min_hessian_eigenvalue']:.6e}",
    )


if __name__ == "__main__":
    main()
