from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.optimizer.milp_solver import fuel_consumption, generate_pwl_breakpoints
from src.ship.kcs_specs import DG_MIN_LOAD_RATIO, DG_SPECS

SFOC_JSON = REPO_ROOT / "config" / "sfoc.json"
OUTPUT_PNG = Path(__file__).resolve().with_name("sfoc_and_pwl_curves.png")
N_PWL_SEGMENTS = 5


def load_sfoc() -> dict:
    with open(SFOC_JSON, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    sfoc_data = load_sfoc()
    dg_names = list(DG_SPECS.keys())
    for dg in dg_names:
        fig, ax = plt.subplots(figsize=(8, 5))
        coeffs = sfoc_data[dg]
        p_max = float(DG_SPECS[dg]["P_max"])
        p_min = p_max * float(DG_MIN_LOAD_RATIO)
        p_curve = np.linspace(p_min, p_max, 400)
        p_curve_kw = 1000.0 * p_curve
        sfoc_curve = (
            float(coeffs["alpha1"]) * p_curve_kw**2
            + float(coeffs["alpha2"]) * p_curve_kw
            + float(coeffs["alpha3"]))
        
        breakpoints = generate_pwl_breakpoints(
            p_min, p_max, float(coeffs["alpha1"]), float(coeffs["alpha2"]), float(coeffs["alpha3"]), n_segments=N_PWL_SEGMENTS)
        
        bp_p = np.array([point[0] for point in breakpoints])
        bp_fc = np.array([point[1] for point in breakpoints])
        bp_sfoc = bp_fc / bp_p

        ax.plot(p_curve, sfoc_curve, color="#1E88E5", linewidth=2)
        ax.plot(bp_p, bp_sfoc, "o--", color="#E53935", linewidth=1.6, markersize=5, label="PWL approximation")
        for point_index, (power, sfoc_value) in enumerate(zip(bp_p, bp_sfoc)):
            ax.annotate(f"k={point_index}", (power, sfoc_value), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_title(f"{dg} SFOC Curve vs PWL")
        ax.set_xlabel("Power (MW)")
        ax.set_ylabel("SFOC (g/kWh)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        plt.tight_layout()
        output_path = Path(__file__).resolve().with_name(f"{dg}_sfoc_curve.png")
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {output_path}")

    # fig, axes = plt.subplots(len(dg_names), 2, figsize=(13, 4.2 * len(dg_names)), squeeze=False)

    # for row_index, dg in enumerate(dg_names):
    #     spec = DG_SPECS[dg]
    #     coeffs = sfoc_data[dg]
    #     p_max = float(spec["P_max"])
    #     p_min = p_max * float(DG_MIN_LOAD_RATIO)

    #     p_curve = np.linspace(p_min, p_max, 400)
    #     p_curve_kw = 1000.0 * p_curve
    #     sfoc_curve = (
    #         float(coeffs["alpha1"]) * p_curve_kw**2
    #         + float(coeffs["alpha2"]) * p_curve_kw
    #         + float(coeffs["alpha3"])
    #     )
    #     fuel_curve = np.array(
    #         [
    #             fuel_consumption(
    #                 p,
    #                 float(coeffs["alpha1"]),
    #                 float(coeffs["alpha2"]),
    #                 float(coeffs["alpha3"]),
    #             )
    #             for p in p_curve
    #         ]
    #     )

    #     breakpoints = generate_pwl_breakpoints(
    #         p_min,
    #         p_max,
    #         float(coeffs["alpha1"]),
    #         float(coeffs["alpha2"]),
    #         float(coeffs["alpha3"]),
    #         n_segments=N_PWL_SEGMENTS,
    #     )
    #     bp_p = np.array([point[0] for point in breakpoints])
    #     bp_fc = np.array([point[1] for point in breakpoints])
    #     bp_sfoc = bp_fc / bp_p

    #     ax = axes[row_index, 0]
    #     ax.plot(p_curve, sfoc_curve, color="#1E88E5", linewidth=2, label="Quadratic SFOC")
    #     ax.plot(bp_p, bp_sfoc, "o--", color="#E53935", linewidth=1.6, markersize=5, label="PWL approximation")
    #     for point_index, (power, sfoc_value) in enumerate(zip(bp_p, bp_sfoc)):
    #         ax.annotate(f"k={point_index}", (power, sfoc_value), textcoords="offset points", xytext=(4, 4), fontsize=8)
    #     ax.set_title(f"{dg} SFOC Curve vs PWL")
    #     ax.set_xlabel("Power (MW)")
    #     ax.set_ylabel("SFOC (g/kWh)")
    #     ax.grid(True, alpha=0.3)
    #     ax.legend(fontsize=9)

    #     ax = axes[row_index, 1]
    #     ax.plot(p_curve, fuel_curve, color="#43A047", linewidth=2, label="Quadratic fuel curve")
    #     ax.plot(bp_p, bp_fc, "o--", color="#FB8C00", linewidth=1.6, markersize=5, label="PWL approximation")
    #     for point_index, (power, fuel_value) in enumerate(zip(bp_p, bp_fc)):
    #         ax.annotate(f"k={point_index}", (power, fuel_value), textcoords="offset points", xytext=(4, 4), fontsize=8)
    #     ax.set_title(f"{dg} Fuel Curve vs PWL")
    #     ax.set_xlabel("Power (MW)")
    #     ax.set_ylabel("Fuel Consumption (kg/h)")
    #     ax.grid(True, alpha=0.3)
    #     ax.legend(fontsize=9)

    # plt.tight_layout()
    # fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
    # plt.close(fig)
    # print(f"Saved plot: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
