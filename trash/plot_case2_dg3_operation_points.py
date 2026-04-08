from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DETAILS_XLSX = REPO_ROOT / "output" / "verification" / "comparison" / "case_comparison_details.xlsx"
SFOC_JSON = REPO_ROOT / "config" / "sfoc.json"
OUTPUT_PNG = Path(__file__).resolve().with_name("Case3 DG1 Operation Points on SFOC.png")


def load_case3_dg1_points() -> pd.DataFrame:
    segments = pd.read_excel(DETAILS_XLSX, sheet_name="segments")
    case3 = segments[segments["case_name"] == "case3_ga_integrated"].copy()
    dg1_ops = case3[case3["milp_DG1_ON"] == 1].copy()
    return dg1_ops[["segment", "milp_DG1_P_MW", "milp_DG1_FC_kgh"]]


def load_dg1_sfoc_coeffs() -> tuple[float, float, float]:
    with open(SFOC_JSON, "r", encoding="utf-8") as handle:
        sfoc = json.load(handle)["DG1"]
    return float(sfoc["alpha1"]), float(sfoc["alpha2"]), float(sfoc["alpha3"])


def main() -> None:
    dg1_ops = load_case3_dg1_points()
    alpha1, alpha2, alpha3 = load_dg1_sfoc_coeffs()

    p_grid = np.linspace(0.0, 14.4, 400)
    p_grid_kw = 1000.0 * p_grid
    sfoc_curve = alpha1 * p_grid_kw**2 + alpha2 * p_grid_kw + alpha3
    fuel_curve = sfoc_curve * p_grid

    fig, axes = plt.subplots(1, 1, figsize=(8, 5))

    ax = axes
    ax.plot(p_grid, sfoc_curve, color="Grey", linewidth=2, label="DG1 SFOC curve")
    if not dg1_ops.empty:
        op_sfoc = dg1_ops["milp_DG1_FC_kgh"].to_numpy() / dg1_ops["milp_DG1_P_MW"].to_numpy()
        scatter = ax.scatter(
            dg1_ops["milp_DG1_P_MW"],
            op_sfoc,
            cmap="viridis",
            s=70,
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
            label="Case3 DG1 operation points",
        )
        for _, row in dg1_ops.iterrows():
            sfoc_value = row["milp_DG1_FC_kgh"] / row["milp_DG1_P_MW"]
            ax.annotate(
                f"t={int(row['segment'])}",
                (row["milp_DG1_P_MW"], sfoc_value),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    ax.set_xlabel("DG1 Power (MW)")
    ax.set_ylabel("SFOC (g/kWh)")
    ax.set_title("DG1 Operation Points on SFOC Curve")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {OUTPUT_PNG}")
    print(dg1_ops.to_string(index=False))


if __name__ == "__main__":
    main()
