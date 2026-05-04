from __future__ import annotations

import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.resistance.modified_dpm import compute_P_req
from src.ship.kcs_specs import KCS_ISO15016


OUTPUT_CSV = Path(__file__).with_name("modified_dpm_design_speed_22.csv")


def build_ship_specs() -> dict:
    ship_specs = dict(KCS_ISO15016)
    ship_specs["DesignSpeed"] = 22.0
    return ship_specs


def collect_rows(speed_min: int = 12, speed_max: int = 24) -> list[dict]:
    ship_specs = build_ship_specs()
    rows: list[dict] = []

    for speed in range(speed_min, speed_max + 1):
        result = compute_P_req(
            v_ship_knots=float(speed),
            v_wind_ms=0.0,
            encounter_angle_deg=0.0,
            P_service=0.0,
            heading_deg=0.0,
            ship_specs=ship_specs,
        )
        rows.append(
            {
                "speed_kts": speed,
                "P_prop_MW": result["P_prop"],
                "BHP_kW": result["BHP_kW"],
                "BHP_kW_raw": result["BHP_kW_raw"],
                "PD_kW": result["PD_kW"],
                "PS_metric_hp": result["BHP_kW"] / 0.73549875,
                "total_resistance_kN": result["total_resistance_N"] / 1000.0,
                "RPM": result["RPM"],
                "eta_D": result["eta_D"],
                "clipped": result["BHP_kW"] < result["BHP_kW_raw"] - 1e-9,
            }
        )

    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict]) -> None:
    print("Modified DPM check with DesignSpeed=22.0 knots")
    print(
        f"{'Speed(kts)':>10} {'P_prop(MW)':>12} {'BHP(kW)':>12} "
        f"{'BHP_raw(kW)':>12} {'PS':>10} {'clip?':>8}"
    )
    for row in rows:
        print(
            f"{row['speed_kts']:10.0f} {row['P_prop_MW']:12.3f} "
            f"{row['BHP_kW']:12.1f} {row['BHP_kW_raw']:12.1f} "
            f"{row['PS_metric_hp']:10.0f} {str(row['clipped']):>8}"
        )


def main() -> None:
    rows = collect_rows()
    write_csv(rows, OUTPUT_CSV)
    print_table(rows)
    print(f"\nSaved CSV to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
