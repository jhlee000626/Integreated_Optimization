"""Download ERA5 and CMEMS data for weather-scenario verification cases.

Scenarios:
  - calm_summer      : 2024-07-15
  - typhoon_bebinca  : 2024-09-14
  - winter_severe    : 2025-01-15

Each scenario uses date-matched ERA5 wind/wave data and CMEMS surface-current
data over the same UTC window. This avoids reusing a current field from a
different date.

Usage:
    python scripts/download_scenario_data.py
    python scripts/download_scenario_data.py --scenario typhoon_bebinca
    python scripts/download_scenario_data.py --scenario typhoon --kind cmems
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.download_cmems import download_cmems
from scripts.download_era5 import download_era5


SCENARIOS = {
    "calm_summer": {
        "start": "2024-07-15T00:00:00Z",
        "end": "2024-07-16T06:00:00Z",
        "label": "Calm Summer (2024-07-15)",
        "era5": "era5_marine_20240715T00_20240716T06.nc",
        "cmems": "cmems_current_20240715T00_20240716T06.nc",
    },
    "typhoon_bebinca": {
        "start": "2024-09-14T00:00:00Z",
        "end": "2024-09-15T06:00:00Z",
        "label": "Typhoon Bebinca (2024-09-14)",
        "era5": "era5_marine_20240914T00_20240915T06.nc",
        "cmems": "cmems_current_20240914T00_20240915T06.nc",
    },
    "winter_severe": {
        "start": "2025-01-15T00:00:00Z",
        "end": "2025-01-16T06:00:00Z",
        "label": "Winter Monsoon (2025-01-15)",
        "era5": "era5_marine_20250115T00_20250116T06.nc",
        "cmems": "cmems_current_20250115T00_20250116T06.nc",
    },
}

SCENARIO_ALIASES = {
    "typhoon": "typhoon_bebinca",
}


def _canonical_scenario_name(name: str) -> str:
    return SCENARIO_ALIASES.get(name, name)


def _skip_existing(path: Path) -> str | None:
    if path.exists():
        print(f"[skip] Already exists: {path}")
        return str(path)
    return None


def download_scenario(
    name: str,
    kind: str = "all",
    cmems_username: str | None = None,
    cmems_password: str | None = None,
) -> dict[str, str]:
    """Download data for a single scenario. Returns output paths by data kind."""
    canonical_name = _canonical_scenario_name(name)
    scenario = SCENARIOS[canonical_name]
    print(f"\n{'=' * 60}")
    print(f"  Scenario: {scenario['label']}")
    print(f"{'=' * 60}")

    paths: dict[str, str] = {}
    if kind in {"all", "era5"}:
        paths["era5"] = download_era5(
            start_datetime=scenario["start"],
            end_datetime=scenario["end"],
            output_name=scenario["era5"],
        )

    if kind in {"all", "cmems"}:
        cmems_path = PROJECT_ROOT / "data" / "cmems" / scenario["cmems"]
        paths["cmems"] = _skip_existing(cmems_path) or download_cmems(
            start_datetime=scenario["start"],
            end_datetime=scenario["end"],
            output_name=scenario["cmems"],
            username=cmems_username,
            password=cmems_password,
        )

    return paths


def main() -> None:
    scenario_choices = list(SCENARIOS.keys()) + list(SCENARIO_ALIASES.keys()) + ["all"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=scenario_choices,
        default="all",
        help="Which scenario to download (default: all).",
    )
    parser.add_argument(
        "--kind",
        choices=("all", "era5", "cmems"),
        default="all",
        help="Which data source to download (default: all).",
    )
    parser.add_argument("--cmems-username", default=None, help="Copernicus Marine username.")
    parser.add_argument("--cmems-password", default=None, help="Copernicus Marine password.")
    args = parser.parse_args()

    if args.scenario == "all":
        targets = list(SCENARIOS.keys())
    else:
        targets = [_canonical_scenario_name(args.scenario)]

    paths: dict[str, dict[str, str]] = {}
    for name in targets:
        paths[name] = download_scenario(
            name,
            kind=args.kind,
            cmems_username=args.cmems_username,
            cmems_password=args.cmems_password,
        )

    print("\n" + "=" * 60)
    print(" Download Summary")
    print("=" * 60)
    for name, source_paths in paths.items():
        print(f"  {name}")
        for source, path in source_paths.items():
            print(f"    {source:5s}: {path}")


if __name__ == "__main__":
    main()
