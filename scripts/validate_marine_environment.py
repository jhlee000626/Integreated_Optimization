"""Validate ERA5 and CMEMS files against the shared marine environment contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.weather import MarineEnvironmentLoader, resolve_marine_dataset_paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era5-path", default=None, help="Optional ERA5 NetCDF path.")
    parser.add_argument("--cmems-path", default=None, help="Optional CMEMS NetCDF path.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    era5_path, cmems_path = resolve_marine_dataset_paths(
        PROJECT_ROOT,
        era5_path=args.era5_path,
        cmems_path=args.cmems_path,
    )

    print("=" * 60)
    print(" Marine environment validation")
    print("=" * 60)
    print(f"  ERA5 : {era5_path}")
    print(f"  CMEMS: {cmems_path}")

    try:
        loader = MarineEnvironmentLoader(era5_path, cmems_path)
    except Exception as exc:
        print(f"\n[error] {exc}")
        raise SystemExit(1) from exc

    times = loader.available_times_utc

    print(f"  Shared UTC timesteps : {len(times)}")
    print(f"  First timestep       : {times[0].isoformat()}")
    print(f"  Last timestep        : {times[-1].isoformat()}")
    if len(times) < 2:
        print("  Warning              : only one shared timestep is available.")
        print("                         Time-varying optimization will effectively see a frozen environment.")
    print("\n[ok] Marine environment datasets are aligned and ready.")


if __name__ == "__main__":
    main()
