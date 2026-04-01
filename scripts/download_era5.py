"""Download ERA5 wind and wave variables for the marine environment loader."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import io
from pathlib import Path
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "era5"
DEFAULT_AREA = [36.0, 125.0, 32.0, 131.0]  # north, west, south, east
ERA5_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "significant_height_of_combined_wind_waves_and_swell",
    "mean_wave_period",
    "mean_wave_direction",
]


def _parse_utc_datetime(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.replace(minute=0, second=0, microsecond=0)


def _hourly_range(start_utc: datetime, end_utc: datetime) -> list[datetime]:
    if end_utc < start_utc:
        raise ValueError("end datetime must be on or after start datetime.")

    values: list[datetime] = []
    current = start_utc
    while current <= end_utc:
        values.append(current)
        current += timedelta(hours=1)
    return values


def _time_buckets(hours: list[datetime]) -> dict[tuple[int, int], dict[str, list[str]]]:
    buckets: dict[tuple[int, int], dict[str, set[str]]] = defaultdict(
        lambda: {"days": set(), "times": set()}
    )
    for current in hours:
        key = (current.year, current.month)
        buckets[key]["days"].add(f"{current.day:02d}")
        buckets[key]["times"].add(f"{current.hour:02d}:00")

    return {
        key: {
            "days": sorted(values["days"]),
            "times": sorted(values["times"]),
        }
        for key, values in buckets.items()
    }


def _default_output_name(start_utc: datetime, end_utc: datetime) -> str:
    start_label = start_utc.strftime("%Y%m%dT%H")
    end_label = end_utc.strftime("%Y%m%dT%H")
    return f"era5_marine_{start_label}_{end_label}.nc"


def _normalize_era5_file(path: Path) -> Path:
    """CDS sometimes returns a zip payload even when the target suffix is .nc."""

    if not path.exists() or not zipfile.is_zipfile(path):
        return path

    import xarray as xr

    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".nc")]
        if not members:
            raise ValueError(f"ERA5 archive does not contain a NetCDF file: {path}")
        datasets = []
        try:
            for name in members:
                payload = io.BytesIO(archive.read(name))
                try:
                    dataset = xr.open_dataset(payload)
                except ValueError:
                    payload.seek(0)
                    dataset = xr.open_dataset(payload, engine="h5netcdf")
                datasets.append(dataset)

            if len(datasets) == 1:
                merged = datasets[0].load()
            else:
                merged = xr.merge(datasets, compat="override", combine_attrs="override").load()
        finally:
            for dataset in datasets:
                dataset.close()

    merged.to_netcdf(path)
    merged.close()
    return path


def _detect_time_name(dataset) -> str:
    for candidate in ("valid_time", "time"):
        if candidate in dataset.coords or candidate in dataset.dims:
            return candidate
    raise ValueError("Downloaded ERA5 file has no time coordinate.")


def download_era5(
    start_datetime: str,
    end_datetime: str,
    *,
    area: list[float] | None = None,
    output_name: str | None = None,
) -> str:
    start_utc = _parse_utc_datetime(start_datetime)
    end_utc = _parse_utc_datetime(end_datetime)
    hours = _hourly_range(start_utc, end_utc)
    buckets = _time_buckets(hours)

    if area is None:
        area = list(DEFAULT_AREA)
    if output_name is None:
        output_name = _default_output_name(start_utc, end_utc)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / output_name
    if output_path.exists():
        _normalize_era5_file(output_path)
        print(f"[skip] Already exists: {output_path}")
        return str(output_path)

    import cdsapi

    client = cdsapi.Client()
    temp_files: list[Path] = []

    print("=" * 60)
    print(" ERA5 wind+wave download")
    print("=" * 60)
    print(f"  Start (UTC): {start_utc.isoformat()}")
    print(f"  End   (UTC): {end_utc.isoformat()}")
    print(f"  Area       : {area}")
    print(f"  Output     : {output_path}")
    print(f"  Variables  : {', '.join(ERA5_VARIABLES)}")

    for (year, month), values in sorted(buckets.items()):
        part_path = output_path.with_name(
            f"{output_path.stem}_{year:04d}{month:02d}.part.nc"
        )
        temp_files.append(part_path)
        print(f"\n[request] {year:04d}-{month:02d} days={len(values['days'])} hours={len(values['times'])}")

        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": ERA5_VARIABLES,
                "year": f"{year:04d}",
                "month": f"{month:02d}",
                "day": values["days"],
                "time": values["times"],
                "area": area,
                "format": "netcdf",
            },
            str(part_path),
        )
        _normalize_era5_file(part_path)

    if len(temp_files) == 1:
        temp_files[0].replace(output_path)
        print(f"\n[ok] Saved: {output_path}")
        return str(output_path)

    import xarray as xr

    datasets = [xr.open_dataset(path) for path in temp_files]
    try:
        time_name = _detect_time_name(datasets[0])
        merged = xr.concat(datasets, dim=time_name).sortby(time_name)
        merged.to_netcdf(output_path)
    finally:
        for dataset in datasets:
            dataset.close()
        for path in temp_files:
            if path.exists():
                path.unlink()

    print(f"\n[ok] Saved merged dataset: {output_path}")
    return str(output_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-datetime",
        default="2025-03-25T00:00:00Z",
        help="UTC start time in ISO format, for example 2025-03-25T00:00:00Z",
    )
    parser.add_argument(
        "--end-datetime",
        default="2025-03-25T23:00:00Z",
        help="UTC end time in ISO format, for example 2025-03-25T23:00:00Z",
    )
    parser.add_argument(
        "--area",
        nargs=4,
        type=float,
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
        default=DEFAULT_AREA,
        help="Bounding box in CDS format: north west south east",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional output filename. Defaults to an auto-generated marine dataset name.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    download_era5(
        start_datetime=args.start_datetime,
        end_datetime=args.end_datetime,
        area=list(args.area),
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()
