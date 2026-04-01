"""Download CMEMS surface current data for the marine environment loader."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import copernicusmarine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "cmems"
DEFAULT_AREA = {
    "lon_min": 125.0,
    "lon_max": 131.0,
    "lat_min": 32.0,
    "lat_max": 36.0,
}
DEFAULT_DATASET_ID = "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m"
DEFAULT_DEPTH_M = 0.49402499198913574


def _patch_cmems_netcdf_writer() -> None:
    """Avoid h5py/h5netcdf ABI issues by forcing a scipy/netCDF3 write path."""

    import copernicusmarine.download_functions.download_zarr as download_zarr

    def _download_dataset_as_netcdf_scipy(
        dataset,
        output_path,
        netcdf_compression_level: int,
        netcdf3_compatible: bool,
    ):
        if netcdf_compression_level > 0:
            print("  Note: compression is ignored for scipy/netCDF3 output.")
        for coord in dataset.coords:
            dataset[coord].encoding["_FillValue"] = None
        return dataset.to_netcdf(
            output_path,
            mode="w",
            format="NETCDF3_CLASSIC",
            engine="scipy",
        )

    download_zarr._download_dataset_as_netcdf = _download_dataset_as_netcdf_scipy


def _parse_utc_datetime(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.replace(microsecond=0)


def _default_output_name(start_utc: datetime, end_utc: datetime) -> str:
    start_label = start_utc.strftime("%Y%m%dT%H")
    end_label = end_utc.strftime("%Y%m%dT%H")
    return f"cmems_current_{start_label}_{end_label}.nc"


def download_cmems(
    start_datetime: str,
    end_datetime: str,
    *,
    dataset_id: str = DEFAULT_DATASET_ID,
    depth_m: float = DEFAULT_DEPTH_M,
    area: dict[str, float] | None = None,
    output_name: str | None = None,
) -> str:
    start_utc = _parse_utc_datetime(start_datetime)
    end_utc = _parse_utc_datetime(end_datetime)
    if end_utc < start_utc:
        raise ValueError("end datetime must be on or after start datetime.")

    if area is None:
        area = dict(DEFAULT_AREA)
    if output_name is None:
        output_name = _default_output_name(start_utc, end_utc)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / output_name

    print("=" * 60)
    print(" CMEMS current download")
    print("=" * 60)
    print(f"  Dataset    : {dataset_id}")
    print(f"  Start (UTC): {start_utc.isoformat()}")
    print(f"  End   (UTC): {end_utc.isoformat()}")
    print(
        "  Area       : "
        f"{area['lon_min']}~{area['lon_max']}E, {area['lat_min']}~{area['lat_max']}N"
    )
    print(f"  Depth (m)  : {depth_m}")
    print(f"  Output     : {output_path}")
    print("  Writer     : scipy / NETCDF3_CLASSIC")

    _patch_cmems_netcdf_writer()
    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=["uo", "vo"],
        minimum_longitude=area["lon_min"],
        maximum_longitude=area["lon_max"],
        minimum_latitude=area["lat_min"],
        maximum_latitude=area["lat_max"],
        start_datetime=start_utc.isoformat(),
        end_datetime=end_utc.isoformat(),
        minimum_depth=depth_m,
        maximum_depth=depth_m,
        output_filename=output_path.name,
        output_directory=str(OUTPUT_DIR),
        overwrite=True,
        file_format="netcdf",
    )

    if not output_path.exists():
        raise FileNotFoundError(f"CMEMS download did not create: {output_path}")

    print(f"\n[ok] Saved: {output_path}")
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
    parser.add_argument("--lon-min", type=float, default=DEFAULT_AREA["lon_min"])
    parser.add_argument("--lon-max", type=float, default=DEFAULT_AREA["lon_max"])
    parser.add_argument("--lat-min", type=float, default=DEFAULT_AREA["lat_min"])
    parser.add_argument("--lat-max", type=float, default=DEFAULT_AREA["lat_max"])
    parser.add_argument("--depth-m", type=float, default=DEFAULT_DEPTH_M)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--output-name", default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    download_cmems(
        start_datetime=args.start_datetime,
        end_datetime=args.end_datetime,
        dataset_id=args.dataset_id,
        depth_m=args.depth_m,
        area={
            "lon_min": args.lon_min,
            "lon_max": args.lon_max,
            "lat_min": args.lat_min,
            "lat_max": args.lat_max,
        },
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()
