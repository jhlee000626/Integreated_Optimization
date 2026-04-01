"""Helpers for locating marine environment datasets on disk."""

from __future__ import annotations

from pathlib import Path
import os


ERA5_ENV_VAR = "MARINE_ERA5_PATH"
CMEMS_ENV_VAR = "MARINE_CMEMS_PATH"

ERA5_PATTERNS = (
    "era5_marine_*.nc",
    "era5_wave_*.nc",
    "era5_wind_*.nc",
    "*.nc",
)
CMEMS_PATTERNS = (
    "cmems_current_*.nc",
    "cmems_current.nc",
    "*.nc",
)


def _project_root(project_root: str | Path | None = None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    return Path(__file__).resolve().parents[2]


def _pick_latest(directory: Path, patterns: tuple[str, ...]) -> Path | None:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(path for path in directory.glob(pattern) if path.is_file())

    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_marine_dataset_paths(
    project_root: str | Path | None = None,
    era5_path: str | Path | None = None,
    cmems_path: str | Path | None = None,
) -> tuple[str, str]:
    """Resolve ERA5 and CMEMS dataset paths from explicit args, env vars, or defaults."""

    root = _project_root(project_root)
    era5_dir = root / "data" / "era5"
    cmems_dir = root / "data" / "cmems"

    era5_candidate = era5_path or os.getenv(ERA5_ENV_VAR)
    cmems_candidate = cmems_path or os.getenv(CMEMS_ENV_VAR)

    if era5_candidate is None:
        era5_file = _pick_latest(era5_dir, ERA5_PATTERNS)
    else:
        era5_file = Path(era5_candidate).expanduser().resolve()

    if cmems_candidate is None:
        cmems_file = _pick_latest(cmems_dir, CMEMS_PATTERNS)
    else:
        cmems_file = Path(cmems_candidate).expanduser().resolve()

    if era5_file is None:
        raise FileNotFoundError(
            f"ERA5 dataset not found under {era5_dir}. "
            f"Set {ERA5_ENV_VAR} or place a file matching {ERA5_PATTERNS[0]} there."
        )
    if cmems_file is None:
        raise FileNotFoundError(
            f"CMEMS dataset not found under {cmems_dir}. "
            f"Set {CMEMS_ENV_VAR} or place a file matching {CMEMS_PATTERNS[0]} there."
        )

    if not era5_file.exists():
        raise FileNotFoundError(f"ERA5 dataset not found: {era5_file}")
    if not cmems_file.exists():
        raise FileNotFoundError(f"CMEMS dataset not found: {cmems_file}")

    return str(era5_file), str(cmems_file)
