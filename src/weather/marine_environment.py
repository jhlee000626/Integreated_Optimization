"""
Spatiotemporal marine environment loader for wind, current, and wave data.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Callable

import numpy as np

from src.resistance.models import EnvironmentData


def _coerce_datetime64_ns(value: datetime | np.datetime64) -> np.datetime64:
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[ns]")

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)

    return np.datetime64(value.replace(tzinfo=None), "ns")


def _datetime64_to_utc(value: np.datetime64) -> datetime:
    dt = value.astype("datetime64[us]").item()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _pick_name(candidates: tuple[str, ...], available: set[str]) -> str | None:
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _ceil_to_hour(value: np.datetime64) -> np.datetime64:
    hour_value = value.astype("datetime64[h]")
    if value.astype("datetime64[ns]") == hour_value.astype("datetime64[ns]"):
        return hour_value.astype("datetime64[ns]")
    return (hour_value + np.timedelta64(1, "h")).astype("datetime64[ns]")


def _floor_to_hour(value: np.datetime64) -> np.datetime64:
    return value.astype("datetime64[h]").astype("datetime64[ns]")


def _normalize_coords(values: np.ndarray, field: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    if values[0] <= values[-1]:
        return values.astype(float), field
    return values[::-1].astype(float), np.flip(field, axis=axis)


def _interp_bounds(coords: np.ndarray, value: float) -> tuple[int, int, float]:
    if value <= coords[0]:
        return 0, 0, 0.0
    if value >= coords[-1]:
        last = len(coords) - 1
        return last, last, 0.0

    upper = int(np.searchsorted(coords, value, side="right"))
    lower = upper - 1
    span = coords[upper] - coords[lower]
    weight = 0.0 if span <= 1e-12 else (value - coords[lower]) / span
    return lower, upper, float(weight)


def _weighted_mean(values: list[tuple[float, float]]) -> float:
    valid = [(weight, value) for weight, value in values if not math.isnan(value)]
    if not valid:
        return float("nan")
    total_weight = sum(weight for weight, _ in valid)
    if total_weight <= 1e-12:
        return valid[0][1]
    return sum(weight * value for weight, value in valid) / total_weight


def _bilinear(field_2d: np.ndarray, lats: np.ndarray, lons: np.ndarray, lat: float, lon: float) -> float:
    lat0, lat1, lat_w = _interp_bounds(lats, lat)
    lon0, lon1, lon_w = _interp_bounds(lons, lon)

    corners = [
        ((1.0 - lat_w) * (1.0 - lon_w), float(field_2d[lat0, lon0])),
        ((1.0 - lat_w) * lon_w, float(field_2d[lat0, lon1])),
        (lat_w * (1.0 - lon_w), float(field_2d[lat1, lon0])),
        (lat_w * lon_w, float(field_2d[lat1, lon1])),
    ]
    return _weighted_mean(corners)


class MarineEnvironmentLoader:
    """Load wind/wave from ERA5 and current from CMEMS on a shared UTC grid."""

    ERA5_REQUIRED_VARS = ("u10", "v10", "swh", "mwp", "mwd")
    CMEMS_REQUIRED_VARS = ("uo", "vo")

    def __init__(self, era5_path: str, cmems_path: str, time_step_hours: int = 1):
        import xarray as xr

        self.era5_path = era5_path
        self.cmems_path = cmems_path
        self.time_step_hours = int(time_step_hours)

        era5 = xr.open_dataset(era5_path)
        cmems = xr.open_dataset(cmems_path)
        try:
            self._initialize_from_datasets(era5, cmems)
        finally:
            era5.close()
            cmems.close()

    @classmethod
    def from_datasets(cls, era5_dataset, cmems_dataset, time_step_hours: int = 1) -> "MarineEnvironmentLoader":
        instance = cls.__new__(cls)
        instance.era5_path = "<in-memory-era5>"
        instance.cmems_path = "<in-memory-cmems>"
        instance.time_step_hours = int(time_step_hours)
        instance._initialize_from_datasets(era5_dataset, cmems_dataset)
        return instance

    def _initialize_from_datasets(self, era5, cmems) -> None:
        diagnostics: list[str] = []

        era5_time_name = _pick_name(("valid_time", "time"), set(era5.coords) | set(era5.dims))
        cmems_time_name = _pick_name(("time", "valid_time"), set(cmems.coords) | set(cmems.dims))
        if era5_time_name is None:
            diagnostics.append("ERA5 dataset has no time coordinate.")
        if cmems_time_name is None:
            diagnostics.append("CMEMS dataset has no time coordinate.")

        missing_era5 = [name for name in self.ERA5_REQUIRED_VARS if name not in era5.data_vars]
        missing_cmems = [name for name in self.CMEMS_REQUIRED_VARS if name not in cmems.data_vars]
        if missing_era5:
            diagnostics.append(f"ERA5 dataset is missing variables: {', '.join(missing_era5)}.")
        if missing_cmems:
            diagnostics.append(f"CMEMS dataset is missing variables: {', '.join(missing_cmems)}.")

        if era5_time_name is not None and cmems_time_name is not None:
            era5_times = era5[era5_time_name].values.astype("datetime64[ns]")
            cmems_times = cmems[cmems_time_name].values.astype("datetime64[ns]")
            overlap_start = max(era5_times[0], cmems_times[0])
            overlap_end = min(era5_times[-1], cmems_times[-1])
            if overlap_start > overlap_end:
                diagnostics.append(
                    "ERA5 and CMEMS time windows do not overlap: "
                    f"{era5_times[0]}..{era5_times[-1]} vs {cmems_times[0]}..{cmems_times[-1]}."
                )

        if diagnostics:
            raise ValueError("Marine environment dataset validation failed:\n- " + "\n- ".join(diagnostics))

        common_start = _ceil_to_hour(max(era5_times[0], cmems_times[0]))
        common_end = _floor_to_hour(min(era5_times[-1], cmems_times[-1]))
        if common_start > common_end:
            raise ValueError(
                "Marine environment datasets overlap in time, but there is no shared hourly UTC grid "
                f"between {common_start} and {common_end}."
            )

        time_step = np.timedelta64(self.time_step_hours, "h")
        common_times = np.arange(common_start, common_end + time_step, time_step).astype("datetime64[ns]")
        self._times = common_times

        era5_std = self._standardize_dataset(era5, era5_time_name)
        cmems_std = self._standardize_dataset(cmems, cmems_time_name)

        era5_std = era5_std.assign(
            mwd_sin=np.sin(np.radians(era5_std["mwd"])),
            mwd_cos=np.cos(np.radians(era5_std["mwd"])),
        )

        era5_interp = era5_std[["u10", "v10", "swh", "mwp", "mwd_sin", "mwd_cos"]].interp(time=common_times)
        cmems_interp = cmems_std[list(self.CMEMS_REQUIRED_VARS)].interp(time=common_times)

        self._era5_lats = era5_interp["latitude"].values.astype(float)
        self._era5_lons = era5_interp["longitude"].values.astype(float)
        self._cmems_lats = cmems_interp["latitude"].values.astype(float)
        self._cmems_lons = cmems_interp["longitude"].values.astype(float)

        self._u10 = era5_interp["u10"].transpose("time", "latitude", "longitude").values.astype(float)
        self._v10 = era5_interp["v10"].transpose("time", "latitude", "longitude").values.astype(float)
        self._swh = era5_interp["swh"].transpose("time", "latitude", "longitude").values.astype(float)
        self._mwp = era5_interp["mwp"].transpose("time", "latitude", "longitude").values.astype(float)
        self._mwd_sin = era5_interp["mwd_sin"].transpose("time", "latitude", "longitude").values.astype(float)
        self._mwd_cos = era5_interp["mwd_cos"].transpose("time", "latitude", "longitude").values.astype(float)

        self._uo = cmems_interp["uo"].transpose("time", "latitude", "longitude").values.astype(float)
        self._vo = cmems_interp["vo"].transpose("time", "latitude", "longitude").values.astype(float)

    @staticmethod
    def _standardize_dataset(dataset, time_name: str):
        renamed = dataset.rename({time_name: "time"})
        lat_name = _pick_name(("latitude", "lat"), set(renamed.coords) | set(renamed.dims))
        lon_name = _pick_name(("longitude", "lon"), set(renamed.coords) | set(renamed.dims))
        if lat_name is None or lon_name is None:
            raise ValueError("Dataset must include latitude/longitude coordinates.")

        renamed = renamed.rename({lat_name: "latitude", lon_name: "longitude"})
        if "depth" in renamed.dims:
            renamed = renamed.isel(depth=0, drop=True)
        return renamed.sortby("time").sortby("latitude").sortby("longitude")

    @property
    def times(self) -> np.ndarray:
        return self._times.copy()

    @property
    def available_times_utc(self) -> tuple[datetime, ...]:
        return tuple(_datetime64_to_utc(value) for value in self._times)

    def get_environment_fn(self) -> Callable[[float, float, datetime], EnvironmentData]:
        return self.get_environment

    def close(self) -> None:
        """Compatibility no-op; datasets are loaded eagerly during initialization."""
        return None

    def _time_bounds(self, when_utc: datetime | np.datetime64) -> tuple[int, int, float]:
        when = _coerce_datetime64_ns(when_utc)
        if when <= self._times[0]:
            return 0, 0, 0.0
        if when >= self._times[-1]:
            last = len(self._times) - 1
            return last, last, 0.0

        upper = int(np.searchsorted(self._times, when, side="right"))
        lower = upper - 1
        lower_time = self._times[lower]
        upper_time = self._times[upper]
        span_ns = (upper_time - lower_time).astype("timedelta64[ns]").astype(np.int64)
        offset_ns = (when - lower_time).astype("timedelta64[ns]").astype(np.int64)
        weight = 0.0 if span_ns == 0 else offset_ns / span_ns
        return lower, upper, float(weight)

    def _sample_field(
        self,
        data: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        lat: float,
        lon: float,
        when_utc: datetime | np.datetime64,
    ) -> float:
        lower, upper, time_weight = self._time_bounds(when_utc)
        lower_value = _bilinear(data[lower], lats, lons, lat, lon)
        if lower == upper:
            return lower_value
        upper_value = _bilinear(data[upper], lats, lons, lat, lon)
        return (1.0 - time_weight) * lower_value + time_weight * upper_value

    def _sample_angle(
        self,
        sin_field: np.ndarray,
        cos_field: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        lat: float,
        lon: float,
        when_utc: datetime | np.datetime64,
    ) -> float:
        sample_sin = self._sample_field(sin_field, lats, lons, lat, lon, when_utc)
        sample_cos = self._sample_field(cos_field, lats, lons, lat, lon, when_utc)
        return math.degrees(math.atan2(sample_sin, sample_cos)) % 360.0

    def get_environment(
        self,
        lat: float,
        lon: float,
        when_utc: datetime | np.datetime64,
    ) -> EnvironmentData:
        u10 = self._sample_field(self._u10, self._era5_lats, self._era5_lons, lat, lon, when_utc)
        v10 = self._sample_field(self._v10, self._era5_lats, self._era5_lons, lat, lon, when_utc)
        swh = self._sample_field(self._swh, self._era5_lats, self._era5_lons, lat, lon, when_utc)
        mwp = self._sample_field(self._mwp, self._era5_lats, self._era5_lons, lat, lon, when_utc)
        mwd = self._sample_angle(self._mwd_sin, self._mwd_cos, self._era5_lats, self._era5_lons, lat, lon, when_utc)

        uo = self._sample_field(self._uo, self._cmems_lats, self._cmems_lons, lat, lon, when_utc)
        vo = self._sample_field(self._vo, self._cmems_lats, self._cmems_lons, lat, lon, when_utc)

        wind_speed_ms = math.hypot(u10, v10)
        wind_dir_deg = (270.0 - math.degrees(math.atan2(v10, u10))) % 360.0

        current_speed_ms = math.hypot(uo, vo)
        current_dir_deg = (90.0 - math.degrees(math.atan2(vo, uo))) % 360.0

        return EnvironmentData(
            wind_speed_ms=wind_speed_ms,
            wind_dir_deg=wind_dir_deg,
            current_speed_ms=current_speed_ms,
            current_dir_deg=current_dir_deg,
            wave_height_m=max(0.0, swh),
            wave_period_s=max(0.0, mwp),
            wave_dir_deg=mwd,
        )


def create_synthetic_environment(
    base_wind_speed: float = 15.0,
    base_wind_dir: float = 315.0,
    base_current_speed_ms: float = 0.0,
    base_current_dir_deg: float = 90.0,
    base_wave_height_m: float = 0.0,
    base_wave_period_s: float = 0.0,
    base_wave_dir_deg: float = 315.0,
) -> Callable[[float, float, datetime], EnvironmentData]:
    """Create a lightweight deterministic environment callback for tests and demos."""

    def env_fn(lat: float, lon: float, when_utc: datetime | np.datetime64) -> EnvironmentData:
        hour = _coerce_datetime64_ns(when_utc).astype("datetime64[h]").astype(int) % 24
        wind_speed = max(0.0, base_wind_speed + 1.5 * math.sin(math.radians(lat * 6.0 + hour * 10.0)))
        wind_dir = (base_wind_dir + 12.0 * math.cos(math.radians(lon * 4.0 - hour * 3.0))) % 360.0
        current_speed = max(0.0, base_current_speed_ms + 0.15 * math.cos(math.radians(lat * 8.0 + lon * 3.0)))
        current_dir = (base_current_dir_deg + 8.0 * math.sin(math.radians(hour * 12.0 + lon))) % 360.0
        wave_height = max(0.0, base_wave_height_m + 0.2 * math.sin(math.radians(lat * 5.0 - hour * 8.0)))
        wave_period = max(0.0, base_wave_period_s + 0.4 * math.cos(math.radians(lon * 4.0 + hour * 9.0)))
        wave_dir = (base_wave_dir_deg + 10.0 * math.sin(math.radians(hour * 7.0 + lat * 2.0))) % 360.0
        return EnvironmentData(
            wind_speed_ms=wind_speed,
            wind_dir_deg=wind_dir,
            current_speed_ms=current_speed,
            current_dir_deg=current_dir,
            wave_height_m=wave_height,
            wave_period_s=wave_period,
            wave_dir_deg=wave_dir,
        )

    return env_fn
