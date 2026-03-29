"""
ERA5 기상 데이터 로더 (.nc 파일 로드)
====================
ERA5 NetCDF 파일에서 바람 데이터를 로드하고,
GA의 weather_fn 인터페이스로 제공한다.

인터페이스:
    weather_fn(lat, lon) → (wind_speed_ms, wind_dir_deg)
"""

import os
import math
import numpy as np
from typing import Tuple, Optional, Callable


class ERA5Loader:
    """
    ERA5 NetCDF 데이터를 로드하여 위치별 바람 정보를 제공.

    method:
        loader = ERA5Loader("data/era5/era5_wind_2024_01.nc")
        wind_speed, wind_dir = loader.get_wind(lat=34.0, lon=127.5)
    """

    def __init__(self, nc_path: str, time_index: int = 0):
        """
        Parameters
        ----------
        nc_path : str
            ERA5 NetCDF 파일 경로
        time_index : int
            사용할 시간 인덱스 (단일 시각 사용 시). default=0 (첫 번째 시각)
        """
        import xarray as xr

        self.ds = xr.open_dataset(nc_path)
        self.time_index = time_index

        ds_sel = self.ds

        # 시간 차원 선택
        time_dim = None
        for dim in ['time', 'valid_time']:
            if dim in ds_sel.dims:
                time_dim = dim
                break
        
        # 시간 차원이 존재하면 time_index에 해당하는 데이터만 선택 (특정 시간의 기상 공간 데이터 추출)
        if time_dim is not None:
            ds_sel = ds_sel.isel({time_dim: time_index})

        # Squeeze를 통해 남은 차원이 lat, lon 2개뿐이도록 확인 (Time 차원이 1이 남으므로 Squeeze로 제거)
        self._u10 = ds_sel['u10'].values.squeeze()
        self._v10 = ds_sel['v10'].values.squeeze()

        # u10, v10이 2D 배열인지 확인 (latitude, longitud 배열인지)
        if self._u10.ndim != 2:
            raise ValueError(f"u10 data is not 2D after selection. Final shape: {self._u10.shape}. "
                             f"Check NetCDF dimensions: {self.ds.dims}")

        # 좌표축 (latitude/longitude 또는 lat/lon) --> 1차원 데이터로 위/경도 추출
        self._lats = ds_sel['latitude'].values if 'latitude' in ds_sel.coords else ds_sel['lat'].values
        self._lons = ds_sel['longitude'].values if 'longitude' in ds_sel.coords else ds_sel['lon'].values

        # ECMWF데이터 Path, Shape[위도, 경도], 위경도의 min,max 값 확인
        print(f"[ERA5] Loaded: {nc_path}")
        print(f"       Shape: {self._u10.shape}")
        print(f"       Lat: {self._lats.min():.2f} ~ {self._lats.max():.2f}")
        print(f"       Lon: {self._lons.min():.2f} ~ {self._lons.max():.2f}")

        # 시간 차원이 존재하면 시간 정보 출력, Time_index에 해당하는 데이터만 선택 일단 nc에는 없음 
        if 'time' in self.ds.dims:
            times = self.ds['time'].values
            print(f"       Time steps: {len(times)}, using index {time_index}")

    def get_wind(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        주어진 좌표의 바람 데이터 반환 (최근접 격자점).

        Parameters
        ----------
        lat : float
            위도 (°N)
        lon : float
            경도 (°E)

        Returns
        -------
        tuple
            (wind_speed_ms, wind_dir_deg)
            wind_dir_deg: 바람이 불어오는 방향 (기상학 관례, 0°=N)
        """
        # 최근접 격자점 인덱스
        # 입력된 lat과 self._lats의 차이의 절댓값이 가장 작은 값의 인덱스 변환 --> Nearest Neighbor의 인덱스
        lat_idx = int(np.argmin(np.abs(self._lats - lat)))
        lon_idx = int(np.argmin(np.abs(self._lons - lon)))

        # 가까운 인덱스에서의 u, v 성분 추출
        u = self._u10[lat_idx, lon_idx].item()
        v = self._v10[lat_idx, lon_idx].item()

        # 풍속 계산
        wind_speed = math.sqrt(u ** 2 + v ** 2)

        # 풍향 (기상학 관례: 바람이 불어오는 방향)
        # meteorological: direction FROM which wind blows
        wind_dir = (270 - math.degrees(math.atan2(v, u))) % 360

        return wind_speed, wind_dir
        
    def get_weather_fn(self) -> Callable:
        """
        GA에 전달할 weather_fn 클로저 반환.

        Returns
        -------
        callable
            (lat, lon) → (wind_speed_ms, wind_dir_deg)
        """
        return self.get_wind

    def close(self):
        self.ds.close()
