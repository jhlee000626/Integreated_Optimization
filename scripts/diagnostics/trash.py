import xarray as xr
import numpy as np
import pandas as pd

# weather_path = 'data\era5\era5_wind_2024_01.nc'

# df = xr.open_dataset(weather_path)
# print(df)

# print('상세 정보')
# print(df.info())

# print('차원')
# print(df.dims)

# print('좌표')
# print(df.coords)

# print('변수')
# print(df.data_vars)

# print('좌표')
# print(df.coords.keys())

# print('전역 메타 데이터')
# print(df.attrs)

# print('헤더')
# print(df.head())

u = np.sqrt(3)
v = 1
wind_direction = np.arctan2(u, v)
wind_deg = np.rad2deg(wind_direction)
wind_deg = np.mod(wind_deg + 360, 360)
print(wind_deg)