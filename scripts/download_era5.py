"""
ERA5 데이터 다운로드 스크립트
=============================
CDS API로 부산→제주 해역의 ERA5 바람 데이터를 다운로드.

사전 준비:
    1. pip install cdsapi
    2. CDS 계정 → https://cds.climate.copernicus.eu
    3. ~/.cdsapirc 파일에 API key 설정:
        url: https://cds.climate.copernicus.eu/api
        key: <YOUR-UID>:<YOUR-API-KEY>

실행: python scripts/download_era5.py
"""

import os

# 다운로드 설정
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "era5")
AREA = [36, 125, 32, 131]  # [North, West, South, East] 위도 32~36, 경도 125~131

# 기본: 2024년 1월 한 달 (테스트용)
DEFAULT_YEAR = "2024"
DEFAULT_MONTH = "01"
DEFAULT_DAYS = [f"{d:02d}" for d in range(1, 32)]
DEFAULT_TIMES = [f"{h:02d}:00" for h in range(0, 24, 1)]  # 1시간 간격


def download_era5(
    year: str = DEFAULT_YEAR,
    month: str = DEFAULT_MONTH,
    days: list = None,
    times: list = None,
    output_name: str = None,
):
    """
    ERA5 10m 바람 + 파랑 데이터 다운로드.

    Variables:
        - u10: 10m U-component of wind (m/s)
        - v10: 10m V-component of wind (m/s)
    """
    import cdsapi

    if days is None:
        days = DEFAULT_DAYS
    if times is None:
        times = DEFAULT_TIMES
    if output_name is None:
        output_name = f"era5_wind_{year}_{month}.nc"

    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, output_name)

    if os.path.exists(output_path):
        print(f"[OK] Already exists: {output_path}")
        return output_path

    print(f"[1] ERA5 다운로드 시작...")
    print(f"    Period: {year}-{month}")
    print(f"    Area: {AREA}")
    print(f"    Variables: u10, v10")

    client = cdsapi.Client()

    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
            ],
            "year": year,
            "month": month,
            "day": days,
            "time": times,
            "area": AREA,
            "format": "netcdf",
        },
        output_path,
    )

    print(f"[OK] Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    download_era5()
