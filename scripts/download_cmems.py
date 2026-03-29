"""
CMEMS 해류 데이터 다운로드 스크립트
====================================
Copernicus Marine Service API를 사용하여
부산-제주 해역(125~130°E, 32~36°N)의 해류(uo, vo) 데이터를 다운로드.

사전 설정:
    pip install copernicusmarine
    copernicusmarine login          # 계정 인증 (최초 1회)

실행:
    python scripts/download_cmems.py
"""

import os
import copernicusmarine

# ─── 설정 ───
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "cmems")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "cmems_current.nc")

# 공간 범위 (GSHHG / ERA5 와 동일)
LON_MIN, LON_MAX = 125.0, 130.0
LAT_MIN, LAT_MAX = 32.0, 36.0

# 시간 (단일 시각)
DATETIME = "2025-03-25T00:00:00"

# 데이터셋 ID (CMEMS Global Ocean Physics Analysis & Forecast, 1/12°, 1h)
DATASET_ID = "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print(" CMEMS 해류 데이터 다운로드")
    print("=" * 60)
    print(f"  Dataset : {DATASET_ID}")
    print(f"  영역    : {LON_MIN}~{LON_MAX}°E, {LAT_MIN}~{LAT_MAX}°N")
    print(f"  시각    : {DATETIME}")
    print(f"  저장 경로: {OUTPUT_FILE}")
    print()

    copernicusmarine.subset(
        dataset_id=DATASET_ID,
        variables=["uo", "vo"],
        minimum_longitude=LON_MIN,
        maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN,
        maximum_latitude=LAT_MAX,
        start_datetime=DATETIME,
        end_datetime=DATETIME,
        minimum_depth=0.49402499198913574,
        maximum_depth=0.49402499198913574,
        output_filename="cmems_current.nc",
        output_directory=OUTPUT_DIR,
        force_download=True,
    )

    if os.path.exists(OUTPUT_FILE):
        size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
        print(f"\n ✅ 다운로드 완료: {OUTPUT_FILE} ({size_mb:.2f} MB)")
    else:
        print("\n ❌ 다운로드 실패. Copernicus Marine 계정 인증을 확인하세요.")
        print("    copernicusmarine login")


if __name__ == "__main__":
    main()
