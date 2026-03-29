"""
Natural Earth 해안선 데이터 다운로드
=====================================
10m 해상도 Land polygons을 다운로드하여 data/ 폴더에 저장.

실행: python scripts/download_coastline.py
"""

import os
import zipfile
import urllib.request

# Natural Earth 10m Land polygons
URL = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ZIP_PATH = os.path.join(DATA_DIR, "ne_10m_land.zip")
SHP_DIR = os.path.join(DATA_DIR, "ne_10m_land")


def download():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 이미 shapefile 존재 확인
    shp_file = os.path.join(SHP_DIR, "ne_10m_land.shp")
    if os.path.exists(shp_file):
        print(f"[OK] Already exists: {shp_file}")
        return shp_file

    # 다운로드
    print(f"[1] Downloading Natural Earth 10m Land...")
    print(f"    URL: {URL}")
    urllib.request.urlretrieve(URL, ZIP_PATH)
    print(f"    Saved: {ZIP_PATH}")

    # 압축 해제
    print(f"[2] Extracting...")
    os.makedirs(SHP_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(SHP_DIR)
    print(f"    Extracted to: {SHP_DIR}")

    # zip 삭제
    os.remove(ZIP_PATH)

    print(f"[OK] Done: {shp_file}")
    return shp_file


if __name__ == "__main__":
    download()
