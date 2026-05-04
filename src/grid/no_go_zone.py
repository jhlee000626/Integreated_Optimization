"""
No-Go Zone 생성 모듈
====================
Natural Earth 10m 해상도 실제 해안선 데이터를 사용하여
정밀한 No-Go Zone을 생성한다.

데이터 소스:
    1. Natural Earth 10m Land shapefile (기본 — geopandas 로드)
    2. GSHHG shapefile (고정밀 옵션)

부산 → 상하이 바운딩 박스:
    lat: 30.0°N ~ 36.0°N
    lon: 121.0°E ~ 131.0°E
"""

import os
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
from typing import Optional


# =============================================================================
# 부산-상하이 해역 바운딩 박스
# =============================================================================

BUSAN_SHANGHAI_BOUNDS = {
    "lat_min": 30.0,
    "lat_max": 36.0,
    "lon_min": 121.0,
    "lon_max": 131.0,
}

# BUSAN_PORT = (35.075, 129.113)   # 원래 좁은 항구 위치 (영도 안쪽)
BUSAN_PORT = (34.950, 129.150)   # (Pilot Station) 탁 트인 대한해협 앞바다로 전진 배치
# SHANGHAI_PORT  = (31.366, 121.614)   # 원래 상하이 항구 위치 (양쯔강 하구 복잡한 수역)
SHANGHAI_PORT  = (31.000, 122.00)   # 상하이 앞바다 외항(Pilot/오픈 바다)으로 적절히 전진 배치
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_NE_SHP_DIR = os.path.join(_PROJECT_ROOT, "data", "ne_10m_land")
_NE_SHP_FILE = os.path.join(_NE_SHP_DIR, "ne_10m_land.shp")


# =============================================================================
# 해안선 로드 (데이터 소스 선택)
# =============================================================================

def load_coastline(
    source: str = "auto",
    shp_path: Optional[str] = None,
    bounds: Optional[dict] = None,
) -> MultiPolygon:
    """
    해안선 데이터 로드.

    Parameters
    ----------
    source : str
        'auto'          : Natural Earth 10m 자동 감지, 없으면 다운로드 시도
        'natural_earth' : Natural Earth shapefile 직접 지정
    shp_path : str, optional
        shapefile 경로 (source='natural_earth' 또는 'gshhg')
    bounds : dict, optional
        바운딩 박스. None이면 BUSAN_SHANGHAI_BOUNDS 사용.

    Returns
    -------
    MultiPolygon
        육지 폴리곤 (bbox로 클리핑됨)
    """
    if bounds is None:
        bounds = BUSAN_SHANGHAI_BOUNDS

    if source == "auto":
        # Natural Earth 10m 자동 감지
        if os.path.exists(_NE_SHP_FILE):
            return _load_from_shapefile(_NE_SHP_FILE, bounds)
        else:
            # 자동 다운로드 시도
            print("[INFO] Natural Earth 10m 데이터 미발견. 다운로드를 시도합니다...")
            downloaded = _download_natural_earth()
            if downloaded and os.path.exists(_NE_SHP_FILE):
                return _load_from_shapefile(_NE_SHP_FILE, bounds)

    elif source == "natural_earth":
        path = shp_path or _NE_SHP_FILE
        return _load_from_shapefile(path, bounds)

    else:
        raise ValueError(f"Unknown source: {source}")


# 데이터 읽기 및 가공
def _load_from_shapefile(shp_path: str, bounds: dict) -> MultiPolygon:
    """geopandas로 shapefile 로드 + bbox 클리핑."""
    import geopandas as gpd

    print(f"[INFO] Loading coastline: {shp_path}")

    # 바운딩 박스 내 육지 데이터 로드 (해안선을 포함하는 영역도 갖고옴)
    gdf = gpd.read_file(
        shp_path,
        bbox=(bounds["lon_min"], bounds["lat_min"],
              bounds["lon_max"], bounds["lat_max"]),
    )

    if gdf.empty:
        print("[WARN] 바운딩 박스 내 육지 데이터 없음.")
        return MultiPolygon()

    # 여기서 겹쳐있는 Polygon 합집합
    land = unary_union(gdf.geometry)

    # bbox 클리핑 (바운딩 박스 밖의 데이터 제거)
    bbox = box(
        bounds["lon_min"], bounds["lat_min"],
        bounds["lon_max"], bounds["lat_max"],
    )

    # 바운딩 박스 내의 데이터만 남김
    land = land.intersection(bbox)

    # 타입 정리 (MultiPolygon으로 변환, Polygon이 여러 개일 경우를 대비)
    land = _ensure_multipolygon(land)

    # Polygon 개수
    n = len(land.geoms) if hasattr(land, 'geoms') else 1
    print(f"[INFO] Loaded {n} land polygon(s)")

    # shapely Polygon은 WGS84 좌표계를 가지고 X,Y 경도 위도 좌표를 가지고 있음
    # 따라서 연속적인 공간의 육지(Polygon)의 좌표값 x,y의 vector data반환
    return land


def _download_natural_earth() -> bool:
    """Natural Earth 10m Land 다운로드."""
    import zipfile
    import urllib.request

    url = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip"
    zip_path = os.path.join(_PROJECT_ROOT, "data", "ne_10m_land.zip")

    try:
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        print(f"    Downloading from: {url}")
        urllib.request.urlretrieve(url, zip_path)

        os.makedirs(_NE_SHP_DIR, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(_NE_SHP_DIR)

        if os.path.exists(zip_path):
            os.remove(zip_path)

        print(f"    Downloaded to: {_NE_SHP_DIR}")
        return True
    except Exception as e:
        print(f"    Download failed: {e}")
        return False

# Geopandas로 데이터 읽어와서 MultiPolygon으로 변환
def _ensure_multipolygon(geom) -> MultiPolygon:
    """Geometry를 MultiPolygon으로 정규화."""
    # Multipolygon instance가 geom이면 그대로 반환
    if isinstance(geom, MultiPolygon):
        return geom
    # Polygon instance가 geom이면 MultiPolygon에 넣어 반환
    elif isinstance(geom, Polygon):
        return MultiPolygon([geom])
    else:
        # GeometryCollection에서 Polygon만 추출
        # Polygon안에 Polyline이나 Point가 있을 경우 Polygon안에 병합
        polys = [g for g in geom.geoms if g.geom_type == 'Polygon']
        return MultiPolygon(polys) if polys else MultiPolygon()


