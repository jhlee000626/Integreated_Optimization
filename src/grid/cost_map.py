"""
Binary Cost Map — 육지 회피 비용 맵
=====================================
고해상도 해안선(Natural Earth) → 이진 육지 마스크 생성

비용 레벨:
    100  : 육지 내부 (Hard violation)
    0    : 원양 (No cost)

GA 적합도에서:
    Fitness = FC_total + λ × Σ segment_violation_cost
"""

import os
import numpy as np


# =============================================================================
# 바운딩 박스
# =============================================================================

# 해당 BOUNDS가지고 LAT, LON GRID COST MAP 생성
BUSAN_JEJU_BOUNDS = {
    "lat_min": 32.0,
    "lat_max": 36.0,
    "lon_min": 125.0,
    "lon_max": 131.0,
}


BUSAN_PORT = (35.075, 129.113)
JEJU_PORT  = (33.545, 126.558)


# =============================================================================
# Cost Map 생성
# =============================================================================

class CostMap:
    """
    Gaussian 기반 육지 회피 비용 맵.

    # 사용방법 (costmap 빌드, 비용 조회, 경로 비용 계산)
    Usage:
        cmap = CostMap(resolution=0.002, sigma=30)
        cmap.build()
        cost = cmap.get_cost(lat=34.0, lon=127.5)
        seg_cost = cmap.segment_cost(lat1, lon1, lat2, lon2)
    """

    def __init__(
        self,
        resolution: float = 0.02,
        land_value: float = 100.0,
        bounds: dict | None=None,
        coastline_source: str = "natural_earth",
    ):
        """
        Parameters
        ----------
        resolution : float
            격자 해상도 (도). 0.002° ≈ 220m
        land_value : float
            육지 내부 비용 값 (기본 100)
        bounds : dict
            바운딩 박스
        coastline_source : str
            해안선 소스 ('auto', 'natural_earth')
        """
        self.resolution = resolution
        self.land_value = land_value
        self.bounds = bounds or BUSAN_JEJU_BOUNDS
        self.coastline_source = coastline_source

        # 격자 생성
        # np.arange(start, end, step) : start부터 end까지 step 간격으로 배열 생성, 약 200m 간격
        self.lats = np.arange(
            self.bounds["lat_min"],
            self.bounds["lat_max"] + resolution,
            resolution,
        )
        self.lons = np.arange(
            self.bounds["lon_min"],
            self.bounds["lon_max"] + resolution,
            resolution,
        )

        # 격자 개수
        self.n_lat = len(self.lats)
        self.n_lon = len(self.lons)

        # 비용 맵 (build() 후 채워짐)
        self.cost_grid = None     # (n_lat, n_lon)
        self.land_mask = None     # (n_lat, n_lon) bool

    # build() 메서드: 비용 맵 구축
    def build(self) -> 'CostMap':
        """
        비용 맵 구축:
            1. 해안선 로드
            2. 이진 육지 마스크 래스터화
        """

        # no_go_zone 모듈에서 load_coastline 함수를 불러옴
        from src.grid.no_go_zone import load_coastline

        # resolution에 따라 격자 개수 출력
        print(f"[CostMap] Building... (res={self.resolution}°, "
              f"grid={self.n_lat}×{self.n_lon})")

        # ── 1. 해안선 로드 ──
        # load_coastline 함수는 WGS84 좌표계의 육지(Polygon)의 vector data를 반환 [DD형식으로 반환]
        land_poly = load_coastline(
            source=self.coastline_source,
            bounds=self.bounds,
        )

        # ── 2. 이진 육지 마스크 래스터화 ──
        # load_coastline에서 반환된 vector data를 rasterize하여 이진 마스크 생성
        print("[CostMap] Rasterizing land mask...")
        # _rasterize_land 메서드를 호출하여 이진 마스크 생성
        # self.land_mask는 인덱스 0과 1(육지)을 포함하는 cost map이 생성
        self.land_mask = self._rasterize_land(land_poly)

        # land_pct는 land_mask에서 1의 개수를 전체 격자 개수로 나눈 값
        land_pct = self.land_mask.sum() / self.land_mask.size * 100

        # 전체 맵에서 Land 비율 출력
        print(f"[CostMap] Land pixels: {self.land_mask.sum()} ({land_pct:.1f}%)")

        # ── 3. 이진 비용 맵 적용 ──
        print("[CostMap] Creating binary cost map...")
        # cost_grid는 land_mask와 동일한 shape, dtype=np.float64
        self.cost_grid = np.zeros_like(self.land_mask, dtype=np.float64)
        # land_mask가 True인 부분의 cost_grid 값을 land_value로 설정 (=100)
        self.cost_grid[self.land_mask] = self.land_value

        print(f"[CostMap] Done. Cost range: {self.cost_grid.min():.1f} ~ {self.cost_grid.max():.1f}")
        return self

    def _rasterize_land(self, land_poly) -> np.ndarray:
        """
        Shapely 폴리곤 → 래스터 이진 마스크.
        벡터화된 contains 검사로 속도 최적화.
        """
        import shapely
        from shapely.vectorized import contains
        # self.lons, self.lats의 range를 이용해서 2차원 배열 H X W 행렬
        # lon_grid는 (H, W) shape, 각 원소는 해당 열의 경도 값
        # lat_grid는 (H, W) shape, 각 원소는 해당 행의 위도 값
        lon_grid, lat_grid = np.meshgrid(self.lons, self.lats)

        # contains를 써서, 같은 인덱스의 (lon, lat) 쌍이 land_poly 안에 있는지 검사
        mask = contains(land_poly, lon_grid, lat_grid)
        return mask

    # ─────────────────────────────────────────────────────────────────
    # 비용 조회
    # ─────────────────────────────────────────────────────────────────

    # (lat, lon) 좌표의 비용 값 반환
    def get_cost(self, lat: float, lon: float) -> float:
        """
        (lat, lon) 좌표의 비용 값 반환.

        Returns
        -------
        float
            0 (안전) or 100 (육지 내부)
        """
        
        if self.cost_grid is None:
            raise RuntimeError("CostMap not built. Call build() first.")

        # 범위 밖 → 높은 페널티
        if (lat < self.bounds["lat_min"] or lat > self.bounds["lat_max"] or
            lon < self.bounds["lon_min"] or lon > self.bounds["lon_max"]):
            return self.land_value

        # 격자 인덱스
        lat_idx = int(round((lat - self.bounds["lat_min"]) / self.resolution))
        lon_idx = int(round((lon - self.bounds["lon_min"]) / self.resolution))

        # 클리핑
        # 인덱스가 범위를 벗어나지 않도록 조정
        lat_idx = np.clip(lat_idx, 0, self.n_lat - 1)
        lon_idx = np.clip(lon_idx, 0, self.n_lon - 1)

        return float(self.cost_grid[lat_idx, lon_idx])

    def segment_cost(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float,
    ) -> float:
        """
        두 점 사이 선분의 비용.
        경로상의 최고 위반 비용을 반환하여, 
        한 점이라도 육지를 관통하면 강하게 페널티를 받도록 함.
        """
        import math
        # 두 점 사이의 유클리드 거리 (도 단위)
        dist_deg = math.hypot(lat2 - lat1, lon2 - lon1)
        
        # 격자 해상도의 절반 간격으로 촘촘하게 샘플링 (섬/육지 뛰어넘기 방지)
        step_deg = self.resolution * 0.5
        n_samples = max(2, int(dist_deg / step_deg) + 1)
        
        max_cost = 0.0
        for k in range(n_samples + 1):
            frac = k / n_samples
            lat = lat1 + frac * (lat2 - lat1)
            lon = lon1 + frac * (lon2 - lon1)
            cost = self.get_cost(lat, lon)
            if cost > max_cost:
                max_cost = cost

        return max_cost

    def route_violation(self, waypoints: list) -> float:
        """
        전체 경로의 violation cost.
        Returns
        -------
        float
            경로 중 가장 위험한 구간의 비용 (Max Segment Cost)
        """
        max_seg_cost = 0.0
        for i in range(len(waypoints) - 1):
            lat1, lon1 = waypoints[i]
            lat2, lon2 = waypoints[i + 1]
            seg_cost = self.segment_cost(lat1, lon1, lat2, lon2)
            if seg_cost > max_seg_cost:
                max_seg_cost = seg_cost
        return max_seg_cost

    # ─────────────────────────────────────────────────────────────────
    # 시각화
    # ─────────────────────────────────────────────────────────────────

    def plot(self, save_path: str = "output/cost_map.png", route_wps=None):
        """비용 맵 시각화."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        fig, ax = plt.subplots(figsize=(14, 10))

        colors = ['white', 'green']
        custom_cmap = ListedColormap(colors)

        # 비용 맵 이미지
        extent = [
            self.bounds["lon_min"], self.bounds["lon_max"],
            self.bounds["lat_min"], self.bounds["lat_max"],
        ]
        im = ax.imshow(
            self.cost_grid, # cost_grid: (H, W) shape, 각 원소는 0 또는 100, Matrix임
            extent=extent, # extent는 행렬의 인덱스를 그래프의 축을 실제 경위도 숫자로 매핑
            origin='lower', # origin='lower'는 행렬의 (0,0)을 그래프의 왼쪽 아래로 설정
            cmap=custom_cmap, # cmap은 색상 맵
            vmin=0, vmax=100, # vmin, vmax는 색상의 최솟값과 최댓값
            aspect='equal', # aspect='equal'은 x축과 y축의 비율을 동일하게 설정
            interpolation='nearest' # interpolation='nearest'는 nearest neighbor 보간법을 사용
        )

        # 이진 맵이므로 레벨은 50 하나만 표시
        X, Y = np.meshgrid(self.lons, self.lats)
        # 등고선 표시
        cs = ax.contour(X, Y, self.cost_grid, levels=[50],
                        colors='black', linewidths=0.5, alpha=0.8)
        
        # ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f')

        # 경로 (위경도 Value 그대로 받아서 ax extent내부에 매칭)
        if route_wps:
            lats = [wp[0] for wp in route_wps]
            lons = [wp[1] for wp in route_wps]
            ax.plot(lons, lats, 'o-', color='blue', linewidth=2,
                    markersize=4, markeredgecolor='blue', label='Route')

        # 항구
        ax.plot(BUSAN_PORT[1], BUSAN_PORT[0], '*', color='orange',
                markersize=15, markeredgecolor='black', label='Busan Port')
        ax.plot(JEJU_PORT[1], JEJU_PORT[0], '*', color='orange',
                markersize=15, markeredgecolor='black', label='Jeju Port')

        # cbar = fig.colorbar(im, ax=ax, shrink=0.7)
        # cbar.set_label("Violation Cost", fontsize=12)
        # cbar.set_ticks([0, 20, 50, 80, 100])
        # cbar.set_ticklabels(['0\n(Open Sea)', '20', '50', '80', '100\n(Land)'])

        ax.set_xlabel("Longitude (°E)", fontsize=12)
        ax.set_ylabel("Latitude (°N)", fontsize=12)
        ax.set_title("Cost Map", fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[SAVED] {save_path}")


# =============================================================================
# 편의 함수
# =============================================================================

def build_cost_map(
    resolution: float = 0.02,
) -> CostMap:
    """CostMap 인스턴스 생성 + 구축."""
    cmap = CostMap(resolution=resolution)
    return cmap.build()

