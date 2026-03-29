import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.grid.cost_map import build_cost_map
from src.grid.no_go_zone import BUSAN_PORT, JEJU_PORT

cmap = build_cost_map(resolution=0.005)

output_path = Path(__file__).with_name("port_coords.txt")

with output_path.open("w", encoding="utf-8") as f:
    def find_sea(port, name):
        f.write(f"Original {name}: {port}, Cost = {cmap.get_cost(*port)}\n")
        # Try finding the closest
        for radius in [0.01, 0.02, 0.05, 0.1]:
            for angle in range(0, 360, 10):
                lat = port[0] + radius * math.sin(math.radians(angle))
                lon = port[1] + radius * math.cos(math.radians(angle))
                if cmap.get_cost(lat, lon) == 0.0:
                    f.write(f"  Found sea at radius {radius}: {(lat, lon)}\n")
                    return
        f.write(f"  Could not find sea near {port}\n")

    find_sea(BUSAN_PORT, "Busan")
    find_sea(JEJU_PORT, "Jeju")
