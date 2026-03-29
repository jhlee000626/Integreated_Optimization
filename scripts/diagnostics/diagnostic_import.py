import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("PYTHONPATH:", sys.path)
try:
    from src.grid.no_go_zone import BUSAN_PORT, JEJU_PORT
    print("SUCCESS: Imported BUSAN_PORT:", BUSAN_PORT)
except Exception as e:
    print("FAILURE:", e)
    import traceback
    traceback.print_exc()
