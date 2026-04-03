from __future__ import annotations

import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.optimizer.milp_solver import MILPSolver
from src.visualization.plotter import plot_sfoc_pwl


def main() -> None:
    output_dir = os.path.join(PROJECT_ROOT, "output", "sfoc_pwl")
    sfoc_path = os.path.join(PROJECT_ROOT, "config", "sfoc.json")
    milp = MILPSolver(sfoc_json_path=sfoc_path, n_pwl_segments=5)
    plot_sfoc_pwl(milp, save_dir=output_dir)
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
