# Legacy Verification Scripts

`Verification/` contains older experiment scripts kept for reference.

These files are not aligned with the current public interfaces and should be
treated as legacy examples until they are explicitly modernized.

Known mismatches with the current codebase:
- `create_synthetic_weather` is referenced, but is not provided by the current weather loader.
- `N_WAYPOINTS` is referenced, but the current GA engine does not export it.
- Several scripts assume the old `setup_ga()` return shape (`toolbox, pop`).
- Several scripts assume the old `decode_route()` return shape (`route, speed_profile`).
- Several scripts call `plot_power_schedule()` with an outdated signature.
- `case4_extreme_weather.py` references `milp.dg_max`, which the current solver does not expose.
- The old root diagnostic `check_cost.py` previously called `build_cost_map(..., sigma=...)`, which is not part of the current interface.

Use `tests/` for the maintained manual verification scripts.
