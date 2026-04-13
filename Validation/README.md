# Verification Scenarios

`Verification/` contains scenario runners for the current research workflow.

These scripts are separate from `tests/test_ga.py`:

- `tests/test_ga.py`
  Runs the maintained integrated GA + MILP flow end-to-end.
- `Verification/case*.py`
  Runs named research scenarios for comparison and interpretation.

## Case Definitions

- `case1_astar_fixed.py`
  A* on the land-mask grid with a fixed-speed baseline, then MILP scheduling.
- `case2_ga_twostage.py`
  GA route search with an energy-only objective, then MILP scheduling on the selected route.
- `case3_ga_integrated.py`
  Integrated GA + MILP. This is the same workflow as `tests/test_ga.py`, kept as a named scenario.
- `case4_extreme_weather.py`
  Integrated GA + MILP under synthetic extreme weather.
- `case5_ess_utility.py`
  Integrated route optimization followed by a SOC sensitivity comparison in MILP scheduling.

## Weather Data Policy

- Cases 1, 2, 3, and 5 use the shared marine environment loader.
- The loader expects:
  - an ERA5 NetCDF containing `u10`, `v10`, `swh`, `mwp`, `mwd`
  - a CMEMS NetCDF containing `uo`, `vo`
  - overlapping UTC time windows between the two datasets
- Dataset paths are auto-discovered from `data/era5` and `data/cmems`, or can be overridden with `MARINE_ERA5_PATH` and `MARINE_CMEMS_PATH`.
- Case 4 uses synthetic weather on purpose.

## Shared Helpers

- `Verification/common.py`
  Shared path resolution, marine environment loading, output directory creation, and route-to-MILP helper utilities.

## Notes

- Verification scripts no longer need `src/optimizer/fitness.py`.
- Legacy fitness helpers may remain in the repository, but the maintained verification path uses `src/optimizer/ga_engine.py`.
