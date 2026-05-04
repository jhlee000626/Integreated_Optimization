# Integrated Optimization

This repository studies route optimization and onboard power scheduling for the Busan -> Shanghai voyage.

## Main Components

- `src/optimizer/ga_engine.py`
  Integrated DEAP-based GA route optimizer with MILP-in-the-loop evaluation.
- `src/optimizer/milp_solver.py`
  Generator + ESS scheduling model.
- `src/weather/era5_loader.py`
  ERA5 weather loading and synthetic-weather helpers.
- `src/grid/cost_map.py`
  Land-mask-based routing penalty map.

## Maintained Execution Paths

- `tests/test_ga.py`
  Main end-to-end manual runner for the integrated GA + MILP workflow.
- `Verification/`
  Named research scenarios for baseline, two-stage optimization, integrated optimization, extreme weather, and SOC sensitivity studies.

## Verification Cases

- `case1`
  A* fixed-route baseline with MILP scheduling.
- `case2`
  Energy-only GA route search followed by MILP scheduling.
- `case3`
  Same integrated workflow as `tests/test_ga.py`.
- `case4`
  Integrated optimization under synthetic extreme weather.
- `case5`
  SOC sensitivity comparison using the optimized route.

## Weather Usage

- ERA5 data is used in Cases 1, 2, 3, and 5.
- Synthetic weather is used only in Case 4.

## Legacy Note

`src/optimizer/fitness.py` is no longer part of the maintained verification path. The current workflow is centered on `src/optimizer/ga_engine.py`.
