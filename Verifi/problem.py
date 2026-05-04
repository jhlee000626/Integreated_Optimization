from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import differential_evolution

from spec import (
    DISTANCES,
    ESS_PARAMS,
    GEN_PARAMS,
    K,
    N,
    P_ser,
    SPEED_MARGINS,
    SPEED_NOMINAL,
    VOYAGE_STAGES,
    dt,
    time_steps,
)

try:
    from docplex.mp.model import Model
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
    Model = None
    _DOCPLEX_IMPORT_ERROR = exc
else:
    _DOCPLEX_IMPORT_ERROR = None


def _ensure_docplex_available():
    if Model is None:
        raise ModuleNotFoundError(
            "docplex is required to build this model. Install it with `pip install docplex`."
        ) from _DOCPLEX_IMPORT_ERROR


def _is_dragging_stage(t, voyage_stages):
    return t in voyage_stages["T_dra"]


def _calculate_true_nonlinear_power_by_mode(v, is_dragging):
    if v <= 0:
        return 0.0

    # ★ 추가: knots 단위의 속도를 m/s로 변환 (1 knot = 0.514444 m/s)
    speed_ms = v * 0.514444

    c1 = 1.67e-3
    c2 = 1.83
    c3 = 0.147
    c4 = 1.74
    c5 = 0.15
    eta_pl = 0.97

    L_tug, B_tug, D_tug, CB_tug = 38.5, 10.6, 3.7, 0.97
    L_ves, B_ves, D_ves, CB_ves = 91.5, 24.5, 2.5, 0.95

    # ★ 수정: 변환된 speed_ms를 공식에 적용
    def get_resistance(L, B, D, CB, speed):
        frictional_resistance = c1 * L * (B + 2 * D) * (speed ** c2)
        residual_resistance = c3 * CB * B * D * (speed ** (c4 + c5 * speed))
        return frictional_resistance + residual_resistance

    total_resistance = get_resistance(L_tug, B_tug, D_tug, CB_tug, speed_ms)

    if is_dragging:
        total_resistance += get_resistance(L_ves, B_ves, D_ves, CB_ves, speed_ms)

    # ★ 수정: Power(kW) = Resistance(kN) * Speed(m/s)
    power_kw = (total_resistance * speed_ms) / eta_pl
    
    # ★ 추가: 발전기 용량과 단위를 맞추기 위해 kW를 MW로 변환
    power_mw = power_kw / 1000.0

    return power_mw

def calculate_true_nonlinear_power(v, t, voyage_stages):
    return _calculate_true_nonlinear_power_by_mode(v, _is_dragging_stage(t, voyage_stages))


@lru_cache(maxsize=None)
def _find_optimal_segments_cached(v_min, v_max, segment_count, is_dragging):
    if segment_count < 1:
        raise ValueError("segment_count must be at least 1.")

    if segment_count == 1:
        return (v_min, v_max)

    num_samples = 100
    v_samples = np.linspace(v_min, v_max, num_samples)
    g_samples = np.array(
        [_calculate_true_nonlinear_power_by_mode(v, is_dragging) for v in v_samples]
    )

    def objective(breakpoints):
        sorted_breakpoints = np.sort(breakpoints)
        x_full = np.concatenate(([v_min], sorted_breakpoints, [v_max]))
        g_full = np.array(
            [_calculate_true_nonlinear_power_by_mode(x, is_dragging) for x in x_full]
        )
        g_interp = np.interp(v_samples, x_full, g_full)
        return float(np.sum((g_samples - g_interp) ** 2))

    bounds = [(v_min + 0.1, v_max - 0.1)] * (segment_count - 1)
    result = differential_evolution(
        objective,
        bounds,
        popsize=50,
        maxiter=1000,
        mutation=0.5,
        recombination=0.5,
        atol=1e-4,
        seed=0,
    )

    best_breakpoints = np.sort(result.x)
    optimal_x_points = np.concatenate(([v_min], best_breakpoints, [v_max]))
    return tuple(optimal_x_points.tolist())


def find_optimal_segments(v_min, v_max, segment_count, t, voyage_stages):
    is_dragging = _is_dragging_stage(t, voyage_stages)
    return list(_find_optimal_segments_cached(v_min, v_max, segment_count, is_dragging))


def _get_speed_bounds(t):
    if t in VOYAGE_STAGES["T_doc"]:
        nominal = SPEED_NOMINAL["doc"]
        margin = SPEED_MARGINS["mu1"]
        return nominal * (1 - margin), nominal * (1 + margin)

    if t in VOYAGE_STAGES["T_cru"]:
        nominal = SPEED_NOMINAL["cru"]
        margin = SPEED_MARGINS["lambda"]
        return nominal * (1 - margin), nominal * (1 + margin)

    if t in VOYAGE_STAGES["T_dra"]:
        nominal = SPEED_NOMINAL["dra"]
        margin = SPEED_MARGINS["epsilon"]
        return nominal * (1 - margin), nominal * (1 + margin)

    if t in VOYAGE_STAGES["T_dep"]:
        nominal = SPEED_NOMINAL["dep"]
        margin = SPEED_MARGINS["mu2"]
        return nominal * (1 - margin), nominal * (1 + margin)

    return 0.0, 0.0


def _solve_status_text(model):
    solve_details = getattr(model, "solve_details", None)
    if solve_details is not None:
        status = getattr(solve_details, "status", None)
        if status:
            return str(status)

    solve_status = getattr(model, "solve_status", None)
    if solve_status is not None:
        return str(solve_status)

    return "unknown"


def solve_tugboat_scheduling():
    _ensure_docplex_available()

    model = Model(name="Electric_Tugboat_MIQP")

    unit_time_keys = [(k, t) for k in range(1, K + 1) for t in time_steps]
    ess_time_keys = [(n, t) for n in range(1, N + 1) for t in time_steps]
    soc_keys = [(n, t) for n in range(1, N + 1) for t in [0] + time_steps]

    segment_count = 7
    piecewise_weight_keys = [
        (t, m) for t in time_steps for m in range(1, segment_count + 2)
    ]
    piecewise_selector_keys = [
        (t, m) for t in time_steps for m in range(1, segment_count + 1)
    ]

    P_shore = model.continuous_var_dict(time_steps, lb=0, name='P_shore')
    u = model.binary_var_dict(unit_time_keys, name="u")
    P_G = model.continuous_var_dict(unit_time_keys, lb=0, name="P_G")
    P_c = model.continuous_var_dict(ess_time_keys, lb=0, name="P_c")
    P_dc = model.continuous_var_dict(ess_time_keys, lb=0, name="P_dc")
    SOC = model.continuous_var_dict(
        soc_keys,
        lb=ESS_PARAMS["SOC_min"],
        ub=ESS_PARAMS["SOC_max"],
        name="SOC",
    )
    w_var = model.continuous_var_dict(piecewise_weight_keys, lb=0, ub=1, name="w")
    l_var = model.binary_var_dict(piecewise_selector_keys, name="l")
    V = model.continuous_var_dict(time_steps, lb=0, name="V")
    P_pl = model.continuous_var_dict(time_steps, lb=0, name="P_pl")

    # ★ 수정: GEN_PARAMS[k]["a2"] * (P_G[k, t] ** 2) 추가
    objective = model.sum(
        GEN_PARAMS[k]["a0"] * u[k, t] 
        + GEN_PARAMS[k]["a1"] * P_G[k, t] 
        + GEN_PARAMS[k]["a2"] * (P_G[k, t] ** 2) 
        for t in time_steps
        for k in range(1, K + 1)
    ) + model.sum(
        ESS_PARAMS["F_B"] * (P_c[n, t] + P_dc[n, t])
        for t in time_steps
        for n in range(1, N + 1)
    )
    
    model.minimize(objective)

    for n in range(1, N + 1):
        model.add_constraint(SOC[n, 0] == 1.0, ctname=f"Initial_SOC_Bat_{n}")

    for t in time_steps:
        model.add_constraint(
            model.sum(P_G[k, t] for k in range(1, K + 1))
            + model.sum(P_dc[n, t] - P_c[n, t] for n in range(1, N + 1))
            + P_shore[t]
            == P_pl[t] + P_ser,
            ctname=f"PowerBalance_t{t}",
        )

        if t not in VOYAGE_STAGES['T_ber']:
            model.add_constraint(P_shore[t] == 0, ctname=f"NoshorePower_t{t}")

        for k in range(1, K + 1):
            model.add_constraint(
                P_G[k, t] >= GEN_PARAMS[k]["Pmin"] * u[k, t],
                ctname=f"GenMin_{k}_t{t}",
            )
            model.add_constraint(
                P_G[k, t] <= GEN_PARAMS[k]["Pmax"] * u[k, t],
                ctname=f"GenMax_{k}_t{t}",
            )

        for n in range(1, N + 1):
            model.add_constraint(
                P_c[n, t] <= ESS_PARAMS["P_c_max"],
                ctname=f"Bat_ChargeMax_{n}_t{t}",
            )
            model.add_constraint(
                P_dc[n, t] <= ESS_PARAMS["P_dc_max"],
                ctname=f"Bat_DischargeMax_{n}_t{t}",
            )
            model.add_constraint(
                SOC[n, t]
                == SOC[n, t - 1]
                + (ESS_PARAMS["eff_c"] * P_c[n, t] * dt) / ESS_PARAMS["Cap"]
                - (P_dc[n, t] * dt) / (ESS_PARAMS["eff_dc"] * ESS_PARAMS["Cap"]),
                ctname=f"SOC_update_{n}_t{t}",
            )

        speed_lb, speed_ub = _get_speed_bounds(t)
        if speed_lb == speed_ub:
            model.add_constraint(V[t] == speed_lb, ctname=f"SpeedFix_t{t}")
        else:
            model.add_constraint(V[t] >= speed_lb, ctname=f"SpeedLB_t{t}")
            model.add_constraint(V[t] <= speed_ub, ctname=f"SpeedUB_t{t}")

        v_min, v_max = 0.0, 12.0
        x_points = find_optimal_segments(v_min, v_max, segment_count, t, VOYAGE_STAGES)
        g_points = [calculate_true_nonlinear_power(x, t, VOYAGE_STAGES) for x in x_points]

        model.add_constraint(
            V[t]
            == model.sum(
                w_var[t, m] * x_points[m - 1] for m in range(1, segment_count + 2)
            ),
            ctname=f"SpeedPiecewise_t{t}",
        )
        model.add_constraint(
            P_pl[t]
            == model.sum(
                w_var[t, m] * g_points[m - 1] for m in range(1, segment_count + 2)
            ),
            ctname=f"PowerPiecewise_t{t}",
        )
        model.add_constraint(
            model.sum(w_var[t, m] for m in range(1, segment_count + 2)) == 1,
            ctname=f"WeightSum_t{t}",
        )
        model.add_constraint(
            model.sum(l_var[t, m] for m in range(1, segment_count + 1)) == 1,
            ctname=f"SelectorSum_t{t}",
        )
        model.add_constraint(w_var[t, 1] <= l_var[t, 1], ctname=f"SOS2_left_t{t}")
        model.add_constraint(
            w_var[t, segment_count + 1] <= l_var[t, segment_count],
            ctname=f"SOS2_right_t{t}",
        )
        for m in range(2, segment_count + 1):
            model.add_constraint(
                w_var[t, m] <= l_var[t, m - 1] + l_var[t, m],
                ctname=f"SOS2_mid_t{t}_{m}",
            )

    dist_AB = model.sum(V[t] * dt for t in VOYAGE_STAGES["T_doc"] + VOYAGE_STAGES["T_cru"])
    model.add_constraint(dist_AB >= DISTANCES["D_AB"] * 0.99, ctname="DistAB_LB")
    model.add_constraint(dist_AB <= DISTANCES["D_AB"] * 1.01, ctname="DistAB_UB")

    total_active_stages = (
        VOYAGE_STAGES["T_doc"]
        + VOYAGE_STAGES["T_cru"]
        + VOYAGE_STAGES["T_dra"]
        + VOYAGE_STAGES["T_dep"]
    )
    dist_AC = model.sum(V[t] * dt for t in total_active_stages)
    model.add_constraint(dist_AC >= DISTANCES["D_AC"] * 0.99, ctname="DistAC_LB")
    model.add_constraint(dist_AC <= DISTANCES["D_AC"] * 1.01, ctname="DistAC_UB")

    try:
        solution = model.solve(log_output=True)
    except Exception as exc:  # pragma: no cover - depends on local solver
        print(f"Optimization failed while solving the docplex model: {exc}")
        return model, P_G, P_c, P_dc, V

    print(f"Optimization Status: {_solve_status_text(model)}")
    if solution is not None:
        print(f"Total Objective Cost: ${solution.objective_value:.2f}")
    else:
        print("Warning: docplex did not return a feasible solution.")

    return model, P_G, P_c, P_dc, V


if __name__ == "__main__":
    solve_tugboat_scheduling()
