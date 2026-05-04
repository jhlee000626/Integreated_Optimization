import matplotlib.pyplot as plt
import numpy as np

from problem import solve_tugboat_scheduling
from spec import K, N, time_steps


def _solve_status_text(model):
    solve_details = getattr(model, "solve_details", None)
    if solve_details is not None:
        status = getattr(solve_details, "status", None)
        if status:
            return str(status)
    return "unknown"


def _solution_value(var):
    try:
        return var.solution_value
    except Exception:
        return 0.0


def plot_results():
    print("Starting optimization. Please wait...")
    try:
        prob, P_G, P_c, P_dc, V = solve_tugboat_scheduling()
    except ModuleNotFoundError as exc:
        print(f"\n[Error] {exc}")
        return

    if prob.solution is None:
        print(f"\n[Error] No feasible solution was returned. (status: {_solve_status_text(prob)})")
        print("Check the model constraints and the docplex/CPLEX solver setup.")
        return

    print("\nOptimization finished. Rendering the figure...")

    gen_outputs = {
        k: [_solution_value(P_G[k, t]) for t in time_steps]
        for k in range(1, K + 1)
    }
    batt_discharge = [
        sum(_solution_value(P_dc[n, t]) for n in range(1, N + 1))
        for t in time_steps
    ]
    batt_charge = [
        -sum(_solution_value(P_c[n, t]) for n in range(1, N + 1))
        for t in time_steps
    ]
    speeds = [_solution_value(V[t]) for t in time_steps]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    width = 0.6
    bottom_pos = np.zeros(len(time_steps))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for k in range(1, K + 1):
        ax1.bar(time_steps, gen_outputs[k], width, bottom=bottom_pos, label=f"G{k}", color=colors[k - 1])
        bottom_pos += np.array(gen_outputs[k])

    ax1.bar(time_steps, batt_discharge, width, bottom=bottom_pos, label="Discharge", color="#d62728")
    ax1.bar(time_steps, batt_charge, width, label="Charge", color="#9467bd")

    ax1.set_xlabel("Time Period (15 min)", fontsize=12)
    ax1.set_ylabel("Generators and ESS Outputs (MW)", fontsize=12)
    ax1.set_xticks(time_steps)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    ax2 = ax1.twinx()
    ax2.plot(
        time_steps,
        speeds,
        color="blue",
        marker=">",
        markersize=8,
        linestyle="--",
        linewidth=2,
        label="Optimized speed",
    )
    ax2.set_ylabel("Speed (kn)", color="blue", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="blue")
    ax2.set_ylim(-1, 12)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=6)

    plt.title("Optimization of power generation and navigation speed scheduling", y=1.15)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_results()
