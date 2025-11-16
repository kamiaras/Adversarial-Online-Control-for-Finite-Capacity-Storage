"""
plots.py
--------

Reusable plotting utilities for visualizing and comparing controller behavior.

Functions:
    - compare_controllers: plot b_t, u_t, a_t, and c_t for multiple algorithms.
"""

from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt

from src._controller_compare import normalize_controller_payload


def compare_controllers(
    *controllers,
    a=None,
    plot_start=0,
    plot_end=None,
    plot_arrival_separately=False,
    color_scheme=None,
):
    """
    Compare two or more controller simulations visually.

    Parameters
    ----------
    controllers : inputs describing controller simulations
        Accepts the classic ``(sim, name)`` positional pairs as well as richer
        payloads such as::

            compare_controllers(
                {"Offline Opt": sim1, "EGPC": sim2},
                a=a,
            )

            compare_controllers(
                [
                    (sim1, "Offline Opt"),
                    {"name": "EGPC", "sim": sim2},
                    ("Uniform", sim3),
                ],
                a=a,
            )

        Every simulation dictionary must contain the keys:
            "b" : np.ndarray, backlog or battery sequence
            "u" : np.ndarray, control/usage sequence
            "c" : np.ndarray, per-step cost sequence
    a : np.ndarray, optional
        Arrival sequence to plot (same horizon length).
    plot_start : int, optional
        Zero-based start index of the time window to visualize (default: 0).
    plot_end : int, optional
        Exclusive end index of the visualization window (default: min(500, T)).
    plot_arrival_separately : bool, optional (keyword-only)
        If True and `a` is provided, draw arrivals on their own figure instead of
        overlaying them on the control plot.
    color_scheme : dict[str, str], optional (keyword-only)
        Custom color mapping, e.g. {"Offline Opt": "tab:red", "EGPC": "tab:blue"}.
    """
    controller_data = normalize_controller_payload(controllers)

    # Determine plotting horizon shared across controllers.
    min_horizon = min(len(sim["b"]) for _, sim in controller_data)
    if plot_end is None:
        plot_end = min(500, min_horizon)
    else:
        plot_end = min(plot_end, min_horizon)

    plot_start = max(int(plot_start), 0)
    if plot_start >= plot_end:
        raise ValueError("plot_start must be smaller than plot_end.")

    if a is not None and len(a) < plot_end:
        raise ValueError(f"Arrival sequence length {len(a)} shorter than requested plot_end {plot_end}.")

    window_slice = slice(plot_start, plot_end)
    t_axis = np.arange(plot_start + 1, plot_end + 1)

    # Build or extend the color scheme.
    if color_scheme is None:
        color_scheme = {}
    cmap = plt.cm.get_cmap("tab10")
    for idx, (name, _) in enumerate(controller_data):
        if name not in color_scheme:
            color_scheme[name] = cmap(idx % cmap.N)
    a_color = color_scheme.get("a_t", "tab:gray")

    line_styles = ["-", "--", "-.", ":"]

    # -------------------- (1) Plot b_t comparison --------------------
    plt.figure(figsize=(9, 4))
    for idx, (name, sim) in enumerate(controller_data):
        plt.plot(
            t_axis,
            sim["b"][window_slice],
            label=f"{name}",
            color=color_scheme[name],
            linewidth=1.8,
            linestyle=line_styles[idx % len(line_styles)],
        )
    plt.title(r"Battery Level $b_t$")
    plt.xlabel("Time $t$")
    plt.ylabel(r"$b_t$")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------- (2) Plot u_t and arrivals --------------------
    plt.figure(figsize=(9, 4))
    if a is not None and not plot_arrival_separately:
        plt.plot(t_axis, a[window_slice], label=r"$a_t$", color=a_color, linewidth=1.2)
    for idx, (name, sim) in enumerate(controller_data):
        plt.plot(
            t_axis,
            sim["u"][window_slice],
            label=f"{name} ",
            color=color_scheme[name],
            linewidth=1.8,
            linestyle=line_styles[idx % len(line_styles)],
        )
    if plot_arrival_separately:
        plt.title(r"Usage $u_t$")
        plt.ylabel(r"$u_t$")
    else:
        plt.ylabel(r"$u_t$ , $a_t$")
        plt.title(r"Usage $u_t$ and Arrivals $a_t$")
    plt.xlabel("Time $t$")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if a is not None and plot_arrival_separately:
        plt.figure(figsize=(9, 3.6))
        plt.plot(t_axis, a[window_slice], label=r"$a_t$", color=a_color, linewidth=1.2)
        plt.title(r"Arrival Sequence $a_t$")
        plt.xlabel("Time $t$")
        plt.ylabel(r"$a_t$")
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # -------------------- (3) Plot cost comparison --------------------
    plt.figure(figsize=(9, 4))
    for idx, (name, sim) in enumerate(controller_data):
        plt.plot(
            t_axis,
            sim["c"][window_slice],
            label=f"{name} cost",
            color=color_scheme[name],
            linewidth=1.8,
            linestyle=line_styles[idx % len(line_styles)],
        )
    plt.title(r"Per-step Cost Comparison")
    plt.xlabel("Time $t$")
    plt.ylabel(r"$c_t(b_t,u_t)$")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------- (4) Print totals --------------------
    total_costs = {name: float(np.sum(sim["c"])) for name, sim in controller_data}
    print("\n──────────────────────────────")
    for name in total_costs:
        print(f"{name:<15} Total cost: {total_costs[name]:.4f}")
    print("──────────────────────────────")

    # Prepare return payload with backwards-compatible keys.
    comparison_summary = {"total_costs": total_costs}
    if len(controller_data) >= 2:
        pairwise_abs_diff = {}
        for (name_i, sim_i), (name_j, sim_j) in combinations(controller_data, 2):
            key = f"{name_i} vs {name_j}"
            pairwise_abs_diff[key] = {
                "b_diff": float(np.mean(np.abs(sim_i["b"][window_slice] - sim_j["b"][window_slice]))),
                "u_diff": float(np.mean(np.abs(sim_i["u"][window_slice] - sim_j["u"][window_slice]))),
                "cost_diff": float(np.mean(np.abs(sim_i["c"][window_slice] - sim_j["c"][window_slice]))),
            }
        comparison_summary["pairwise_abs_diff"] = pairwise_abs_diff

        # Maintain legacy return keys based on the first pair.
        first_two_names = [controller_data[0][0], controller_data[1][0]]
        first_pair_key = f"{first_two_names[0]} vs {first_two_names[1]}"
        comparison_summary["total_cost_1"] = total_costs[first_two_names[0]]
        comparison_summary["total_cost_2"] = total_costs[first_two_names[1]]
        if first_pair_key in pairwise_abs_diff:
            comparison_summary["b_diff"] = pairwise_abs_diff[first_pair_key]["b_diff"]
            comparison_summary["u_diff"] = pairwise_abs_diff[first_pair_key]["u_diff"]
            comparison_summary["cost_diff"] = pairwise_abs_diff[first_pair_key]["cost_diff"]
    comparison_summary["plot_window"] = {"start": plot_start, "end": plot_end}

    return comparison_summary


# Example usage
if __name__ == "__main__":
    # Mock data
    T = 300
    t = np.arange(T)
    a = 0.5 * (1 + np.sin(2 * np.pi * t / 50))
    sim1 = {
        "b": np.sin(0.1 * t) + 1,
        "u": np.cos(0.1 * t) + 1,
        "c": np.sin(0.1 * t) ** 2,
    }
    sim2 = {
        "b": np.sin(0.1 * t + 0.5) + 1,
        "u": np.cos(0.1 * t + 0.3) + 1,
        "c": np.sin(0.1 * t + 0.2) ** 2,
    }
    sim3 = {
        "b": np.sin(0.1 * t + 1.0) + 1,
        "u": np.cos(0.1 * t + 0.6) + 1,
        "c": np.sin(0.1 * t + 0.4) ** 2,
    }

    compare_controllers(sim1, "Offline Opt", sim2, "EGPC", sim3, "Uniform", a=a)
