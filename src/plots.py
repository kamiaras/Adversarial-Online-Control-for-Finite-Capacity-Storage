"""
plots.py
--------

Reusable plotting utilities for visualizing and comparing controller behavior.

Functions:
    - compare_controllers: plot b_t, u_t, a_t, and c_t for two algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt


def compare_controllers(
    sim1,
    name1,
    sim2,
    name2,
    a=None,
    plot_len=None,
    color_scheme=None,
):
    """
    Compare two controller simulations visually.

    Parameters
    ----------
    sim1, sim2 : dict
        Output dictionaries containing keys:
            "b" : np.ndarray, backlog or battery sequence
            "u" : np.ndarray, control/usage sequence
            "c" : np.ndarray, per-step cost sequence
    name1, name2 : str
        Display names for the two controllers.
    a : np.ndarray, optional
        Arrival sequence to plot (same horizon length).
    plot_len : int, optional
        Number of time steps to plot (default: min(500, T)).
    color_scheme : dict[str, str], optional
        Custom color mapping, e.g. {"Offline Opt": "tab:red", "EGPC": "tab:blue"}.
    """
    # --- Basic validation ---
    if sim1 is None or sim2 is None:
        raise ValueError("Both sim1 and sim2 results must be provided.")
    if "b" not in sim1 or "b" not in sim2:
        raise ValueError("Both simulations must include key 'b' for backlog sequence.")
    if "u" not in sim1 or "u" not in sim2:
        raise ValueError("Both simulations must include key 'u' for control sequence.")
    if "c" not in sim1 or "c" not in sim2:
        raise ValueError("Both simulations must include key 'c' for per-step cost sequence.")

    b1, b2 = sim1["b"], sim2["b"]
    u1, u2 = sim1["u"], sim2["u"]
    c1, c2 = sim1["c"], sim2["c"]
    T = min(len(b1), len(b2))

    if plot_len is None:
        plot_len = min(500, T)
    t_axis = np.arange(1, plot_len + 1)

    if color_scheme is None:
        color_scheme = {
            name1: "tab:red",
            name2: "tab:blue",
            "a_t": "tab:gray",
        }

    # -------------------- (1) Plot b_t comparison --------------------
    plt.figure(figsize=(9, 4))
    plt.plot(t_axis, b1[:plot_len], label=f"{name1} $b_t$", color=color_scheme[name1], linewidth=1.8)
    plt.plot(t_axis, b2[:plot_len], label=f"{name2} $b_t$", color=color_scheme[name2], linestyle="--", linewidth=1.8)
    plt.title(r"Backlog / Battery Level $b_t$")
    plt.xlabel("Time $t$")
    plt.ylabel(r"$b_t$")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------- (2) Plot u_t and arrivals --------------------
    plt.figure(figsize=(9, 4))
    if a is not None:
        plt.plot(t_axis, a[:plot_len], label=r"$a_t$", color=color_scheme["a_t"], linewidth=1.2)
    plt.plot(t_axis, u1[:plot_len], label=f"{name1} $u_t$", color=color_scheme[name1], linewidth=1.8)
    plt.plot(t_axis, u2[:plot_len], label=f"{name2} $u_t$", color=color_scheme[name2], linestyle="--", linewidth=1.8)
    plt.title(r"Control Action $u_t$ and Arrivals $a_t$")
    plt.xlabel("Time $t$")
    plt.ylabel(r"$u_t$ / $a_t$")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------- (3) Plot cost comparison --------------------
    plt.figure(figsize=(9, 4))
    plt.plot(t_axis, c1[:plot_len], label=f"{name1} cost", color=color_scheme[name1], linewidth=1.8)
    plt.plot(t_axis, c2[:plot_len], label=f"{name2} cost", color=color_scheme[name2], linestyle="--", linewidth=1.8)
    plt.title(r"Per-step Cost Comparison")
    plt.xlabel("Time $t$")
    plt.ylabel(r"$c_t(b_t,u_t)$")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------- (4) Print totals --------------------
    total1 = np.sum(c1)
    total2 = np.sum(c2)
    print("\n──────────────────────────────")
    print(f"{name1:<15} Total cost: {total1:.4f}")
    print(f"{name2:<15} Total cost: {total2:.4f}")
    print("──────────────────────────────")

    return {
        "total_cost_1": total1,
        "total_cost_2": total2,
        "b_diff": np.mean(np.abs(b1[:plot_len] - b2[:plot_len])),
        "u_diff": np.mean(np.abs(u1[:plot_len] - u2[:plot_len])),
        "cost_diff": np.mean(np.abs(c1[:plot_len] - c2[:plot_len])),
    }


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

    compare_controllers(sim1, "Offline Opt", sim2, "EGPC", a=a)
