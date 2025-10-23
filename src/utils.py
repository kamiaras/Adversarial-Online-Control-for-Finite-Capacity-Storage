"""
utils.py
--------

General-purpose utility functions for the Adversarial Storage Control project.

This module is fully function-agnostic:
- No hardcoded arrival or cost registries.
- Works seamlessly with any external callable cost or arrival function.
- Provides reusable helpers for array operations, normalization,
  zero-padding, and simplex projections.
"""

from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  Basic helpers
# ======================================================================

def zero_padded(a, t):
    """
    Returns a_t with zero padding for t <= 0.

    Parameters
    ----------
    a : np.ndarray
        Arrival sequence of length T.
    t : int
        Time index (1-indexed).
    """
    return a[t - 1] if t > 0 else 0.0


def normalize(v, eps=1e-12):
    """
    Normalize vector to sum to 1 (simplex projection).

    Parameters
    ----------
    v : np.ndarray
        Input vector.
    eps : float
        Small offset to avoid division by zero.

    Returns
    -------
    np.ndarray
        Normalized vector with nonnegative entries summing to 1.
    """
    v = np.maximum(v, 0)
    s = np.sum(v)
    if s < eps:
        return np.ones_like(v) / len(v)
    return v / s


def project_to_simplex(v):
    """
    Euclidean projection onto the probability simplex:
        minimize ||x - v||₂²  subject to  x >= 0,  sum(x) = 1.

    Implementation based on:
        Wang & Carreira-Perpiñán, "Projection onto the probability simplex:
        An efficient algorithm with a simple proof," arXiv:1309.1541.

    Parameters
    ----------
    v : np.ndarray
        Input vector.

    Returns
    -------
    np.ndarray
        Projected vector on the simplex.
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0)
    return w


# ======================================================================
#  Matrix construction utilities (Φ, Ψ)
# ======================================================================

def compute_phi(a, T, H, kappa):
    """
    Compute matrix Φ with entries:
        φ_t^(j) = κ ∑_{i=1}^{j-1} (1-κ)^{i-1} a_{t-i}
                  + (1-κ)^{j-1} a_{t-j}
    using zero padding for a_t when t <= 0.

    Returns
    -------
    np.ndarray of shape (T, H)
    """
    Phi = np.zeros((T, H))
    for t in range(1, T + 1):
        for j in range(1, H + 1):
            term1 = kappa * sum((1 - kappa) ** (i - 1) * zero_padded(a, t - i)
                                for i in range(1, j))
            term2 = (1 - kappa) ** (j - 1) * zero_padded(a, t - j)
            Phi[t - 1, j - 1] = term1 + term2
    return Phi


def compute_psi(a, T, H, kappa):
    """
    Compute matrix Ψ with entries:
        ψ_t^(j) = ∑_{i=1}^j (1-κ)^{i-1} a_{t-i}
    using zero padding for a_t when t <= 0.

    Returns
    -------
    np.ndarray of shape (T, H)
    """
    Psi = np.zeros((T, H))
    for t in range(1, T + 1):
        for j in range(1, H + 1):
            Psi[t - 1, j - 1] = sum((1 - kappa) ** (i - 1) * zero_padded(a, t - i)
                                    for i in range(1, j + 1))
    return Psi


# ======================================================================
#  Simulation utilities
# ======================================================================

def compute_sequences(w, Phi, Psi):
    """
    Compute u_t(w) and b_t(w) sequences for t = 1..T.

    Parameters
    ----------
    w : np.ndarray
        Weight vector (length H).
    Phi, Psi : np.ndarray
        Feature matrices.

    Returns
    -------
    (u_seq, b_seq)
    """
    T = Phi.shape[0]
    u_seq = np.array([np.dot(w, Phi[t, :]) for t in range(T)])
    b_seq = np.array([np.dot(w, Psi[t, :]) for t in range(T)])
    return u_seq, b_seq


def simulate_storage_dynamics(Phi, Psi, a, w, cost_fn, cost_kwargs=None):
    """
    Compute full trajectory for a given w and cost function.

    Parameters
    ----------
    Phi, Psi : np.ndarray
        Feature matrices (T × H).
    a : np.ndarray
        Arrival sequence.
    w : np.ndarray
        Control vector (probability simplex).
    cost_fn : callable
        c_t(b, u, t, **kwargs)
    cost_kwargs : dict, optional
        Additional keyword arguments for cost_fn.

    Returns
    -------
    dict with keys {b, u, c}
    """
    if cost_kwargs is None:
        cost_kwargs = {}

    u_seq = Phi @ w
    b_seq = Psi @ w
    T = len(a)
    c_seq = np.array([cost_fn(b_seq[t], u_seq[t], t + 1, **cost_kwargs) for t in range(T)])
    return {"b": b_seq, "u": u_seq, "c": c_seq}


# ======================================================================
#  Plotting helpers
# ======================================================================

def plot_sequence(t, seq, label, color="tab:blue", xlabel="t", ylabel="value", title=None):
    """
    Quick line plot for a single sequence.
    """
    plt.figure(figsize=(8, 3))
    plt.plot(t, seq, label=label, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()


def compare_controllers(
    *controllers,
    a=None,
    plot_start=0,
    plot_end=None,
    plot_arrival_separately=False,
    color_scheme=None,
):
    """
    Compare two or more controllers' b_t, u_t, c_t, and optionally arrivals.

    Parameters
    ----------
    controllers : (sim, name) pairs, variadic
        Ordered pairs of simulation output dictionaries and display names.
    a : np.ndarray, optional
        Arrival sequence.
    plot_start : int, optional
        Zero-based start index of the visualization window (default: 0).
    plot_end : int, optional
        Exclusive end index for visualization (default=min(500, T)).
    plot_arrival_separately : bool, optional
        If True and `a` is provided, arrivals are plotted on a separate figure.
    color_scheme : dict[str, str], optional
        Custom colors keyed by controller name.
    """
    if len(controllers) < 4 or len(controllers) % 2 != 0:
        raise ValueError(
            "Provide controller data as (sim, name) pairs. "
            "Example: compare_controllers(sim1, 'A', sim2, 'B', sim3, 'C')."
        )

    controller_data = []
    for idx in range(0, len(controllers), 2):
        sim = controllers[idx]
        name = controllers[idx + 1]
        if sim is None:
            raise ValueError(f"Simulation data for controller '{name}' is None.")
        if not isinstance(sim, dict):
            raise TypeError(f"Simulation for '{name}' must be a dict of arrays.")
        for required_key in ("b", "u", "c"):
            if required_key not in sim:
                raise KeyError(f"Simulation for '{name}' missing key '{required_key}'.")
        controller_data.append((name, sim))

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

    if color_scheme is None:
        color_scheme = {}
    cmap = plt.cm.get_cmap("tab10")
    for idx, (name, _) in enumerate(controller_data):
        if name not in color_scheme:
            color_scheme[name] = cmap(idx % cmap.N)
    a_color = color_scheme.get("a_t", "tab:gray")

    line_styles = ["-", "--", "-.", ":"]

    plt.figure(figsize=(9, 4))
    for idx, (name, sim) in enumerate(controller_data):
        plt.plot(
            t_axis,
            sim["b"][window_slice],
            label=f"{name} b_t",
            color=color_scheme[name],
            linewidth=1.8,
            linestyle=line_styles[idx % len(line_styles)],
        )
    plt.title("Backlog / Battery Level")
    plt.xlabel("t")
    plt.ylabel("b_t")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 4))
    if a is not None and not plot_arrival_separately:
        plt.plot(t_axis, a[window_slice], label="a_t", color=a_color, linewidth=1.2)
    for idx, (name, sim) in enumerate(controller_data):
        plt.plot(
            t_axis,
            sim["u"][window_slice],
            label=f"{name} u_t",
            color=color_scheme[name],
            linewidth=1.8,
            linestyle=line_styles[idx % len(line_styles)],
        )
    plt.title("Usage and Arrivals" if not plot_arrival_separately else "Usage")
    plt.xlabel("t")
    plt.ylabel("u_t" if plot_arrival_separately else "u_t / a_t")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if a is not None and plot_arrival_separately:
        plt.figure(figsize=(9, 3.6))
        plt.plot(t_axis, a[window_slice], label="a_t", color=a_color, linewidth=1.2)
        plt.title("Arrival Sequence")
        plt.xlabel("t")
        plt.ylabel("a_t")
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()

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
    plt.title("Per-step Cost Comparison")
    plt.xlabel("t")
    plt.ylabel("c_t")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    total_costs = {name: float(np.sum(sim["c"])) for name, sim in controller_data}
    print("\n──────────────────────────────")
    for name in total_costs:
        print(f"{name:<15} Total cost: {total_costs[name]:.4f}")
    print("──────────────────────────────")

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


# ======================================================================
#  Example standalone run
# ======================================================================

if __name__ == "__main__":
    print("✅ utils.py loaded successfully — running demo...")

    # Toy example
    T, H, kappa = 200, 10, 0.1
    rng = np.random.default_rng(0)
    a = rng.random(T)
    Phi = compute_phi(a, T, H, kappa)
    Psi = compute_psi(a, T, H, kappa)
    w = np.ones(H) / H

    def cost_fn(b, u, t): return b + (1 - u) ** 2

    sim = simulate_storage_dynamics(Phi, Psi, a, w, cost_fn)
    t_axis = np.arange(1, T + 1)
    plot_sequence(t_axis, sim["b"], "Battery Level", ylabel="b_t", title="Demo Battery Level")
    print("Demo complete.")
