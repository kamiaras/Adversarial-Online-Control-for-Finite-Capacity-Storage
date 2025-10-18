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


def compare_controllers(sim1, name1, sim2, name2, a=None, plot_len=None):
    """
    Compare two controllers' b_t, u_t, c_t, and optionally arrivals.

    Parameters
    ----------
    sim1, sim2 : dict
        Must include keys 'b', 'u', 'c'.
    name1, name2 : str
        Labels for controllers.
    a : np.ndarray, optional
        Arrival sequence.
    plot_len : int, optional
        Number of points to visualize (default=min(500, T)).
    """
    b1, b2 = sim1["b"], sim2["b"]
    u1, u2 = sim1["u"], sim2["u"]
    c1, c2 = sim1["c"], sim2["c"]
    T = min(len(b1), len(b2))
    if plot_len is None:
        plot_len = min(500, T)
    t_axis = np.arange(1, plot_len + 1)

    plt.figure(figsize=(9, 4))
    plt.plot(t_axis, b1[:plot_len], label=f"{name1} b_t", color="tab:red", linewidth=1.8)
    plt.plot(t_axis, b2[:plot_len], label=f"{name2} b_t", color="tab:blue", linestyle="--", linewidth=1.8)
    plt.title("Backlog / Battery Level")
    plt.xlabel("t")
    plt.ylabel("b_t")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 4))
    if a is not None:
        plt.plot(t_axis, a[:plot_len], label="a_t", color="tab:gray", linewidth=1.2)
    plt.plot(t_axis, u1[:plot_len], label=f"{name1} u_t", color="tab:red", linewidth=1.8)
    plt.plot(t_axis, u2[:plot_len], label=f"{name2} u_t", color="tab:blue", linestyle="--", linewidth=1.8)
    plt.title("Control Action and Arrivals")
    plt.xlabel("t")
    plt.ylabel("u_t / a_t")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.plot(t_axis, c1[:plot_len], label=f"{name1} cost", color="tab:red", linewidth=1.8)
    plt.plot(t_axis, c2[:plot_len], label=f"{name2} cost", color="tab:blue", linestyle="--", linewidth=1.8)
    plt.title("Per-step Cost Comparison")
    plt.xlabel("t")
    plt.ylabel("c_t")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    total1, total2 = np.sum(c1), np.sum(c2)
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
