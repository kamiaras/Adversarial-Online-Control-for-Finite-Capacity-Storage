"""
offline_opt.py
--------------

Offline convex optimization module for solving the static problem:

    w* = argmin_{w in Δ_H} ∑_t c_t(b_t(w), u_t(w))

This version is function-agnostic: accepts direct callables for
the arrival and cost functions.
"""

import numpy as np
import cvxpy as cp
from .features import compute_phi, compute_psi
from .utils import simulate_storage_dynamics
from .arrivals import ARRIVAL_FUNCTIONS
from .costs import COST_FUNCTIONS


def solve_optimal_w(
    a_fn=None,
    cost_fn=None,
    *,
    a_mode=None,
    cost_key=None,
    cost_kwargs=None,
    a_kwargs=None,
    T=1000,
    H=20,
    kappa=0.1,
    amplitude=1.0,
    verbose=False
):
    """
    Solve for the optimal w* minimizing ∑_t c_t(b_t(w), u_t(w)).

    Parameters
    ----------
    a_fn : callable, optional
        Arrival generator of the form a_fn(T, **a_kwargs) → np.ndarray of length T.
        Provide this or specify a_mode to look up a predefined generator.
    cost_fn : callable, optional
        Cost function of the form c_t(b, u, t, **cost_kwargs).
        Provide this or specify cost_key to look up a predefined cost.
    a_mode : str, optional
        Key into ARRIVAL_FUNCTIONS. Ignored if a_fn is provided.
    cost_key : str, optional
        Key into COST_FUNCTIONS. Ignored if cost_fn is provided.
    cost_kwargs : dict, optional
        Additional parameters for the cost function.
    a_kwargs : dict, optional
        Additional parameters for the arrival generator.
    T : int
        Horizon length.
    H : int
        Feature dimension.
    kappa : float
        Decay parameter (0 ≤ κ ≤ 1).
    amplitude : float
        Optional scaling for arrivals.
    verbose : bool
        Whether to show solver output.

    Returns
    -------
    dict with keys:
        {
            "a": a,
            "Phi": Phi,
            "Psi": Psi,
            "w_star": w_star,
            "objective": objective_value,
        }
    """
    import warnings
    if not verbose:
        warnings.filterwarnings("ignore", message="Objective contains too many subexpressions")

    if cost_kwargs is None:
        cost_kwargs = {}
    else:
        cost_kwargs = dict(cost_kwargs)
    if a_kwargs is None:
        a_kwargs = {}
    else:
        a_kwargs = dict(a_kwargs)

    if a_fn is not None and a_mode is not None:
        raise ValueError("Provide either 'a_fn' or 'a_mode', not both.")
    if cost_fn is not None and cost_key is not None:
        raise ValueError("Provide either 'cost_fn' or 'cost_key', not both.")

    if a_fn is None:
        if a_mode is None:
            raise ValueError("❌ Provide an arrival generator via 'a_fn' or choose a predefined one using 'a_mode'.")
        if a_mode not in ARRIVAL_FUNCTIONS:
            raise ValueError(f"❌ Unknown a_mode '{a_mode}'. Available: {list(ARRIVAL_FUNCTIONS.keys())}")
        a_fn = ARRIVAL_FUNCTIONS[a_mode]
    a_label = a_mode if a_mode is not None else getattr(a_fn, "__name__", "arrival_fn")

    if cost_fn is None:
        if cost_key is None:
            raise ValueError("❌ Provide a cost function via 'cost_fn' or choose a predefined one using 'cost_key'.")
        if cost_key not in COST_FUNCTIONS:
            raise ValueError(f"❌ Unknown cost_key '{cost_key}'. Available: {list(COST_FUNCTIONS.keys())}")
        cost_fn = COST_FUNCTIONS[cost_key]
    cost_label = cost_key if cost_key is not None else getattr(cost_fn, "__name__", "cost_fn")

    if not callable(a_fn):
        raise ValueError("❌ 'a_fn' must be a valid callable arrival generator.")
    if not callable(cost_fn):
        raise ValueError("❌ 'cost_fn' must be a valid callable cost function.")

    # --- 1. Generate arrivals ---
    try:
        a = a_fn(T, amplitude=amplitude, **a_kwargs)
    except TypeError:
        a = a_fn(T, **a_kwargs)
    if not isinstance(a, np.ndarray):
        raise ValueError("Arrival function must return a NumPy array.")

    # --- 2. Compute Φ and Ψ ---
    Phi = compute_phi(a, T, H, kappa)
    Psi = compute_psi(a, T, H, kappa)

    # --- 3. Solve convex optimization problem ---
    w = cp.Variable(H, nonneg=True)
    constraints = [cp.sum(w) == 1]
    terms = [cost_fn(Psi[t, :] @ w, Phi[t, :] @ w, t + 1, **cost_kwargs) for t in range(T)]
    total_cost = cp.sum(cp.hstack(terms))
    prob = cp.Problem(cp.Minimize(total_cost), constraints)
    prob.solve(verbose=verbose)

    # --- 4. Extract results ---
    w_star = np.array(w.value).flatten()
    print("\n──────────────────────────────")
    print(f"Optimal weight vector found for '{a_label}' | cost='{cost_label}'")
    print(f"Objective value: {prob.value:.4f}")
    print("w* (rounded):", np.round(w_star, 4))
    print(f"Sum(w*): {np.sum(w_star):.4f}")
    print("──────────────────────────────")

    return {
        "a": a,
        "Phi": Phi,
        "Psi": Psi,
        "w_star": w_star,
        "objective": prob.value,
        "a_fn": a_fn,
        "cost_fn": cost_fn,
        "a_mode": a_label if a_mode is not None else None,
        "cost_key": cost_label if cost_key is not None else None,
        "a_kwargs": a_kwargs,
        "cost_kwargs": cost_kwargs,
    }
