"""
simulate_storage_dynamics.py
----------------------------

Simulate storage system dynamics (battery / queue) for any given weight vector w.

This function computes:
    - u_t(w) = <w, φ_t>
    - b_t(w) = <w, ψ_t>
    - c_t(b_t, u_t) for a given cost function

It is fully function-agnostic and works with any provided cost_fn.
"""

import numpy as np
from .costs import COST_FUNCTIONS


def simulate_storage_dynamics(
    Phi,
    Psi,
    a,
    w,
    cost_fn=None,
    *,
    cost_key=None,
    cost_kwargs=None,
):
    """
    Simulate the system dynamics and per-step costs for a fixed weight vector w.

    Parameters
    ----------
    Phi : np.ndarray, shape (T, H)
        Feature matrix φ_t used for u_t(w) = <w, φ_t>.
    Psi : np.ndarray, shape (T, H)
        Feature matrix ψ_t used for b_t(w) = <w, ψ_t>.
    a : np.ndarray, shape (T,)
        Arrival sequence.
    w : np.ndarray, shape (H,)
        Fixed weight vector (typically optimal or learned).
    cost_fn : callable, optional
        Function of the form c_t(b, u, t, **cost_kwargs). Provide this or specify cost_key.
    cost_key : str, optional
        Key into COST_FUNCTIONS if cost_fn is not provided.
    cost_kwargs : dict, optional
        Additional keyword arguments passed to cost_fn.

    Returns
    -------
    dict with keys:
        {
            "b": np.ndarray,     # backlog or storage level sequence
            "u": np.ndarray,     # usage or control sequence
            "c": np.ndarray,     # per-step costs
            "total_cost": float  # total cost over horizon
        }
    """
    if cost_fn is not None and cost_key is not None:
        raise ValueError("Provide either 'cost_fn' or 'cost_key', not both.")
    if cost_fn is None:
        if cost_key is None:
            raise ValueError("❌ Provide a cost function via 'cost_fn' or choose a predefined one using 'cost_key'.")
        if cost_key not in COST_FUNCTIONS:
            raise ValueError(f"❌ Unknown cost_key '{cost_key}'. Available: {list(COST_FUNCTIONS.keys())}")
        cost_fn = COST_FUNCTIONS[cost_key]
    if not callable(cost_fn):
        raise ValueError("❌ 'cost_fn' must be a callable cost function of (b, u, t, **kwargs).")

    if cost_kwargs is None:
        cost_kwargs = {}

    # --- Dimensions ---
    T, H = Phi.shape
    if Psi.shape != (T, H):
        raise ValueError(f"Shape mismatch: Phi {Phi.shape}, Psi {Psi.shape}. Must match (T, H).")
    if len(w) != H:
        raise ValueError(f"Shape mismatch: w has length {len(w)}, expected {H}.")
    if len(a) != T:
        raise ValueError(f"Shape mismatch: arrivals 'a' must have length {T}.")

    # --- Compute trajectories ---
    b = Psi @ w
    u = Phi @ w
    c = np.array([cost_fn(b[t], u[t], t + 1, **cost_kwargs) for t in range(T)])
    total_cost = float(np.sum(c))

    return {"b": b, "u": u, "c": c, "total_cost": total_cost}


# Example direct usage (for quick testing)
if __name__ == "__main__":
    # Mock data
    T, H = 10, 3
    np.random.seed(0)
    a = np.random.rand(T)
    Phi = np.random.rand(T, H)
    Psi = np.random.rand(T, H)
    w = np.array([0.4, 0.3, 0.3])

    def c_example(b, u, t, s1=1.0, s2=0.5):
        return s1 * b + s2 * (1 - u) ** 2

    result = simulate_storage_dynamics(Phi, Psi, a, w, cost_fn=c_example)
    print("Total cost:", result["total_cost"])
    print("b:", np.round(result["b"], 3))
    print("u:", np.round(result["u"], 3))
