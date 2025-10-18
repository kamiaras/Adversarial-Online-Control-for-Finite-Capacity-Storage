"""
egpc.py
-------

Implements the Exponentiated-Gradient Perturbation Controller (EGPC)
for online storage control.

This controller updates a probability weight vector w_t ∈ Δ_H each round
based on observed arrivals and costs, using exponentiated-gradient updates.

It is fully function-agnostic:
    - Receives callable arrival and cost functions
    - Uses externally provided feature matrices (Φ_t, Ψ_t)
"""

import numpy as np
from .features import compute_phi, compute_psi
from .arrivals import ARRIVAL_FUNCTIONS
from .costs import COST_FUNCTIONS


def run_egpc(
    a_fn=None,
    cost_fn=None,
    *,
    a_mode=None,
    cost_key=None,
    eta=0.05,
    T=1000,
    H=20,
    kappa=0.1,
    amplitude=1.0,
    a_kwargs=None,
    cost_kwargs=None,
    verbose=False,
):
    """
    Run the Exponentiated-Gradient Perturbation Controller (EGPC).

    Parameters
    ----------
    a_fn : callable, optional
        Arrival generator of the form a_fn(T, **a_kwargs) → np.ndarray of length T.
        Provide this or specify a_mode to use a predefined generator.
    cost_fn : callable, optional
        Cost function of the form c_t(b, u, t, **cost_kwargs).
        Provide this or specify cost_key to use a predefined cost.
    a_mode : str, optional
        Key into ARRIVAL_FUNCTIONS. Ignored if a_fn is provided.
    cost_key : str, optional
        Key into COST_FUNCTIONS. Ignored if cost_fn is provided.
    eta : float
        Learning rate (step size) for exponentiated gradient update.
    T : int
        Time horizon.
    H : int
        Number of experts / dimensions in w_t.
    kappa : float
        Decay parameter (0 ≤ κ ≤ 1).
    amplitude : float
        Scaling factor for arrivals.
    a_kwargs : dict, optional
        Extra parameters for arrival generator.
    cost_kwargs : dict, optional
        Extra parameters for cost function.
    verbose : bool
        If True, prints running progress and diagnostics.

    Returns
    -------
    dict with keys:
        {
            "a": np.ndarray,         # arrivals
            "w_t": np.ndarray,       # full weight trajectory (T+1, H)
            "b": np.ndarray,         # storage level sequence
            "u": np.ndarray,         # usage/control sequence
            "c": np.ndarray,         # per-step costs
            "total_cost": float,     # total accumulated cost
        }
    """

    if a_kwargs is None:
        a_kwargs = {}
    else:
        a_kwargs = dict(a_kwargs)
    if cost_kwargs is None:
        cost_kwargs = {}
    else:
        cost_kwargs = dict(cost_kwargs)

    if a_fn is not None and a_mode is not None:
        raise ValueError("Provide either 'a_fn' or 'a_mode', not both.")
    if cost_fn is not None and cost_key is not None:
        raise ValueError("Provide either 'cost_fn' or 'cost_key', not both.")

    if a_fn is None:
        if a_mode is None:
            raise ValueError("❌ Provide an arrival generator via 'a_fn' or choose one using 'a_mode'.")
        if a_mode not in ARRIVAL_FUNCTIONS:
            raise ValueError(f"❌ Unknown a_mode '{a_mode}'. Available: {list(ARRIVAL_FUNCTIONS.keys())}")
        a_fn = ARRIVAL_FUNCTIONS[a_mode]
    a_label = a_mode if a_mode is not None else getattr(a_fn, "__name__", "arrival_fn")

    if cost_fn is None:
        if cost_key is None:
            raise ValueError("❌ Provide a cost function via 'cost_fn' or choose one using 'cost_key'.")
        if cost_key not in COST_FUNCTIONS:
            raise ValueError(f"❌ Unknown cost_key '{cost_key}'. Available: {list(COST_FUNCTIONS.keys())}")
        cost_fn = COST_FUNCTIONS[cost_key]
    cost_label = cost_key if cost_key is not None else getattr(cost_fn, "__name__", "cost_fn")

    if not callable(a_fn):
        raise ValueError("❌ 'a_fn' must be a callable arrival generator.")
    if not callable(cost_fn):
        raise ValueError("❌ 'cost_fn' must be a callable cost function.")
    if eta <= 0:
        raise ValueError("❌ 'eta' (learning rate) must be positive.")

    # --- 1. Generate arrival sequence ---
    try:
        a = a_fn(T, amplitude=amplitude, **a_kwargs)
    except TypeError:
        a = a_fn(T, **a_kwargs)

    # --- 2. Precompute feature matrices ---
    Phi = compute_phi(a, T, H, kappa)
    Psi = compute_psi(a, T, H, kappa)

    # --- 3. Initialize state and weights ---
    w_t = np.ones(H) / H
    w_hist = np.zeros((T + 1, H))
    w_hist[0] = w_t

    b_t = np.zeros(T + 1)  # storage levels (b_0 = 0)
    u_t = np.zeros(T)
    c_t = np.zeros(T)

    # --- 4. Main loop ---
    for t in range(T):
        # Compute tentative and feasible usage
        a_vec = np.array([(1 - kappa) ** (i + 1) * a[t - i - 1] if t - i - 1 >= 0 else 0.0 for i in range(H)])
        u_hat = kappa * b_t[t] + np.dot(w_t, a_vec)
        u_t[t] = min(u_hat, b_t[t] + a[t])  # feasible use within available energy

        # Update storage level
        b_t[t + 1] = b_t[t] + a[t] - u_t[t]

        # Compute surrogate gradients (∂ℓ/∂w) ≈ [∂c_t/∂b * ψ_t + ∂c_t/∂u * φ_t]
        b_val = np.dot(w_t, Psi[t, :])
        u_val = np.dot(w_t, Phi[t, :])
        c_t[t] = cost_fn(b_val, u_val, t + 1, **cost_kwargs)

        # Simple numerical gradient estimate (finite differences)
        eps = 1e-5
        grad = np.zeros(H)
        for j in range(H):
            w_perturb = w_t.copy()
            w_perturb[j] += eps
            b_pert = np.dot(w_perturb, Psi[t, :])
            u_pert = np.dot(w_perturb, Phi[t, :])
            c_pert = cost_fn(b_pert, u_pert, t + 1, **cost_kwargs)
            grad[j] = (c_pert - c_t[t]) / eps

        # Exponentiated-Gradient update
        w_t = w_t * np.exp(-eta * grad)
        w_t = np.maximum(w_t, 1e-12)
        w_t /= np.sum(w_t)
        w_hist[t + 1] = w_t

        if verbose and (t % 100 == 0 or t == T - 1):
            print(f"[t={t:4d}] b_t={b_t[t]:.4f} u_t={u_t[t]:.4f} cost={c_t[t]:.4f}")

    total_cost = float(np.sum(c_t))

    if verbose:
        print("\n──────────────────────────────")
        print(f"EGPC completed for T={T}, H={H}, η={eta}")
        print(f"Arrival mode: {a_label}")
        print(f"Cost function: {cost_label}")
        print(f"Total cost: {total_cost:.4f}")
        print("Final w_T (rounded):", np.round(w_t, 4))
        print("──────────────────────────────")

    return {
        "a": a,
        "w_t": w_hist,
        "b": b_t[:-1],
        "u": u_t,
        "c": c_t,
        "total_cost": total_cost,
        "a_fn": a_fn,
        "cost_fn": cost_fn,
        "a_mode": a_label if a_mode is not None else None,
        "cost_key": cost_label if cost_key is not None else None,
        "a_kwargs": a_kwargs,
        "cost_kwargs": cost_kwargs,
    }


# Example direct usage (for quick testing)
if __name__ == "__main__":
    def arrivals_random(T, amplitude=1.0, seed=0):
        rng = np.random.default_rng(seed)
        return amplitude * rng.random(T)

    def c_weighted_combo(b, u, t, s1=1.0, s2=0.5, target=1.0):
        return s1 * b + s2 * (target - u) ** 2

    result = run_egpc(
        a_fn=arrivals_random,
        cost_fn=c_weighted_combo,
        eta=0.05,
        T=300,
        H=10,
        kappa=0.1,
        verbose=True,
    )

    print("\nTotal cost:", result["total_cost"])
    print("Final w_T:", np.round(result["w_t"][-1], 3))
