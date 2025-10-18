"""
experiments.py
---------------

Batch utilities for running optimization experiments over multiple
(arrival, cost) function pairs.

All functions are fully function-agnostic — they take callables directly
rather than relying on global registries or string lookups.

Dependencies:
    - NumPy
    - solve_optimal_w (from offline_opt.py)
    - simulate_storage_dynamics (from simulate_storage_dynamics.py)
"""

import numpy as np
from itertools import product
from pathlib import Path
from .offline_opt import solve_optimal_w
from .simulate_storage_dynamics import simulate_storage_dynamics


# ----------------------------------------------------------------------
# 1. Batch search for non-degenerate w*
# ----------------------------------------------------------------------

def run_all_combinations(
    arrivals,
    costs,
    T=1000,
    H=20,
    kappa=0.1,
    verbose=False,
    save_path="interesting_combinations.txt",
    tol=1e-6,
):
    """
    Run all (arrival, cost) combinations, identify non-degenerate w*.

    Parameters
    ----------
    arrivals : dict[str, callable]
        Dict of arrival generator functions: {"name": fn(T, **kwargs) -> np.ndarray}
    costs : dict[str, callable]
        Dict of cost functions: {"name": fn(b, u, t, **kwargs) -> float}
    T : int
        Horizon length.
    H : int
        Feature dimension.
    kappa : float
        Decay parameter.
    verbose : bool
        Print solver details.
    save_path : str
        Path to save the list of interesting (a_mode, cost_key).
    tol : float
        Numerical tolerance for detecting one-hot w*.

    Returns
    -------
    (dict, list)
        all_results, interesting_combos
    """

    all_results = {}
    interesting_combos = []

    total_runs = len(arrivals) * len(costs)
    print(f"🚀 Running {total_runs} experiments (|arrivals|={len(arrivals)}, |costs|={len(costs)})\n")

    for i, (a_name, c_name) in enumerate(product(arrivals.keys(), costs.keys()), start=1):
        print(f"─── ({i}/{total_runs}) Running a_fn='{a_name}' | cost_fn='{c_name}' ───")
        a_fn = arrivals[a_name]
        cost_fn = costs[c_name]
        try:
            result = solve_optimal_w(
                a_fn=a_fn,
                cost_fn=cost_fn,
                T=T,
                H=H,
                kappa=kappa,
                verbose=verbose,
            )

            w_star = np.array(result["w_star"]).flatten()
            is_onehot = (
                np.allclose(w_star, np.eye(H)[0], atol=tol)
                or np.allclose(w_star, np.eye(H)[-1], atol=tol)
            )

            if not is_onehot:
                interesting_combos.append((a_name, c_name))
                all_results[(a_name, c_name)] = result
                print(f"✅ Non-degenerate w*: saved ({a_name}, {c_name})")
            else:
                print(f"⚪ Degenerate w*: skipped ({a_name}, {c_name})")

        except Exception as e:
            print(f"❌ Error on ({a_name}, {c_name}): {e}\n")

    # Save interesting pairs
    if interesting_combos:
        path = Path(save_path)
        with open(path, "w") as f:
            f.write("# Non-degenerate combinations (w* not one-hot)\n")
            for a_name, c_name in interesting_combos:
                f.write(f"{a_name},{c_name}\n")
        print(f"\n💾 Saved {len(interesting_combos)} interesting combos → {path}")
    else:
        print("\n⚠️ No non-degenerate w* combinations found.")

    print("🏁 Finished all combinations.\n")
    return all_results, interesting_combos


# ----------------------------------------------------------------------
# 2. Rerun saved combinations
# ----------------------------------------------------------------------

def run_from_saved_combos(
    arrivals,
    costs,
    file_path="interesting_combinations.txt",
    T=1000,
    H=20,
    kappa=0.1,
    verbose=False,
):
    """
    Re-run experiments using (arrival, cost) pairs saved in a text file.

    Parameters
    ----------
    arrivals : dict[str, callable]
        Dictionary of arrival functions.
    costs : dict[str, callable]
        Dictionary of cost functions.
    file_path : str
        Path to file with saved combinations (comma-separated).
    T : int
        Horizon length.
    H : int
        Feature dimension.
    kappa : float
        Decay parameter.
    verbose : bool
        If True, print solver progress.

    Returns
    -------
    dict[(str, str), dict]
        Mapping of (a_name, c_name) to result dictionary.
    """

    combos = []
    path = Path(file_path)
    if not path.exists():
        print(f"⚠️ File not found: {file_path}")
        return {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) == 2:
                combos.append((parts[0].strip(), parts[1].strip()))

    if not combos:
        print(f"⚠️ No valid combinations found in {file_path}.")
        return {}

    print(f"📂 Loaded {len(combos)} combinations from {file_path}.\n")

    results = {}
    for i, (a_name, c_name) in enumerate(combos, start=1):
        print(f"─── ({i}/{len(combos)}) Running a_fn='{a_name}' | cost_fn='{c_name}' ───")
        if a_name not in arrivals or c_name not in costs:
            print(f"⚠️ Missing function for ({a_name}, {c_name}); skipping.")
            continue
        a_fn = arrivals[a_name]
        cost_fn = costs[c_name]
        try:
            result = solve_optimal_w(
                a_fn=a_fn,
                cost_fn=cost_fn,
                T=T,
                H=H,
                kappa=kappa,
                verbose=verbose,
            )
            results[(a_name, c_name)] = result
        except Exception as e:
            print(f"❌ Error for ({a_name}, {c_name}): {e}\n")

    print("🏁 Finished re-running saved combinations.\n")
    return results


# ----------------------------------------------------------------------
# 3. Example usage
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Example: Define a few toy arrivals and costs for demonstration
    def arrivals_random(T, amplitude=1.0, seed=42):
        rng = np.random.default_rng(seed)
        return amplitude * rng.random(T)

    def arrivals_bursty(T, amplitude=1.0, seed=123):
        rng = np.random.default_rng(seed)
        a = np.zeros(T)
        t = 0
        while t < T:
            burst_len = rng.integers(20, 40)
            a[t:t + burst_len] = rng.uniform(0.7, 1.0)
            t += burst_len + rng.integers(30, 60)
        return np.clip(a, 0.0, 1.0) * amplitude

    def c_linear(b, u, t):
        return b + (1 - u) ** 2

    def c_weighted(b, u, t, s1=1.0, s2=0.5):
        return s1 * b + s2 * (1 - u) ** 2

    ARRIVALS = {
        "random": arrivals_random,
        "bursty": arrivals_bursty,
    }
    COSTS = {
        "linear": c_linear,
        "weighted": c_weighted,
    }

    # Run full experiment
    all_results, combos = run_all_combinations(
        ARRIVALS,
        COSTS,
        T=300,
        H=10,
        kappa=0.1,
        verbose=False,
    )

    # Re-run interesting combinations
    subset = run_from_saved_combos(
        ARRIVALS,
        COSTS,
        file_path="interesting_combinations.txt",
        T=300,
        H=10,
        kappa=0.1,
    )
