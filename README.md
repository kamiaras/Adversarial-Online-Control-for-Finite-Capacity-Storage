# Adversarial Online Control for Finite-Capacity Storage

Algorithms, experiments, and notebooks for studying online control of batteries, queues, and other finite-capacity storage systems under stochastic or adversarial arrivals. The codebase supports both the convex offline benchmark used in regret analyses and the disturbance--action controllers that act online.

## Online Decision-Making and Control with Finite-Capacity Storage

Another central line of my research studies online decision-making for systems with finite-capacity storage—such as energy-harvesting devices, batteries, or finite buffers—where a controller must allocate limited resources in the presence of stochastic or adversarial arrivals. Here the performance metric is regret: we seek algorithms whose total cost is within $O(\sqrt{T})$ of an ideal benchmark policy.

In earlier work with Neely [Asgari & Neely, POMACS 2020], we considered random (i.i.d.) arrivals and convex losses that depend only on the allocation vector. We designed allocation algorithms that compete with the best fixed allocation in hindsight, obtaining optimal $O(\sqrt{T})$ regret in an OCO-style framework while respecting the finite-capacity constraint. This work connects classical Lyapunov/queueing ideas with OCO-style regret guarantees for stochastic resource allocation.

In ongoing work, I push this line of research into the fully adversarial, nonstochastic control regime by adopting the disturbance–action (DAC) framework from recent work on online control. I introduce a simplex-constrained disturbance–action policy class (SDAC) tailored to finite-capacity storage: the DAC parameters are required to lie in a probability simplex, which simultaneously enforces non-negativity, conservation of mass, and prevents allocations that exceed the available inventory. Unlike the earlier POMACS model, the stage cost now depends on both the storage level (state) and the allocation (control), and the exogenous arrivals may be adversarial. For this SDAC class I design and analyze an online controller that achieves $O(\sqrt{T})$ regret with respect to the best fixed SDAC policy in hindsight, under convex Lipschitz costs and bounded disturbances. Conceptually, this shows that regret-optimal control is possible even when the system is constrained by finite storage and adversarial inputs. I initiated this extension from fixed allocations to simplex-constrained DAC policies and developed the corresponding regret analysis and proofs; it illustrates how viewing OCO algorithms as modular components can lead to new, structure-aware policy classes for online control.

These models capture, for example, batteries interacting with uncertain renewable generation, energy-harvesting communication systems, or inventory/queueing systems where one must decide how to route or consume limited stored resources over time.

## Repository layout

```
.
├── src/
│   ├── arrivals.py              # Arrival process generators (stochastic & adversarial)
│   ├── costs.py                 # Convex stage-cost models
│   ├── features.py              # Φ, Ψ feature construction for DAC policies
│   ├── offline_opt.py           # CVXPY solver for the simplex-constrained offline benchmark
│   ├── simulate_storage_dynamics.py
│   ├── utils.py                 # Simplex projections, plotting, controller comparison helpers
│   ├── experiments.py           # Batch search + reruns over arrival/cost combinations
│   └── plots.py, egpc.py, etc.  # Extra utilities used in notebooks
├── notebook/                    # Reproducible demos (Jupyter)
├── images1/                     # Figures exported from the notebooks
└── LICENSE
```

## Getting started

Create a virtual environment and install the scientific Python stack (Python ≥3.10 recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy cvxpy matplotlib scipy tqdm
```

### Solve the offline benchmark

The offline program minimizes $\sum_t c_t(b_t(w), u_t(w))$ over the simplex-constrained disturbance–action weights. You can supply any custom arrival generator or cost function as a callable.

```python
from src.offline_opt import solve_optimal_w
from src.arrivals import arrivals_random
from src.costs import c_weighted_combo

result = solve_optimal_w(
    a_fn=arrivals_random,
    cost_fn=c_weighted_combo,
    T=400,
    H=25,
    kappa=0.1,
    cost_kwargs={"s1": 0.25, "s2": 4.0, "target": 0.8},
)
w_star = result["w_star"]
```

### Simulate a controller

```python
from src.simulate_storage_dynamics import simulate_storage_dynamics

sim = simulate_storage_dynamics(
    Phi=result["Phi"],
    Psi=result["Psi"],
    a=result["a"],
    w=w_star,
    cost_fn=result["cost_fn"],
    cost_kwargs=result["cost_kwargs"],
)
print("Total cost:", sim["total_cost"])
```

### Batch search for interesting regimes

Use `src/experiments.py` to scan across multiple arrival/cost pairs and log the ones whose offline solutions are non-degenerate.

```python
from src.arrivals import ARRIVAL_FUNCTIONS
from src.costs import COST_FUNCTIONS
from src.experiments import run_all_combinations

run_all_combinations(
    arrivals={k: ARRIVAL_FUNCTIONS[k] for k in ("random", "bursty", "adversarial")},
    costs={k: COST_FUNCTIONS[k] for k in ("tracking", "sin_randomized")},
    T=600,
    H=30,
    kappa=0.05,
    save_path="interesting_combinations.txt",
)
```

### Visualize competing controllers

After simulating two or more controllers, call `utils.compare_controllers(sim_a, "ALG-A", sim_b, "ALG-B")` to compare battery levels, usage, and per-step costs with automatically generated Matplotlib figures plus a numerical summary of total costs and pairwise differences.

## Notebooks

The `notebook/` directory contains exploration-friendly Jupyter notebooks:

- `demo.ipynb` — walkthrough of the SDAC construction and plotting utilities.
- `three_controller_comparison.ipynb` — reproduces the figures from the adversarial-vs-stochastic comparison.
- `cost_vs_T.ipynb`, `linear_gain_vs_optimal.ipynb`, `multi_linear_egpc_comparison.ipynb` — ablation studies on horizon length, gains, and controller families.

Launch them with `jupyter lab` (or `jupyter notebook`) after activating the environment.

## Reusing or extending the code

- **Arrivals**: Add new generators to `ARRIVAL_FUNCTIONS` for scenario-specific inputs (e.g., renewable traces, bursts, switches).
- **Costs**: Drop in any convex stage cost via `COST_FUNCTIONS` or by passing your own callable into `solve_optimal_w` / `simulate_storage_dynamics`.
- **Controllers**: Plug alternate disturbance–action parameterizations or online learning rules into the feature matrices `(Φ, Ψ)`. The utilities accept any vector on the simplex, so algorithms such as Hedge, Mirror Descent, or projected gradient updates can be prototyped quickly.

## Citation

If you use this repository in academic work, please cite Asgari & Neely (POMACS 2020) for the stochastic finite-capacity OCO model and the ongoing manuscript described above for the simplex-constrained DAC extension. A BibTeX entry will be added once the adversarial-control manuscript is public.
