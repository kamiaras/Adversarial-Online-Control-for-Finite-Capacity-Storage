import numpy as np


# --- Cell 3: Define Convex Cost Functions c_t(b, u) ---

def c_linear_neg_b(b, u, t=None):
    """c_t(b, u) = -b  (maximize throughput / minimize negative backlog)."""
    return -b


def c_linear_pos_b(b, u, t=None):
    """c_t(b, u) = +b  (minimize backlog)."""
    return b


def c_quadratic_tracking(b, u, t=None, target=1.0):
    """c_t(b, u) = b + (target - u)^2  (track u_t toward target)."""
    return b + (target - u) ** 2


def c_weighted_combo(b, u, t=None, s1=0.1, s2=2, target=1.0):
    """c_t(b, u) = s1*b + s2*(target - u)^2."""
    return s1 * b + s2 * (target - u) ** 2


def c_time_variant_sin(b, u, t, s1=1.0, s2=0.5, period=200):
    """c_t(b, u) = (s1 + 0.5*sin(2πt/period))*b + s2*(1 - u)^2 (periodic cost variation)."""
    weight_t = s1 + 0.5 * np.sin(2 * np.pi * t / period)
    return weight_t * b + s2 * (1 - u) ** 2


def c_time_variant_adversarial(b, u, t, s1=1.0, s2=0.5):
    """c_t(b, u) = s1*(-1)^t * b + s2*(1 - u)^2 (alternating adversarial sign)."""
    return ((-1) ** t) * s1 * b + s2 * (1 - u) ** 2


# --- NEW Burst and Unpredictable Cost Functions ---

def c_burst_fluctuating(b, u, t, s1=1.0, s2=50, burst_period=15, burst_strength=10.0):
    """
    Sudden cost spikes every few bursts.
    c_t(b,u) = s1*(1 + burst_strength*I_burst)*b + s2*(1 - u)^2
    """
    burst = int((t // burst_period) % 2 == 0)  # bursts every other period
    s1_t = s1 * (1 + burst_strength * burst)
    return s1_t * b + s2 * (1 - u) ** 2


def c_burst_randomized(b, u, t, s1=1.0, s2=0.5, seed=123):
    """
    Random bursty cost changes using pseudo-random multipliers.
    c_t(b,u) = s1*(1 + noise_t)*b + s2*(1 - u)^2
    """
    rng = np.random.default_rng(seed + t)
    noise_t = rng.uniform(-0.8, 1.2)  # random multiplier each step
    s1_t = s1 * (1 + max(-0.9, noise_t))  # cap extreme negatives
    return s1_t * b + s2 * (1 - u) ** 2


def c_burst_switching(b, u, t, s1=1.0, s2=0.5, switch_len=50, levels=(0.5, 5)):
    """
    Cost switches unpredictably between low/high regimes every 'switch_len' steps.
    c_t(b,u) = s1_t * b + s2*(1 - u)^2,  s1_t alternates between levels
    """
    phase = (t // switch_len) % 4
    s1_t = levels[0] if phase in (0, 3) else levels[1]
    return s1_t * b + s2 * (1 - u) ** 2


def c_burst_decay(b, u, t, s1=1.0, s2=5, reset_period=30):
    """
    Decaying cost that resets periodically.
    c_t(b,u) = s1*(1 + exp(-((t mod reset_period)/50)))*b + s2*(1 - u)^2
    """
    phase = t % reset_period
    s1_t = s1 * (1 + np.exp(-phase / 50))
    return s1_t * b + s2 * (1 - u) ** 2


# --- Registry of available cost functions ---
COST_FUNCTIONS = {
    "linear_-b": c_linear_neg_b,
    "linear_+b": c_linear_pos_b,
    "tracking": c_quadratic_tracking,
    "weighted": c_weighted_combo,
    "sin_time_variant": c_time_variant_sin,
    "adversarial_time_variant": c_time_variant_adversarial,
    "burst_fluctuating": c_burst_fluctuating,
    "burst_randomized": c_burst_randomized,
    "burst_switching": c_burst_switching,
    "burst_decay": c_burst_decay,
}

