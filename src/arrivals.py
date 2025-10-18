import numpy as np

def arrivals_fixed(T, amplitude=1.0, value=0.5):
    """Fixed constant arrival sequence a_t = value for all t."""
    return np.full(T, amplitude * value)


def arrivals_random(T, amplitude=1.0, seed=42):
    """i.i.d. Uniform[0,1) arrivals scaled by amplitude."""
    rng = np.random.default_rng(seed)
    return amplitude * rng.random(T)


def arrivals_gaussian(T, amplitude=1.0, mean=0.5, std=0.15, seed=42):
    """Gaussian arrivals clipped to [0,1]."""
    rng = np.random.default_rng(seed)
    a = rng.normal(mean, std, T)
    return np.clip(amplitude * a, 0.0, 1.0)


def arrivals_sinusoidal(T, amplitude=1.0, period=100, phase=0):
    """Smooth periodic sinusoidal arrivals."""
    t = np.arange(T)
    a = 0.5 * (1 + np.sin(2 * np.pi * t / period + phase))
    return amplitude * a


def arrivals_decaying(T, amplitude=1.0, decay_rate=0.002):
    """Exponentially decaying arrivals."""
    t = np.arange(T)
    return amplitude * np.exp(-decay_rate * t)


def arrivals_bursty(T, amplitude=1.0, burst_length=(20, 60), quiet_length=(40, 100), seed=123):
    """Random bursts of activity separated by quiet periods."""
    rng = np.random.default_rng(seed)
    a = np.zeros(T)
    t = 0
    while t < T:
        # burst
        blen = rng.integers(burst_length[0], burst_length[1])
        a[t:t+blen] = rng.uniform(0.7, 1.0)
        t += blen
        # quiet
        qlen = rng.integers(quiet_length[0], quiet_length[1])
        t += qlen
    return np.clip(a, 0.0, 1.0) * amplitude


def arrivals_adversarial(T, amplitude=1.0, noise_level=0.1, seed=321):
    """Alternating high/low arrivals with small noise (adversarial pattern)."""
    rng = np.random.default_rng(seed)
    a = np.array([(i % 2) for i in range(T)], dtype=float)
    a += noise_level * rng.standard_normal(T)
    return np.clip(amplitude * a, 0.0, 1.0)


def arrivals_switching_regime(T, amplitude=1.0, switch_period=200, seed=999):
    """Switches between low/high mean regimes."""
    rng = np.random.default_rng(seed)
    a = np.zeros(T)
    for t in range(T):
        regime = (t // switch_period) % 2
        mean = 0.2 if regime == 0 else 0.8
        a[t] = np.clip(rng.normal(mean, 0.1), 0, 1)
    return amplitude * a


def arrivals_spike_decay(T, amplitude=1.0, spike_every=150, decay_len=50):
    """Occasional spikes followed by exponential decay."""
    a = np.zeros(T)
    for spike_start in range(0, T, spike_every):
        for k in range(decay_len):
            if spike_start + k < T:
                a[spike_start + k] = amplitude * np.exp(-0.1 * k)
    return np.clip(a, 0.0, 1.0)


def arrivals_piecewise_linear(T, amplitude=1.0, segments=5, seed=456):
    """Piecewise linear random segments (smooth but nonstationary)."""
    rng = np.random.default_rng(seed)
    breakpoints = np.linspace(0, T, segments + 1, dtype=int)
    a = np.zeros(T)
    for i in range(segments):
        start, end = breakpoints[i], breakpoints[i + 1]
        start_val = rng.uniform(0, 1)
        end_val = rng.uniform(0, 1)
        a[start:end] = np.linspace(start_val, end_val, end - start)
    return amplitude * np.clip(a, 0.0, 1.0)


# --- Registry of available arrivals ---
ARRIVAL_FUNCTIONS = {
    "fixed": arrivals_fixed,
    "random": arrivals_random,
    "gaussian": arrivals_gaussian,
    "sinusoidal": arrivals_sinusoidal,
    "decaying": arrivals_decaying,
    "bursty": arrivals_bursty,
    "adversarial": arrivals_adversarial,
    "switching": arrivals_switching_regime,
    "spike_decay": arrivals_spike_decay,
    "piecewise_linear": arrivals_piecewise_linear,
}
