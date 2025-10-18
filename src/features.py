import numpy as np

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


def compute_phi(a, T, H, kappa):
    """
    Compute matrix Phi with entries:
        phi_t^(j) = kappa * sum_{i=1}^{j-1} (1-kappa)^(i-1) * a_{t-i}
                    + (1-kappa)^(j-1) * a_{t-j}
    with zero padding for a_t when t <= 0.

    Returns
    -------
    Phi : np.ndarray of shape (T, H)
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
    Compute matrix Psi with entries:
        psi_t^(j) = sum_{i=1}^j (1-kappa)^(i-1) * a_{t-i}
    with zero padding for a_t when t <= 0.

    Returns
    -------
    Psi : np.ndarray of shape (T, H)
    """
    Psi = np.zeros((T, H))
    for t in range(1, T + 1):
        for j in range(1, H + 1):
            Psi[t - 1, j - 1] = sum((1 - kappa) ** (i - 1) * zero_padded(a, t - i)
                                    for i in range(1, j + 1))
    return Psi

