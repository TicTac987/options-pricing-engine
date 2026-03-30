import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def simulate_gbm_paths(S0, mu, sigma, T, n_steps, n_paths, seed=None):
    """
    Simulate n_paths of Geometric Brownian Motion.

    Parameters
    ----------
    S0      : float  — initial stock price  (S_0)
    mu      : float  — drift               (μ)
    sigma   : float  — volatility          (σ)
    T       : float  — terminal time
    n_steps : int    — number of time steps
    n_paths : int    — number of independent paths
    seed    : int    — optional RNG seed

    Returns
    -------
    t : np.ndarray shape (n_steps+1,)         — time grid
    S : np.ndarray shape (n_paths, n_steps+1) — simulated paths
    """
    rng = np.random.default_rng(seed)

    dt = T / n_steps

    # Draw a (n_paths × n_steps) array of standard normals Z_{i,k}
    Z = rng.standard_normal((n_paths, n_steps))

    # Compute log-increments: (μ - σ²/2)Δt + σ√Δt · Z
    log_increments = (mu - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * Z

    # Cumulative sum along axis=1 gives log(S_t / S_0) at each step
    # Prepend a column of zeros for t=0
    zeros = np.zeros((n_paths, 1))
    log_paths = np.concatenate([zeros, np.cumsum(log_increments, axis=1)], axis=1)

    # Exponentiate and scale by S0
    S = S0 * np.exp(log_paths)

    t = np.linspace(0, T, n_steps + 1)
    return t, S