"""
exotics.py — closed-form and Monte Carlo pricers for path-dependent options.
 
Currently implements the discretely monitored *geometric* Asian option under
Black-Scholes dynamics. The monitoring grid is t_i = i*T/n for i = 1..n
(stop-inclusive: the final monitoring date coincides with expiry T). The
analytic and Monte Carlo pricers MUST share this grid so they are directly
comparable for testing.
 
Mathematical background (geometric Asian, fixed strike)
-------------------------------------------------------
Under risk-neutral GBM, log S_{t_i} is Gaussian, and the log of the geometric
average  G = (prod_i S_{t_i})^{1/n}  is therefore also Gaussian:
 
    log G ~ Normal(mu_G, sigma_G^2)
 
    mu_G     = log S0 + (r - 0.5 sigma^2) * t_bar          (t_bar = mean(t_i))
    sigma_G^2 = (sigma^2 / n^2) * sum_{i,j} min(t_i, t_j)
 
Because G is lognormal, the option reduces to a Black-Scholes-style formula
with forward F = E[G] = exp(mu_G + 0.5 sigma_G^2).
"""

from __future__ import annotations

import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm


# z-score for a two-sided 95% confidence interval (~1.95996).
_Z_95 = float(norm.ppf(0.975))


def _validate_common(S0: float, K: float, T: float, sigma: float,
                     option_type: str) -> str:
    """
    Shared input checks. Returns the normalised (lowercased) option_type.
    """
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive")
    if T < 0:
        raise ValueError("T must be non-negative")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    return option_type


def geometric_asian_analytic(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    n_monitor: int,
) -> float:
    """Closed-form price of a discretely monitored geometric Asian option.
 
    Parameters
    ----------
    S0, K : float
        Spot and strike (both must be > 0).
    T : float
        Time to maturity in years (>= 0).
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility (>= 0).
    option_type : {'call', 'put'}
        Case-insensitive.
    n_monitor : int
        Number of monitoring dates (>= 1), placed at t_i = i*T/n_monitor.
 
    Returns
    -------
    float
        Present value of the option.
    """
    option_type = _validate_common(S0, K, T, sigma, option_type) 
    if n_monitor < 1:
        raise ValueError("n_monitor must be >= 1")
        
    # Edge case: expiry now -> intrinsic value on the spot
    if T == 0:
        intrinsic = max(S0 - K, 0.0) if option_type == "call" else max(K - S0, 0.0)
        return float(intrinsic)
    
    # Edge case: zero vol -> deterministic geometric average 
    # With sigma = 0, S_{t_i} = S0 * exp(r t_i), so log G = log S0 + r * t_bar,
    # where t_bar = mean_i (i T / n) = T (n+1) / (2n).
    if sigma == 0:
        t_bar = T * (n_monitor + 1) / (2 * n_monitor)
        G_det = S0 * exp(r * t_bar)
        payoff = max(G_det - K, 0.0) if option_type == "call" else max(K - G_det, 0.0)
        return exp(-r * T) * payoff
    
    
    # General case
    t = np.arange(1, n_monitor + 1, dtype=float) * (T / n_monitor)
    t_bar = float(t.mean())
    
    # Var(sum_i W_{t_i}) = sum_{i,j} Cov(W_{t_i}, W_{t_j})
    #                    = sum_{i,j} min(t_i,t_j).
    cov_sum = float(np.minimum.outer(t, t).sum())
    
    sigma_G2 = sigma**2 * cov_sum / n_monitor**2
    sigma_G = sqrt(sigma_G2)
    
    mu_G = log(S0) + (r - 0.5 * sigma**2) * t_bar
    
    # Lognormal (Black-Scholes) transform on the geometric average.
    d1 = (mu_G - log(K) + sigma_G2) / sigma_G
    d2 = d1 - sigma_G
    forward = exp(mu_G + 0.5 * sigma_G2)  # E[G]
    
    
    if option_type == "call":
        price = exp(-r * T) * (forward * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        price = exp(-r * T) * (K * norm.cdf(-d2) - forward * norm.cdf(-d1))
        
    return float(price)



def _simulate_discounted_payoffs(
    rng: np.random.Generator,
    S0: float, K: float, T: float, r: float, sigma: float,
    option_type: str, n_monitor: int, n_paths: int,
) -> np.ndarray:
    """
    Simulate `n_paths` discounted geometric-Asian payoffs.
 
    Works entirely in log space for numerical stability:
        log S_{t_{k+1}} = log S0 + sum_{j<=k} [(r - 0.5 sigma^2) dt + sigma dW_j]
        
    so column k of `log_S` is exactly monitoring date t_{k+1}; no S0 column is
    needed because the geometric average is taken over t_1..t_n only.
    """ 
    
    dt = T / n_monitor
    dW = np.sqrt(dt) * rng.standard_normal((n_paths, n_monitor))
    drift = (r - 0.5 * sigma**2) * dt
    
    log_S = np.log(S0) + np.cumsum(drift + sigma * dW, axis=1)
    
    G = np.exp(log_S.mean(axis=1))  # geometric mean per path
 
    if option_type == "call":
        payoff = np.maximum(G - K, 0.0)
    else:
        payoff = np.maximum(K - G, 0.0)
 
    return np.exp(-r * T) * payoff



def geometric_asian_mc(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    n_monitor: int,
    n_paths: int,
    seed: int | None = None,
    batch_size: int | None = None,
) -> dict:
    """
    Monte Carlo price of a discretely monitored geometric Asian option.
 
    Parameters
    ----------
    (S0, K, T, r, sigma, option_type, n_monitor): as in `geometric_asian_analytic`.
    n_paths : int
        Number of simulated paths (>= 2 for a defined standard error).
    seed : int | None
        Seed for the random number generator (reproducibility).
    batch_size : int | None
        If set, paths are simulated in chunks of this size and the statistics
        accumulated incrementally, capping peak memory at ~O(batch_size *
        n_monitor) instead of O(n_paths * n_monitor). `None` simulates in one
        vectorised batch (fastest for moderate sizes).
 
    Returns
    -------
    dict with keys 'price', 'stderr', 'ci_95'.
    """
    
    option_type = _validate_common(S0, K, T, sigma, option_type)
    
    if n_monitor < 1 or n_paths < 1:
        raise ValueError("n_monitor and n_paths must be >= 1")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be >= 1 or None")
 
    # Edge case: expiry now -> deterministic intrinsic, zero MC error.
    if T == 0:
        intrinsic = max(S0 - K, 0.0) if option_type == "call" else max(K - S0, 0.0)
        return {"price": float(intrinsic), "stderr": 0.0,
                "ci_95": (float(intrinsic), float(intrinsic))}
 
    rng = np.random.default_rng(seed)
    
    if batch_size is None:
        # Single vecotrised batch
        disc = _simulate_discounted_payoffs(
            rng, S0, K, T, r, sigma, option_type, n_monitor, n_paths)
        price = float(disc.mean())
        sample_std = float(disc.std(ddof=1)) if n_paths > 1 else float("nan")
        se = sample_std / sqrt(n_paths)
        
    else:
        # Memory-bounded streaming: accumulate count, sum, and sum-of-squares.
        # (For non-negative O(price) payoffs this is numerically fine.
        count, total, total_sq = 0, 0.0, 0.0
        remaining = n_paths
        while remaining > 0:
            m = min(batch_size, remaining)
            disc = _simulate_discounted_payoffs(
                rng, S0, K, T, r, sigma, option_type, n_monitor, m)
            count += m
            total += float(disc.sum())
            total_sq += float(np.square(disc).sum())
            remaining -= m
            
        price = total / count
        if count > 1:
            variance = (total_sq - total**2 / count) / (count - 1)
            se = sqrt(max(variance, 0.0) / count)  # clamp tiny negative roundoff
            
        else:
            se = float("nan")
            
    
    half_width = _Z_95 * se
    return {
        "price": price,
        "stderr": se,
        "ci_95": (price - half_width, price + half_width),
    }


    
       
    