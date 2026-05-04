from __future__ import annotations

from math import sqrt, exp
from typing import Literal, TypedDict

import numpy as np

BS_TIME_EPSILON = 1e-12
Z_975 = 1.9599639845400545

class MCResult(TypedDict):
    """
    Structured return type for `mc_price`.
    
    Keys
    ----
    price : float
        Discounted MC estimate of the present value:
        V_hat = e^{-rT} * (1/N) * sum_i g(S_T^{(i)}).
    std_error : float
        Standard error of the *price* estimator:
        SE = e^{-rT} * sigma_hat_X / sqrt(N), where sigma_hat_X is the
        sample standard deviation (ddof=1) of the undiscounted payoffs.
        Note: SE of the mean, NOT the std of the payoffs themselves —
        they differ by a factor of sqrt(N).
    ci_95 : tuple[float, float]
        Two-sided 95% normal-approximation confidence interval,
        (price - Z_975 * SE, price + Z_975 * SE). Coverage is asymptotic
        (justified by the CLT) and may be poor for very small N or for
        deep-OTM options where payoffs are zero with high probability.
    n_paths : int
        Number of MC paths used; echoed back for logging and reproducibility.
    """
    
    price: float
    std_error: float
    ci_95: tuple[float, float]
    n_paths: int


def _simulate_terminal_prices(S0: float, T: float, r: float, sigma: float,
    n_paths: int, seed: int | None) -> np.ndarray:
    
    # simulation under the RISK-NEUTRAL measure Q 
    # CRITICAL: drift is r, NOT a real-world mu. This is the *only* place
    # the change of measure enters the code. Pricing is E^Q[ e^{-rT} g(S_T) ].
    
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    log_S_T = np.log(S0) + (r - 0.5 * sigma**2) * T + sigma * sqrt(T) * z
    return np.exp(log_S_T)


def mc_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"],
    n_paths: int,
    seed: int | None = None,
) -> MCResult:
    """
    Monte Carlo price for a European call or put under risk-neutral GBM.

    Simulates N independent terminal prices S_T under the risk-neutral
    measure Q, where  dS_t / S_t = r dt + sigma dW_t^Q,  and estimates

        V_0 = e^{-rT} * E^Q[ g(S_T) ]

    via  V_hat_N = e^{-rT} * (1/N) * sum_i g(S_T^{(i)}),
    with standard error  SE = e^{-rT} * sigma_hat_X / sqrt(N).

    Parameters
    ----------
    S0 : float
        Spot price at valuation time. Must be > 0.
    K : float
        Strike price. Must be > 0.
    T : float
        Time to maturity in years. Must be >= 0. If T < BS_TIME_EPSILON
        the function short-circuits to the deterministic intrinsic value
        with zero standard error (avoids 0/0 in the SE formula).
    r : float
        Continuously compounded risk-free rate. Also serves as the drift
        of the simulated GBM — this is the *only* place where the change
        of measure from P (real-world) to Q (risk-neutral) enters.
    sigma : float
        Annualized volatility. Must be >= 0. sigma == 0 collapses the
        diffusion and gives S_T = S_0 * exp(r*T) deterministically; SE = 0.
    option_type : {"call", "put"}
        Payoff type. Call pays max(S_T - K, 0); put pays max(K - S_T, 0).
    n_paths : int
        Number of independent MC paths. Must be a Python `int` (or
        `np.integer`); floats like 1e5 are rejected because they break
        ddof=1 arithmetic in unexpected ways. Must be > 1.
    seed : int or None, optional
        Seed for `np.random.default_rng`. 
        
    Returns
    -------
    MCResult
        See `MCResult` for the schema.

    Notes
    -----
    - Estimator is unbiased; SE uses Bessel-corrected sample std (ddof=1).
    - The 95% CI uses 1.96 (the 0.975 standard-normal quantile) under
      the CLT approximation that (V_hat - V_true) / SE -> N(0, 1).
    - Coverage is asymptotic and can be poor for very small N or deep-OTM
      options.
    """
    
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if K <= 0:
        raise ValueError("K must be positive.")
    if T < 0:
        raise ValueError("T must be non-negative.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if not isinstance(n_paths, (int, np.integer)):
        raise TypeError("n_paths must be an integer (got float or other).")
    if n_paths <= 1:
        raise ValueError("n_paths must be an integer >= 2 for a valid standard error estimate.")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be either 'call' or 'put'.")
        
    discount_factor = exp(-r * T)
    
    # T -> 0 branch: payoff is deterministic intrinsic value
    if T < BS_TIME_EPSILON:
        intrinsic = max(S0 - K, 0.0) if option_type == "call" else max(K - S0, 0.0)
        return {
            "price": float(intrinsic),
            "std_error": 0.0,
            "ci_95": (float(intrinsic), float(intrinsic)),
            "n_paths": int(n_paths),
        }
    
    
    S_T = _simulate_terminal_prices(S0, T, r, sigma, n_paths, seed)
    
    
    # When sigma=0, every simulated path is identical (deterministic), so
    # the payoff has zero variance. We return the exact value directly to
    # avoid floating-point noise in std(ddof=1)
    
    if sigma == 0.0:
    # S_T = S0 * exp(r*T) for all paths; discount cancels exp(r*T)
        intrinsic_fwd = ( (S0 - K * discount_factor) if option_type == "call"
                        else (K * discount_factor - S0) )
        
        price_det = float(max(intrinsic_fwd, 0.0))
        
        return {
        "price": price_det,
        "std_error": 0.0,
        "ci_95": (price_det, price_det),
        "n_paths": int(n_paths),
    }
    
        
    # payoff (vectorised)
    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)
        
    # estimator and standard error
    mean_payoff = float(payoffs.mean())
    std_payoff  = float(payoffs.std(ddof=1))  # Bessel correction
    
    price     = discount_factor * mean_payoff
    std_error = discount_factor * std_payoff / sqrt(n_paths)
    
    half_width = Z_975 * std_error
    
    return {
        "price": price,
        "std_error": std_error,
        "ci_95": (price - half_width, price + half_width),
        "n_paths": int(n_paths),
    }
