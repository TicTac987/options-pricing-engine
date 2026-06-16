import numpy as np

from scipy.stats import norm
from typing import Literal

# Constants
BS_TIME_EPSILON = 1e-9
BS_VOL_EPSILON  = 1e-12

def bs_price(
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float, 
        option_type: Literal["call", "put"]
    ) -> float:
    """
    Black-Scholes price for European options (vectorised).

    Handles edge cases explicitly:
    - T → 0  → payoff
    - σ → 0  → deterministic forward payoff

    Returns
    -------
    np.ndarray or float
    """
    
    # Normalise inputs
    S0 = np.asarray(S0, dtype=float)
    K  = np.asarray(K, dtype=float)
    T  = np.asarray(T, dtype=float)
    
    is_call = option_type.lower() == "call"
    
    # Validation
    if np.any(S0 <= 0):
        raise ValueError("S0 must be positive")
    if np.any(K <= 0):
        raise ValueError("K must be positive")        
    if np.any(T < 0):   
        raise ValueError("T must be non-negative")        
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
        
    # Expiry handling
    if np.all(T <= BS_TIME_EPSILON):
        return np.maximum(S0 - K, 0.0) if is_call else np.maximum(K - S0, 0.0)
    
    # Pre compute
    sqrtT = np.sqrt(T)
    disc  = np.exp(-r * T)
    
    # Zero volatility limit
    if sigma <= BS_VOL_EPSILON:
        # Deterministic forward evolution
        forward = S0 * np.exp(r * T)
        payoff  = np.maximum(forward - K, 0.0) if is_call else np.maximum(K - forward, 0.0)
                                                
        return disc * payoff
    
    # Black-Scholes core    
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) *T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    
    if is_call:
        return S0 * norm.cdf(d1) - K * disc * norm.cdf(d2)
    
    else:
        return K * disc * norm.cdf(-d2) - S0 * norm.cdf(-d1)
               
    
def bs_greeks(S0: float, K: float, T: float, r: float,
              sigma: float, option_type: Literal["call", "put"]) -> dict:
    """
    Compute Black-Scholes Greeks analytically (European options).

    Parameters
    ----------
    S0 : float
        Spot price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate (continuous compounding)
    sigma : float
        Volatility (annualised)
    option_type : {'call', 'put'}

    Returns
    -------
    dict
        {
            'delta': float,
            'gamma': float,
            'vega': float,   # per 1.0 vol (NOT per 1%)
            'theta': float,  # dV/dt (calendar time)
            'rho': float
        }
    """
    
    # Input validation
    if S0 <= 0:
        raise ValueError(f"S0 must be positive, got {S0}")
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if T < 0:
        raise ValueError(f"T cannot be negative, got {T}")
    if sigma < 0:
        raise ValueError(f"sigma cannot be negative, got {sigma}")

    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
    
    
    #Edge case: T -> 0 
    # φ(d1) → 0, so Gamma/Vega vanish; Delta becomes a step function on spot.
    if T < BS_TIME_EPSILON:
        if option_type == "call":
            delta = 1.0 if S0 > K else (0.5 if S0 == K else 0.0)
        else:
            delta = -1.0 if S0 < K else (-0.5 if S0 == K else 0.0)
        
        return {
            "delta": delta,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": 0.0,
        }
    
    
    # Edge case: σ → 0 
    # Terminal price is deterministic: F = S₀eʳᵀ.
    # Gamma/Vega vanish; Theta/Rho from discounted intrinsic value.
    if sigma < BS_VOL_EPSILON:
        disc = np.exp(-r * T)
        fwd  = S0 / disc          

        if option_type == "call":
            itm   = fwd > K
            delta = 1.0 if itm else (0.5 if fwd == K else 0.0)
            theta = -r * K * disc if itm else 0.0
            rho   =  K * T * disc if itm else 0.0
        else:
            itm   = fwd < K
            delta = -1.0 if itm else (-0.5 if fwd == K else 0.0)
            theta =  r * K * disc if itm else 0.0
            rho   = -K * T * disc if itm else 0.0

        return {
            "delta": delta,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": theta,
            "rho": rho,
        }
    
    
    # Standard Black-Scholes 
    sqrtT  = np.sqrt(T)
    d1     = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2     = d1 - sigma * sqrtT
    disc   = np.exp(-r * T)

    Phi_d1 = norm.cdf(d1)
    Phi_d2 = norm.cdf(d2)
    phi_d1 = norm.pdf(d1)

    # Gamma and Vega are the same for calls and puts (put-call parity)
    gamma = phi_d1 / (S0 * sigma * sqrtT)
    vega  = S0 * phi_d1 * sqrtT

    if option_type == "call":
        delta = Phi_d1
        theta = -(S0 * phi_d1 * sigma) / (2.0 * sqrtT) - r * K * disc * Phi_d2
        rho   =  K * T * disc * Phi_d2
    else:
        delta = Phi_d1 - 1.0          # put-call parity: Δ_put = Δ_call − 1
        theta = -(S0 * phi_d1 * sigma) / (2.0 * sqrtT) + r * K * disc * norm.cdf(-d2)
        rho   = -K * T * disc * norm.cdf(-d2)
    
        
    
    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }
    