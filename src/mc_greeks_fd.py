"""
mc_greeks_fd.py
============
 
Monte Carlo Greeks via central finite differences with Common Random Numbers
(CRN) variance reduction.
 
Computes Delta, Gamma, Vega, Theta, and Rho for a European vanilla option by
bumping each input parameter and re-pricing with `mc_price`. Variance
reduction comes from passing the same `seed` to every paired evaluation
(CRN), which shares the underlying Brownian increments across the two
simulations and so makes the *difference* of payoffs much less noisy than
the payoffs themselves.
 
Mathematical formulas
---------------------
For first-order Greeks (Delta, Vega, Theta, Rho), the central-difference
stencil is
 
    dV/dθ  ≈  ( V(θ + h) − V(θ − h) ) / ( 2 h ),     truncation error O(h²).
 
For Gamma, the three-point second-derivative stencil is
 
    d²V/dS₀²  ≈  ( V(S₀ + h) − 2 V(S₀) + V(S₀ − h) ) / h².
 
Theta uses the financial-convention sign Θ = ∂V/∂t = −∂V/∂T (calendar time
plus time-to-maturity is constant). We compute the FD with respect to T and
negate.
 
Notes
-----
CRN is doing the heavy lifting here. Without it, the noise floor on a
finite-difference Greek scales as O(σ_payoff / h), which blows up as h → 0.
With CRN, the payoff difference is computed *per path* and most of the noise
cancels, leaving the O(h²) truncation bias as the dominant error.
 
Correctness depends on `mc_price` being deterministic in `seed`: the same
seed must produce the same underlying random draws. If the implementation
draws from a global RNG state, CRN will silently fail and the Greeks will
be unusable.
"""

from __future__ import annotations
from typing import Literal, TypedDict
from monte_carlo import mc_price

__all__ = ["GreeksResult", "mc_greeks_fd"]


class GreeksResult(TypedDict):
    """The five Greeks returned by `mc_greeks_fd`."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    

def mc_greeks_fd(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"],
    n_paths: int,
    seed: int | None = None,
    *,
    h_S_rel: float = 1e-2,
    h_sigma: float = 1e-2,
    h_r: float = 1e-2,
    h_T: float = 1.0 / 365.0,
) -> GreeksResult:
    """
    Monte Carlo Greeks via central finite differences with CRN.
 
    Parameters
    ----------
    S0, K, T, r, sigma, option_type, n_paths, seed
        Forwarded to `mc_price` exactly as in the analytical pricer. `seed`
        must be set, and `mc_price` must be deterministic in it, for CRN to
        actually reduce variance.
    h_S_rel : float, default 1e-2
        Relative bump for S₀; the absolute bump used is ``h_S = h_S_rel * S0``.
        Recommended range: [1e-3, 1e-1]. Must be < 1 to keep S₀ − h_S > 0.
    h_sigma : float, default 1e-2
        Absolute bump for σ. Sigma is O(1) in typical use, so an absolute
        bump is appropriate. Must satisfy ``sigma - h_sigma > 0``.
    h_r : float, default 1e-2
        Absolute bump for r. Sign of r is unconstrained (negative rates are
        permitted), so no boundary check is needed.
    h_T : float, default 1/365
        Absolute bump for T, in years. Default is one calendar day. Must
        satisfy ``T - h_T > 0``.
 
    Returns
    -------
    GreeksResult
        TypedDict with keys 'delta', 'gamma', 'vega', 'theta', 'rho'.
 
    Raises
    ------
    ValueError
        If any input is out of domain, or if a bumped parameter would leave
        its valid domain (e.g. ``T - h_T <= 0``).
    """
    _validate_inputs(S0, K, T, sigma, n_paths, h_S_rel, h_sigma, h_r, h_T)
 
    # Concrete absolute bump for S₀. The other three bumps are already absolute.
    h_S = h_S_rel * S0
 
    # Closure captures every invariant of the pricing call. Crucially, `seed`
    # is captured here, so every call below uses the same random draws => CRN.
    def price(S0_: float, sigma_: float, T_: float, r_: float) -> float:
        return mc_price(
            S0=S0_,
            K=K,
            T=T_,
            r=r_,
            sigma=sigma_,
            option_type=option_type,
            n_paths=n_paths,
            seed=seed,
        )["price"]
 
    # Delta and Gamma share the V(S₀) center evaluation.
    V_S_plus = price(S0 + h_S, sigma, T, r)
    V_S_0 = price(S0, sigma, T, r)
    V_S_minus = price(S0 - h_S, sigma, T, r)
 
    delta = (V_S_plus - V_S_minus) / (2.0 * h_S)
    gamma = (V_S_plus - 2.0 * V_S_0 + V_S_minus) / (h_S * h_S)
 
    # Vega
    V_sig_plus = price(S0, sigma + h_sigma, T, r)
    V_sig_minus = price(S0, sigma - h_sigma, T, r)
    vega = (V_sig_plus - V_sig_minus) / (2.0 * h_sigma)
 
    # Theta = -dV/dT  (financial convention)
    V_T_plus = price(S0, sigma, T + h_T, r)
    V_T_minus = price(S0, sigma, T - h_T, r)
    theta = -(V_T_plus - V_T_minus) / (2.0 * h_T)
 
    # Rho
    V_r_plus = price(S0, sigma, T, r + h_r)
    V_r_minus = price(S0, sigma, T, r - h_r)
    rho = (V_r_plus - V_r_minus) / (2.0 * h_r)
 
    return GreeksResult(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
    
    
    
def _validate_inputs(
    S0: float,
    K: float,
    T: float,
    sigma: float,
    n_paths: int,
    h_S_rel: float,
    h_sigma: float,
    h_r: float,
    h_T: float,
) -> None:
    """Validate inputs and ensure every bumped parameter stays in-domain.
 
    `r` is intentionally allowed to be any sign; negative rates are real.
    Every other parameter must be strictly positive, and bumps must not push
    the bumped parameter to ≤ 0.
    """
    if S0 <= 0:
        raise ValueError(f"S0 must be positive, got {S0}.")
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}.")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}.")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    if n_paths < 2:
        raise ValueError(f"n_paths must be >= 2, got {n_paths}.")
 
    for name, h in (
        ("h_S_rel", h_S_rel),
        ("h_sigma", h_sigma),
        ("h_r", h_r),
        ("h_T", h_T),
    ):
        if h <= 0:
            raise ValueError(f"{name} must be positive, got {h}.")
 
    # In-domain bump checks: every bumped parameter must remain valid.
    if h_S_rel >= 1.0:
        raise ValueError(
            f"h_S_rel must be < 1 (else S0 - h_S <= 0), got {h_S_rel}. "
            f"Reduce h_S_rel."
        )
    if sigma - h_sigma <= 0:
        raise ValueError(
            f"Bump would make sigma non-positive: sigma - h_sigma = "
            f"{sigma - h_sigma:.4g}. Reduce h_sigma or increase sigma."
        )
    if T - h_T <= 0:
        raise ValueError(
            f"Bump would make T non-positive: T - h_T = "
            f"{T - h_T:.4g}. Reduce h_T or increase T."
        )
 
 
    
if __name__ == "__main__":
    # Demo: ATM 1-year European call, side-by-side MC vs analytic.
    from black_scholes import bs_greeks
 
    params = dict(
        S0=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.20,
        option_type="call",
    )
 
    mc = mc_greeks_fd(**params, n_paths=100_000, seed=42)
    bs = bs_greeks(**params)
 
    header = f"{'Greek':<8}{'MC (FD+CRN)':>14}{'BS (analytic)':>16}{'|diff|':>12}"
    print(header)
    print("-" * len(header))
    for g in ("delta", "gamma", "vega", "theta", "rho"):
        mc_v, bs_v = mc[g], bs[g]
        print(f"{g:<8}{mc_v:>14.6f}{bs_v:>16.6f}{abs(mc_v - bs_v):>12.6f}")