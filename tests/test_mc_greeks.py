"""
test_mc_greeks.py
=================
 
Pytest suite for `mc_greeks_fd`.
 
Test taxonomy
-------------
1. Convergence to BS analytic Greeks at N=1e5 (per-Greek tolerances).
2. Put-call symmetry under shared seed (CRN-tight tolerances).
       Δ_C - Δ_P = 1     (MC noise scale, ~1e-3 at N=1e5)
       Γ_C = Γ_P         (machine precision: per-path identity, since S_T is
                          linear in S₀ and so its second derivative is zero)
3. Theta sign matches BS (the easiest bug to ship: ±dV/dT confusion).
4. Boundary raises: bumps that would exit the valid domain.
5. Reproducibility: same inputs + same seed => identical output dict.
6. Input validation: bad S₀/K/T/σ/n_paths/bumps all raise ValueError.
 
The expensive convergence fixtures are module-scoped so each (call, put,
seed) MC computation runs once and is reused across parametrised cases.
"""

from __future__ import annotations
 
import math
import pytest

from optpricing.black_scholes import bs_greeks
from optpricing.mc_greeks_fd import mc_greeks_fd



# Standard ATM 1-year European option used as the baseline test case.
BASE = dict(
    S0=100.0,
    K=100.0,
    T=1.0,
    r=0.05,
    sigma=0.20,
)
 
N_PATHS_CONV = 100_000
N_PATHS_FAST = 10_000
SEED         = 42

TOL_PUT_CALL_DELTA = 5e-3   # ~ MC standard error of (Δ_C - Δ_P)
TOL_PUT_CALL_GAMMA = 1e-10  # essentially machine precision


# Per-Greek tolerances at N=1e5 with default bumps.
# - rel_tol covers Greeks with large magnitude (vega, rho), where MC noise
#   scales with the value itself (~0.5% at N=1e5 is normal).
# - abs_tol covers Greeks that can be small or pass through zero (delta,
#   gamma, theta near ATM).
# math.isclose passes if EITHER threshold is met.
TOL_BS = {
    "delta": dict(rel_tol=0.0,  abs_tol=5e-3),
    "gamma": dict(rel_tol=0.0,  abs_tol=5e-3),
    "vega":  dict(rel_tol=1e-2, abs_tol=5e-2),
    "theta": dict(rel_tol=1e-2, abs_tol=5e-2),
    "rho":   dict(rel_tol=1e-2, abs_tol=5e-2),
}
 


@pytest.fixture(scope="module")
def mc_call() -> dict:
    return mc_greeks_fd(
        **BASE, option_type="call", n_paths=N_PATHS_CONV, seed=SEED
    )


@pytest.fixture(scope="module")
def mc_put() -> dict:
    return mc_greeks_fd(
        **BASE, option_type="put", n_paths=N_PATHS_CONV, seed=SEED
    )


@pytest.fixture(scope="module")
def bs_call_greeks() -> dict:
    return bs_greeks(**BASE, option_type="call")
 
 
@pytest.fixture(scope="module")
def bs_put_greeks() -> dict:
    return bs_greeks(**BASE, option_type="put")
 

    
# 1. Convergence to BS analytic Greeks

@pytest.mark.parametrize("greek",  ["delta", "gamma", "vega", "theta", "rho"])
def test_call_converges_to_bs(greek, mc_call, bs_call_greeks):
    assert math.isclose(
        mc_call[greek], bs_call_greeks[greek], **TOL_BS[greek]
    ), (
        f"call {greek}: MC={mc_call[greek]:.6f}, "
        f"BS={bs_call_greeks[greek]:.6f}, "
        f"|diff|={abs(mc_call[greek] - bs_call_greeks[greek]):.6f}"
    )

    
@pytest.mark.parametrize("greek", ["delta", "gamma", "vega", "theta", "rho"])
def test_put_converges_to_bs(greek, mc_put, bs_put_greeks):
    assert math.isclose(
        mc_put[greek], bs_put_greeks[greek], **TOL_BS[greek]
    ), (
        f"put {greek}: MC={mc_put[greek]:.6f}, BS={bs_put_greeks[greek]:.6f}, "
        f"|diff|={abs(mc_put[greek] - bs_put_greeks[greek]):.6f}"
    )
    
    
    
# 2. Put-call symmetry under shared seed

def test_put_call_delta_identity(mc_call, mc_put):
    """Δ_C - Δ_P = 1, exact in BS, MC-noisy here."""
    diff = mc_call["delta"] - mc_put["delta"]
    assert math.isclose(diff, 1.0, abs_tol=TOL_PUT_CALL_DELTA), (
        f"Δ_C - Δ_P = {diff:.8f}, expected 1.0 ± {TOL_PUT_CALL_DELTA}"
    )
    

def test_put_call_gamma_identity(mc_call, mc_put):
    """Γ_C = Γ_P, exact per path under CRN (S_T linear in S₀)."""
    diff = mc_call["gamma"] - mc_put["gamma"]
    assert abs(diff) < TOL_PUT_CALL_GAMMA, (
        f"Γ_C - Γ_P = {diff:.2e}, expected ~0 to machine precision"
    )
    

# 3. Theta sign matches BS


def test_call_theta_sign_matches_bs(mc_call, bs_call_greeks):
    assert math.copysign(1.0, mc_call["theta"]) == math.copysign(
        1.0, bs_call_greeks["theta"]
    ), f"call theta sign mismatch: MC={mc_call['theta']}, BS={bs_call_greeks['theta']}"
 
 
def test_put_theta_sign_matches_bs(mc_put, bs_put_greeks):
    assert math.copysign(1.0, mc_put["theta"]) == math.copysign(
        1.0, bs_put_greeks["theta"]
    ), f"put theta sign mismatch: MC={mc_put['theta']}, BS={bs_put_greeks['theta']}"
    

# 4. Boundary raises: bumps that exit the valid domain

def test_raises_when_T_le_h_T():
    """T = h_T pushes T - h_T to 0, which must raise."""
    params = {**BASE, "option_type": "call", "T": 1.0 / 365.0}  # = default h_T
    with pytest.raises(ValueError, match="T"):
        mc_greeks_fd(**params, n_paths=1_000, seed=SEED)
 
 
def test_raises_when_sigma_le_h_sigma():
    """σ < h_sigma pushes σ - h_sigma below 0."""
    params = {**BASE, "option_type": "call", "sigma": 0.005}  # < default 1e-2
    with pytest.raises(ValueError, match="sigma"):
        mc_greeks_fd(**params, n_paths=1_000, seed=SEED)
 
 
def test_raises_when_h_S_rel_ge_one():
    """h_S_rel ≥ 1 would push S₀ - h_S to ≤ 0."""
    params = {**BASE, "option_type": "call"}
    with pytest.raises(ValueError, match="h_S_rel"):
        mc_greeks_fd(**params, n_paths=1_000, seed=SEED, h_S_rel=1.0)
        
        
# 5. Reproducibility


def test_same_seed_gives_identical_output():
    params = {**BASE, "option_type": "call"}
    g1 = mc_greeks_fd(**params, n_paths=N_PATHS_FAST, seed=SEED)
    g2 = mc_greeks_fd(**params, n_paths=N_PATHS_FAST, seed=SEED)
    assert g1 == g2, f"non-reproducible:\n  g1={g1}\n  g2={g2}"
    

def test_different_seeds_give_different_output():
    """Sanity check: distinct seeds must actually exercise different draws."""
    params = {**BASE, "option_type": "call"}
    g1 = mc_greeks_fd(**params, n_paths=N_PATHS_FAST, seed=1)
    g2 = mc_greeks_fd(**params, n_paths=N_PATHS_FAST, seed=2)
    assert g1["delta"] != g2["delta"], (
        "Greeks are identical across seeds — `mc_price` may be ignoring `seed`."
    )
    

# 6. Input validation

@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("S0",    0.0),
        ("S0",  -100.0),
        ("K",     0.0),
        ("K",   -50.0),
        ("T",     0.0),
        ("T",    -1.0),
        ("sigma", 0.0),
        ("sigma", -0.2),
    ],
)
def test_invalid_primary_inputs_raise(field, bad_value):
    params = {**BASE, "option_type": "call", field: bad_value}
    with pytest.raises(ValueError, match=field):
        mc_greeks_fd(**params, n_paths=1_000, seed=SEED)
 
 
def test_invalid_n_paths_raises():
    params = {**BASE, "option_type": "call"}
    with pytest.raises(ValueError, match="n_paths"):
        mc_greeks_fd(**params, n_paths=0, seed=SEED)
 
 
@pytest.mark.parametrize("bump_kw", ["h_S_rel", "h_sigma", "h_r", "h_T"])
def test_nonpositive_bump_raises(bump_kw):
    params = {**BASE, "option_type": "call"}
    with pytest.raises(ValueError, match=bump_kw):
        mc_greeks_fd(**params, n_paths=1_000, seed=SEED, **{bump_kw: 0.0})
 
 
def test_negative_rate_is_allowed():
    """Negative interest rates are real; r is intentionally unconstrained."""
    params = {**BASE, "option_type": "call", "r": -0.01}
    g = mc_greeks_fd(**params, n_paths=N_PATHS_FAST, seed=SEED)
    assert all(math.isfinite(v) for v in g.values())
    